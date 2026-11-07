
import sqlite3
from pathlib import Path

import pandas as pd


# ============================================================
# PATHS AND CONFIGURATION
# ============================================================

PROCESSED_DIR = Path("data/processed")
DB_DIR = Path("database")
DB_PATH = DB_DIR / "instamart.db"

DB_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_csv(filename):
    """Load a processed CSV file and validate its existence."""

    path = PROCESSED_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Required dataset not found: {path}")

    return pd.read_csv(path)


def normalize_nulls(df):
    """Convert pandas missing values into Python None for SQL NULL."""

    return df.astype(object).where(pd.notna(df), None)


def get_table_count(connection, table_name):
    """Return the number of rows in a database table."""

    return connection.execute(
        f'SELECT COUNT(*) FROM "{table_name}"'
    ).fetchone()[0]


# ============================================================
# DATABASE CREATION
# ============================================================

def create_database():

    print("=" * 65)
    print("INSTAMART OPERATIONAL DATABASE BUILDER")
    print("=" * 65)

    # --------------------------------------------------------
    # 1. LOAD PROCESSED DATASETS
    # --------------------------------------------------------

    print("\n[1/6] Loading processed datasets...")

    customers = load_csv("customers.csv")
    products = load_csv("products.csv")
    orders = load_csv("orders.csv")
    tickets = load_csv("tickets.csv")
    scenarios = load_csv("agent_scenarios.csv")
    policy_eval = load_csv("policy_eval.csv")

    datasets = {
        "customers": customers,
        "products": products,
        "orders": orders,
        "tickets": tickets,
        "agent_scenarios": scenarios,
        "policy_eval": policy_eval,
    }

    for name, df in datasets.items():
        print(f"  {name}: {len(df)} rows")

    # --------------------------------------------------------
    # 2. VALIDATE REQUIRED COLUMNS
    # --------------------------------------------------------

    print("\n[2/6] Validating dataset columns...")

    required_columns = {
        "customers": [
            "customer_id", "city", "signup_date", "age", "gender"
        ],
        "products": [
            "product_id", "product_name", "category", "price_inr"
        ],
        "orders": [
            "order_id", "customer_id", "product_id", "quantity",
            "order_ts", "status", "discount_pct", "order_amount",
            "delivery_minutes"
        ],
        "tickets": [
            "ticket_id", "customer_id", "order_id",
            "text", "true_category", "text_missing"
        ],
        "agent_scenarios": [
            "scenario_id", "customer_id", "message",
            "expected_outcome", "expected_refund_inr"
        ],
        "policy_eval": [
            "question", "expected_doc", "expected_fact",
            "expected_fact_missing"
        ],
    }

    for name, columns in required_columns.items():

        missing_columns = set(columns) - set(datasets[name].columns)

        if missing_columns:
            raise ValueError(
                f"{name} is missing columns: {sorted(missing_columns)}"
            )

    print("  All required columns are present.")

    # --------------------------------------------------------
    # 3. NORMALIZE MISSING VALUES
    # --------------------------------------------------------

    print("\n[3/6] Normalizing missing values...")

    customers = normalize_nulls(customers)
    products = normalize_nulls(products)
    orders = normalize_nulls(orders)
    tickets = normalize_nulls(tickets)
    scenarios = normalize_nulls(scenarios)
    policy_eval = normalize_nulls(policy_eval)

    print("  Missing values converted to SQL NULL.")

    # --------------------------------------------------------
    # 4. CREATE DATABASE AND TABLES
    # --------------------------------------------------------

    print("\n[4/6] Creating SQLite database...")

    # Remove old database only after datasets are loaded
    # and validated.
    if DB_PATH.exists():
        DB_PATH.unlink()

    connection = sqlite3.connect(DB_PATH)

    try:
        connection.execute("PRAGMA foreign_keys = ON")

        # Customers
        connection.execute("""
            CREATE TABLE customers (
                customer_id TEXT PRIMARY KEY,
                city TEXT,
                signup_date TEXT,
                age REAL,
                gender TEXT
            )
        """)

        # Products
        connection.execute("""
            CREATE TABLE products (
                product_id TEXT PRIMARY KEY,
                product_name TEXT,
                category TEXT,
                price_inr REAL
            )
        """)

        # Order event history
        connection.execute("""
            CREATE TABLE order_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_id TEXT NOT NULL,
                customer_id TEXT NOT NULL,
                product_id TEXT NOT NULL,
                quantity INTEGER,
                order_ts TEXT,
                status TEXT,
                discount_pct REAL,
                order_amount REAL,
                delivery_minutes REAL,

                FOREIGN KEY (customer_id)
                    REFERENCES customers(customer_id),

                FOREIGN KEY (product_id)
                    REFERENCES products(product_id)
            )
        """)

        # Support tickets
        connection.execute("""
            CREATE TABLE tickets (
                ticket_id TEXT PRIMARY KEY,
                customer_id TEXT NOT NULL,
                order_id TEXT,
                text TEXT,
                true_category TEXT,
                text_missing INTEGER,

                FOREIGN KEY (customer_id)
                    REFERENCES customers(customer_id)
            )
        """)

        # Agent test scenarios
        connection.execute("""
            CREATE TABLE agent_scenarios (
                scenario_id TEXT PRIMARY KEY,
                customer_id TEXT,
                message TEXT,
                expected_outcome TEXT,
                expected_refund_inr REAL,

                FOREIGN KEY (customer_id)
                    REFERENCES customers(customer_id)
            )
        """)

        # Policy evaluation data
        connection.execute("""
            CREATE TABLE policy_eval (
                evaluation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT,
                expected_doc TEXT,
                expected_fact TEXT,
                expected_fact_missing INTEGER
            )
        """)

        # ----------------------------------------------------
        # 5. INSERT DATA
        # ----------------------------------------------------

        print("\n[5/6] Inserting datasets...")

        customers.to_sql(
            "customers",
            connection,
            if_exists="append",
            index=False
        )

        products.to_sql(
            "products",
            connection,
            if_exists="append",
            index=False
        )

        orders.to_sql(
            "order_events",
            connection,
            if_exists="append",
            index=False
        )

        tickets.to_sql(
            "tickets",
            connection,
            if_exists="append",
            index=False
        )

        scenarios.to_sql(
            "agent_scenarios",
            connection,
            if_exists="append",
            index=False
        )

        policy_eval.to_sql(
            "policy_eval",
            connection,
            if_exists="append",
            index=False
        )

        # ----------------------------------------------------
        # 6. CREATE CURRENT ORDERS VIEW
        # ----------------------------------------------------

        connection.execute("""
            CREATE VIEW current_orders AS

            SELECT *
            FROM (
                SELECT
                    *,
                    ROW_NUMBER() OVER (
                        PARTITION BY order_id
                        ORDER BY order_ts DESC, event_id DESC
                    ) AS row_num

                FROM order_events
            )

            WHERE row_num = 1
        """)

        # ----------------------------------------------------
        # CREATE INDEXES
        # ----------------------------------------------------

        connection.execute("""
            CREATE INDEX idx_orders_customer
            ON order_events(customer_id)
        """)

        connection.execute("""
            CREATE INDEX idx_orders_product
            ON order_events(product_id)
        """)

        connection.execute("""
            CREATE INDEX idx_orders_order_id
            ON order_events(order_id)
        """)

        connection.execute("""
            CREATE INDEX idx_tickets_customer
            ON tickets(customer_id)
        """)

        connection.execute("""
            CREATE INDEX idx_scenarios_customer
            ON agent_scenarios(customer_id)
        """)

        connection.commit()

        # ----------------------------------------------------
        # DATABASE VALIDATION
        # ----------------------------------------------------

        print("\n[6/6] Validating database...")

        integrity = connection.execute(
            "PRAGMA integrity_check"
        ).fetchone()[0]

        if integrity != "ok":
            raise RuntimeError(
                f"Database integrity failed: {integrity}"
            )

        foreign_key_errors = connection.execute(
            "PRAGMA foreign_key_check"
        ).fetchall()

        if foreign_key_errors:
            raise RuntimeError(
                f"Foreign-key violations: {foreign_key_errors}"
            )

        expected_counts = {
            "customers": len(customers),
            "products": len(products),
            "order_events": len(orders),
            "tickets": len(tickets),
            "agent_scenarios": len(scenarios),
            "policy_eval": len(policy_eval),
        }

        for table, expected in expected_counts.items():

            actual = get_table_count(connection, table)

            if actual != expected:
                raise RuntimeError(
                    f"{table}: expected {expected}, got {actual}"
                )

        # Validate current_orders view
        current_count = get_table_count(
            connection, "current_orders"
        )

        unique_orders = orders["order_id"].nunique()

        if current_count != unique_orders:
            raise RuntimeError(
                f"current_orders view mismatch: "
                f"expected {unique_orders}, got {current_count}"
            )

        # ----------------------------------------------------
        # FINAL SUMMARY
        # ----------------------------------------------------

        print("\n" + "=" * 65)
        print("DATABASE CREATED AND VALIDATED SUCCESSFULLY")
        print("=" * 65)

        print("\nTABLE SUMMARY")

        for table, expected in expected_counts.items():
            print(f"{table:<20} {expected:>8} rows")

        print(f"{'current_orders':<20} {current_count:>8} rows")

        print("\nIntegrity check:", integrity)
        print("Foreign-key violations:", len(foreign_key_errors))
        print("Database path:", DB_PATH.resolve())

        print("\nALL DATABASE CHECKS PASSED")

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    create_database()
