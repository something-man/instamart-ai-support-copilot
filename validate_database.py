
import sqlite3
from pathlib import Path

DB_PATH = Path("database/instamart.db")


def validate_database():

    if not DB_PATH.exists():
        raise FileNotFoundError(DB_PATH)

    connection = sqlite3.connect(DB_PATH)
    connection.execute("PRAGMA foreign_keys = ON")

    print("\nINSTAMART DATABASE VALIDATION")
    print("=" * 55)

    # 1. SQLite integrity
    integrity = connection.execute(
        "PRAGMA integrity_check"
    ).fetchone()[0]

    print("\n1. DATABASE INTEGRITY")
    print("Integrity check:", integrity)

    # 2. Foreign key validation
    foreign_errors = connection.execute(
        "PRAGMA foreign_key_check"
    ).fetchall()

    print("\n2. FOREIGN KEY VALIDATION")
    print("Foreign key violations:", len(foreign_errors))

    for error in foreign_errors[:10]:
        print(error)

    # 3. Table row counts
    tables = [
        "customers",
        "products",
        "order_events",
        "tickets",
        "agent_scenarios",
        "policy_eval"
    ]

    print("\n3. TABLE ROW COUNTS")

    for table in tables:
        count = connection.execute(
            f"SELECT COUNT(*) FROM {table}"
        ).fetchone()[0]

        print(f"{table}: {count}")

    # 4. Current orders view
    current_count = connection.execute(
        "SELECT COUNT(*) FROM current_orders"
    ).fetchone()[0]

    unique_orders = connection.execute(
        "SELECT COUNT(DISTINCT order_id) FROM order_events"
    ).fetchone()[0]

    print("\n4. CURRENT ORDER VIEW")
    print("Current order rows:", current_count)
    print("Distinct order IDs:", unique_orders)
    print("View matches distinct IDs:", current_count == unique_orders)

    # 5. NULL value checks
    print("\n5. NULL VALUE CHECKS")

    checks = {
        "customers.age": """
            SELECT COUNT(*) FROM customers WHERE age IS NULL
        """,
        "orders.order_amount": """
            SELECT COUNT(*) FROM order_events
            WHERE order_amount IS NULL
        """,
        "orders.delivery_minutes": """
            SELECT COUNT(*) FROM order_events
            WHERE delivery_minutes IS NULL
        """,
        "tickets.text": """
            SELECT COUNT(*) FROM tickets
            WHERE text IS NULL OR TRIM(text) = ''
        """
    }

    for label, query in checks.items():
        count = connection.execute(query).fetchone()[0]
        print(f"{label}: {count}")

    # 6. Status distribution
    print("\n6. ORDER STATUS DISTRIBUTION")

    results = connection.execute("""
        SELECT status, COUNT(*)
        FROM current_orders
        GROUP BY status
        ORDER BY status
    """).fetchall()

    for status, count in results:
        print(f"{status}: {count}")

    connection.close()

    print("\n" + "=" * 55)
    print("DATABASE VALIDATION COMPLETED")


if __name__ == "__main__":
    validate_database()
