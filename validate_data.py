
from pathlib import Path
import sqlite3
import pandas as pd

CLEAN_DIR = Path("data/clean")
DB_PATH = Path("data/instamart.db")

print("\nINSTAMART PART A - FORMAL VALIDATION")
print("=" * 55)

customers = pd.read_csv(CLEAN_DIR / "customers.csv")
products = pd.read_csv(CLEAN_DIR / "products.csv")
orders = pd.read_csv(CLEAN_DIR / "orders.csv")

checks_passed = 0


def check(description, condition):
    global checks_passed

    assert condition, f"FAILED: {description}"
    checks_passed += 1
    print(f"PASS {checks_passed}: {description}")


# 1. Dataset row counts
check("Customer count is 1200", len(customers) == 1200)
check("Product count is 40", len(products) == 40)
check("Order count is 12309", len(orders) == 12309)


# 2. Unique identifiers
check(
    "Customer IDs are unique",
    customers["customer_id"].is_unique
)
check(
    "Product IDs are unique",
    products["product_id"].is_unique
)
check(
    "Order IDs are unique",
    orders["order_id"].is_unique
)


# 3. Customer cleaning
expected_cities = {
    "Pune": 227,
    "Delhi": 210,
    "Bengaluru": 197,
    "Chennai": 191,
    "Hyderabad": 191,
    "Mumbai": 184
}

check(
    "Exactly six standardized cities",
    customers["city"].nunique() == 6
)

check(
    "City distribution matches reference",
    customers["city"].value_counts().to_dict() == expected_cities
)

expected_gender = {
    "female": 610,
    "male": 560,
    "unknown": 30
}

check(
    "Gender distribution matches reference",
    customers["gender"].value_counts().to_dict() == expected_gender
)

check(
    "No missing customer ages",
    customers["age"].isna().sum() == 0
)

check(
    "All customer ages are between 10 and 100",
    customers["age"].between(10, 100).all()
)


# 4. Order cleaning
expected_status = {
    "completed": 11110,
    "cancelled": 1199
}

check(
    "Order status distribution matches reference",
    orders["status"].value_counts().to_dict() == expected_status
)

check(
    "No pending orders",
    not orders["status"].eq("pending").any()
)

check(
    "No missing order amounts",
    orders["order_amount"].isna().sum() == 0
)

completed = orders["status"].eq("completed")
cancelled = orders["status"].eq("cancelled")

check(
    "Completed orders have no missing delivery times",
    orders.loc[completed, "delivery_minutes"].isna().sum() == 0
)

check(
    "All cancelled orders have missing delivery times",
    orders.loc[cancelled, "delivery_minutes"].isna().all()
)

check(
    "Completed revenue matches INR 2,998,668.80",
    round(orders.loc[completed, "order_amount"].sum(), 2)
    == 2998668.80
)


# 5. Referential integrity in CSV data
check(
    "Every order references an existing customer",
    orders["customer_id"].isin(customers["customer_id"]).all()
)

check(
    "Every order references an existing product",
    orders["product_id"].isin(products["product_id"]).all()
)


# 6. SQLite database validation
with sqlite3.connect(DB_PATH) as conn:

    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }

    check(
        "Database contains exactly the three required tables",
        tables == {"customers", "products", "orders"}
    )

    check(
        "SQLite integrity check is OK",
        conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    )

    check(
        "No SQLite foreign-key violations",
        len(conn.execute("PRAGMA foreign_key_check").fetchall()) == 0
    )

    for table, expected_count in {
        "customers": 1200,
        "products": 40,
        "orders": 12309
    }.items():

        actual_count = conn.execute(
            f"SELECT COUNT(*) FROM {table}"
        ).fetchone()[0]

        check(
            f"Database table {table} contains {expected_count} rows",
            actual_count == expected_count
        )


print("\n" + "=" * 55)
print(f"CHECKPOINTS PASSED: {checks_passed}")
print("ALL CHECKPOINTS PASSED")
print("=" * 55)
