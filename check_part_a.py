
import pandas as pd
import sqlite3

customers = pd.read_csv("data/clean/customers.csv")
orders = pd.read_csv("data/clean/orders.csv")

print("CUSTOMER CHECKS")
print("\nCities:")
print(customers["city"].value_counts().to_string())

print("\nGender:")
print(customers["gender"].value_counts().to_string())

print("\nAge range:", customers["age"].min(), "-", customers["age"].max())
print("Missing ages:", customers["age"].isna().sum())

print("\nORDER CHECKS")
print("Missing amounts:", orders["order_amount"].isna().sum())

completed = orders["status"] == "completed"
cancelled = orders["status"] == "cancelled"

print(
    "Completed missing delivery:",
    orders.loc[completed, "delivery_minutes"].isna().sum()
)
print(
    "Cancelled missing delivery:",
    orders.loc[cancelled, "delivery_minutes"].isna().sum()
)

print("\nDATABASE CHECKS")

with sqlite3.connect("data/instamart.db") as conn:
    tables = pd.read_sql_query(
        "SELECT name FROM sqlite_master WHERE type='table'",
        conn
    )

    print("Tables:", tables["name"].tolist())
    print("Integrity:", conn.execute("PRAGMA integrity_check").fetchone()[0])
    print(
        "Foreign key violations:",
        len(conn.execute("PRAGMA foreign_key_check").fetchall())
    )
