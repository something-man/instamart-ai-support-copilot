
from pathlib import Path
import pandas as pd
import sqlite3

RAW_DIR = Path("data/raw")
CLEAN_DIR = Path("data/clean")
CLEAN_DIR.mkdir(parents=True, exist_ok=True)


# 1. LOAD AND AUDIT RAW DATA

def load_data():
    customers = pd.read_csv(RAW_DIR / "customers.csv")
    products = pd.read_csv(RAW_DIR / "products.csv")
    orders = pd.read_csv(RAW_DIR / "orders.csv")

    # Preserve blank ticket text if tickets are loaded later.
    tickets = pd.read_csv(
        RAW_DIR / "tickets.csv",
        keep_default_na=False
    )

    return customers, products, orders, tickets


def audit_data(datasets):
    print("\nRAW DATA AUDIT")
    for name, df in datasets.items():
        print(f"\n{'=' * 55}\n{name.upper()}")
        print("Shape:", df.shape)
        print("\nData types:\n", df.dtypes)
        missing = df.isna().sum()
        report = pd.DataFrame({
            "missing_count": missing,
            "missing_percent": (missing / len(df) * 100).round(2)
        })
        print("\nMissing values:\n", report)


# 2. CLEAN CUSTOMERS

def clean_customers(df):
    df = df.drop_duplicates().copy()

    df["city"] = (
        df["city"].astype("string").str.strip().str.title()
    )

    df["gender"] = (
        df["gender"].astype("string").str.strip().str.lower()
    )

    gender_map = {
        "m": "male",
        "male": "male",
        "man": "male",
        "f": "female",
        "female": "female",
        "woman": "female"
    }

    df["gender"] = df["gender"].map(gender_map).fillna("unknown")

    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df.loc[~df["age"].between(10, 100), "age"] = pd.NA

    df["age"] = df["age"].fillna(
        df.groupby("city")["age"].transform("median")
    )

    df["signup_date"] = pd.to_datetime(
        df["signup_date"], errors="coerce"
    )

    return df


# 3. CLEAN PRODUCTS

def clean_products(df):
    df = df.drop_duplicates().copy()

    df["product_name"] = (
        df["product_name"].astype("string").str.strip()
    )
    df["category"] = (
        df["category"].astype("string").str.strip().str.title()
    )
    df["price_inr"] = pd.to_numeric(
        df["price_inr"], errors="coerce"
    )

    return df


# 4. CLEAN ORDERS

def clean_orders(df, products):
    df = df.copy()

    # Convert timestamp before selecting latest order record.
    df["order_ts"] = pd.to_datetime(
        df["order_ts"], errors="coerce"
    )

    # Remove exact duplicate rows.
    df = df.drop_duplicates()

    numeric_columns = [
        "quantity", "discount_pct",
        "order_amount", "delivery_minutes"
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["status"] = (
        df["status"].astype("string").str.strip().str.lower()
    )

    # Keep latest event for each order.
    df = df.sort_values(
        ["order_id", "order_ts"],
        kind="stable"
    )
    df = df.drop_duplicates(
        subset=["order_id"], keep="last"
    ).copy()

    # Merge product price and category.
    product_info = products[
        ["product_id", "price_inr", "category"]
    ]

    df = df.merge(
        product_info,
        on="product_id",
        how="left",
        validate="many_to_one"
    )

    # Recalculate missing order amounts.
    missing_amount = df["order_amount"].isna()

    calculated_amount = (
        df["price_inr"]
        * df["quantity"]
        * (1 - df["discount_pct"] / 100)
    ).round(2)

    df.loc[missing_amount, "order_amount"] = (
        calculated_amount[missing_amount]
    )

    # Remove invalid negative delivery times.
    df.loc[
        df["delivery_minutes"] < 0,
        "delivery_minutes"
    ] = pd.NA

    # Fill missing delivery times only for completed orders.
    completed = df["status"].eq("completed")

    category_median = (
        df.loc[completed]
        .groupby("category")["delivery_minutes"]
        .transform("median")
    )

    completed_index = df.index[completed]
    missing_delivery = df.loc[
        completed_index, "delivery_minutes"
    ].isna()

    df.loc[
        completed_index[missing_delivery],
        "delivery_minutes"
    ] = category_median.loc[
        missing_delivery
    ].values

    # Cancelled orders must have missing delivery time.
    df.loc[
        df["status"].eq("cancelled"),
        "delivery_minutes"
    ] = pd.NA

    # Remove temporary product lookup columns.
    df = df.drop(columns=["price_inr", "category"])

    return df.sort_values("order_id").reset_index(drop=True)


# 5. SAVE CLEAN CSV FILES

def save_clean_data(customers, products, orders):
    customers.to_csv(CLEAN_DIR / "customers.csv", index=False)
    products.to_csv(CLEAN_DIR / "products.csv", index=False)
    orders.to_csv(CLEAN_DIR / "orders.csv", index=False)


# 6. STORE IN SQLITE

def create_database(customers, products, orders):
    db_path = Path("data/instamart.db")

    with sqlite3.connect(db_path) as conn:
        customers.to_sql(
            "customers", conn, if_exists="replace", index=False
        )
        products.to_sql(
            "products", conn, if_exists="replace", index=False
        )

        db_orders = orders.copy()
        db_orders["order_ts"] = (
            db_orders["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
        )

        db_orders.to_sql(
            "orders", conn, if_exists="replace", index=False
        )

        conn.execute("CREATE INDEX IF NOT EXISTS idx_orders_customer ON orders(customer_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_orders_product ON orders(product_id)")


# 7. MAIN

def main():
    print("INSTAMART PART A - DATA FOUNDATION")

    customers_raw, products_raw, orders_raw, tickets = load_data()

    audit_data({
        "customers": customers_raw,
        "products": products_raw,
        "orders": orders_raw,
        "tickets": tickets
    })

    customers = clean_customers(customers_raw)
    products = clean_products(products_raw)
    orders = clean_orders(orders_raw, products)

    save_clean_data(customers, products, orders)
    create_database(customers, products, orders)

    print("\nCLEANING SUMMARY")
    print("Customers:", len(customers))
    print("Products:", len(products))
    print("Orders:", len(orders))
    print("\nOrder status counts:\n", orders["status"].value_counts())
    print("\nCompleted revenue INR:",
          round(orders.loc[
              orders["status"].eq("completed"), "order_amount"
          ].sum(), 2))

    print("\nPART A DATA CLEANING AND DATABASE CREATION COMPLETED")


if __name__ == "__main__":
    main()
