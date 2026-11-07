
import pandas as pd

orders = pd.read_csv("data/raw/orders.csv")
customers = pd.read_csv("data/raw/customers.csv")
tickets = pd.read_csv("data/raw/tickets.csv")

print("=== ORDER DUPLICATES ===")
print("Exact duplicate rows:", orders.duplicated().sum())
print("Rows with repeated order IDs:", orders["order_id"].duplicated(keep=False).sum())

repeated = orders[
    orders["order_id"].duplicated(keep=False)
].sort_values("order_id")

print("\nFirst 15 rows with repeated order IDs:")
print(repeated.head(15).to_string(index=False))

print("\n=== INVALID CUSTOMER AGES ===")
print(
    customers.loc[
        customers["age"].notna()
        & ~customers["age"].between(18, 100),
        ["customer_id", "age"]
    ].to_string(index=False)
)

print("\n=== INVALID DELIVERY TIMES ===")
print(
    orders.loc[
        orders["delivery_minutes"].notna()
        & (orders["delivery_minutes"] < 0),
        ["order_id", "delivery_minutes"]
    ].head(20).to_string(index=False)
)

print("\n=== MISSING TICKET TEXT ===")
print(
    tickets.loc[
        tickets["text"].isna(),
        ["ticket_id", "customer_id", "order_id"]
    ].to_string(index=False)
)