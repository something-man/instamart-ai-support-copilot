
import pandas as pd
import sqlite3

# Load cleaned datasets
customers = pd.read_csv("data/clean/customers.csv")
products = pd.read_csv("data/clean/products.csv")
orders = pd.read_csv("data/clean/orders.csv")

completed = orders[orders["status"] == "completed"].copy()

# Merge city information
completed = completed.merge(
    customers[["customer_id", "city"]],
    on="customer_id",
    how="left"
)

completed = completed.merge(
    products[["product_id", "category"]],
    on="product_id",
    how="left"
)

print("\nPART A - PANDAS EXPLORATORY ANALYSIS")
print("=" * 65)


# 1. PIVOT TABLE: REVENUE BY CITY AND CATEGORY

print("\n1. REVENUE PIVOT TABLE")

pivot = pd.pivot_table(
    completed,
    values="order_amount",
    index="city",
    columns="category",
    aggfunc="sum",
    fill_value=0,
    margins=True,
    margins_name="All"
)

print(pivot.round(2).to_string())

pivot_total = pivot.loc["All", "All"]
print("\nPivot grand total INR:", round(pivot_total, 2))

with sqlite3.connect("data/instamart.db") as conn:
    sql_total = pd.read_sql_query(
        """
        SELECT SUM(order_amount) AS revenue
        FROM orders
        WHERE status = 'completed'
        """,
        conn
    ).iloc[0]["revenue"]

print("SQL grand total INR:", round(sql_total, 2))
print("Pivot matches SQL:", round(pivot_total, 2) == round(sql_total, 2))


# 2. CITY-WISE REVENUE ANALYSIS

print("\n2. CITY-WISE REVENUE")

city_summary = (
    completed.groupby("city")
    .agg(
        total_revenue=("order_amount", "sum"),
        average_order_value=("order_amount", "mean"),
        completed_orders=("order_amount", "count")
    )
    .round(2)
    .sort_values("total_revenue", ascending=False)
)

print(city_summary.to_string())


# 3. DESCRIPTIVE STATISTICS AND OUTLIERS

print("\n3. ORDER AMOUNT DISTRIBUTION")

amount = completed["order_amount"]

mean = amount.mean()
median = amount.median()
skewness = amount.skew()

q1 = amount.quantile(0.25)
q3 = amount.quantile(0.75)
iqr = q3 - q1

lower_fence = q1 - 1.5 * iqr
upper_fence = q3 + 1.5 * iqr

below_outliers = int((amount < lower_fence).sum())
above_outliers = int((amount > upper_fence).sum())

print("Mean INR:", round(mean, 2))
print("Median INR:", round(median, 2))
print("Skewness:", round(skewness, 2))
print("Q1 INR:", round(q1, 2))
print("Q3 INR:", round(q3, 2))
print("IQR INR:", round(iqr, 2))
print("Lower fence INR:", round(lower_fence, 2))
print("Upper fence INR:", round(upper_fence, 2))
print("Outliers below lower fence:", below_outliers)
print("Outliers above upper fence:", above_outliers)


# 4. CORRELATION ANALYSIS

print("\n4. CORRELATION MATRIX")

correlation_columns = [
    "order_amount",
    "discount_pct",
    "quantity",
    "delivery_minutes"
]

correlation = completed[correlation_columns].corr()

print(correlation.round(3).to_string())

# Find strongest absolute off-diagonal correlation.
absolute_corr = correlation.abs()
for column in absolute_corr.columns:
    absolute_corr.loc[column, column] = 0

strongest_pair = absolute_corr.stack().idxmax()
strongest_value = correlation.loc[
    strongest_pair[0], strongest_pair[1]
]

print("\nStrongest absolute correlation:")
print(strongest_pair, "=", round(strongest_value, 3))

print("\nPANDAS ANALYSIS COMPLETED")
