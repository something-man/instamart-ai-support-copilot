
from pathlib import Path
import re
import sqlite3
import pandas as pd

SQL_FILE = Path("sql/queries.sql")
DB_FILE = Path("data/instamart.db")

sql_text = SQL_FILE.read_text(encoding="utf-8")

# Split the file into seven numbered SQL queries.
parts = re.split(r"(?=-- Q[1-7]\.)", sql_text)

queries = {}

for part in parts:
    match = re.match(r"-- Q([1-7])\.", part.strip())
    if match:
        number = int(match.group(1))
        query = re.sub(r"^--.*$", "", part, flags=re.MULTILINE).strip()
        if query:
            queries[number] = query

assert len(queries) == 7, (
    f"Expected 7 SQL queries, found {len(queries)}"
)

results = {}

with sqlite3.connect(DB_FILE) as conn:
    for number, query in queries.items():
        results[number] = pd.read_sql_query(query, conn)

print("\nINSTAMART SQL ANALYSIS")
print("=" * 60)

for number, df in results.items():
    print(f"\nQ{number} RESULT")
    print("Rows:", len(df))

    if number == 6:
        print(df.head(10).to_string(index=False))
    elif number == 7:
        print(df.tail(5).to_string(index=False))
    else:
        print(df.to_string(index=False))


# Reference checkpoint validation

checks = []

def check(description, condition):
    assert condition, f"FAILED: {description}"
    checks.append(description)
    print("PASS:", description)


q1 = results[1]
q2 = results[2]
q3 = results[3]
q4 = results[4]
q5 = results[5]
q6 = results[6]
q7 = results[7]

check("Q1 returns 30 customers", len(q1) == 30)

check("Q2 returns 30 city-category combinations", len(q2) == 30)

check(
    "Q2 total revenue matches INR 2,998,668.80",
    round(q2["revenue"].sum(), 2) == 2998668.80
)

largest_cell = q2.loc[q2["revenue"].idxmax()]

check(
    "Q2 largest revenue cell is Bengaluru / Electronics",
    largest_cell["city"] == "Bengaluru"
    and largest_cell["category"] == "Electronics"
    and round(largest_cell["revenue"], 2) == 233909.45
)

check("Q3 returns 11 products", len(q3) == 11)

check("Q4 returns 25 customers", len(q4) == 25)

check("Q5 returns 550 customers", len(q5) == 550)

check(
    "Q5 average customer spend is INR 2,552.06",
    round(q5["average_spend"].iloc[0], 2) == 2552.06
)

expected_top = {
    "Bengaluru": ("C0082", 12856.20),
    "Chennai": ("C0930", 10151.40),
    "Delhi": ("C0291", 13742.60),
    "Hyderabad": ("C0743", 12730.05),
    "Mumbai": ("C0693", 14766.55),
    "Pune": ("C0426", 10377.10)
}

actual_top = q6[q6["spend_rank"] == 1]

top_matches = True

for city, (customer_id, spend) in expected_top.items():
    row = actual_top[actual_top["city"] == city]

    if len(row) != 1:
        top_matches = False
        break

    if (
        row.iloc[0]["customer_id"] != customer_id
        or round(row.iloc[0]["total_spend"], 2) != spend
    ):
        top_matches = False
        break

check("Q6 top customer in each city matches reference", top_matches)

rank_differences = int((q6["spend_rank"] != q6["row_num"]).sum())

check(
    "Q6 has exactly 8 rows where RANK differs from ROW_NUMBER",
    rank_differences == 8
)

check("Q7 returns 183 daily rows", len(q7) == 183)

check(
    "Q7 final cumulative revenue matches INR 2,998,668.80",
    round(q7["cumulative_revenue"].iloc[-1], 2) == 2998668.80
)

print("\n" + "=" * 60)
print(f"SQL CHECKPOINTS PASSED: {len(checks)}")
print("ALL SQL CHECKPOINTS PASSED")
print("=" * 60)
