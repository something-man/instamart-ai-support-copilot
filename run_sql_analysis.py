
from pathlib import Path
import sqlite3
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "instamart.db"
SQL_PATH = BASE_DIR / "sql" / "queries.sql"

sql_text = SQL_PATH.read_text(encoding="utf-8")

queries = [
    query.strip()
    for query in sql_text.split(";")
    if query.strip()
]

conn = sqlite3.connect(DB_PATH)

for i, query in enumerate(queries, start=1):

    print("\n" + "=" * 65)
    print(f"Q{i} RESULTS")
    print("=" * 65)

    df = pd.read_sql_query(query, conn)

    print("Rows returned:", len(df))

    if i == 6:
        print("\nTop customer in each city:")
        print(
            df[df["spend_rank"] == 1].to_string(index=False)
        )

        print("\nRanking comparison:")
        print(
            "RANK vs ROW_NUMBER ties:",
            int((df["spend_rank"] != df["row_num"]).sum())
        )

    else:
        print(df.to_string(index=False))

    if i == 2:
        print(
            "\nTotal revenue:",
            round(df["revenue"].sum(), 2)
        )

    if i == 5:
        print(
            "\nAverage customer spend:",
            df["average_spend"].iloc[0] if not df.empty else 0
        )

    if i == 7 and not df.empty:
        print(
            "\nFinal cumulative revenue:",
            df["cumulative_revenue"].iloc[-1]
        )

conn.close()

print("\nALL SEVEN SQL QUERIES EXECUTED.")
