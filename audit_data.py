from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw")


def print_file_audit(path):
    df = pd.read_csv(path)

    print("\n" + "=" * 65)
    print(f"FILE: {path.name}")
    print(f"Rows: {len(df)} | Columns: {len(df.columns)}")

    # Missing values
    missing = df.isna().sum()
    print("\nMissing values:")
    print(missing[missing > 0].to_string() if missing.any() else "None")

    # Exact duplicate rows
    print("\nExact duplicate rows:", int(df.duplicated().sum()))

    # Identifier checks
    id_columns = [
        col for col in df.columns
        if col == "id" or col.endswith("_id")
    ]

    print("\nIdentifier checks:")
    if id_columns:
        for col in id_columns:
            print(
                f"  {col}: "
                f"{int(df[col].duplicated().sum())} duplicate values"
            )
    else:
        print("  No identifier-like columns")

    # Numeric ranges
    print("\nNumeric ranges:")
    numeric_columns = df.select_dtypes(include="number").columns

    for col in numeric_columns:
        print(f"  {col}: min={df[col].min()}, max={df[col].max()}")


def audit_instamart_specific():
    print("\n" + "=" * 65)
    print("INSTAMART-SPECIFIC DATA QUALITY CHECKS")

    # Customer checks
    customers = pd.read_csv(
        RAW_DIR / "customers.csv",
        dtype={"age": "string"}
    )

    age = pd.to_numeric(customers["age"], errors="coerce")

    print("\nCUSTOMERS")
    print("Invalid/non-numeric ages:", int(age.isna().sum()))
    print("Ages outside 10-100:", int((~age.between(10, 100)).sum()))

    # Order checks
    orders = pd.read_csv(
        RAW_DIR / "orders.csv",
        dtype={
            "order_amount": "string",
            "delivery_minutes": "string"
        }
    )

    amount = pd.to_numeric(orders["order_amount"], errors="coerce")
    delivery = pd.to_numeric(
        orders["delivery_minutes"], errors="coerce"
    )

    print("\nORDERS")
    print("Exact duplicate rows:", int(orders.duplicated().sum()))
    print("Duplicate order IDs:", int(orders["order_id"].duplicated().sum()))

    print(
        "Missing order amounts:",
        int(orders["order_amount"].isna().sum())
    )
    print(
        "Non-numeric order amounts:",
        int((orders["order_amount"].notna() & amount.isna()).sum())
    )

    print(
        "Blank/missing delivery times:",
        int(orders["delivery_minutes"].isna().sum())
    )
    print(
        "Non-numeric delivery times:",
        int((orders["delivery_minutes"].notna() & delivery.isna()).sum())
    )
    print("Negative delivery times:", int((delivery < 0).sum()))

    if "status" in orders.columns:
        print("\nOrder status distribution:")
        print(orders["status"].value_counts(dropna=False).to_string())
        print(
            "Pending orders:",
            int(
                orders["status"]
                .astype("string")
                .str.strip()
                .str.lower()
                .eq("pending")
                .sum()
            )
        )


if __name__ == "__main__":
    for csv_path in sorted(RAW_DIR.glob("*.csv")):
        print_file_audit(csv_path)

    audit_instamart_specific()