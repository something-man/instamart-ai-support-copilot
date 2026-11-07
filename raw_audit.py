
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"

files = sorted(RAW_DIR.glob("*.csv"))

summary = []
column_audit = []

for file in files:
    df = pd.read_csv(file)

    rows, cols = df.shape

    summary.append({
        "Dataset": file.name,
        "Rows": rows,
        "Columns": cols
    })

    for column in df.columns:
        missing = int(df[column].isna().sum())

        column_audit.append({
            "Dataset": file.name,
            "Column": column,
            "Data Type": str(df[column].dtype),
            "Missing Count": missing,
            "Missing Percentage": round(
                (missing / rows) * 100, 2
            ) if rows else 0
        })

print("\n========== DATASET DIMENSIONS ==========")
print(pd.DataFrame(summary).to_string(index=False))

print("\n========== COLUMN-LEVEL DATA AUDIT ==========")
print(pd.DataFrame(column_audit).to_string(index=False))

print("\n========== TOTAL MISSING VALUES ==========")

for file in files:
    df = pd.read_csv(file)
    print(
        f"{file.name}: "
        f"{df.isna().sum().sum()} missing values"
    )

# Save the audit results for project documentation.
pd.DataFrame(summary).to_csv(
    BASE_DIR / "outputs_raw_dimensions.csv",
    index=False
)

pd.DataFrame(column_audit).to_csv(
    BASE_DIR / "outputs_raw_column_audit.csv",
    index=False
)

print("\nAudit CSV files saved successfully.")
