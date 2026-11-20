
# ============================================================
# DELIVERIQ - FOOD DELIVERY ANALYTICS
# Stage 1: Data Collection & Initial Data Inspection
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path

# ------------------------------------------------------------
# 1. PROJECT PATHS
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "raw" / "Zomato Dataset.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("\n" + "=" * 65)
print("             DELIVERIQ - FOOD DELIVERY ANALYTICS")
print("=" * 65)

# ------------------------------------------------------------
# 2. DATA COLLECTION
# ------------------------------------------------------------

print("\n[1] DATA COLLECTION")

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"\nDataset not found: {DATA_PATH}\n"
        "Place Zomato Dataset.csv inside data/raw/"
    )

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("File Name:", DATA_PATH.name)

# ------------------------------------------------------------
# 3. DATASET DIMENSIONS
# ------------------------------------------------------------

print("\n[2] DATASET DIMENSIONS")

print("Total Rows:", df.shape[0])
print("Total Columns:", df.shape[1])

# ------------------------------------------------------------
# 4. COLUMN INFORMATION
# ------------------------------------------------------------

print("\n[3] COLUMN INFORMATION")

for number, column in enumerate(df.columns, start=1):
    print(f"{number}. {column}")

# ------------------------------------------------------------
# 5. DATA TYPES
# ------------------------------------------------------------

print("\n[4] DATA TYPES")

print(df.dtypes.to_string())

# ------------------------------------------------------------
# 6. MISSING VALUES ANALYSIS
# ------------------------------------------------------------

print("\n[5] MISSING VALUES ANALYSIS")

missing_report = pd.DataFrame({
    "Column": df.columns,
    "Missing_Values": df.isnull().sum().values,
    "Missing_Percentage": (
        df.isnull().mean().values * 100
    ).round(2)
})

print(missing_report.to_string(index=False))

missing_report.to_csv(
    OUTPUT_DIR / "missing_values_report.csv",
    index=False
)

# ------------------------------------------------------------
# 7. DUPLICATE RECORDS
# ------------------------------------------------------------

print("\n[6] DUPLICATE RECORD ANALYSIS")

duplicate_count = df.duplicated().sum()

print("Total Duplicate Rows:", duplicate_count)

# ------------------------------------------------------------
# 8. STATISTICAL SUMMARY
# ------------------------------------------------------------

print("\n[7] STATISTICAL SUMMARY")

print(df.describe(include="all").T.to_string())

# ------------------------------------------------------------
# 9. UNIQUE VALUES
# ------------------------------------------------------------

print("\n[8] UNIQUE VALUES ANALYSIS")

categorical_columns = df.select_dtypes(
    include=["object"]
).columns

for column in categorical_columns:

    print(f"\n{column}")
    print("Unique Values:", df[column].nunique())

    print(
        "Sample Categories:",
        df[column].dropna().unique()[:10]
    )

# ------------------------------------------------------------
# 10. FIRST FIVE RECORDS
# ------------------------------------------------------------

print("\n[9] SAMPLE DATA")

print(df.head().to_string(index=False))

# ------------------------------------------------------------
# 11. LAST FIVE RECORDS
# ------------------------------------------------------------

print("\n[10] LAST FIVE RECORDS")

print(df.tail().to_string(index=False))

# ------------------------------------------------------------
# 12. DATASET MEMORY USAGE
# ------------------------------------------------------------

print("\n[11] MEMORY USAGE")

memory_usage = df.memory_usage(deep=True).sum() / (1024 ** 2)

print(f"Dataset Memory Usage: {memory_usage:.2f} MB")

# ------------------------------------------------------------
# 13. SAVE INSPECTION REPORT
# ------------------------------------------------------------

report_path = OUTPUT_DIR / "stage1_inspection_report.txt"

with open(report_path, "w", encoding="utf-8") as report:

    report.write("DELIVERIQ - DATA INSPECTION REPORT\n")
    report.write("=" * 55 + "\n\n")

    report.write(f"Total Rows: {df.shape[0]}\n")
    report.write(f"Total Columns: {df.shape[1]}\n")
    report.write(f"Duplicate Rows: {duplicate_count}\n")
    report.write(f"Memory Usage: {memory_usage:.2f} MB\n\n")

    report.write("COLUMN INFORMATION\n")
    report.write(df.dtypes.to_string())

    report.write("\n\nMISSING VALUES REPORT\n")
    report.write(missing_report.to_string(index=False))

    report.write("\n\nSTATISTICAL SUMMARY\n")
    report.write(df.describe(include="all").T.to_string())

print("\nInspection reports saved in outputs/ folder.")

# ------------------------------------------------------------
# 14. FINAL STATUS
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("       STAGE 1 - COMPLETED SUCCESSFULLY!")
print("=" * 65)

print("\nGenerated Files:")
print("- outputs/missing_values_report.csv")
print("- outputs/stage1_inspection_report.txt")

print("\nNext Stage: Data Understanding & Quality Assessment")
print("=" * 65)
