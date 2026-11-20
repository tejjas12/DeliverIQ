
# ============================================================
# DELIVERIQ
# STAGE 2: DATA UNDERSTANDING & QUALITY ASSESSMENT
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "raw" / "Zomato Dataset.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

print("\n" + "=" * 65)
print("       DELIVERIQ - DATA QUALITY ASSESSMENT")
print("=" * 65)

# Load original dataset
df = pd.read_csv(DATA_PATH)

# ------------------------------------------------------------
# 1. BASIC DATASET INFORMATION
# ------------------------------------------------------------

print("\n[1] DATASET INFORMATION")

print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\nColumn Data Types:")
print(df.dtypes.to_string())

# ------------------------------------------------------------
# 2. MISSING VALUE ANALYSIS
# ------------------------------------------------------------

print("\n[2] MISSING VALUE ANALYSIS")

missing_report = pd.DataFrame({
    "Column": df.columns,
    "Missing_Count": df.isnull().sum().values,
    "Missing_Percentage": (
        df.isnull().mean().values * 100
    ).round(2)
})

print(missing_report.to_string(index=False))

# ------------------------------------------------------------
# 3. DUPLICATE ANALYSIS
# ------------------------------------------------------------

print("\n[3] DUPLICATE ANALYSIS")

print("Duplicate Rows:", df.duplicated().sum())
print("Duplicate Order IDs:", df["ID"].duplicated().sum())

# ------------------------------------------------------------
# 4. NUMERICAL FEATURE ANALYSIS
# ------------------------------------------------------------

print("\n[4] NUMERICAL FEATURE ANALYSIS")

numerical_columns = df.select_dtypes(
    include=np.number
).columns

print(df[numerical_columns].describe().T.to_string())

# ------------------------------------------------------------
# 5. AGE VALIDATION
# ------------------------------------------------------------

print("\n[5] DELIVERY PERSON AGE VALIDATION")

age = df["Delivery_person_Age"]

print("Minimum Age:", age.min())
print("Maximum Age:", age.max())

print("Age Below 18:", (age < 18).sum())
print("Age Above 60:", (age > 60).sum())

# ------------------------------------------------------------
# 6. RATING VALIDATION
# ------------------------------------------------------------

print("\n[6] DELIVERY RATING VALIDATION")

rating = df["Delivery_person_Ratings"]

print("Minimum Rating:", rating.min())
print("Maximum Rating:", rating.max())

print("Ratings Below 1:", (rating < 1).sum())
print("Ratings Above 5:", (rating > 5).sum())

# ------------------------------------------------------------
# 7. DELIVERY TIME VALIDATION
# ------------------------------------------------------------

print("\n[7] DELIVERY TIME VALIDATION")

delivery_time = df["Time_taken (min)"]

print("Minimum Delivery Time:", delivery_time.min())
print("Maximum Delivery Time:", delivery_time.max())

print("Non-positive Delivery Time:", (delivery_time <= 0).sum())

# ------------------------------------------------------------
# 8. COORDINATE VALIDATION
# ------------------------------------------------------------

print("\n[8] LOCATION COORDINATE VALIDATION")

coordinate_columns = [
    "Restaurant_latitude",
    "Restaurant_longitude",
    "Delivery_location_latitude",
    "Delivery_location_longitude"
]

zero_coordinates = (df[coordinate_columns] == 0).any(axis=1)

print("Records with Zero Coordinates:", zero_coordinates.sum())

print("Records with Invalid Latitude:",
      ((df["Restaurant_latitude"].abs() > 90) |
       (df["Delivery_location_latitude"].abs() > 90)).sum())

print("Records with Invalid Longitude:",
      ((df["Restaurant_longitude"].abs() > 180) |
       (df["Delivery_location_longitude"].abs() > 180)).sum())

# ------------------------------------------------------------
# 9. DATE FORMAT VALIDATION
# ------------------------------------------------------------

print("\n[9] DATE FORMAT VALIDATION")

parsed_dates = pd.to_datetime(
    df["Order_Date"],
    format="%d-%m-%Y",
    errors="coerce"
)

print("Invalid Order Dates:", parsed_dates.isna().sum())

print("Date Range:",
      parsed_dates.min(),
      "to",
      parsed_dates.max())

# ------------------------------------------------------------
# 10. TIME FORMAT VALIDATION
# ------------------------------------------------------------

print("\n[10] TIME FORMAT VALIDATION")

for column in ["Time_Orderd", "Time_Order_picked"]:

    values = df[column].dropna().astype(str).str.strip()

    decimal_times = values.str.match(
        r"^\d*\.?\d+$"
    )

    invalid_times = ~(
        values.str.match(r"^\d{1,2}:\d{2}$") |
        decimal_times
    )

    print(f"\n{column}")
    print("Decimal Time Values:", decimal_times.sum())
    print("Other Unrecognized Formats:", invalid_times.sum())

# ------------------------------------------------------------
# 11. CATEGORICAL FEATURE ANALYSIS
# ------------------------------------------------------------

print("\n[11] CATEGORICAL FEATURE ANALYSIS")

categorical_columns = df.select_dtypes(
    include="object"
).columns

for column in categorical_columns:

    print(f"\n{column}")
    print("Unique Categories:", df[column].nunique())

    print(df[column].value_counts(dropna=False).head(10))

# ------------------------------------------------------------
# 12. OUTLIER DETECTION USING IQR
# ------------------------------------------------------------

print("\n[12] OUTLIER DETECTION")

outlier_columns = [
    "Delivery_person_Age",
    "Delivery_person_Ratings",
    "Time_taken (min)"
]

outlier_results = []

for column in outlier_columns:

    values = df[column].dropna()

    Q1 = values.quantile(0.25)
    Q3 = values.quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    outliers = (
        (values < lower_limit) |
        (values > upper_limit)
    ).sum()

    outlier_results.append({
        "Column": column,
        "Lower Limit": round(lower_limit, 2),
        "Upper Limit": round(upper_limit, 2),
        "Outlier Count": int(outliers)
    })

    print(f"\n{column}")
    print("Lower Limit:", round(lower_limit, 2))
    print("Upper Limit:", round(upper_limit, 2))
    print("Outliers:", outliers)

outlier_report = pd.DataFrame(outlier_results)

# ------------------------------------------------------------
# 13. SAVE QUALITY REPORTS
# ------------------------------------------------------------

missing_report.to_csv(
    OUTPUT_DIR / "stage2_missing_report.csv",
    index=False
)

outlier_report.to_csv(
    OUTPUT_DIR / "stage2_outlier_report.csv",
    index=False
)

# Save complete quality summary
with open(
    OUTPUT_DIR / "stage2_quality_report.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write("DELIVERIQ - STAGE 2 QUALITY REPORT\n")
    file.write("=" * 50 + "\n")

    file.write(f"\nTotal Rows: {df.shape[0]}\n")
    file.write(f"Total Columns: {df.shape[1]}\n")

    file.write(
        f"Duplicate Rows: {df.duplicated().sum()}\n"
    )

    file.write(
        f"Age Below 18: {(age < 18).sum()}\n"
    )

    file.write(
        f"Ratings Above 5: {(rating > 5).sum()}\n"
    )

    file.write(
        f"Zero Coordinate Records: {zero_coordinates.sum()}\n"
    )

    file.write("\nMISSING VALUE REPORT\n")
    file.write(missing_report.to_string(index=False))

    file.write("\n\nOUTLIER REPORT\n")
    file.write(outlier_report.to_string(index=False))

print("\n" + "=" * 65)
print("STAGE 2 QUALITY ASSESSMENT COMPLETED!")
print("=" * 65)

print("\nGenerated Reports:")
print("1. stage2_missing_report.csv")
print("2. stage2_outlier_report.csv")
print("3. stage2_quality_report.txt")
