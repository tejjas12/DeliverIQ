
# ============================================================
# DELIVERIQ - STAGE 3: DATA CLEANING
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "raw" / "Zomato Dataset.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

OUTPUT_DIR.mkdir(exist_ok=True)
PROCESSED_DIR.mkdir(exist_ok=True)

print("\n" + "=" * 60)
print("       DELIVERIQ - DATA CLEANING")
print("=" * 60)

# 1. Load original dataset
df = pd.read_csv(DATA_PATH)

original_rows = len(df)

print("\nOriginal Dataset:", df.shape)

# 2. Clean column names and text spaces
df.columns = df.columns.str.strip()

text_columns = df.select_dtypes(include="object").columns

for col in text_columns:
    df[col] = df[col].str.strip()

# Convert blank strings to missing values
df.replace(r"^\s*$", np.nan, regex=True, inplace=True)

# 3. Remove duplicate records
duplicates_removed = df.duplicated().sum()
df.drop_duplicates(inplace=True)

# 4. Convert numerical columns
numeric_columns = [
    "Delivery_person_Age",
    "Delivery_person_Ratings",
    "Restaurant_latitude",
    "Restaurant_longitude",
    "Delivery_location_latitude",
    "Delivery_location_longitude",
    "Vehicle_condition",
    "multiple_deliveries",
    "Time_taken (min)"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# 5. Handle invalid age
invalid_age_count = (df["Delivery_person_Age"] < 18).sum()

df.loc[
    (df["Delivery_person_Age"] < 18) |
    (df["Delivery_person_Age"] > 60),
    "Delivery_person_Age"
] = np.nan

age_median = df["Delivery_person_Age"].median()

df["Delivery_person_Age"] = (
    df["Delivery_person_Age"].fillna(age_median)
)

# 6. Handle invalid ratings
invalid_rating_count = (
    (df["Delivery_person_Ratings"] < 1) |
    (df["Delivery_person_Ratings"] > 5)
).sum()

df.loc[
    (df["Delivery_person_Ratings"] < 1) |
    (df["Delivery_person_Ratings"] > 5),
    "Delivery_person_Ratings"
] = np.nan

rating_median = df["Delivery_person_Ratings"].median()

df["Delivery_person_Ratings"] = (
    df["Delivery_person_Ratings"].fillna(rating_median)
)

# 7. Fill numerical missing values
numeric_fill_columns = [
    "multiple_deliveries"
]

for col in numeric_fill_columns:
    df[col] = df[col].fillna(df[col].mode()[0])

# 8. Fill categorical missing values
categorical_fill_columns = [
    "Weather_conditions",
    "Road_traffic_density",
    "Festival",
    "City"
]

for col in categorical_fill_columns:
    df[col] = df[col].fillna(df[col].mode()[0])

# 9. Handle missing order time
df["Time_Orderd"] = df["Time_Orderd"].fillna("Unknown")

# 10. Convert decimal time values to HH:MM
def convert_time(value):

    if pd.isna(value):
        return np.nan

    value = str(value).strip()

    if value == "Unknown":
        return "Unknown"

    if ":" in value:
        try:
            parsed = pd.to_datetime(value, format="%H:%M")
            return parsed.strftime("%H:%M")
        except ValueError:
            return "Unknown"

    try:
        number = float(value)

        # Spreadsheet time fraction (0 to 1)
        if 0 <= number <= 1:
            total_minutes = round(number * 1440) % 1440
            hours = total_minutes // 60
            minutes = total_minutes % 60

            return f"{hours:02d}:{minutes:02d}"

    except ValueError:
        pass

    return "Unknown"


for col in ["Time_Orderd", "Time_Order_picked"]:
    df[col] = df[col].apply(convert_time)

# 11. Convert Order Date
df["Order_Date"] = pd.to_datetime(
    df["Order_Date"],
    format="%d-%m-%Y",
    errors="coerce"
)

# 12. Handle zero coordinates safely
coordinate_columns = [
    "Restaurant_latitude",
    "Restaurant_longitude",
    "Delivery_location_latitude",
    "Delivery_location_longitude"
]

zero_coordinate_mask = (df[coordinate_columns] == 0).any(axis=1)

zero_coordinate_records = int(zero_coordinate_mask.sum())

df["Invalid_Coordinates_Flag"] = zero_coordinate_mask

# Replace zero coordinates with NaN, do not invent locations
df[coordinate_columns] = df[coordinate_columns].replace(0, np.nan)

# 13. Save cleaned dataset
cleaned_path = PROCESSED_DIR / "deliveriq_cleaned.csv"

df.to_csv(cleaned_path, index=False)

# 14. Create cleaning summary
summary = pd.DataFrame({
    "Metric": [
        "Original Rows",
        "Final Rows",
        "Duplicate Rows Removed",
        "Invalid Ages Corrected",
        "Invalid Ratings Corrected",
        "Zero Coordinate Records Flagged",
        "Remaining Missing Values"
    ],
    "Value": [
        original_rows,
        len(df),
        duplicates_removed,
        invalid_age_count,
        invalid_rating_count,
        zero_coordinate_records,
        int(df.isnull().sum().sum())
    ]
})

summary.to_csv(
    OUTPUT_DIR / "stage3_cleaning_summary.csv",
    index=False
)

print("\nCLEANING SUMMARY")
print(summary.to_string(index=False))

print("\nRemaining Missing Values:")
print(df.isnull().sum()[df.isnull().sum() > 0])

print("\nCleaned Dataset Saved:")
print(cleaned_path)

print("\n" + "=" * 60)
print("STAGE 3 COMPLETED SUCCESSFULLY!")
print("=" * 60)
