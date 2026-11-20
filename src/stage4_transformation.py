
# ============================================================
# DELIVERIQ - STAGE 4
# DATA TRANSFORMATION & FEATURE ENGINEERING
# ============================================================

import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

# ------------------------------------------------------------
# 1. PROJECT PATHS
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = BASE_DIR / "data" / "processed" / "deliveriq_cleaned.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
FINAL_DIR = BASE_DIR / "data" / "final"

OUTPUT_DIR.mkdir(exist_ok=True)
FINAL_DIR.mkdir(exist_ok=True)

print("\n" + "=" * 65)
print("       DELIVERIQ - DATA TRANSFORMATION")
print("=" * 65)

# ------------------------------------------------------------
# 2. LOAD CLEANED DATASET
# ------------------------------------------------------------

if not INPUT_PATH.exists():
    raise FileNotFoundError(
        "Cleaned dataset not found. Run Stage 3 first."
    )

df = pd.read_csv(INPUT_PATH)

print("\nCleaned Dataset Loaded!")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

# ------------------------------------------------------------
# 3. DATE TRANSFORMATION
# ------------------------------------------------------------

print("\n[1] DATE TRANSFORMATION")

df["Order_Date"] = pd.to_datetime(
    df["Order_Date"],
    errors="coerce"
)

df["Order_Day"] = df["Order_Date"].dt.day
df["Order_Month"] = df["Order_Date"].dt.month
df["Order_Year"] = df["Order_Date"].dt.year
df["Order_Weekday"] = df["Order_Date"].dt.dayofweek

df["Is_Weekend"] = (
    df["Order_Weekday"].isin([5, 6]).astype(int)
)

print("Day, Month, Year and Weekday features created.")

# ------------------------------------------------------------
# 4. TIME TRANSFORMATION
# ------------------------------------------------------------

print("\n[2] TIME TRANSFORMATION")

def time_to_minutes(value):

    if pd.isna(value):
        return np.nan

    value = str(value).strip()

    if value.lower() in ["unknown", "nan", "none", ""]:
        return np.nan

    try:
        if ":" in value:
            hour, minute = map(int, value.split(":"))

            if 0 <= hour <= 23 and 0 <= minute <= 59:
                return hour * 60 + minute

        number = float(value)

        if 0 <= number <= 1:
            return round(number * 1440)

    except (ValueError, TypeError):
        pass

    return np.nan


for col in ["Time_Orderd", "Time_Order_picked"]:

    if col in df.columns:

        df[col + "_Minutes"] = df[col].apply(time_to_minutes)

        df[col + "_Hour"] = (
            df[col + "_Minutes"] // 60
        )

        df[col + "_Minute"] = (
            df[col + "_Minutes"] % 60
        )

print("Order and pickup time features created.")

# ------------------------------------------------------------
# 5. FEATURE ENGINEERING - DELIVERY DISTANCE
# ------------------------------------------------------------

print("\n[3] DELIVERY DISTANCE FEATURE")

coordinate_columns = [
    "Restaurant_latitude",
    "Restaurant_longitude",
    "Delivery_location_latitude",
    "Delivery_location_longitude"
]

for col in coordinate_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

lat1 = np.radians(df["Restaurant_latitude"])
lon1 = np.radians(df["Restaurant_longitude"])

lat2 = np.radians(df["Delivery_location_latitude"])
lon2 = np.radians(df["Delivery_location_longitude"])

dlat = lat2 - lat1
dlon = lon2 - lon1

a = (
    np.sin(dlat / 2) ** 2
    + np.cos(lat1) * np.cos(lat2)
    * np.sin(dlon / 2) ** 2
)

a = a.clip(0, 1)

df["Delivery_Distance_km"] = (
    6371 * 2 * np.arcsin(np.sqrt(a))
)

print("Delivery distance calculated using Haversine formula.")

# ------------------------------------------------------------
# 6. LABEL ENCODING - TRAFFIC DENSITY
# ------------------------------------------------------------

print("\n[4] LABEL ENCODING")

traffic_mapping = {
    "low": 1,
    "medium": 2,
    "high": 3,
    "jam": 4
}

df["Traffic_Level_Encoded"] = (
    df["Road_traffic_density"]
    .astype("string")
    .str.strip()
    .str.lower()
    .map(traffic_mapping)
)

print("Traffic Density encoded successfully.")
print(traffic_mapping)

# ------------------------------------------------------------
# 7. ONE-HOT ENCODING
# ------------------------------------------------------------

print("\n[5] ONE-HOT ENCODING")

categorical_columns = [
    "Weather_conditions",
    "Type_of_order",
    "Type_of_vehicle",
    "Festival",
    "City"
]

available_columns = [
    col for col in categorical_columns
    if col in df.columns
]

for col in available_columns:
    df[col] = df[col].fillna("Unknown").astype(str).str.strip()

df = pd.get_dummies(
    df,
    columns=available_columns,
    dtype=int,
    prefix=available_columns
)

print("One-hot encoding completed.")

# ------------------------------------------------------------
# 8. SELECT NUMERICAL FEATURES FOR SCALING
# ------------------------------------------------------------

print("\n[6] FEATURE SCALING")

scaling_features = [
    "Delivery_person_Age",
    "Delivery_person_Ratings",
    "Vehicle_condition",
    "multiple_deliveries",
    "Order_Day",
    "Order_Month",
    "Order_Year",
    "Order_Weekday",
    "Is_Weekend",
    "Time_Orderd_Hour",
    "Time_Orderd_Minute",
    "Time_Order_picked_Hour",
    "Time_Order_picked_Minute",
    "Delivery_Distance_km",
    "Traffic_Level_Encoded"
]

scaling_features = [
    col for col in scaling_features
    if col in df.columns
]

# Convert selected features to numeric
for col in scaling_features:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# ------------------------------------------------------------
# 9. HANDLE MISSING VALUES FOR SCALING
# ------------------------------------------------------------

imputer = SimpleImputer(strategy="median")

scaled_input = imputer.fit_transform(
    df[scaling_features]
)

# ------------------------------------------------------------
# 10. STANDARDIZATION
# ------------------------------------------------------------

standard_scaler = StandardScaler()

standardized_data = standard_scaler.fit_transform(
    scaled_input
)

standardized_df = pd.DataFrame(
    standardized_data,
    columns=scaling_features
)

# ------------------------------------------------------------
# 11. NORMALIZATION
# ------------------------------------------------------------

minmax_scaler = MinMaxScaler()

normalized_data = minmax_scaler.fit_transform(
    scaled_input
)

normalized_df = pd.DataFrame(
    normalized_data,
    columns=scaling_features
)

# ------------------------------------------------------------
# 12. SAVE TRANSFORMED DATASETS
# ------------------------------------------------------------

transformed_path = (
    FINAL_DIR / "deliveriq_transformed.csv"
)

standardized_path = (
    FINAL_DIR / "deliveriq_standardized.csv"
)

normalized_path = (
    FINAL_DIR / "deliveriq_normalized.csv"
)

df.to_csv(transformed_path, index=False)

standardized_df.to_csv(
    standardized_path, index=False
)

normalized_df.to_csv(
    normalized_path, index=False
)

# ------------------------------------------------------------
# 13. TRANSFORMATION SUMMARY
# ------------------------------------------------------------

summary = pd.DataFrame({
    "Transformation": [
        "Date Feature Extraction",
        "Time Feature Extraction",
        "Delivery Distance",
        "Traffic Label Encoding",
        "One-Hot Encoding",
        "Standardization",
        "Normalization"
    ],
    "Status": ["Completed"] * 7
})

summary.to_csv(
    OUTPUT_DIR / "stage4_transformation_summary.csv",
    index=False
)

print("\nTRANSFORMATION SUMMARY")
print(summary.to_string(index=False))

print("\nOriginal Transformed Shape:", df.shape)
print("Standardized Shape:", standardized_df.shape)
print("Normalized Shape:", normalized_df.shape)

print("\nFiles Saved Successfully!")

print("\n" + "=" * 65)
print("STAGE 4 COMPLETED SUCCESSFULLY!")
print("=" * 65)
