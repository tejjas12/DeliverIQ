
# ============================================================
# DELIVERIQ - STAGE 5
# DATA INTEGRATION USING PANDAS
# ============================================================

import pandas as pd
from pathlib import Path

# ------------------------------------------------------------
# 1. PROJECT PATHS
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = (
    BASE_DIR / "data" / "processed" / "deliveriq_cleaned.csv"
)

REFERENCE_DIR = BASE_DIR / "data" / "reference"
FINAL_DIR = BASE_DIR / "data" / "final"
OUTPUT_DIR = BASE_DIR / "outputs"

for folder in [REFERENCE_DIR, FINAL_DIR, OUTPUT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

print("\n" + "=" * 65)
print("       DELIVERIQ - DATA INTEGRATION")
print("=" * 65)

# ------------------------------------------------------------
# 2. LOAD MAIN CLEANED DATASET
# ------------------------------------------------------------

if not INPUT_PATH.exists():
    raise FileNotFoundError(
        "Cleaned dataset not found. Run Stage 3 first."
    )

df = pd.read_csv(INPUT_PATH)

print("\nMain Dataset Loaded Successfully!")
print("Main Dataset Shape:", df.shape)

# Check required columns
required_columns = [
    "Weather_conditions",
    "Road_traffic_density"
]

for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Required column missing: {col}")

original_rows = len(df)

# ------------------------------------------------------------
# 3. CREATE WEATHER REFERENCE DATASET
# ------------------------------------------------------------

print("\n[1] CREATING WEATHER REFERENCE DATASET")

weather_data = {
    "Weather_conditions": [
        "Sunny",
        "Cloudy",
        "Windy",
        "Fog",
        "Stormy",
        "Sandstorms",
        "Unknown"
    ],
    "Weather_Severity_Score": [
        1, 2, 2, 3, 5, 4, 0
    ],
    "Weather_Category": [
        "Clear",
        "Moderate",
        "Moderate",
        "Low Visibility",
        "Severe",
        "Severe",
        "Unclassified"
    ]
}

weather_df = pd.DataFrame(weather_data)

weather_path = (
    REFERENCE_DIR / "weather_lookup.csv"
)

weather_df.to_csv(weather_path, index=False)

print("Weather Reference Created!")
print(weather_df.to_string(index=False))

# ------------------------------------------------------------
# 4. CREATE TRAFFIC REFERENCE DATASET
# ------------------------------------------------------------

print("\n[2] CREATING TRAFFIC REFERENCE DATASET")

traffic_data = {
    "Road_traffic_density": [
        "Low",
        "Medium",
        "High",
        "Jam",
        "Unknown"
    ],
    "Traffic_Congestion_Score": [
        1, 2, 3, 4, 0
    ],
    "Traffic_Category": [
        "Smooth",
        "Moderate",
        "Heavy",
        "Severe",
        "Unclassified"
    ]
}

traffic_df = pd.DataFrame(traffic_data)

traffic_path = (
    REFERENCE_DIR / "traffic_lookup.csv"
)

traffic_df.to_csv(traffic_path, index=False)

print("Traffic Reference Created!")
print(traffic_df.to_string(index=False))

# ------------------------------------------------------------
# 5. NORMALIZE JOIN KEYS
# ------------------------------------------------------------

print("\n[3] PREPARING DATA FOR INTEGRATION")

df["Weather_Key"] = (
    df["Weather_conditions"]
    .fillna("Unknown")
    .astype(str)
    .str.strip()
    .str.casefold()
)

weather_df["Weather_Key"] = (
    weather_df["Weather_conditions"]
    .astype(str)
    .str.strip()
    .str.casefold()
)

df["Traffic_Key"] = (
    df["Road_traffic_density"]
    .fillna("Unknown")
    .astype(str)
    .str.strip()
    .str.casefold()
)

traffic_df["Traffic_Key"] = (
    traffic_df["Road_traffic_density"]
    .astype(str)
    .str.strip()
    .str.casefold()
)

# ------------------------------------------------------------
# 6. MERGE WEATHER REFERENCE DATA
# ------------------------------------------------------------

print("\n[4] INTEGRATING WEATHER DATA")

df = df.merge(
    weather_df[
        [
            "Weather_Key",
            "Weather_Severity_Score",
            "Weather_Category"
        ]
    ],
    on="Weather_Key",
    how="left",
    validate="many_to_one",
    indicator="Weather_Merge_Status"
)

weather_unmatched = (
    df["Weather_Merge_Status"] == "left_only"
).sum()

df.drop(
    columns=["Weather_Key", "Weather_Merge_Status"],
    inplace=True
)

print("Weather Integration Completed!")
print("Unmatched Weather Records:", weather_unmatched)

# ------------------------------------------------------------
# 7. MERGE TRAFFIC REFERENCE DATA
# ------------------------------------------------------------

print("\n[5] INTEGRATING TRAFFIC DATA")

df = df.merge(
    traffic_df[
        [
            "Traffic_Key",
            "Traffic_Congestion_Score",
            "Traffic_Category"
        ]
    ],
    on="Traffic_Key",
    how="left",
    validate="many_to_one",
    indicator="Traffic_Merge_Status"
)

traffic_unmatched = (
    df["Traffic_Merge_Status"] == "left_only"
).sum()

df.drop(
    columns=["Traffic_Key", "Traffic_Merge_Status"],
    inplace=True
)

print("Traffic Integration Completed!")
print("Unmatched Traffic Records:", traffic_unmatched)

# ------------------------------------------------------------
# 8. VERIFY INTEGRATED DATASET
# ------------------------------------------------------------

print("\n[6] INTEGRATION VERIFICATION")

print("Original Rows:", original_rows)
print("Integrated Rows:", len(df))
print("Original Columns:", len(required_columns))
print("Integrated Columns:", df.shape[1])

if len(df) != original_rows:
    raise ValueError(
        "Row count changed during integration!"
    )

# ------------------------------------------------------------
# 9. SAVE INTEGRATED DATASET
# ------------------------------------------------------------

integrated_path = (
    FINAL_DIR / "deliveriq_integrated.csv"
)

df.to_csv(integrated_path, index=False)

# ------------------------------------------------------------
# 10. CREATE INTEGRATION REPORT
# ------------------------------------------------------------

report = pd.DataFrame({
    "Integration": [
        "Main Zomato Dataset",
        "Weather Reference Dataset",
        "Traffic Reference Dataset"
    ],
    "Rows": [
        original_rows,
        len(weather_df),
        len(traffic_df)
    ]
})

report.to_csv(
    OUTPUT_DIR / "stage5_integration_report.csv",
    index=False
)

print("\nINTEGRATION SUMMARY")
print(report.to_string(index=False))

print("\nIntegrated Dataset Saved:")
print(integrated_path)

print("\nNew Columns Added:")
print([
    "Weather_Severity_Score",
    "Weather_Category",
    "Traffic_Congestion_Score",
    "Traffic_Category"
])

print("\n" + "=" * 65)
print("STAGE 5 COMPLETED SUCCESSFULLY!")
print("=" * 65)
