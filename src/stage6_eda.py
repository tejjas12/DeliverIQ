
# ============================================================
# DELIVERIQ - STAGE 6
# EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

# ------------------------------------------------------------
# 1. PROJECT PATHS
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = BASE_DIR / "data" / "final" / "deliveriq_integrated.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
GRAPH_DIR = OUTPUT_DIR / "eda_graphs"

GRAPH_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")

print("\n" + "=" * 65)
print("       DELIVERIQ - EXPLORATORY DATA ANALYSIS")
print("=" * 65)

# ------------------------------------------------------------
# 2. LOAD INTEGRATED DATASET
# ------------------------------------------------------------

if not INPUT_PATH.exists():
    raise FileNotFoundError(
        "Integrated dataset not found. Run Stage 5 first."
    )

df = pd.read_csv(INPUT_PATH)

print("\nIntegrated Dataset Loaded Successfully!")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

# Convert important numerical columns
numeric_columns = [
    "Time_taken (min)",
    "Delivery_person_Age",
    "Delivery_person_Ratings",
    "Vehicle_condition",
    "multiple_deliveries",
    "Weather_Severity_Score",
    "Traffic_Congestion_Score"
]

for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Remove invalid target values for analysis only
eda_df = df[
    df["Time_taken (min)"].notna() &
    (df["Time_taken (min)"] > 0)
].copy()

print("Records used for EDA:", len(eda_df))

# ------------------------------------------------------------
# 3. DESCRIPTIVE STATISTICS
# ------------------------------------------------------------

print("\n[1] DESCRIPTIVE STATISTICS")

print(
    eda_df["Time_taken (min)"].describe().round(2)
)

# Save descriptive statistics
eda_df.describe(include="all").T.to_csv(
    OUTPUT_DIR / "eda_descriptive_statistics.csv"
)

# ------------------------------------------------------------
# 4. GRAPH 1 - DELIVERY TIME DISTRIBUTION
# ------------------------------------------------------------

print("\n[2] DELIVERY TIME DISTRIBUTION")

plt.figure(figsize=(10, 6))

sns.histplot(
    data=eda_df,
    x="Time_taken (min)",
    bins=30,
    kde=True
)

plt.title("Distribution of Food Delivery Time")
plt.xlabel("Delivery Time (Minutes)")
plt.ylabel("Number of Deliveries")

plt.tight_layout()
plt.savefig(
    GRAPH_DIR / "01_delivery_time_distribution.png",
    dpi=150
)
plt.close()

# ------------------------------------------------------------
# 5. GRAPH 2 - WEATHER VS DELIVERY TIME
# ------------------------------------------------------------

print("\n[3] WEATHER VS DELIVERY TIME")

weather_analysis = (
    eda_df.groupby("Weather_conditions")["Time_taken (min)"]
    .mean()
    .sort_values(ascending=False)
)

print(weather_analysis.round(2))

plt.figure(figsize=(10, 6))

weather_analysis.plot(kind="bar")

plt.title("Average Delivery Time by Weather")
plt.xlabel("Weather Conditions")
plt.ylabel("Average Delivery Time (Minutes)")
plt.xticks(rotation=30)

plt.tight_layout()
plt.savefig(
    GRAPH_DIR / "02_weather_vs_delivery_time.png",
    dpi=150
)
plt.close()

# ------------------------------------------------------------
# 6. GRAPH 3 - TRAFFIC VS DELIVERY TIME
# ------------------------------------------------------------

print("\n[4] TRAFFIC VS DELIVERY TIME")

traffic_order = ["Low", "Medium", "High", "Jam"]

traffic_analysis = (
    eda_df.groupby("Road_traffic_density")["Time_taken (min)"]
    .mean()
    .reindex(traffic_order)
)

print(traffic_analysis.round(2))

plt.figure(figsize=(9, 6))

traffic_analysis.plot(kind="bar")

plt.title("Average Delivery Time by Traffic Density")
plt.xlabel("Traffic Density")
plt.ylabel("Average Delivery Time (Minutes)")
plt.xticks(rotation=0)

plt.tight_layout()
plt.savefig(
    GRAPH_DIR / "03_traffic_vs_delivery_time.png",
    dpi=150
)
plt.close()

# ------------------------------------------------------------
# 7. GRAPH 4 - CITY WISE DELIVERY ANALYSIS
# ------------------------------------------------------------

print("\n[5] CITY-WISE DELIVERY ANALYSIS")

city_analysis = (
    eda_df.groupby("City")["Time_taken (min)"]
    .mean()
    .sort_values(ascending=False)
)

print(city_analysis.round(2))

plt.figure(figsize=(9, 6))

city_analysis.plot(kind="bar")

plt.title("Average Delivery Time by City")
plt.xlabel("City Category")
plt.ylabel("Average Delivery Time (Minutes)")
plt.xticks(rotation=0)

plt.tight_layout()
plt.savefig(
    GRAPH_DIR / "04_city_wise_delivery.png",
    dpi=150
)
plt.close()

# ------------------------------------------------------------
# 8. GRAPH 5 - VEHICLE TYPE ANALYSIS
# ------------------------------------------------------------

print("\n[6] VEHICLE TYPE ANALYSIS")

vehicle_analysis = (
    eda_df.groupby("Type_of_vehicle")["Time_taken (min)"]
    .mean()
    .sort_values(ascending=False)
)

print(vehicle_analysis.round(2))

plt.figure(figsize=(9, 6))

vehicle_analysis.plot(kind="bar")

plt.title("Average Delivery Time by Vehicle Type")
plt.xlabel("Vehicle Type")
plt.ylabel("Average Delivery Time (Minutes)")
plt.xticks(rotation=20)

plt.tight_layout()
plt.savefig(
    GRAPH_DIR / "05_vehicle_analysis.png",
    dpi=150
)
plt.close()

# ------------------------------------------------------------
# 9. GRAPH 6 - DELIVERY PERSON RATINGS
# ------------------------------------------------------------

print("\n[7] DELIVERY PERSON RATINGS")

plt.figure(figsize=(9, 6))

sns.histplot(
    data=eda_df,
    x="Delivery_person_Ratings",
    bins=15,
    kde=True
)

plt.title("Distribution of Delivery Person Ratings")
plt.xlabel("Delivery Person Rating")
plt.ylabel("Frequency")

plt.tight_layout()
plt.savefig(
    GRAPH_DIR / "06_rating_distribution.png",
    dpi=150
)
plt.close()

# ------------------------------------------------------------
# 10. GRAPH 7 - CORRELATION HEATMAP
# ------------------------------------------------------------

print("\n[8] CORRELATION ANALYSIS")

correlation_columns = [
    "Delivery_person_Age",
    "Delivery_person_Ratings",
    "Vehicle_condition",
    "multiple_deliveries",
    "Time_taken (min)",
    "Weather_Severity_Score",
    "Traffic_Congestion_Score"
]

available_columns = [
    col for col in correlation_columns
    if col in eda_df.columns
]

correlation = eda_df[available_columns].corr()

print(correlation.round(2))

plt.figure(figsize=(11, 8))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)

plt.title("Food Delivery Data Correlation Heatmap")

plt.tight_layout()
plt.savefig(
    GRAPH_DIR / "07_correlation_heatmap.png",
    dpi=150
)
plt.close()

# ------------------------------------------------------------
# 11. GRAPH 8 - ORDER TYPE ANALYSIS
# ------------------------------------------------------------

print("\n[9] ORDER TYPE ANALYSIS")

order_analysis = (
    eda_df["Type_of_order"]
    .value_counts()
)

print(order_analysis)

plt.figure(figsize=(9, 6))

order_analysis.plot(
    kind="bar",
    color="teal"
)

plt.title("Food Delivery Orders by Order Type")
plt.xlabel("Order Type")
plt.ylabel("Number of Orders")
plt.xticks(rotation=20)

plt.tight_layout()
plt.savefig(
    GRAPH_DIR / "08_order_type_analysis.png",
    dpi=150
)
plt.close()

# ------------------------------------------------------------
# 12. SAVE ANALYSIS TABLES
# ------------------------------------------------------------

weather_analysis.to_csv(
    OUTPUT_DIR / "weather_analysis.csv"
)

traffic_analysis.to_csv(
    OUTPUT_DIR / "traffic_analysis.csv"
)

city_analysis.to_csv(
    OUTPUT_DIR / "city_analysis.csv"
)

vehicle_analysis.to_csv(
    OUTPUT_DIR / "vehicle_analysis.csv"
)

# ------------------------------------------------------------
# 13. AUTOMATIC KEY FINDINGS
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("                 KEY FINDINGS")
print("=" * 65)

print(
    "\nAverage Delivery Time:",
    round(eda_df["Time_taken (min)"].mean(), 2),
    "minutes"
)

print(
    "Maximum Delivery Time:",
    eda_df["Time_taken (min)"].max(),
    "minutes"
)

print(
    "Minimum Delivery Time:",
    eda_df["Time_taken (min)"].min(),
    "minutes"
)

print(
    "\nSlowest Weather:",
    weather_analysis.idxmax()
)

print(
    "Slowest Traffic:",
    traffic_analysis.idxmax()
)

print(
    "Highest Average City Category:",
    city_analysis.idxmax()
)

print("\nAll EDA graphs saved in:")
print(GRAPH_DIR)

print("\n" + "=" * 65)
print("STAGE 6 COMPLETED SUCCESSFULLY!")
print("=" * 65)
