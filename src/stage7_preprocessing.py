
# ============================================================
# DELIVERIQ - STAGE 7
# DATA PREPROCESSING FOR MACHINE LEARNING
# ============================================================

import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ------------------------------------------------------------
# 1. PROJECT PATHS
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = (
    BASE_DIR / "data" / "final" / "deliveriq_integrated.csv"
)

FINAL_DIR = BASE_DIR / "data" / "final"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = BASE_DIR / "models"

for folder in [FINAL_DIR, OUTPUT_DIR, MODEL_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

print("\n" + "=" * 65)
print("       DELIVERIQ - DATA PREPROCESSING")
print("=" * 65)

# ------------------------------------------------------------
# 2. LOAD INTEGRATED DATASET
# ------------------------------------------------------------

if not INPUT_PATH.exists():
    raise FileNotFoundError(
        "Integrated dataset not found. Run Stage 5 first."
    )

df = pd.read_csv(INPUT_PATH)

print("\nDataset Loaded Successfully!")
print("Original Shape:", df.shape)

# ------------------------------------------------------------
# 3. TARGET VARIABLE
# ------------------------------------------------------------

TARGET = "Time_taken (min)"

if TARGET not in df.columns:
    raise ValueError(f"Target column not found: {TARGET}")

df[TARGET] = pd.to_numeric(
    df[TARGET], errors="coerce"
)

# Keep only valid positive target values
df = df[
    df[TARGET].notna() &
    (df[TARGET] > 0)
].copy()

print("Records after target validation:", len(df))

# ------------------------------------------------------------
# 4. FEATURE ENGINEERING
# ------------------------------------------------------------

print("\n[1] FEATURE ENGINEERING")

# Calculate delivery distance from coordinates
coordinate_columns = [
    "Restaurant_latitude",
    "Restaurant_longitude",
    "Delivery_location_latitude",
    "Delivery_location_longitude"
]

for col in coordinate_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(
            df[col], errors="coerce"
        )
        df[col] = df[col].replace(0, np.nan)

if all(col in df.columns for col in coordinate_columns):

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

# Extract order date features
if "Order_Date" in df.columns:

    order_date = pd.to_datetime(
        df["Order_Date"],
        errors="coerce",
        dayfirst=True
    )

    df["Order_Month"] = order_date.dt.month
    df["Order_Weekday"] = order_date.dt.dayofweek
    df["Is_Weekend"] = (
        order_date.dt.dayofweek.isin([5, 6])
        .astype(int)
    )

# Extract hour from order time
def extract_hour(value):

    if pd.isna(value):
        return np.nan

    value = str(value).strip()

    try:
        if ":" in value:
            hour = int(value.split(":")[0])

            if 0 <= hour <= 23:
                return hour

        number = float(value)

        if 0 <= number <= 1:
            return int((number * 1440) // 60) % 24

    except (ValueError, TypeError):
        pass

    return np.nan


if "Time_Orderd" in df.columns:
    df["Order_Hour"] = df["Time_Orderd"].apply(
        extract_hour
    )

# ------------------------------------------------------------
# 5. FEATURE SELECTION
# ------------------------------------------------------------

print("\n[2] FEATURE SELECTION")

numerical_features = [
    "Delivery_person_Age",
    "Delivery_person_Ratings",
    "Vehicle_condition",
    "multiple_deliveries",
    "Weather_Severity_Score",
    "Traffic_Congestion_Score",
    "Delivery_Distance_km",
    "Order_Month",
    "Order_Weekday",
    "Is_Weekend",
    "Order_Hour"
]

categorical_features = [
    "Weather_conditions",
    "Road_traffic_density",
    "Type_of_order",
    "Type_of_vehicle",
    "Festival",
    "City"
]

numerical_features = [
    col for col in numerical_features
    if col in df.columns
]

categorical_features = [
    col for col in categorical_features
    if col in df.columns
]

feature_columns = (
    numerical_features + categorical_features
)

if not feature_columns:
    raise ValueError("No valid input features were found.")

X = df[feature_columns].copy()
y = df[TARGET].copy()

# Make categorical columns consistent
for col in categorical_features:
    X[col] = X[col].astype("string").fillna(np.nan)
    X[col] = X[col].replace("nan", np.nan)

# Ensure numeric columns are numeric
for col in numerical_features:
    X[col] = pd.to_numeric(
        X[col], errors="coerce"
    )

print("Numerical Features:", numerical_features)
print("Categorical Features:", categorical_features)
print("Total Input Features:", len(feature_columns))

# ------------------------------------------------------------
# 6. TRAIN-TEST SPLIT
# ------------------------------------------------------------

print("\n[3] TRAIN-TEST SPLIT")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Training Records:", len(X_train))
print("Testing Records:", len(X_test))

# ------------------------------------------------------------
# 7. NUMERICAL PREPROCESSING
# ------------------------------------------------------------

print("\n[4] NUMERICAL PREPROCESSING")

numerical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

# ------------------------------------------------------------
# 8. CATEGORICAL PREPROCESSING
# ------------------------------------------------------------

print("\n[5] CATEGORICAL PREPROCESSING")

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)

# ------------------------------------------------------------
# 9. COMBINE PREPROCESSING
# ------------------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_pipeline, numerical_features),
        ("cat", categorical_pipeline, categorical_features)
    ],
    remainder="drop"
)

# Fit only on training data to prevent data leakage
X_train_processed = preprocessor.fit_transform(X_train)

# Transform test data using training-fitted preprocessing
X_test_processed = preprocessor.transform(X_test)

print("\nPreprocessing Completed!")
print("Training Matrix Shape:", X_train_processed.shape)
print("Testing Matrix Shape:", X_test_processed.shape)

# ------------------------------------------------------------
# 10. GET TRANSFORMED FEATURE NAMES
# ------------------------------------------------------------

feature_names = preprocessor.get_feature_names_out()

# ------------------------------------------------------------
# 11. SAVE PREPROCESSED DATASETS
# ------------------------------------------------------------

train_df = pd.DataFrame(
    X_train_processed,
    columns=feature_names
)

train_df[TARGET] = y_train.reset_index(drop=True)

test_df = pd.DataFrame(
    X_test_processed,
    columns=feature_names
)

test_df[TARGET] = y_test.reset_index(drop=True)

train_path = FINAL_DIR / "deliveriq_train.csv"
test_path = FINAL_DIR / "deliveriq_test.csv"

train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)

# Save the fitted preprocessing pipeline
pipeline_path = MODEL_DIR / "deliveriq_preprocessor.joblib"

joblib.dump(preprocessor, pipeline_path)

# ------------------------------------------------------------
# 12. PREPROCESSING SUMMARY
# ------------------------------------------------------------

summary = pd.DataFrame({
    "Metric": [
        "Original Valid Records",
        "Training Records",
        "Testing Records",
        "Input Features Before Encoding",
        "Features After Encoding",
        "Target Variable",
        "Train-Test Ratio"
    ],
    "Value": [
        len(X),
        len(X_train),
        len(X_test),
        len(feature_columns),
        len(feature_names),
        TARGET,
        "80:20"
    ]
})

summary.to_csv(
    OUTPUT_DIR / "stage7_preprocessing_summary.csv",
    index=False
)

print("\nPREPROCESSING SUMMARY")
print(summary.to_string(index=False))

print("\nSaved Files:")
print(train_path)
print(test_path)
print(pipeline_path)

print("\n" + "=" * 65)
print("STAGE 7 COMPLETED SUCCESSFULLY!")
print("=" * 65)
