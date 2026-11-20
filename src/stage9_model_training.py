
"""
DeliverIQ - Stage 9
Machine Learning Model Development
"""

import os
import pandas as pd
import numpy as np
import sklearn
import joblib

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

print("=" * 60)
print("DELIVERIQ - MACHINE LEARNING MODEL DEVELOPMENT")
print("=" * 60)

# 1. Load training and testing datasets
train_path = "data/final/deliveriq_train.csv"
test_path = "data/final/deliveriq_test.csv"

if not os.path.exists(train_path) or not os.path.exists(test_path):
    raise FileNotFoundError("Training or testing dataset not found.")

train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

print("\nTraining Dataset Shape:", train_data.shape)
print("Testing Dataset Shape:", test_data.shape)

# 2. Identify target column
target = "Time_taken (min)"

if target not in train_data.columns or target not in test_data.columns:
    raise ValueError(
        f"Target column '{target}' not found. "
        "Please verify the dataset columns."
    )

# 3. Separate features and target
X_train = train_data.drop(columns=[target])
y_train = train_data[target]

X_test = test_data.drop(columns=[target])
y_test = test_data[target]

# 4. Validate feature compatibility
if list(X_train.columns) != list(X_test.columns):
    raise ValueError("Training and testing features do not match.")

if not all(pd.api.types.is_numeric_dtype(X_train[c])
           for c in X_train.columns):
    raise ValueError(
        "Non-numeric features found. "
        "Use the saved preprocessing pipeline first."
    )

# 5. Handle invalid values
X_train = X_train.replace([np.inf, -np.inf], np.nan)
X_test = X_test.replace([np.inf, -np.inf], np.nan)

if X_train.isnull().any().any() or X_test.isnull().any().any():
    raise ValueError(
        "Missing values found. Check the preprocessing output."
    )

# 6. Define regression models
models = {
    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        max_depth=12,
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
}

# 7. Train and evaluate models
results = []
trained_models = {}

for name, model in models.items():

    print(f"\nTraining: {name}")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2 Score": r2
    })

    trained_models[name] = model

    print("MAE:", round(mae, 3))
    print("RMSE:", round(rmse, 3))
    print("R2 Score:", round(r2, 3))

# 8. Compare model performance
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    by="MAE",
    ascending=True
)

os.makedirs("models", exist_ok=True)
os.makedirs("outputs/model_evaluation", exist_ok=True)

results_df.to_csv(
    "outputs/model_evaluation/model_comparison.csv",
    index=False
)

# 9. Select best model using MAE
best_model_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_model_name]

joblib.dump(
    best_model,
    "models/deliveriq_best_model.joblib"
)

# 10. Save actual and predicted values
best_predictions = best_model.predict(X_test)

prediction_results = pd.DataFrame({
    "Actual Delivery Time": y_test.values,
    "Predicted Delivery Time": best_predictions
})

prediction_results.to_csv(
    "outputs/model_evaluation/predictions.csv",
    index=False
)

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(results_df.to_string(index=False))

print("\nBest Model:", best_model_name)
print("Saved: models/deliveriq_best_model.joblib")
print("Saved: outputs/model_evaluation/model_comparison.csv")
print("Saved: outputs/model_evaluation/predictions.csv")

print("\nSTAGE 9 COMPLETED")
