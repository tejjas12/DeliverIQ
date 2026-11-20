
"""
DeliverIQ - Stage 10
Model Evaluation and Data Visualization
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# Create output folder
OUTPUT_DIR = "outputs/model_evaluation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load model comparison results
results = pd.read_csv(
    f"{OUTPUT_DIR}/model_comparison.csv"
)

# Load actual and predicted values
predictions = pd.read_csv(
    f"{OUTPUT_DIR}/predictions.csv"
)

# Load best trained model
model = joblib.load(
    "models/deliveriq_best_model.joblib"
)

print("=" * 55)
print("DELIVERIQ - MODEL VISUALIZATION")
print("=" * 55)

# ---------------------------------------
# 1. Model Performance Comparison
# ---------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(17, 5))

metrics = [
    ("MAE", "MAE Comparison", "Error (Minutes)"),
    ("RMSE", "RMSE Comparison", "Error (Minutes)"),
    ("R2 Score", "R2 Score Comparison", "R2 Score")
]

for ax, (column, title, ylabel) in zip(axes, metrics):

    sorted_data = results.sort_values(
        column,
        ascending=(column != "R2 Score")
    )

    bars = ax.bar(
        sorted_data["Model"],
        sorted_data[column]
    )

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=9
        )

plt.suptitle(
    "DeliverIQ - Machine Learning Model Comparison",
    fontsize=16,
    fontweight="bold"
)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/01_model_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ---------------------------------------
# 2. Actual vs Predicted Delivery Time
# ---------------------------------------

actual = predictions["Actual Delivery Time"]
predicted = predictions["Predicted Delivery Time"]

plt.figure(figsize=(9, 6))

plt.scatter(
    actual,
    predicted,
    alpha=0.45
)

minimum = min(actual.min(), predicted.min())
maximum = max(actual.max(), predicted.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--",
    linewidth=2,
    label="Perfect Prediction"
)

plt.xlabel("Actual Delivery Time (Minutes)")
plt.ylabel("Predicted Delivery Time (Minutes)")
plt.title(
    "Actual vs Predicted Delivery Time",
    fontsize=15,
    fontweight="bold"
)

plt.legend()
plt.grid(alpha=0.25)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/02_actual_vs_predicted.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ---------------------------------------
# 3. Prediction Error Distribution
# ---------------------------------------

errors = actual - predicted

plt.figure(figsize=(9, 6))

plt.hist(
    errors,
    bins=30,
    edgecolor="black",
    alpha=0.8
)

plt.axvline(
    0,
    linestyle="--",
    linewidth=2,
    label="Zero Error"
)

plt.xlabel("Prediction Error (Minutes)")
plt.ylabel("Number of Orders")
plt.title(
    "Prediction Error Distribution",
    fontsize=15,
    fontweight="bold"
)

plt.legend()
plt.grid(axis="y", alpha=0.25)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/03_error_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ---------------------------------------
# 4. Feature Importance
# ---------------------------------------

if hasattr(model, "feature_importances_"):

    train_data = pd.read_csv(
        "data/final/deliveriq_train.csv"
    )

    feature_names = train_data.drop(
        columns=["Time_taken (min)"]
    ).columns

    importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    })

    importance = importance.sort_values(
        "Importance",
        ascending=False
    ).head(12)

    plt.figure(figsize=(10, 7))

    plt.barh(
        importance["Feature"][::-1],
        importance["Importance"][::-1]
    )

    plt.xlabel("Importance Score")
    plt.title(
        "Top 12 Feature Importance - Random Forest",
        fontsize=15,
        fontweight="bold"
    )

    plt.grid(axis="x", alpha=0.25)
    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/04_feature_importance.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

else:
    print("Feature importance is not available.")

print("\nAll visualization graphs generated!")

print("\nGenerated Files:")

for file in sorted(os.listdir(OUTPUT_DIR)):
    if file.endswith(".png"):
        print("-", file)

print("\nSTAGE 10 COMPLETED")
