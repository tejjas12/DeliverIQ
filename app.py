
from flask import Flask, render_template, jsonify, request, send_from_directory
from pathlib import Path
import pandas as pd
import joblib

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = BASE_DIR / "data/final/deliveriq_integrated.csv"
MODEL_FILE = BASE_DIR / "models/deliveriq_best_model.joblib"
PREPROCESSOR_FILE = BASE_DIR / "models/deliveriq_preprocessor.joblib"

MODEL_RESULTS_FILE = BASE_DIR / "outputs/model_evaluation/model_comparison.csv"
PREDICTIONS_FILE = BASE_DIR / "outputs/model_evaluation/predictions.csv"
GRAPH_DIR = BASE_DIR / "outputs/model_evaluation"

# Load trained model and preprocessor
model = joblib.load(MODEL_FILE)
preprocessor = joblib.load(PREPROCESSOR_FILE)

# Actual project lookup mappings
WEATHER_SCORES = {
    "Sunny": 1,
    "Cloudy": 2,
    "Windy": 2,
    "Fog": 2,
    "Stormy": 3,
    "Sandstorms": 5
}

TRAFFIC_SCORES = {
    "Low": 1,
    "Medium": 2,
    "High": 3,
    "Jam": 4
}

CATEGORICAL_OPTIONS = {
    "Weather_conditions": list(WEATHER_SCORES),
    "Road_traffic_density": list(TRAFFIC_SCORES),
    "Type_of_order": ["Buffet", "Drinks", "Meal", "Snack"],
    "Type_of_vehicle": [
        "bicycle",
        "electric_scooter",
        "motorcycle",
        "scooter"
    ],
    "Festival": ["No", "Yes"],
    "City": ["Metropolitian", "Semi-Urban", "Urban"]
}

NUMERIC_FIELDS = [
    "Delivery_person_Age",
    "Delivery_person_Ratings",
    "Vehicle_condition",
    "multiple_deliveries",
    "Delivery_Distance_km",
    "Order_Hour",
    "Order_Month",
    "Order_Weekday",
    "Is_Weekend"
]


def load_csv(path):
    if path.exists():
        return pd.read_csv(path)
    return None


def clean_records(data):
    return data.astype(object).where(
        pd.notna(data), None
    ).to_dict(orient="records")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/overview")
def overview():
    df = load_csv(DATA_FILE)

    if df is None:
        return jsonify({"error": "Dataset not found"}), 404

    target = "Time_taken (min)"

    return jsonify({
        "total_records": int(len(df)),
        "total_columns": int(len(df.columns)),
        "average_delivery_time": round(float(df[target].mean()), 2),
        "minimum_delivery_time": int(df[target].min()),
        "maximum_delivery_time": int(df[target].max()),
        "dataset_status": "Available"
    })


@app.route("/api/analytics")
def analytics():
    df = load_csv(DATA_FILE)

    if df is None:
        return jsonify({"error": "Dataset not found"}), 404

    target = "Time_taken (min)"

    categories = [
        "Weather_conditions",
        "Road_traffic_density",
        "City",
        "Type_of_vehicle",
        "Type_of_order"
    ]

    result = {}

    for column in categories:
        if column in df.columns:
            grouped = (
                df.groupby(column, dropna=False)[target]
                .mean()
                .round(2)
                .reset_index()
            )
            result[column] = clean_records(grouped)

    return jsonify(result)


@app.route("/api/models")
def models():
    df = load_csv(MODEL_RESULTS_FILE)

    if df is None:
        return jsonify({"error": "Model results not found"}), 404

    return jsonify({
        "results": clean_records(df),
        "total_models": int(len(df))
    })


@app.route("/api/predictions")
def predictions():
    df = load_csv(PREDICTIONS_FILE)

    if df is None:
        return jsonify({"error": "Predictions not found"}), 404

    return jsonify({
        "total_predictions": int(len(df)),
        "sample": clean_records(df.head(100))
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True) or {}

        # Validate categorical fields
        for field, allowed in CATEGORICAL_OPTIONS.items():
            value = data.get(field)

            if value not in allowed:
                return jsonify({
                    "error": f"Invalid value for {field}"
                }), 400

        # Validate numeric fields
        values = {}

        for field in NUMERIC_FIELDS:
            try:
                value = float(data[field])
            except (KeyError, TypeError, ValueError):
                return jsonify({
                    "error": f"Valid numeric value required for {field}"
                }), 400

            if not pd.notna(value):
                return jsonify({
                    "error": f"Invalid number for {field}"
                }), 400

            values[field] = value

        if not 0 <= values["Order_Hour"] <= 23:
            return jsonify({"error": "Order hour must be 0-23"}), 400

        if not 1 <= values["Order_Month"] <= 12:
            return jsonify({"error": "Order month must be 1-12"}), 400

        if not 0 <= values["Order_Weekday"] <= 6:
            return jsonify({"error": "Order weekday must be 0-6"}), 400

        if values["Is_Weekend"] not in [0, 1]:
            return jsonify({"error": "Is_Weekend must be 0 or 1"}), 400

        if values["Delivery_Distance_km"] < 0:
            return jsonify({"error": "Distance cannot be negative"}), 400

        # Construct original 17-feature input
        input_data = {
            "Delivery_person_Age": values["Delivery_person_Age"],
            "Delivery_person_Ratings": values["Delivery_person_Ratings"],
            "Vehicle_condition": values["Vehicle_condition"],
            "multiple_deliveries": values["multiple_deliveries"],
            "Weather_Severity_Score": WEATHER_SCORES[data["Weather_conditions"]],
            "Traffic_Congestion_Score": TRAFFIC_SCORES[data["Road_traffic_density"]],
            "Delivery_Distance_km": values["Delivery_Distance_km"],
            "Order_Hour": values["Order_Hour"],
            "Order_Month": values["Order_Month"],
            "Order_Weekday": values["Order_Weekday"],
            "Is_Weekend": values["Is_Weekend"],
            "Weather_conditions": data["Weather_conditions"],
            "Road_traffic_density": data["Road_traffic_density"],
            "Type_of_order": data["Type_of_order"],
            "Type_of_vehicle": data["Type_of_vehicle"],
            "Festival": data["Festival"],
            "City": data["City"]
        }

        input_df = pd.DataFrame([input_data])

        # Transform raw input using the saved preprocessor
        transformed = preprocessor.transform(input_df)

        # Preserve transformed feature names when available
        if hasattr(preprocessor, "get_feature_names_out"):
            feature_names = preprocessor.get_feature_names_out()
            transformed_df = pd.DataFrame(
                transformed,
                columns=feature_names
            )
            prediction = model.predict(transformed_df)[0]
        else:
            prediction = model.predict(transformed)[0]

        prediction = round(float(prediction), 2)

        return jsonify({
            "success": True,
            "predicted_delivery_time": prediction,
            "unit": "minutes",
            "model": "Random Forest Regressor"
        })

    except Exception as e:
        app.logger.exception("Prediction failed")
        return jsonify({
            "error": "Prediction failed. Check server logs.",
            "details": str(e)
        }), 500


@app.route("/graphs/<path:filename>")
def graphs(filename):
    return send_from_directory(GRAPH_DIR, filename)


@app.route("/api/health")
def health():
    return jsonify({
        "status": "running",
        "project": "DeliverIQ",
        "model": type(model).__name__
    })


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
