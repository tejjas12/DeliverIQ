
# ============================================================
# DELIVERIQ - STAGE 8
# FINAL DATA QUALITY VERIFICATION
# ============================================================

import pandas as pd
from pathlib import Path
from datetime import datetime

# ------------------------------------------------------------
# 1. PROJECT PATHS
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_PATH = BASE_DIR / "data" / "raw" / "Zomato Dataset.csv"
CLEAN_PATH = BASE_DIR / "data" / "processed" / "deliveriq_cleaned.csv"
INTEGRATED_PATH = BASE_DIR / "data" / "final" / "deliveriq_integrated.csv"

TRAIN_PATH = BASE_DIR / "data" / "final" / "deliveriq_train.csv"
TEST_PATH = BASE_DIR / "data" / "final" / "deliveriq_test.csv"

OUTPUT_DIR = BASE_DIR / "outputs"
GRAPH_DIR = OUTPUT_DIR / "eda_graphs"

OUTPUT_DIR.mkdir(exist_ok=True)

print("\n" + "=" * 65)
print("       DELIVERIQ - FINAL VERIFICATION")
print("=" * 65)

# ------------------------------------------------------------
# 2. VERIFY REQUIRED FILES
# ------------------------------------------------------------

print("\n[1] FILE VERIFICATION")

required_files = {
    "Raw Dataset": RAW_PATH,
    "Cleaned Dataset": CLEAN_PATH,
    "Integrated Dataset": INTEGRATED_PATH,
    "Training Dataset": TRAIN_PATH,
    "Testing Dataset": TEST_PATH,
    "Stage 1 Report": OUTPUT_DIR / "stage1_inspection_report.txt",
    "Stage 2 Report": OUTPUT_DIR / "stage2_quality_report.txt",
    "Stage 3 Summary": OUTPUT_DIR / "stage3_cleaning_summary.csv",
    "Stage 4 Summary": OUTPUT_DIR / "stage4_transformation_summary.csv",
    "Stage 5 Summary": OUTPUT_DIR / "stage5_integration_report.csv",
    "Stage 7 Summary": OUTPUT_DIR / "stage7_preprocessing_summary.csv",
}

file_results = []

for name, path in required_files.items():

    exists = path.exists()

    file_results.append({
        "File": name,
        "Status": "Available" if exists else "Missing",
        "Path": str(path.relative_to(BASE_DIR))
    })

    print(f"{name}: {'Available' if exists else 'MISSING'}")

# ------------------------------------------------------------
# 3. LOAD DATASETS
# ------------------------------------------------------------

print("\n[2] DATASET VERIFICATION")

datasets = {}

for name, path in {
    "Raw": RAW_PATH,
    "Cleaned": CLEAN_PATH,
    "Integrated": INTEGRATED_PATH,
    "Train": TRAIN_PATH,
    "Test": TEST_PATH
}.items():

    if path.exists():
        datasets[name] = pd.read_csv(path)

        print(
            f"{name} Dataset:",
            datasets[name].shape
        )

# ------------------------------------------------------------
# 4. VERIFY ROW COUNTS
# ------------------------------------------------------------

print("\n[3] ROW COUNT VERIFICATION")

row_checks = []

if "Raw" in datasets and "Cleaned" in datasets:

    row_checks.append({
        "Check": "Cleaned rows do not exceed raw rows",
        "Result": len(datasets["Cleaned"]) <= len(datasets["Raw"])
    })

if "Cleaned" in datasets and "Integrated" in datasets:

    row_checks.append({
        "Check": "Integration preserves row count",
        "Result": len(datasets["Cleaned"]) == len(datasets["Integrated"])
    })

if "Train" in datasets and "Test" in datasets:

    split_rows = (
        len(datasets["Train"]) +
        len(datasets["Test"])
    )

    row_checks.append({
        "Check": "Train and test rows are accounted for",
        "Result": split_rows > 0
    })

for check in row_checks:
    print(
        check["Check"],
        ":", "PASS" if check["Result"] else "FAIL"
    )

# ------------------------------------------------------------
# 5. VERIFY INTEGRATED DATASET
# ------------------------------------------------------------

print("\n[4] DATA INTEGRATION VERIFICATION")

integration_checks = []

if "Integrated" in datasets:

    integrated = datasets["Integrated"]

    expected_columns = [
        "Weather_Severity_Score",
        "Weather_Category",
        "Traffic_Congestion_Score",
        "Traffic_Category"
    ]

    for col in expected_columns:

        exists = col in integrated.columns

        integration_checks.append({
            "Check": f"Integrated column: {col}",
            "Result": exists
        })

        print(
            col,
            ":", "PASS" if exists else "FAIL"
        )

    for col in [
        "Weather_Severity_Score",
        "Traffic_Congestion_Score"
    ]:

        if col in integrated.columns:

            print(
                f"Unmatched values in {col}:",
                integrated[col].isna().sum()
            )

# ------------------------------------------------------------
# 6. VERIFY TRAINING AND TESTING DATA
# ------------------------------------------------------------

print("\n[5] PREPROCESSING VERIFICATION")

preprocessing_checks = []

for name in ["Train", "Test"]:

    if name not in datasets:
        continue

    data = datasets[name]

    missing_count = int(data.isnull().sum().sum())

    preprocessing_checks.append({
        "Check": f"{name} dataset has no missing values",
        "Result": missing_count == 0
    })

    print(f"{name} Missing Values:", missing_count)

    target_exists = "Time_taken (min)" in data.columns

    preprocessing_checks.append({
        "Check": f"{name} target column exists",
        "Result": target_exists
    })

    print("Target Column Available:", target_exists)

if "Train" in datasets and "Test" in datasets:

    train_rows = len(datasets["Train"])
    test_rows = len(datasets["Test"])

    total_split = train_rows + test_rows

    if total_split > 0:

        train_percentage = train_rows / total_split * 100
        test_percentage = test_rows / total_split * 100

        print(f"Training Split: {train_percentage:.2f}%")
        print(f"Testing Split: {test_percentage:.2f}%")

# ------------------------------------------------------------
# 7. VERIFY EDA GRAPHS
# ------------------------------------------------------------

print("\n[6] EDA GRAPH VERIFICATION")

expected_graphs = [
    "01_delivery_time_distribution.png",
    "02_weather_vs_delivery_time.png",
    "03_traffic_vs_delivery_time.png",
    "04_city_wise_delivery.png",
    "05_vehicle_analysis.png",
    "06_rating_distribution.png",
    "07_correlation_heatmap.png",
    "08_order_type_analysis.png"
]

graph_results = []

for graph in expected_graphs:

    exists = (GRAPH_DIR / graph).exists()

    graph_results.append({
        "Graph": graph,
        "Status": "Available" if exists else "Missing"
    })

    print(
        graph,
        ":", "Available" if exists else "MISSING"
    )

# ------------------------------------------------------------
# 8. COMBINE VERIFICATION RESULTS
# ------------------------------------------------------------

all_checks = (
    row_checks +
    integration_checks +
    preprocessing_checks
)

all_results = [
    item["Status"] == "Available"
    for item in file_results
]

all_results += [
    item["Result"] for item in all_checks
]

all_results += [
    item["Status"] == "Available"
    for item in graph_results
]

passed = sum(all_results)
total = len(all_results)

overall_status = (
    "PASS" if passed == total else "REVIEW REQUIRED"
)

# ------------------------------------------------------------
# 9. SAVE FINAL REPORTS
# ------------------------------------------------------------

file_report = pd.DataFrame(file_results)
check_report = pd.DataFrame(all_checks)
graph_report = pd.DataFrame(graph_results)

file_report.to_csv(
    OUTPUT_DIR / "stage8_file_verification.csv",
    index=False
)

check_report.to_csv(
    OUTPUT_DIR / "stage8_quality_checks.csv",
    index=False
)

graph_report.to_csv(
    OUTPUT_DIR / "stage8_eda_verification.csv",
    index=False
)

report_path = OUTPUT_DIR / "stage8_final_report.txt"

with open(report_path, "w", encoding="utf-8") as file:

    file.write("DELIVERIQ - FINAL VERIFICATION REPORT\n")
    file.write("=" * 55 + "\n")

    file.write(
        f"\nReport Date: {datetime.now().strftime('%d-%m-%Y %H:%M')}\n"
    )

    file.write(f"\nChecks Passed: {passed}/{total}\n")
    file.write(f"Overall Status: {overall_status}\n")

    file.write("\nFILE VERIFICATION\n")
    file.write(file_report.to_string(index=False))

    file.write("\n\nQUALITY CHECKS\n")
    file.write(check_report.to_string(index=False))

    file.write("\n\nEDA GRAPH VERIFICATION\n")
    file.write(graph_report.to_string(index=False))

print("\n" + "=" * 65)
print("FINAL VERIFICATION SUMMARY")
print("=" * 65)

print(f"Checks Passed: {passed}/{total}")
print("Overall Status:", overall_status)

print("\nFinal Reports Saved in outputs/")

print("\nSTAGE 8 COMPLETED!")
print("=" * 65)
