
# 🚚 DeliverIQ – Food Delivery Analytics

### Data Wrangling and Preprocessing Mini Project

DeliverIQ is a Python-based Data Wrangling and Preprocessing project that analyzes food delivery operations using the Zomato Delivery Dataset.

The project demonstrates data collection, data cleaning, data transformation, data integration, exploratory data analysis (EDA), and data preprocessing.

---

## 📌 Project Overview

| Details | Information |
|---|---|
| Project Name | DeliverIQ |
| Domain | Food Delivery Analytics |
| Dataset | Zomato Delivery Dataset |
| Original Records | 45,584 |
| Original Features | 20 |
| Programming Language | Python |
| IDE | Visual Studio Code |
| Project Type | DWP Mini Project |

## 🎯 Objectives

- Collect and understand food delivery data.
- Identify and handle missing values.
- Detect invalid and duplicate records.
- Transform numerical and categorical features.
- Integrate weather and traffic reference data.
- Perform exploratory data analysis.
- Apply encoding and feature scaling.
- Prepare machine learning-ready datasets.

## 🛠️ Technology Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- VS Code

## 📂 Project Structure

```text
DeliverIQ/
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── reference/
│   └── final/
│
├── models/
├── notebooks/
├── outputs/
│   └── eda_graphs/
│
├── reports/
│   └── DeliverIQ_Project_Report.md
│
├── src/
│   ├── stage2_quality.py
│   ├── stage3_cleaning.py
│   ├── stage4_transformation.py
│   ├── stage5_integration.py
│   ├── stage6_eda.py
│   ├── stage7_preprocessing.py
│   └── stage8_final_verification.py
│
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

## ⚙️ Installation & Setup

### 1. Clone Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Navigate to Project

```bash
cd DeliverIQ
```

### 3. Create Virtual Environment

```bash
python3 -m venv venv
```

### 4. Activate Environment

Mac/Linux:

```bash
source venv/bin/activate
```

Windows:

```bash
venv\Scripts\activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

## 🚀 Project Execution

Run the stages in the following order:

```bash
python main.py
python src/stage2_quality.py
python src/stage3_cleaning.py
python src/stage4_transformation.py
python src/stage5_integration.py
python src/stage6_eda.py
python src/stage7_preprocessing.py
python src/stage8_final_verification.py
```

Run Stage 1 first using `main.py`.

## 📊 Exploratory Data Analysis

The project includes the following visualizations:

1. Delivery Time Distribution
2. Weather vs Delivery Time
3. Traffic vs Delivery Time
4. City Category Analysis
5. Vehicle Analysis
6. Delivery Person Rating Distribution
7. Correlation Heatmap
8. Order Type Analysis

## 📁 Generated Outputs

- Cleaned Dataset
- Integrated Dataset
- Training Dataset
- Testing Dataset
- Weather and Traffic Lookup Tables
- EDA Visualizations
- Data Quality Reports
- Preprocessing Pipeline

## 🎯 Future Scope

- Food delivery time prediction.
- Interactive analytics dashboard.
- Real-time weather integration.
- Advanced delivery performance analytics.

## 👨‍💻 Author

**Tejas Patil**

B.Tech – Computer Science and Engineering (Data Science)

Data Wrangling and Preprocessing Mini Project

---

## 📜 License

This project is developed for academic and educational purposes.
