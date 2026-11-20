
# DeliverIQ – Food Delivery Analytics

## 1. Abstract

DeliverIQ is a Data Wrangling and Preprocessing mini project developed using Python. The project focuses on analyzing food delivery data and applying various data preparation techniques.

The project uses a Zomato food delivery dataset containing approximately 45,584 records and 20 original columns.

It demonstrates data collection, data cleaning, data transformation, data integration, exploratory data analysis, and data preprocessing.

## 2. Problem Statement

Food delivery platforms generate large amounts of data related to orders, delivery partners, weather conditions, traffic density, and delivery duration.

Raw data often contains missing values, inconsistent formats, invalid records, and unstructured information.

The objective of DeliverIQ is to clean, transform, integrate, and analyze food delivery data to generate meaningful insights and prepare a structured dataset for further analysis and machine learning.

## 3. Objectives

1. Collect food delivery data from Kaggle.
2. Identify and handle missing values.
3. Detect duplicate and invalid records.
4. Transform date, time, and numerical features.
5. Integrate weather and traffic reference datasets.
6. Perform exploratory data analysis.
7. Apply encoding and feature scaling.
8. Prepare training and testing datasets.

## 4. Dataset Description

Dataset Name: Zomato Delivery Dataset

Original Records: Approximately 45,584

Original Attributes: 20

Source: Kaggle

Important attributes:

- Delivery Person Age
- Delivery Person Ratings
- Weather Conditions
- Road Traffic Density
- Vehicle Condition
- Type of Order
- Type of Vehicle
- Restaurant Location
- Delivery Location
- Order Date
- Delivery Time

Target Variable for future prediction:
Time_taken (min)

## 5. Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- VS Code

## 6. Project Methodology

### Stage 1 – Data Collection

Collected the Zomato food delivery dataset from Kaggle and performed initial inspection.

### Stage 2 – Data Understanding

Examined data types, missing values, duplicate records, invalid values, and outliers.

### Stage 3 – Data Cleaning

Handled missing values, invalid ratings, age values, inconsistent date and time formats, and invalid coordinates.

### Stage 4 – Data Transformation

Performed date and time transformation, feature engineering, label encoding, one-hot encoding, normalization, and standardization.

### Stage 5 – Data Integration

Created weather and traffic reference datasets and integrated them with the main dataset using Pandas merge operations.

### Stage 6 – Exploratory Data Analysis

Analyzed delivery time, weather conditions, traffic density, city categories, vehicle types, ratings, and order types using visualizations.

### Stage 7 – Data Preprocessing

Performed feature selection, missing value imputation, categorical encoding, numerical scaling, and train-test splitting.

### Stage 8 – Final Verification

Verified generated datasets, reports, EDA graphs, and preprocessing outputs.

## 7. Expected Outcomes

- A structured and cleaned food delivery dataset.
- Integrated weather and traffic information.
- Visual insights into food delivery operations.
- Processed numerical and categorical features.
- Separate training and testing datasets.
- Reusable preprocessing pipeline.

## 8. Applications

- Food delivery performance analysis.
- Traffic impact analysis.
- Weather impact analysis.
- Delivery partner performance analysis.
- Future delivery time prediction.

## 9. Limitations

- The dataset represents the available historical records.
- Weather and traffic lookup scores are project-defined analytical values.
- Actual road travel distance may differ from calculated geographical distance.
- Predictive model performance has not yet been evaluated.

## 10. Conclusion

DeliverIQ demonstrates the practical implementation of major Data Wrangling and Preprocessing techniques using Python.

The project transforms raw food delivery data into structured and analysis-ready datasets through cleaning, transformation, integration, visualization, and preprocessing.

The resulting datasets can be used for further machine learning experiments and food delivery analytics.

## 11. Future Scope

- Develop a delivery time prediction model.
- Create an interactive analytics dashboard.
- Integrate real-time weather information.
- Analyze delivery performance across different regions.
