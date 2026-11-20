🚀 DeliverIQ – Food Delivery Analytics

Data Wrangling, Exploratory Data Analysis & Machine Learning Dashboard

DeliverIQ is a food delivery analytics project developed using Python, Flask, Pandas, and Machine Learning. The project focuses on transforming raw food delivery data into clean, structured, integrated, and analysis-ready data.

It provides an interactive analytics dashboard to explore delivery performance, understand factors affecting delivery time, visualize patterns, and demonstrate machine learning predictions.

🌐 Live Demo: [https://deliveriq-1upc.onrender.com](https://deliveriq-lupc.onrender.com)

📂 GitHub Repository: https://github.com/tejjas12/DeliverIQ

---

📌 Project Overview

Food delivery services generate large amounts of data related to orders, delivery partners, weather conditions, traffic, vehicles, and delivery time.

Analyzing this data can help identify patterns and understand the factors associated with delivery performance.

DeliverIQ implements a complete data processing workflow, including:

- Data Collection
- Data Understanding and Quality Assessment
- Data Cleaning
- Data Transformation
- Data Integration
- Exploratory Data Analysis (EDA)
- Data Preprocessing
- Final Data Verification
- Machine Learning Model Training and Visualization

The project also includes a Flask-based web dashboard for interactive data exploration and model-related insights.

🎯 Project Objectives

- Collect and understand real-world food delivery data.
- Identify and handle missing values, duplicate records, and inconsistent data.
- Perform data cleaning and transformation.
- Integrate additional weather and traffic reference information.
- Perform Exploratory Data Analysis using visualizations.
- Prepare numerical and categorical features for machine learning.
- Train and evaluate delivery-time prediction models.
- Present analytical results through an interactive dashboard.

📊 Dataset Description

Dataset: Zomato Food Delivery Dataset

The dataset contains approximately 45,584 records and 20 original attributes.

Important features include:

Feature| Description
Delivery_person_Age| Age of delivery partner
Delivery_person_Ratings| Delivery partner rating
Weather_conditions| Weather during delivery
Road_traffic_density| Traffic conditions
Type_of_order| Food order category
Type_of_vehicle| Vehicle used for delivery
Vehicle_condition| Condition of delivery vehicle
multiple_deliveries| Number of deliveries handled
City| City category
Order_Date| Order date
Time_Orderd| Order placement time
Time_taken (min)| Delivery time in minutes

Target Variable: "Time_taken (min)"

🛠️ Technologies Used

Technology| Purpose
Python| Core programming
Pandas| Data manipulation
NumPy| Numerical operations
Matplotlib| Data visualization
Seaborn| Statistical visualization
Scikit-learn| Data preprocessing and machine learning
Joblib| Model serialization
Flask| Backend web framework
HTML & CSS| Frontend interface
JavaScript| Dashboard interactions
Plotly| Interactive visualizations
Git & GitHub| Version control
Render| Web application deployment

⚙️ Project Methodology

Stage 1 – Data Collection

Collected the food delivery dataset and organized the raw data for processing.

Stage 2 – Data Understanding & Quality Assessment

Examined dataset structure, data types, missing values, duplicate records, and data quality issues.

Stage 3 – Data Cleaning

Handled missing values, duplicate records, inconsistent values, and unnecessary data.

Stage 4 – Data Transformation

Applied data transformation techniques, including feature creation, standardization, and normalization.

Stage 5 – Data Integration

Integrated weather and traffic reference information with the main dataset using Pandas merge operations.

Stage 6 – Exploratory Data Analysis (EDA)

Created visualizations to understand delivery-time patterns and relationships between different variables.

Stage 7 – Data Preprocessing

Prepared numerical and categorical features using imputation, scaling, encoding, and train-test splitting.

Stage 8 – Final Data Verification

Performed final verification of generated files, data quality, preprocessing outputs, and visualizations.

Stage 9 – Model Training & Visualization

Implemented machine learning model training and model evaluation visualizations for delivery-time prediction.

📈 Dashboard Features

DeliverIQ includes an interactive web dashboard with the following sections:

- Dashboard Overview: Displays dataset statistics and delivery performance.
- Analytics: Explores weather, traffic, city category, and vehicle-related delivery patterns.
- ML Models: Presents machine learning model evaluation results.
- AI Predictor: Provides a delivery-time prediction interface.
- Insights: Displays analytical findings and useful observations.
- About Project: Provides project information and methodology.

🤖 Machine Learning

The project includes a trained Random Forest Regression model for delivery-time prediction.

The model uses a preprocessing pipeline to prepare input features before generating predictions.

Model artifacts:

- "deliveriq_best_model.joblib"
- "deliveriq_preprocessor.joblib"

Model evaluation results and prediction outputs are stored separately in the project outputs directory.

📂 Project Structure

DeliverIQ/
│
├── app.py
├── main.py
├── requirements.txt
├── .python-version
├── .gitignore
├── README.md
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── final/
│   │   └── deliveriq_integrated.csv
│   └── reference/
│
├── models/
│   ├── deliveriq_best_model.joblib
│   └── deliveriq_preprocessor.joblib
│
├── outputs/
│   ├── eda_graphs/
│   └── model_evaluation/
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
│   ├── stage8_final_verification.py
│   ├── stage9_model_training.py
│   └── stage10_model_visualization.py
│
├── static/
│   ├── css/
│   └── js/
│
└── templates/
    └── index.html

Note: The structure above represents the main project files. Some generated files and local datasets may not be included in the repository.

💻 Installation & Setup

1. Clone the Repository

git clone https://github.com/tejjas12/DeliverIQ.git

2. Navigate to the Project Folder

cd DeliverIQ

3. Create a Virtual Environment

python -m venv venv

Activate it:

Mac/Linux:

source venv/bin/activate

Windows:

venv\Scripts\activate

4. Install Dependencies

pip install -r requirements.txt

5. Run the Application

python app.py

Open your browser:

http://127.0.0.1:5000

Note: The integrated dataset and model artifacts must be available at their configured paths before running the application.

📊 Key Project Outputs

- Cleaned dataset
- Transformed dataset
- Standardized and normalized datasets
- Integrated dataset
- EDA graphs and analysis reports
- Preprocessed training and testing datasets
- Trained machine learning model
- Model evaluation visualizations
- Interactive Flask dashboard

🔍 Project Verification

The final verification stage completed 30 out of 30 checks successfully, confirming the configured project files, data quality checks, and EDA outputs.

🌐 Deployment

The Flask application is deployed using Render.

Live Application:
https://deliveriq-1upc.onrender.com

🚀 Future Scope

- Real-time food delivery data integration.
- Advanced delivery-time prediction.
- Live weather and traffic API integration.
- Interactive location-based delivery analytics.
- Improved model comparison and optimization.
- Cloud-based data storage and automated pipelines.

👨‍💻 Author

Tejas Patil

B.Tech – Computer Science and Engineering (Data Science)

D Y Patil International University, Pune

- GitHub: https://github.com/tejjas12
- Project: https://github.com/tejjas12/DeliverIQ

---

📜 License

This project is developed for academic and educational purposes as part of the Data Wrangling and Preprocessing (DWP) internal assessment.

---

⭐ If you find this project interesting, consider giving the repository a star!

DeliverIQ – Transforming Food Delivery Data into Meaningful Insights.
