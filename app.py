import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# ==========================
# PAGE CONFIG
# ==========================

st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Employee Attrition Prediction")
st.markdown(
    "Predict employee attrition using multiple machine learning models."
)

# ==========================
# LOAD MODELS
# ==========================

models = {
    "Logistic Regression":
        joblib.load("model/logistic_regression.pkl"),

    "Decision Tree":
        joblib.load("model/decision_tree.pkl"),

    "KNN":
        joblib.load("model/knn.pkl"),

    "Naive Bayes":
        joblib.load("model/naive_bayes.pkl"),

    "Random Forest":
        joblib.load("model/random_forest.pkl")
}

# ==========================
# LOAD MODELS CONFUSION MATRICES
# ==========================
models_cm = {
    "Logistic Regression":
        joblib.load("model/logistic_regression_cm.pkl"),

    "Decision Tree":
        joblib.load("model/decision_tree_cm.pkl"),

    "KNN":
        joblib.load("model/knn_cm.pkl"),

    "Naive Bayes":
        joblib.load("model/naive_bayes_cm.pkl"),

    "Random Forest":
        joblib.load("model/random_forest_cm.pkl")
}

# ==========================
# LOAD SCALER AND ENCODERS
# ==========================
scaler = joblib.load("model/scaler.pkl")
encoders = joblib.load("model/label_encoders.pkl")

# ==========================
# REQUIRED FEATURES
# ==========================

feature_order = [
    'Age',
    'BusinessTravel',
    'DailyRate',
    'Department',
    'DistanceFromHome',
    'Education',
    'EducationField',
    'EnvironmentSatisfaction',
    'Gender',
    'HourlyRate',
    'JobInvolvement',
    'JobLevel',
    'JobRole',
    'JobSatisfaction',
    'MaritalStatus',
    'MonthlyIncome',
    'MonthlyRate',
    'NumCompaniesWorked',
    'OverTime',
    'PercentSalaryHike',
    'PerformanceRating',
    'RelationshipSatisfaction',
    'StockOptionLevel',
    'TotalWorkingYears',
    'TrainingTimesLastYear',
    'WorkLifeBalance',
    'YearsAtCompany',
    'YearsInCurrentRole',
    'YearsSinceLastPromotion',
    'YearsWithCurrManager'
]

categorical_columns = [
    "BusinessTravel",
    "Department",
    "EducationField",
    "Gender",
    "JobRole",
    "MaritalStatus",
    "OverTime"
]

# ==========================
# SIDEBAR
# ==========================

st.sidebar.header("Settings")

selected_model = st.sidebar.selectbox(
    "Select Model",
    list(models.keys())
)

prediction_mode = st.sidebar.radio(
    "Prediction Mode",
    [
        "Single Employee",
        "CSV Upload"
    ]
)

def get_encoder_help(column):

    mapping = dict(
        zip(
            encoders[column].classes_,
            encoders[column].transform(
                encoders[column].classes_
            )
        )
    )

    return "\n".join(
        [f"{k} = {v}" for k, v in mapping.items()]
    )

# ========================================================
# SINGLE EMPLOYEE MODE
# ========================================================

if prediction_mode == "Single Employee":

    st.subheader("Employee Information")

    col1, col2 = st.columns(2)

    with col1:

        age = st.number_input("*Age*",18, 60, 30)

        business_travel = st.selectbox(
            "Business Travel",
            list(encoders["BusinessTravel"].classes_),
            help=get_encoder_help("BusinessTravel")
        )

        daily_rate = st.number_input(
            "*Daily Rate*",
            100,
            2000,
            800
        )

        department = st.selectbox(
            "Department",
            list(encoders["Department"].classes_),
            help=get_encoder_help("Department")
        )

        distance = st.number_input(
            "*Distance From Home*",
            0,
            50,
            5
        )

        education = st.selectbox(
            "Education",
            {
                "Below College":1,
                "College":2,
                "Bachelor":3,
                "Master":4,
                "Doctor":5
            }
        )

        education_field = st.selectbox(
            "Education Field",
            list(encoders["EducationField"].classes_)
        )

        environment = st.selectbox(
            "Environment Satisfaction",
            [1,2,3,4],
            help="""
            1 = Low
            2 = Medium
            3 = High
            4 = Very High
            """
        )

        gender = st.selectbox(
            "Gender",
            list(encoders["Gender"].classes_),
            help=get_encoder_help("Gender")
        )

        hourly_rate = st.number_input(
            "Hourly Rate",
            10,
            100,
            50
        )

        job_involvement = st.selectbox(
            "Job Involvement",
            [1,2,3,4],
            help="""
            1 = Low
            2 = Medium
            3 = High
            4 = Very High
            """
        )

        job_level = st.number_input(
            "Job Level",
            1,
            5,
            2
        )

        job_role = st.selectbox(
            "Job Role",
            list(encoders["JobRole"].classes_),
            help=get_encoder_help("JobRole")
        )

        job_satisfaction = st.selectbox(
            "Job Satisfaction",
            [1,2,3,4],
            help="""
            1 = Low
            2 = Medium
            3 = High
            4 = Very High
            """
        )

    with col2:

        marital_status = st.selectbox(
            "Marital Status",
            list(encoders["MaritalStatus"].classes_),
            help=get_encoder_help("MaritalStatus")
        )

        monthly_income = st.number_input(
            "*Monthly Income*",
            1000,
            50000,
            5000
        )

        monthly_rate = st.number_input(
            "Monthly Rate",
            1000,
            30000,
            10000
        )

        companies = st.number_input(
            "Num Companies Worked",
            0,
            15,
            2
        )

        overtime = st.selectbox(
            "*Over Time*",
            [0, 1]
        )

        salary_hike = st.number_input(
            "Percent Salary Hike",
            0,
            40,
            15
        )

        performance_rating_map = {
            "Low": 1,
            "Good": 2,
            "Excellent": 3,
            "Outstanding": 4
        }

        performance_rating = st.selectbox(
            "Performance Rating",
            performance_rating_map.keys(),
            help="""
            1 = Low
            2 = Good
            3 = Excellent
            4 = Outstanding
            """
        )

        performance_rating = performance_rating_map[performance_rating]

        relationship = st.selectbox(
            "Relationship Satisfaction",
            [1,2,3,4],
            help="""
            1 = Low
            2 = Medium
            3 = High
            4 = Very High
            """
        )

        stock_option_map = {
            "None": 0,
            "Low": 1,
            "Medium": 2,
            "High": 3
        }

        stock_option_level = st.selectbox(
            "Stock Option Level",
            stock_option_map.keys(),
            help="""
            0 = None
            1 = Low
            2 = Medium
            3 = High
            """
        )

        stock_option_level = stock_option_map[stock_option_level]

        total_years = st.number_input(
            "*Total Working Years*",
            0,
            40,
            10
        )

        training = st.number_input(
            "Training Times Last Year",
            0,
            10,
            2
        )

        work_life_map = {
            "Bad": 1,
            "Good": 2,
            "Better": 3,
            "Best": 4
        }

        work_life_balance = st.selectbox(
            "Work Life Balance",
            work_life_map.keys(),
            help="""
            1 = Bad
            2 = Good
            3 = Better
            4 = Best
            """
        )

        work_life_balance = work_life_map[work_life_balance]

        years_company = st.number_input(
            "*Years At Company*",
            0,
            40,
            5
        )

        years_role = st.number_input(
            "Years In Current Role",
            0,
            20,
            5
        )

        years_promotion = st.number_input(
            "Years Since Last Promotion",
            0,
            20,
            2
        )

        years_manager = st.number_input(
            "*Years With Current Manager*",
            0,
            20,
            4
        )

    if st.button("Predict Attrition"):

        row = pd.DataFrame({

            "Age":[age],

            "BusinessTravel":[
                encoders["BusinessTravel"]
                .transform([business_travel])[0]
            ],

            "DailyRate":[daily_rate],

            "Department":[
                encoders["Department"]
                .transform([department])[0]
            ],

            "DistanceFromHome":[distance],

            "Education":[education],

            "EducationField":[
                encoders["EducationField"]
                .transform([education_field])[0]
            ],

            "EnvironmentSatisfaction":[environment],

            "Gender":[
                encoders["Gender"]
                .transform([gender])[0]
            ],

            "HourlyRate":[hourly_rate],

            "JobInvolvement":[job_involvement],

            "JobLevel":[job_level],

            "JobRole":[
                encoders["JobRole"]
                .transform([job_role])[0]
            ],

            "JobSatisfaction":[job_satisfaction],

            "MaritalStatus":[
                encoders["MaritalStatus"]
                .transform([marital_status])[0]
            ],

            "MonthlyIncome":[monthly_income],

            "MonthlyRate":[monthly_rate],

            "NumCompaniesWorked":[companies],

            "OverTime":[overtime],

            "PercentSalaryHike":[salary_hike],

            "PerformanceRating":[performance],

            "RelationshipSatisfaction":[relationship],

            "StockOptionLevel":[stock_option],

            "TotalWorkingYears":[total_years],

            "TrainingTimesLastYear":[training],

            "WorkLifeBalance":[work_life],

            "YearsAtCompany":[years_company],

            "YearsInCurrentRole":[years_role],

            "YearsSinceLastPromotion":[years_promotion],

            "YearsWithCurrManager":[years_manager]

        })

        row = row[feature_order]

        model = models[selected_model]

        if selected_model in ["Logistic Regression", "KNN"]:

            row_scaled = scaler.transform(row)

            prediction = model.predict(row_scaled)

            probability = model.predict_proba(row_scaled)[0][1]

        else:

            prediction = model.predict(row)

            probability = model.predict_proba(row)[0][1]

        st.subheader("Prediction Result")

        st.metric("Attrition Probability",f"{probability*100:.2f}%")

        if prediction[0] == 1:
            st.error("Employee Likely To Leave")
        else:
            st.success("Employee Likely To Stay")

# ========================================================
# CSV MODE
# ========================================================

else:

    st.subheader("Batch Prediction Using CSV")

    uploaded_file = st.file_uploader(
        "Upload CSV File",
        type=["csv"]
    )

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file)

        st.write("Uploaded Data")

        st.dataframe(df.head())

        missing_columns = [
            col
            for col in feature_order
            if col not in df.columns
        ]

        if missing_columns:

            st.error(
                f"Missing Columns: {missing_columns}"
            )

        else:

            for col in categorical_columns:

                if df[col].dtype == object:

                    df[col] = (
                        encoders[col]
                        .transform(df[col])
                    )

            model = models[selected_model]

            if selected_model in [
                "Logistic Regression",
                "KNN"
            ]:

                transformed = scaler.transform(
                    df[feature_order]
                )

                predictions = (
                    model.predict(transformed)
                )

                probabilities = (
                    model.predict_proba(
                        transformed
                    )[:,1]
                )

            else:

                predictions = (
                    model.predict(
                        df[feature_order]
                    )
                )

                probabilities = (
                    model.predict_proba(
                        df[feature_order]
                    )[:,1]
                )

            df["Prediction"] = predictions
            df["Attrition_Probability"] = probabilities

            st.success(
                "Prediction Completed"
            )

            st.dataframe(df.head())

            csv = df.to_csv(
                index=False
            )

            st.download_button(
                "Download Prediction Results",
                csv,
                "attrition_predictions.csv",
                "text/csv"
            )




col1, col2 = st.columns([2,1])

with col1:
    st.subheader("Model Evalution Metrics")
    metrics_df = pd.read_csv("model_comparison.csv")
    selected_metrics = metrics_df[metrics_df["Model"] == selected_model].iloc[0]
    st.dataframe(selected_metrics)

with col2:
    st.subheader("Confusion Matrix")
    cm = models_cm[selected_model]
    fig, ax = plt.subplots(figsize=(3,3))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=True,
        ax=ax
    )

    st.pyplot(fig)