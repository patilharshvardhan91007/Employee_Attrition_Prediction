# Employee_Attrition_Prediction
Employee Attrition Prediction using Machine Learning and Streamlit

**Name:** Harshavardan Patil  
**BITS Email ID:** 2025ac05711@wilp.bits-pilani.ac.in

---

# 1. Problem Statement

Employee attrition is one of the major challenges faced by organizations, leading to increased recruitment costs, loss of organizational knowledge, and reduced productivity. The objective of this project is to develop machine learning models capable of predicting employee attrition using HR analytics data.

The developed system allows organizations to identify employees at risk of leaving and enables better workforce planning and retention strategies.

---

# 2. Dataset Description

Predictive Employee Attrition Analysis - IBM HR. (kaggle)

### Dataset Characteristics

| Attribute | Value |
|------------|---------|
| Total Records | 1470 |
| Total Features | 35 |
| Numerical Features | 28 |
| Categorical Features | 7 |
| Missing Values | 0 |
| Target Variable | Attrition |

### Target Variable

| Value | Meaning |
|---------|----------|
| 0 | Employee Stays |
| 1 | Employee Leaves |

### Preprocessing Performed

- Missing value check
- Label Encoding of categorical features
- Feature importance analysis using Random Forest
- Removal of non-informative features:
  - EmployeeCount
  - EmployeeNumber
  - Over18
  - StandardHours
- Feature scaling using StandardScaler

---

# 3. GitHub Repository

**Repository Link:**

> Add your GitHub repository URL here after uploading the project.

Example:

https://github.com/yourusername/Employee-Attrition-Prediction

Repository Contents:

- Streamlit Application
- Trained Machine Learning Models
- Model Comparison Results
- Test Dataset
- Scaler
- Label Encoders
- Project Documentation

---

# 4. Models Implemented

The following machine learning algorithms were implemented and evaluated:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Naive Bayes
5. Random Forest (Ensemble)

---

# 5. Model Comparison

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---------------|----------|----------|----------|----------|----------|----------|
| Logistic Regression | 0.874 | 0.805 | 0.692 | 0.383 | 0.493 | 0.453 |
| Decision Tree | 0.765 | 0.593 | 0.296 | 0.340 | 0.317 | 0.177 |
| KNN | 0.847 | 0.666 | 0.625 | 0.106 | 0.182 | 0.212 |
| Naive Bayes | 0.738 | 0.742 | 0.333 | 0.638 | 0.438 | 0.314 |
| Random Forest (Ensemble) | 0.830 | 0.804 | 0.385 | 0.106 | 0.167 | 0.132 |

---

# 6. Feature Importance Analysis

Random Forest feature importance analysis identified the following predictors as the most influential:

1. Monthly Income
2. Age
3. Total Working Years
4. Daily Rate
5. Hourly Rate
6. Monthly Rate
7. Distance From Home
8. OverTime
9. Years At Company
10. Years With Current Manager

These attributes contributed most significantly to employee attrition prediction.

---

# 7. Model Performance Observations

| ML Model Name | Observation |
|---------------|-------------|
| Logistic Regression | Achieved the highest overall accuracy (87.4%) and best MCC score. Demonstrated the most balanced performance among all models. |
| Decision Tree | Easy to interpret but showed lower accuracy and weaker generalization compared to other models. |
| KNN | Produced competitive accuracy but suffered from very low recall, indicating difficulty identifying employees who leave. |
| Naive Bayes | Achieved the highest recall among all models, making it effective at identifying attrition cases but resulted in lower overall accuracy. |
| Random Forest (Ensemble) | Delivered strong AUC performance and captured important predictors effectively but showed low recall in this dataset. |
| Overall Winner | **Logistic Regression** was selected as the best performing model because it achieved the highest accuracy, highest MCC value, and a strong AUC score while maintaining balanced classification performance. |

---

# 8. Streamlit Application Features

The Streamlit application includes:

### Single Employee Prediction

- Employee information input form
- Attrition prediction
- Attrition probability score

### Batch Prediction

- CSV upload functionality
- Bulk employee attrition prediction
- Downloadable prediction results

### Model Selection

Users can select:

- Logistic Regression
- Decision Tree
- KNN
- Naive Bayes
- Random Forest

### Evaluation Metrics Display

- Accuracy
- Precision
- Recall
- F1 Score
- AUC
- MCC

### Confusion Matrix Visualization

Model-wise confusion matrix is displayed dynamically based on the selected model.

---

# 9. Conclusion

This project successfully developed multiple machine learning models for employee attrition prediction using HR analytics data.

After preprocessing, feature selection, and model evaluation, Logistic Regression achieved the best predictive performance with an accuracy of 87.4%, AUC of 0.805, and MCC of 0.453.

Feature importance analysis identified MonthlyIncome, Age, TotalWorkingYears, OverTime, and YearsAtCompany as major factors influencing employee attrition. The developed Streamlit application enables both individual and batch prediction capabilities and can support organizations in implementing proactive employee retention strategies.

---
