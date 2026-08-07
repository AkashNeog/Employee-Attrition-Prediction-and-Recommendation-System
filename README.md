# Employee Attrition Prediction and Recommendation System

## Overview

This project presents a Machine Learning-based Employee Attrition Prediction and Recommendation System developed using the IBM HR Analytics Employee Attrition dataset. The objective is to analyze employee data, identify factors influencing attrition, predict employees who are likely to leave the organization, and provide actionable business insights through an interactive Power BI dashboard.

The project combines Exploratory Data Analysis (EDA), Machine Learning, and Business Intelligence to support HR professionals in making informed employee retention decisions.

---

## Objectives

- Analyze employee attrition patterns.
- Perform data preprocessing and exploratory data analysis.
- Train and compare multiple machine learning models.
- Predict employee attrition.
- Generate employee prediction reports.
- Build an interactive Power BI dashboard for HR analytics.

---

## Dataset

**Dataset Used:** IBM HR Analytics Employee Attrition Dataset

The dataset contains employee information such as:

- Age
- Gender
- Department
- Job Role
- Monthly Income
- Education
- Marital Status
- Overtime
- Job Satisfaction
- Environment Satisfaction
- Work-Life Balance
- Years at Company
- Attrition Status

---

## Technologies Used

### Programming Language
- Python

### Libraries
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib

### Visualization
- Microsoft Power BI

### Development Environment
- Google Colab
- Jupyter Notebook

---

## Project Workflow

1. Data Collection
2. Data Preprocessing
3. Exploratory Data Analysis (EDA)
4. Feature Engineering
5. Machine Learning Model Training
6. Model Evaluation
7. Employee Attrition Prediction
8. Power BI Dashboard Development

---

## Exploratory Data Analysis

The following analyses were performed:

- Missing Value Analysis
- Duplicate Record Detection
- Correlation Matrix
- Histograms
- Boxplots
- Attrition Distribution
- Department-wise Analysis
- Job Role Analysis
- Gender Analysis
- Monthly Income Distribution
- Overtime Analysis
- Job Satisfaction Analysis
- Work-Life Balance Analysis

---

## Machine Learning Models

The following supervised learning algorithms were implemented:

- Logistic Regression
- Decision Tree
- Random Forest

---

## Model Evaluation Metrics

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- ROC Curve
- Feature Importance

---

## Model Performance

| Model | Accuracy |
|--------|---------:|
| Logistic Regression | **87.41%** |
| Random Forest | **82.99%** |
| Decision Tree | **76.19%** |

**Best Performing Model:** Logistic Regression

---

## Power BI Dashboard

The dashboard includes:

- Total Employees KPI
- Average Monthly Income
- Average Job Satisfaction
- Average Work-Life Balance
- Employee Attrition Distribution
- Department-wise Attrition Analysis
- Job Role-wise Attrition Analysis
- Overtime Analysis
- Employee Recommendation Table
- Interactive Slicers

---

## Project Structure

```text
Employee-Attrition-Prediction-and-Recommendation-System/

├── README.md
├── LICENSE
├── requirements.txt

├── notebooks/
│   └── Employee_Attrition_Recommendation_System.ipynb

├── dataset/
│   ├── IBM_HR_Employee_Attrition.csv
│   ├── employee_predictions.csv
│   └── processed_employee_data.csv

├── powerbi/
│   └── Employee_Attrition_Analytics_Dashboard.pbix

├── model/
│   └── employee_attrition_model.pkl
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/Employee-Attrition-Prediction-and-Recommendation-System.git
```

Navigate to the project folder:

```bash
cd Employee-Attrition-Prediction-and-Recommendation-System
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Open and run the notebook using Jupyter Notebook or Google Colab.

---

## Project Outputs

The project generates the following files:

- processed_employee_data.csv
- employee_predictions.csv
- employee_attrition_model.pkl

---

## Business Recommendations

Based on the analysis, organizations should:

- Improve employee engagement.
- Enhance work-life balance.
- Reduce excessive overtime.
- Review salary and compensation policies.
- Provide career development opportunities.
- Conduct regular employee feedback sessions.
- Use HR analytics for continuous workforce monitoring.

---

## Future Scope

Future enhancements may include:

- Deep Learning models
- XGBoost and LightGBM
- Explainable AI (SHAP/LIME)
- Real-time prediction system
- Web application deployment
- Cloud deployment
- Integration with HR Management Systems

---

## License

This project is licensed under the MIT License.

---

## Author

**Akash Jyoti Neog**

MBA Major Project

Employee Attrition Prediction and Recommendation System

---

If you found this project useful, consider giving it a ⭐ on GitHub.
