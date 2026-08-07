# ==========================================================
# EMPLOYEE ATTRITION PREDICTION AND PERFORMANCE ANALYSIS
# PART 1 - IMPORTS, DATA LOADING & PREPROCESSING
# ==========================================================

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# Train Test Split
from sklearn.model_selection import train_test_split

# Machine Learning Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Evaluation Metrics
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

# Save Model
import joblib

# Plot Style
plt.style.use("ggplot")
sns.set(font_scale=1)

print("="*50)
print("Libraries Imported Successfully")
print("="*50)

# ==========================================================
# LOAD DATASET
# ==========================================================

df = pd.read_csv("IBM_HR_Employee_Attrition.csv")

print("\nDataset Loaded Successfully")
print("="*50)

# First Five Rows
print(df.head())

# Last Five Rows
print(df.tail())

# Dataset Shape
print("\nDataset Shape")
print(df.shape)

print("\nRows :", df.shape[0])
print("Columns :", df.shape[1])

# ==========================================================
# BASIC INFORMATION
# ==========================================================

print("\nColumn Names")
print(df.columns.tolist())

print("\nDataset Information")
df.info()

print("\nStatistical Summary")
print(df.describe().T)

print("\nMissing Values")
print(df.isnull().sum())

print("\nDuplicate Rows :", df.duplicated().sum())

# ==========================================================
# TARGET VARIABLE
# ==========================================================

print("\nAttrition Distribution")
print(df["Attrition"].value_counts())

plt.figure(figsize=(6,5))

sns.countplot(
    data=df,
    x="Attrition"
)

plt.title("Employee Attrition Distribution")
plt.show()

# ==========================================================
# NUMERICAL & CATEGORICAL FEATURES
# ==========================================================

numerical_columns = df.select_dtypes(include=np.number).columns

categorical_columns = df.select_dtypes(include="object").columns

print("\nNumerical Features")
for col in numerical_columns:
    print(col)

print("\nCategorical Features")
for col in categorical_columns:
    print(col)

# ==========================================================
# CORRELATION MATRIX
# ==========================================================

plt.figure(figsize=(18,12))

corr = df.corr(numeric_only=True)

sns.heatmap(
    corr,
    cmap="coolwarm",
    annot=False
)

plt.title("Correlation Matrix")
plt.show()

# ==========================================================
# HISTOGRAMS
# ==========================================================

df.hist(
    figsize=(20,18),
    bins=25
)

plt.tight_layout()
plt.show()

# ==========================================================
# BOXPLOTS
# ==========================================================

plt.figure(figsize=(18,10))

for i, col in enumerate(numerical_columns):

    plt.subplot(5,6,i+1)

    sns.boxplot(
        y=df[col]
    )

    plt.title(col)

plt.tight_layout()
plt.show()

# ==========================================================
# ATTRITION PIE CHART
# ==========================================================

attrition = df["Attrition"].value_counts()

plt.figure(figsize=(6,6))

plt.pie(
    attrition.values,
    labels=attrition.index,
    autopct="%1.1f%%",
    shadow=True,
    startangle=90
)

plt.title("Employee Attrition Percentage")

plt.show()

# ==========================================================
# COMPLETE SUMMARY
# ==========================================================
print(df.describe(include="all"))

# ==========================================================
# SAVE ORIGINAL DATASET
# ==========================================================

df.to_csv(
    "initial_dataset_backup.csv",
    index=False
)

print("\nInitial Dataset Saved Successfully")

# ==========================================================
# DATA PREPROCESSING
# ==========================================================

data = df.copy()

print("\nDataset Copied Successfully")

print("\nMissing Values")
print(data.isnull().sum())

print("\nDuplicate Rows :", data.duplicated().sum())

# Remove Duplicates
data = data.drop_duplicates()

print("\nNew Dataset Shape")
print(data.shape)

# ==========================================================
# LABEL ENCODING
# ==========================================================

data["Attrition"] = data["Attrition"].map({
    "Yes":1,
    "No":0
})

encoder = LabelEncoder()

categorical_columns = data.select_dtypes(include="object").columns

for col in categorical_columns:

    data[col] = encoder.fit_transform(data[col])

print("\nCategorical Encoding Completed")

data.info()

# ==========================================================
# FEATURE CORRELATION
# ==========================================================

plt.figure(figsize=(10,12))

data.corr()["Attrition"].sort_values().plot(
    kind="barh"
)

plt.title("Feature Correlation with Attrition")

plt.show()

print("\nPart 1 Completed Successfully")


# ==========================================================
# PART 2 - EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================================

print("="*60)
print("EXPLORATORY DATA ANALYSIS")
print("="*60)

# ==========================================================
# AGE DISTRIBUTION
# ==========================================================

plt.figure(figsize=(8,5))

sns.histplot(
    df["Age"],
    bins=20,
    kde=True,
    color="steelblue"
)

plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Employees")
plt.show()

# ==========================================================
# GENDER DISTRIBUTION
# ==========================================================

plt.figure(figsize=(6,5))

sns.countplot(
    data=df,
    x="Gender"
)

plt.title("Gender Distribution")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.show()

# ==========================================================
# ATTRITION BY GENDER
# ==========================================================

plt.figure(figsize=(6,5))

sns.countplot(
    data=df,
    x="Gender",
    hue="Attrition"
)

plt.title("Employee Attrition by Gender")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.show()

# ==========================================================
# DEPARTMENT ANALYSIS
# ==========================================================

plt.figure(figsize=(8,5))

sns.countplot(
    data=df,
    x="Department",
    hue="Attrition"
)

plt.xticks(rotation=15)

plt.title("Department-wise Attrition")
plt.xlabel("Department")
plt.ylabel("Employees")

plt.show()

# ==========================================================
# JOB ROLE ANALYSIS
# ==========================================================

plt.figure(figsize=(12,6))

sns.countplot(
    data=df,
    x="JobRole",
    hue="Attrition"
)

plt.xticks(rotation=45, ha="right")

plt.title("Job Role-wise Attrition")

plt.show()

# ==========================================================
# MONTHLY INCOME DISTRIBUTION
# ==========================================================

plt.figure(figsize=(8,5))

sns.histplot(
    df["MonthlyIncome"],
    bins=30,
    kde=True
)

plt.title("Monthly Income Distribution")
plt.xlabel("Monthly Income")

plt.show()

# ==========================================================
# MONTHLY INCOME VS ATTRITION
# ==========================================================

plt.figure(figsize=(8,5))

sns.boxplot(
    data=df,
    x="Attrition",
    y="MonthlyIncome"
)

plt.title("Monthly Income vs Attrition")

plt.show()

# ==========================================================
# JOB SATISFACTION
# ==========================================================

plt.figure(figsize=(7,5))

sns.countplot(
    data=df,
    x="JobSatisfaction",
    hue="Attrition"
)

plt.title("Job Satisfaction vs Attrition")

plt.show()

# ==========================================================
# ENVIRONMENT SATISFACTION
# ==========================================================

plt.figure(figsize=(7,5))

sns.countplot(
    data=df,
    x="EnvironmentSatisfaction",
    hue="Attrition"
)

plt.title("Environment Satisfaction vs Attrition")

plt.show()

# ==========================================================
# WORK LIFE BALANCE
# ==========================================================

plt.figure(figsize=(7,5))

sns.countplot(
    data=df,
    x="WorkLifeBalance",
    hue="Attrition"
)

plt.title("Work-Life Balance vs Attrition")

plt.show()

# ==========================================================
# OVERTIME ANALYSIS
# ==========================================================

plt.figure(figsize=(7,5))

sns.countplot(
    data=df,
    x="OverTime",
    hue="Attrition"
)

plt.title("OverTime vs Attrition")

plt.show()

# ==========================================================
# YEARS AT COMPANY
# ==========================================================

plt.figure(figsize=(8,5))

sns.histplot(
    df["YearsAtCompany"],
    bins=20,
    kde=True
)

plt.title("Years at Company Distribution")

plt.show()

# ==========================================================
# DISTANCE FROM HOME
# ==========================================================

plt.figure(figsize=(8,5))

sns.histplot(
    df["DistanceFromHome"],
    bins=20,
    kde=True
)

plt.title("Distance From Home Distribution")

plt.show()

# ==========================================================
# EDUCATION
# ==========================================================

plt.figure(figsize=(7,5))

sns.countplot(
    data=df,
    x="Education"
)

plt.title("Education Level Distribution")

plt.show()

# ==========================================================
# MARITAL STATUS
# ==========================================================

plt.figure(figsize=(7,5))

sns.countplot(
    data=df,
    x="MaritalStatus",
    hue="Attrition"
)

plt.title("Marital Status vs Attrition")

plt.show()

# ==========================================================
# BUSINESS TRAVEL
# ==========================================================

plt.figure(figsize=(8,5))

sns.countplot(
    data=df,
    x="BusinessTravel",
    hue="Attrition"
)

plt.xticks(rotation=20)

plt.title("Business Travel vs Attrition")

plt.show()

# ==========================================================
# CORRELATION HEATMAP
# ==========================================================

plt.figure(figsize=(16,12))

sns.heatmap(
    data.corr(),
    cmap="RdBu_r",
    center=0
)

plt.title("Feature Correlation Heatmap")

plt.show()

# ==========================================================
# PAIRPLOT
# ==========================================================

sns.pairplot(

    df[
        [
            "Age",
            "MonthlyIncome",
            "YearsAtCompany",
            "JobSatisfaction",
            "Attrition"
        ]
    ],

    hue="Attrition"

)

plt.show()

print("="*60)
print("EDA COMPLETED SUCCESSFULLY")
print("="*60)
# ==========================================================
# PART 3 - MACHINE LEARNING MODELS
# ==========================================================

print("="*60)
print("MODEL TRAINING")
print("="*60)

# ==========================================================
# FEATURE SCALING
# ==========================================================

scaler = StandardScaler()

X = data.drop("Attrition", axis=1)
y = data["Attrition"]

X_scaled = scaler.fit_transform(X)

# ==========================================================
# TRAIN TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training Data :", X_train.shape)
print("Testing Data  :", X_test.shape)

# ==========================================================
# SAVE PROCESSED DATASET
# ==========================================================

processed_data = pd.DataFrame(
    X_scaled,
    columns=X.columns
)

processed_data["Attrition"] = y.values

processed_data.to_csv(
    "processed_employee_data.csv",
    index=False
)

print("\nProcessed Dataset Saved Successfully")

# ==========================================================
# LOGISTIC REGRESSION
# ==========================================================

print("\n")
print("="*60)
print("LOGISTIC REGRESSION")
print("="*60)

lr = LogisticRegression(random_state=42)

lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)

lr_accuracy = accuracy_score(y_test, y_pred_lr)

print("Accuracy :", lr_accuracy)

print("\nClassification Report")

print(classification_report(
    y_test,
    y_pred_lr
))

cm_lr = confusion_matrix(
    y_test,
    y_pred_lr
)

plt.figure(figsize=(5,4))

sns.heatmap(
    cm_lr,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Logistic Regression Confusion Matrix")

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

# ==========================================================
# DECISION TREE
# ==========================================================

print("\n")
print("="*60)
print("DECISION TREE")
print("="*60)

dt = DecisionTreeClassifier(
    random_state=42
)

dt.fit(
    X_train,
    y_train
)

y_pred_dt = dt.predict(X_test)

dt_accuracy = accuracy_score(
    y_test,
    y_pred_dt
)

print("Accuracy :", dt_accuracy)

print("\nClassification Report")

print(classification_report(
    y_test,
    y_pred_dt
))

cm_dt = confusion_matrix(
    y_test,
    y_pred_dt
)

plt.figure(figsize=(5,4))

sns.heatmap(
    cm_dt,
    annot=True,
    fmt="d",
    cmap="Oranges"
)

plt.title("Decision Tree Confusion Matrix")

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

# ==========================================================
# RANDOM FOREST
# ==========================================================

print("\n")
print("="*60)
print("RANDOM FOREST")
print("="*60)

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    random_state=42
)

rf.fit(
    X_train,
    y_train
)

y_pred_rf = rf.predict(X_test)

rf_accuracy = accuracy_score(
    y_test,
    y_pred_rf
)

print("Accuracy :", rf_accuracy)

print("\nClassification Report")

print(classification_report(
    y_test,
    y_pred_rf
))

cm_rf = confusion_matrix(
    y_test,
    y_pred_rf
)

plt.figure(figsize=(5,4))

sns.heatmap(
    cm_rf,
    annot=True,
    fmt="d",
    cmap="Greens"
)

plt.title("Random Forest Confusion Matrix")

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

# ==========================================================
# MODEL COMPARISON
# ==========================================================

results = pd.DataFrame({

    "Model":[

        "Logistic Regression",
        "Decision Tree",
        "Random Forest"

    ],

    "Accuracy":[

        lr_accuracy,
        dt_accuracy,
        rf_accuracy

    ]

})

print("\nModel Comparison")

print(results)

# ==========================================================
# BAR CHART
# ==========================================================

plt.figure(figsize=(8,5))

sns.barplot(
    data=results,
    x="Model",
    y="Accuracy"
)

plt.ylim(0.75,1)

plt.title("Model Accuracy Comparison")

plt.xlabel("Machine Learning Models")
plt.ylabel("Accuracy")

plt.show()

print("="*60)
print("MODEL TRAINING COMPLETED")
print("="*60)
# ==========================================================
# PART 4 - FEATURE IMPORTANCE, PREDICTIONS & MODEL SAVING
# ==========================================================

print("="*60)
print("MODEL EVALUATION & PREDICTIONS")
print("="*60)

# ==========================================================
# FEATURE IMPORTANCE
# ==========================================================

importance = pd.DataFrame({

    "Feature": X.columns,

    "Importance": rf.feature_importances_

})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 15 Important Features")
print(importance.head(15))

plt.figure(figsize=(10,7))

sns.barplot(

    data=importance.head(15),

    x="Importance",

    y="Feature"

)

plt.title("Top 15 Important Features")

plt.xlabel("Importance Score")

plt.ylabel("Features")

plt.show()

# ==========================================================
# ROC CURVE
# ==========================================================

probability_scores = rf.predict_proba(X_test)[:,1]

fpr, tpr, thresholds = roc_curve(
    y_test,
    probability_scores
)

auc = roc_auc_score(
    y_test,
    probability_scores
)

print("\nROC-AUC Score :", round(auc,4))

plt.figure(figsize=(7,6))

plt.plot(
    fpr,
    tpr,
    linewidth=2,
    label=f"AUC = {auc:.3f}"
)

plt.plot(
    [0,1],
    [0,1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.show()

# ==========================================================
# EMPLOYEE PREDICTIONS
# ==========================================================

prediction_probability = rf.predict_proba(X_test)

prediction_df = pd.DataFrame({

    "Actual": y_test.values,

    "Prediction": y_pred_rf,

    "Probability_of_Leaving": prediction_probability[:,1]

})

print("\nPrediction Sample")

print(prediction_df.head(10))

# ==========================================================
# RISK CLASSIFICATION
# ==========================================================

def risk(prob):

    if prob >= 0.75:
        return "High"

    elif prob >= 0.50:
        return "Medium"

    else:
        return "Low"

prediction_df["Risk"] = prediction_df[
    "Probability_of_Leaving"
].apply(risk)

print("\nRisk Categories")

print(prediction_df.head())

# ==========================================================
# RISK DISTRIBUTION
# ==========================================================

plt.figure(figsize=(6,5))

sns.countplot(

    data=prediction_df,

    x="Risk",

    order=["Low","Medium","High"]

)

plt.title("Employee Risk Categories")

plt.xlabel("Risk Level")

plt.ylabel("Number of Employees")

plt.show()

# ==========================================================
# HIGH RISK EMPLOYEES
# ==========================================================

high_risk = prediction_df[
    prediction_df["Risk"]=="High"
]

print("\nHigh Risk Employees")

print(high_risk.head(20))

# ==========================================================
# SAVE PREDICTIONS
# ==========================================================

prediction_df.to_csv(

    "employee_predictions.csv",

    index=False

)

print("\nPredictions Saved Successfully")

# ==========================================================
# SAVE MODEL
# ==========================================================

joblib.dump(

    rf,

    "employee_attrition_model.pkl"

)

print("Random Forest Model Saved Successfully")

# ==========================================================
# FINAL RESULTS
# ==========================================================

print("\nFinal Model Accuracy")

print(results)

print("\nTop 10 Important Features")

print(importance.head(10))

# ==========================================================
# COMPLETED
# ==========================================================

print("="*60)
print("PROJECT COMPLETED SUCCESSFULLY")
print("="*60)

print("""
Files Generated:

1. initial_dataset_backup.csv
2. processed_employee_data.csv
3. employee_predictions.csv
4. employee_attrition_model.pkl

Ready for Power BI Dashboard
Ready for MBA Report
Ready for Viva
""")