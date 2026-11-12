# GetIntern Week 5 — Automated ML Data Preprocessor

## Project Overview

This project prepares the IBM Telco Customer Churn dataset for machine learning by performing categorical feature encoding, numerical feature scaling, and train-test partitioning.

The processed datasets are exported as CSV files for downstream model development.

## Tools and Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Google Colab

## Dataset

IBM Telco Customer Churn dataset.

Original dataset shape: 7,043 rows × 21 columns.

After removing the customer identifier and separating the target:
- Feature records: 7,043
- Original input features: 19
- Target: Churn

## Preprocessing Methodology

### 1. Data Cleaning
- Removed duplicate customer records.
- Converted `TotalCharges` into numeric format.
- Replaced missing `TotalCharges` values with zero.
- Removed `customerID` because it is an identifier.

### 2. Categorical Encoding
- Applied `pd.get_dummies()` to categorical input features.
- Used `drop_first=True` to reduce redundant dummy columns.
- Encoded the target variable `Churn`: No = 0 and Yes = 1.

### 3. Numerical Feature Scaling
- Used Scikit-learn's `StandardScaler`.
- Scaled original numerical columns: `SeniorCitizen`, `tenure`, `MonthlyCharges`, and `TotalCharges`.
- Fitted the scaler on training data only and transformed testing data using the same fitted scaler.

### 4. Train-Test Split
- Training partition: 80%
- Testing partition: 20%
- Random state: 42
- Stratified using the target variable to preserve class proportions.

## Before and After Feature Summary

| Description | Before | After |
|---|---:|---:|
| Original dataset rows | 7,043 | 7,043 |
| Original dataset columns | 21 | — |
| Input features | 19 | 30 |
| Categorical input columns | 15 | 0 |
| Original numerical input columns | 4 | 4 scaled |
| Target classes | Yes / No | 1 / 0 |

## Train-Test Partition Summary

| Output File | Expected Shape |
|---|---|
| X_train.csv | 5,634 × 30 |
| X_test.csv | 1,409 × 30 |
| y_train.csv | 5,634 × 1 |
| y_test.csv | 1,409 × 1 |

## Generated Artifacts

- `X_train.csv` — Scaled training features.
- `X_test.csv` — Scaled testing features.
- `y_train.csv` — Training target labels.
- `y_test.csv` — Testing target labels.

## Conclusion

This project demonstrates the essential preprocessing steps required before machine learning model training. Categorical variables were encoded, numerical features were standardized, and the dataset was split into training and testing partitions.

The four exported CSV files are ready to be used as input for a downstream machine learning workflow.

## Dataset Source

IBM Telco Customer Churn dataset:
https://github.com/IBM/telco-customer-churn-on-icp4d
