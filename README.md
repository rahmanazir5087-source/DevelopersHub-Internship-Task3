# DevelopersHub-Internship-Task3
Churn prediction model built with Logistic Regression and Random Forest on 10,000 bank customer records. Random Forest achieved 86% accuracy. Age was the strongest churn predictor, and customers with 3+ products churned at a significantly higher rate. Python, pandas, scikit-learn, seaborn.
# Customer Churn Prediction (Bank Customers)

Part of the Data Science & Analytics Internship at DevelopersHub Corporation (Task 3).

## Objective

Identify which bank customers are likely to leave, using personal and account-level details such as age, balance, number of products, and active membership status.

## Dataset

Churn Modelling Dataset, 10,000 bank customers from France, Spain, and Germany with 14 columns. Target column is `Exited` (1 = churned, 0 = stayed). Around 20% of customers in the dataset churned.

## Approach

- Dropped non-predictive columns: `RowNumber`, `CustomerId`, `Surname`
- Label encoded `Gender` (two values, binary)
- One-hot encoded `Geography` (three countries, to avoid implying false numeric ordering)
- Explored churn patterns by age, balance, number of products, and active membership status
- Split 80/20 with stratification to preserve the churn ratio in both sets
- Trained two classifiers: Logistic Regression (with StandardScaler) and Random Forest (100 estimators, no scaling needed)
- Evaluated with accuracy, confusion matrix, and classification report
- Extracted feature importances from the Random Forest to identify key churn drivers

## Results

| Model | Accuracy |
|---|---|
| Logistic Regression | 81% |
| Random Forest | 86% |

Random Forest outperformed logistic regression, particularly in recall on the churned class, meaning it caught more actual churners.

## Key insights

- Age is the strongest single predictor of churn, customers between 40 and 60 exit at a much higher rate than younger customers
- Customers with 3 or 4 products churn at a sharply higher rate than those with 1 or 2, pointing to over-selling as a retention risk
- Inactive members churn at roughly twice the rate of active ones, making inactivity an early warning signal
- Credit score and balance contribute to churn prediction but are not the dominant drivers

## Files

- `Customer_Churn_Prediction.ipynb` - full notebook (data prep, EDA, modeling, feature importance, evaluation)
- `churn_data.csv` - dataset used

## Tools

Python, pandas, matplotlib, seaborn, scikit-learn
