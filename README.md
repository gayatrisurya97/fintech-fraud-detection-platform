# FinTech Fraud Detection Platform

A production-style data platform built to detect fraudulent payment 
transactions across 100,000+ records — designed to mirror how a real 
FinTech analytics team would approach transaction monitoring.

---

## What This Project Does

Financial fraud is one of the biggest operational challenges facing UK 
FinTechs. This platform ingests raw payment transaction data, transforms 
it through a structured dbt pipeline, scores each transaction using a 
trained machine learning model, and surfaces the results through an 
executive Power BI dashboard.

The goal was to build something that resembles a real company's data 
infrastructure.

**Results:**
- ROC-AUC of 0.977 on fraud detection
- 88% precision on flagged transactions
- 100,000 transactions processed through automated pipeline
- Risk scores saved back to database and visualised in dashboard

---

## Architecture
```
Raw CSV (PaySim Dataset)
        |
Python Ingestion Script
        |
PostgreSQL — raw_transactions table
        |
dbt Transformations
        |-- stg_transactions        (clean and standardise)
        |-- int_transaction_features (engineer ML features)
        |-- int_customer_aggregates  (customer behaviour)
        |-- mart_transactions        (final reporting table)
        |
        |                    |
Power BI Dashboard     Random Forest Model
                             |
                      fraud_predictions table
                             |
                       Power BI Dashboard
```

---

## Tech Stack

- **Python** — data ingestion, feature engineering, machine learning
- **PostgreSQL** — raw and transformed data storage
- **dbt** — transformation pipeline with staging, intermediate, and mart layers
- **Scikit-learn** — Random Forest classifier with balanced class weights
- **SHAP** — model explainability and feature importance
- **Power BI** — four page executive dashboard with DAX measures
- **Git** — version control

---

## Project Structure
```
fintech-fraud-detection-platform/
|
|-- ingestion/
|   |-- load_data.py              
|
|-- dbt_project/
|   |-- models/
|       |-- staging/
|       |   |-- stg_transactions.sql
|       |-- intermediate/
|       |   |-- int_transaction_features.sql
|       |   |-- int_customer_aggregates.sql
|       |-- marts/
|           |-- mart_transactions.sql
|
|-- ml/
|   |-- fraud_model.py            
|   |-- confusion_matrix.png      
|   |-- shap_importance.png       
|
|-- dashboard/
|   |-- fraud_dashboard.pbix      
|
|-- .gitignore
|-- README.md
```

---

## Key Findings

TRANSFER and CASH_OUT transactions account for every single fraud case 
in the dataset. No fraud occurred in PAYMENT, CASH_IN, or DEBIT 
transactions — which makes sense given how mobile money fraud typically 
operates.

The strongest fraud signal turned out to be the sender balance dropping 
to exactly zero after a transaction. Legitimate transactions rarely drain 
an account completely. This single feature contributed most to the model's 
predictions according to SHAP analysis.

Large and very large transactions carry significantly higher fraud risk, 
though most fraudulent transactions were not the largest in absolute value 
— suggesting fraudsters deliberately keep amounts below obvious detection 
thresholds.

---

## Machine Learning

The model is a Random Forest classifier trained on engineered features 
from the dbt transformation layer. Class weights were balanced to handle 
the severe imbalance — only 0.1% of transactions are fraudulent.

| Metric | Score |
|--------|-------|
| ROC-AUC | 0.977 |
| Precision (Fraud) | 0.88 |
| Recall (Fraud) | 0.30 |
| F1 Score (Fraud) | 0.45 |

Recall is relatively low at 0.30 — a known limitation when working with 
only 23 fraud cases in the test set. The next iteration would apply SMOTE 
oversampling to generate synthetic fraud examples, tune the classification 
threshold below the default 0.5, and evaluate XGBoost as an alternative 
algorithm.

SHAP values were used to explain individual predictions, making the model 
interpretable for non-technical stakeholders. Each flagged transaction can 
be traced back to the specific features that drove the risk score.

---

## Dashboard

Four pages covering transaction overview, fraud analysis, customer 
intelligence, and model performance. Built on top of the dbt mart tables 
so the dashboard always reflects the latest pipeline output.





## Author

**Gayathri Kanchi**
MSc Data Science — Nottingham Trent University (2025)
[https://www.linkedin.com/in/gayathri-kanchi-245154307]