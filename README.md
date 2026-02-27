# 🏦 FinTech Fraud Detection Platform

> A production-style data platform that ingests, transforms, and analyses 
> 100,000+ payment transactions to detect fraud using Python, PostgreSQL, 
> dbt, Machine Learning, and Power BI.

---

## 📊 Project Overview

Financial fraud costs the UK economy billions annually. This project builds 
an end-to-end fraud detection platform simulating how a real FinTech company 
like Monzo or Wise would approach transaction monitoring — from raw data 
ingestion through to an executive dashboard with ML-powered risk scoring.

**Key Results:**
- 🎯 97.7% ROC-AUC fraud detection score
- 🔍 Processed 100,000 transactions through automated pipeline
- ⚡ 88% precision on fraud predictions
- 📈 4-page executive Power BI dashboard

---

## 🏗️ Architecture
```
Raw CSV Data (PaySim)
      ↓
Python Ingestion Script
      ↓
PostgreSQL (raw_transactions)
      ↓
dbt Transformations
├── stg_transactions (staging)
├── int_transaction_features (intermediate)
├── int_customer_aggregates (intermediate)
└── mart_transactions (mart)
      ↓                    ↓
Power BI Dashboard    ML Fraud Model (Random Forest)
                           ↓
                    fraud_predictions table
                           ↓
                      Power BI Dashboard
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Data Ingestion | Python, Pandas, SQLAlchemy |
| Database | PostgreSQL |
| Transformations | dbt (staging, intermediate, marts) |
| Machine Learning | Scikit-learn, Random Forest, SHAP |
| Visualisation | Power BI, DAX |
| Version Control | Git, GitHub |

---

## 📁 Project Structure
```
fintech-fraud-detection-platform/
│
├── ingestion/
│   └── load_data.py          # Loads raw CSV into PostgreSQL
│
├── dbt_project/
│   └── models/
│       ├── staging/
│       │   └── stg_transactions.sql
│       ├── intermediate/
│       │   ├── int_transaction_features.sql
│       │   └── int_customer_aggregates.sql
│       └── marts/
│           └── mart_transactions.sql
│
├── ml/
│   ├── fraud_model.py         # Random Forest + SHAP
│   ├── confusion_matrix.png   # Model evaluation
│   └── shap_importance.png    # Feature importance
│
├── dashboard/
│   └── fraud_dashboard.pbix  # Power BI dashboard
│
├── .env                       # Credentials (not on GitHub)
├── .gitignore
└── README.md
```

---

## 🔍 Key Business Insights

- **TRANSFER and CASH_OUT** transactions account for 100% of fraud cases
- Fraudulent transactions show a distinct pattern of **draining sender 
  balance to exactly zero**
- **Large and Very Large** amount categories carry significantly higher 
  fraud risk
- Fraud clusters during **specific hours** suggesting automated fraud bots

---

## 🤖 Machine Learning Model

**Algorithm:** Random Forest Classifier
**Handling Imbalance:** Class weights balanced (0.1% fraud rate)
**Train/Test Split:** 80/20 with stratification

| Metric | Score |
|--------|-------|
| ROC-AUC | 0.977 |
| Precision (Fraud) | 0.88 |
| Recall (Fraud) | 0.30 |
| F1 Score (Fraud) | 0.45 |

**Why Recall is 0.30:**
The dataset contains only 116 fraud cases out of 100,000 transactions. 
With only 23 fraud cases in the test set, the model has limited examples 
to learn from. Improvements would include SMOTE oversampling, threshold 
tuning, and XGBoost.

---

## 📸 Dashboard Screenshots

![Transaction Overview]
![Fraud Analysis]
![Customer Intelligence]
![Model Performance]

---

## ⚙️ Setup Instructions

1. Clone the repository
```bash
git clone https://github.com/gayatrisurya97/fintech-fraud-detection-platform.git
```

2. Install dependencies
```bash
pip install pandas sqlalchemy psycopg2-binary scikit-learn shap matplotlib seaborn python-dotenv dbt-postgres
```

3. Create `.env` file with your PostgreSQL credentials
```
DB_HOST=localhost
DB_PORT=5432
DB_NAME=fintech_db
DB_USER=postgres
DB_PASSWORD=your_password
```

4. Run ingestion
```bash
python ingestion/load_data.py
```

5. Run dbt transformations
```bash
cd dbt_project
dbt run
```

6. Run ML model
```bash
python ml/fraud_model.py
```

7. Open `dashboard/fraud_dashboard.pbix` in Power BI Desktop

---

## 🎓 What I Learned

- Professional data engineering patterns — raw, staging, intermediate, mart layers
- dbt best practices — refs, tests, documentation, lineage
- Handling imbalanced datasets in fraud detection
- Making ML explainable using SHAP values
- Building executive-ready dashboards in Power BI

---

## 👩‍💻 Author

**Gayathri Kanchi**
MSc Data Science — Nottingham Trent University (2025)
[https://www.linkedin.com/in/gayathri-kanchi-245154307]