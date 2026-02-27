# ml/fraud_model.py
# PURPOSE: Build a fraud detection model using Random Forest
# We read from our dbt mart table -- clean, engineered data
# We use SHAP to explain predictions -- making ML business ready

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)
from sklearn.preprocessing import LabelEncoder
import shap
import warnings
warnings.filterwarnings('ignore')

# ── STEP 1: Load credentials and connect ──
load_dotenv()

engine = create_engine(
    f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@"
    f"{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)

def load_data():
    """
    Load data from our dbt mart table
    This is clean, engineered data -- ready for ML
    """
    print("Loading data from mart_transactions...")

    query = """
        select
            amount,
            sender_balance_before,
            sender_balance_after,
            sender_balance_diff,
            sender_balance_zero,
            receiver_balance_before,
            receiver_balance_after,
            receiver_balance_diff,
            receiver_is_merchant,
            balance_mismatch,
            transaction_type,
            amount_category,
            time_of_day,
            is_fraud
        from public.mart_transactions
    """

    df = pd.read_sql(query, engine)
    print(f"Loaded {len(df):,} rows")
    print(f"Fraud cases: {df['is_fraud'].sum():,}")
    print(f"Legitimate cases: {(df['is_fraud']==0).sum():,}")
    return df

def prepare_features(df):
    """
    Prepare features for ML model
    Convert categorical columns to numbers using Label Encoding
    ML models only understand numbers, not text
    """
    print("\nPreparing features...")

    # Encode categorical columns
    # Label encoding converts text to numbers
    # e.g. TRANSFER=0, CASH_OUT=1, PAYMENT=2 etc.
    le = LabelEncoder()

    df['transaction_type_encoded'] = le.fit_transform(df['transaction_type'])
    df['amount_category_encoded'] = le.fit_transform(df['amount_category'])
    df['time_of_day_encoded'] = le.fit_transform(df['time_of_day'])

    # Define features (X) and target (y)
    # Features = what the model learns from
    # Target = what the model predicts
    feature_columns = [
        'amount',
        'sender_balance_before',
        'sender_balance_after',
        'sender_balance_diff',
        'sender_balance_zero',
        'receiver_balance_before',
        'receiver_balance_after',
        'receiver_balance_diff',
        'receiver_is_merchant',
        'balance_mismatch',
        'transaction_type_encoded',
        'amount_category_encoded',
        'time_of_day_encoded'
    ]

    X = df[feature_columns]
    y = df['is_fraud']

    print(f"Features: {feature_columns}")
    print(f"Target distribution:\n{y.value_counts()}")

    return X, y, feature_columns

def train_models(X_train, X_test, y_train, y_test):
    """
    Train two models and compare them
    Logistic Regression = simple baseline
    Random Forest = our main model
    """

    print("\n" + "="*50)
    print("TRAINING MODELS")
    print("="*50)

    # ── Model 1: Logistic Regression (Baseline) ──
    print("\nTraining Logistic Regression...")
    lr = LogisticRegression(
        class_weight='balanced',  # handles imbalanced data
        max_iter=1000,
        random_state=42
    )
    lr.fit(X_train, y_train)
    lr_preds = lr.predict(X_test)
    lr_proba = lr.predict_proba(X_test)[:, 1]

    print("\nLogistic Regression Results:")
    print(classification_report(y_test, lr_preds))
    print(f"ROC-AUC: {roc_auc_score(y_test, lr_proba):.4f}")

    # ── Model 2: Random Forest (Main Model) ──
    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=100,         # 100 decision trees
        class_weight='balanced',  # handles imbalanced fraud data
        random_state=42,
        n_jobs=-1                 # use all CPU cores
    )
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_proba = rf.predict_proba(X_test)[:, 1]

    print("\nRandom Forest Results:")
    print(classification_report(y_test, rf_preds))
    print(f"ROC-AUC: {roc_auc_score(y_test, rf_proba):.4f}")

    return rf, rf_preds, rf_proba

def plot_confusion_matrix(y_test, rf_preds):
    """
    Visualise how many transactions were correctly classified
    True Positive = correctly identified fraud
    False Negative = missed fraud -- most dangerous in real world
    """
    cm = confusion_matrix(y_test, rf_preds)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Legitimate', 'Fraud'],
        yticklabels=['Legitimate', 'Fraud']
    )
    plt.title('Fraud Detection Model — Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('ml/confusion_matrix.png', dpi=150)
    plt.show()
    print("Saved: ml/confusion_matrix.png")

def explain_with_shap(rf, X_test, feature_columns):
    """
    SHAP explains WHY the model made each prediction
    This is what makes ML business ready
    Instead of a black box we can say:
    'This transaction was flagged because the balance dropped to zero
    AND the amount was unusually large'
    """
    print("\nGenerating SHAP explanations...")
    print("This may take a minute...")

    # Create SHAP explainer
    explainer = shap.TreeExplainer(rf)

    # Calculate SHAP values for first 500 test samples
    # Full dataset takes too long for demonstration
    shap_values = explainer.shap_values(X_test[:500])

    # Plot feature importance
    plt.figure()
    shap.summary_plot(
        shap_values[:, :, 1] if len(shap_values.shape) == 3 else shap_values[1],
        X_test[:500],
        feature_names=feature_columns,
        show=False
    )
    plt.title('SHAP Feature Importance — What Drives Fraud?', fontsize=12)
    plt.tight_layout()
    plt.savefig('ml/shap_importance.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: ml/shap_importance.png")

def save_predictions(rf, X, df_original):
    """
    Save fraud probability scores back to PostgreSQL
    Power BI will read these to show risk scores per transaction
    """
    print("\nSaving predictions to PostgreSQL...")

    fraud_probability = rf.predict_proba(X)[:, 1]
    fraud_prediction = rf.predict(X)

    results_df = pd.DataFrame({
        'sender_id': df_original['sender_id'] if 'sender_id' in df_original.columns else range(len(X)),
        'amount': df_original['amount'],
        'transaction_type': df_original['transaction_type'],
        'actual_fraud': df_original['is_fraud'],
        'predicted_fraud': fraud_prediction,
        'fraud_probability': fraud_probability,
        'risk_category': pd.cut(
            fraud_probability,
            bins=[0, 0.3, 0.6, 0.8, 1.0],
            labels=['Low Risk', 'Medium Risk', 'High Risk', 'Critical Risk']
        )
    })

    results_df.to_sql(
        name='fraud_predictions',
        con=engine,
        schema='public',
        if_exists='replace',
        index=False
    )

    print(f"Saved {len(results_df):,} predictions to fraud_predictions table")
    print("\nRisk category distribution:")
    print(results_df['risk_category'].value_counts())

    return results_df

if __name__ == '__main__':

    # Load data
    df = load_data()

    # Prepare features
    X, y, feature_columns = prepare_features(df)

    # Split into training and test sets
    # 80% train, 20% test -- standard split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y  # maintain fraud ratio in both sets
    )

    print(f"\nTraining set: {len(X_train):,} rows")
    print(f"Test set: {len(X_test):,} rows")

    # Train models
    rf, rf_preds, rf_proba = train_models(X_train, X_test, y_train, y_test)

    # Visualise results
    plot_confusion_matrix(y_test, rf_preds)

    # Explain with SHAP
    explain_with_shap(rf, X_test, feature_columns)

    # Save predictions back to PostgreSQL
    save_predictions(rf, X, df)

    print("\n✅ ML Pipeline Complete!")
    print("Check ml/ folder for confusion_matrix.png and shap_importance.png")
    print("Check pgAdmin for fraud_predictions table")