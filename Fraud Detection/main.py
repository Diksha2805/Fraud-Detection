import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import json

# ==============================================
# 💳 Financial Fraud Detection System - PRO VERSION
# ==============================================

sys.stdout.reconfigure(encoding='utf-8')  # Ensure UTF-8 console prints


# ================================
# 🔥 STEP 1: Data Simulation & Preprocessing
# ================================
def generate_transaction_data(n=500, seed=42):
    """
    Simulate realistic transaction data with missing values and categorical fields.
    """
    np.random.seed(seed)
    data = {
        'TransactionID': [f"T{i+1:04d}" for i in range(n)],
        'Amount': np.random.exponential(scale=100, size=n),
        'Time': np.random.randint(0, 86400, size=n),
        'Location': np.random.choice(['Delhi', 'Mumbai', 'Chennai', 'Online'], size=n),
        'Device': np.random.choice(['Mobile', 'Web', 'ATM'], size=n),
        'Class': np.random.choice([0, 1], p=[0.95, 0.05], size=n)
    }
    df = pd.DataFrame(data)

    # Introduce missing values randomly
    for col in ['Amount', 'Location']:
        df.loc[df.sample(frac=0.05).index, col] = np.nan
    return df


def preprocess_data(df):
    """
    Handle missing values, encode categorical variables, scale numeric features.
    """
    # Fix: No inplace (Pandas 3.0 safe)
    df['Amount'] = df['Amount'].fillna(df['Amount'].median())
    df['Location'] = df['Location'].fillna('Unknown')

    # Feature engineering: TransactionAmountCategory
    df['AmountCategory'] = pd.cut(
        df['Amount'], bins=[-1, 50, 200, np.inf],
        labels=['Low', 'Medium', 'High']
    )

    # One-hot encode categorical variables
    df_encoded = pd.get_dummies(df, columns=['Location', 'Device', 'AmountCategory'], drop_first=True)

    # Scale numerical features
    scaler = StandardScaler()
    features = df_encoded.drop(['TransactionID', 'Class'], axis=1)
    X_scaled = scaler.fit_transform(features)

    return X_scaled, df['Class'], df_encoded


# ================================
# 🤖 STEP 2: Fraud Detection Models
# ================================
def get_model(model_name="RandomForest"):
    """
    Return a ML model based on the chosen algorithm.
    """
    if model_name == "RandomForest":
        return RandomForestClassifier(class_weight='balanced', random_state=1)
    elif model_name == "LogisticRegression":
        return LogisticRegression(solver='liblinear', class_weight='balanced')
    elif model_name == "IsolationForest":
        return IsolationForest(contamination=0.05, random_state=1)
    else:
        raise ValueError("Unsupported model: Choose RandomForest, LogisticRegression, or IsolationForest")


def train_and_predict(X, y, model_name="RandomForest"):
    """
    Train the selected model and return predictions + risk scores.
    """
    model = get_model(model_name)

    # Fix: Handle imbalanced data with SMOTE
    smote = SMOTE(random_state=1)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    if model_name == "IsolationForest":
        model.fit(X_resampled)
        predictions = model.predict(X)
        predictions = np.where(predictions == -1, 1, 0)
        risk_scores = np.random.uniform(0.5, 1.0, size=len(predictions)) * predictions
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X_resampled, y_resampled, test_size=0.2, random_state=1
        )
        model.fit(X_train, y_train)
        predictions = model.predict(X)
        risk_scores = model.predict_proba(X)[:, 1]
        print("\n📊 Model Performance (After SMOTE Balancing):")
        print(classification_report(y_test, model.predict(X_test)))

    return model, predictions, risk_scores


# ================================
# 🚨 STEP 3: Alert System (Simulated)
# ================================
def send_alert(transaction_id, amount, location, risk_score):
    """
    Simulate sending an email alert for high-risk transactions.
    """
    print(f"🚨 ALERT: Fraud suspected! Transaction {transaction_id} | Amount: ₹{amount:.2f} | Location: {location} | RiskScore: {risk_score:.2f}")


# ================================
# 📦 STEP 4: Export Data
# ================================
def assign_risk_level(score):
    """
    Classify RiskScore into High, Medium, Low.
    """
    if score >= 0.8:
        return "High"
    elif score >= 0.5:
        return "Medium"
    else:
        return "Low"


def export_data(df_original, df_encoded, predictions, risk_scores, filename="fraud_predictions.csv"):
    """
    Merge results and export CSV + JSON for dashboards.
    """
    df_encoded['Prediction'] = predictions
    df_encoded['RiskScore'] = risk_scores
    df_encoded['RiskLevel'] = df_encoded['RiskScore'].apply(assign_risk_level)

    df_final = pd.concat([
        df_original[['TransactionID', 'Amount', 'Time', 'Location', 'Device', 'Class']],
        df_encoded[['Prediction', 'RiskScore', 'RiskLevel']]
    ], axis=1)

    # Save CSV
    df_final.to_csv(filename, index=False, encoding='utf-8-sig')
    print(f"✅ CSV exported: {filename}")

    # Save JSON
    json_file = filename.replace('.csv', '.json')
    df_final.to_json(json_file, orient='records', indent=2)
    print(f"✅ JSON exported: {json_file}")

    # Trigger alerts for high-risk transactions
    high_risk = df_final[df_final['RiskLevel'] == 'High']
    for _, row in high_risk.iterrows():
        send_alert(row['TransactionID'], row['Amount'], row['Location'], row['RiskScore'])

    return df_final


# ================================
# 📊 STEP 5: Visualization
# ================================
def plot_risk_distribution(df_final):
    """
    Show fraud predictions and risk levels visually.
    """
    plt.figure(figsize=(10, 6))
    sns.countplot(x='RiskLevel', hue='Prediction', data=df_final, palette='coolwarm')
    plt.title("Fraud Prediction Counts by Risk Level")
    plt.xlabel("Risk Level")
    plt.ylabel("Number of Transactions")
    plt.legend(title="Prediction (1=Fraud)")
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.histplot(df_final['RiskScore'], bins=30, kde=True, color='purple')
    plt.title("Risk Score Distribution")
    plt.xlabel("Risk Score")
    plt.ylabel("Frequency")
    plt.show()


# ================================
# 🚀 MAIN EXECUTION
# ================================
def main():
    print("\n" + "=" * 60)
    print("💳 FINANCIAL FRAUD DETECTION SYSTEM - PRO VERSION")
    print("=" * 60 + "\n")

    # Generate and preprocess data
    df = generate_transaction_data()
    print(f"📥 Loaded dataset with {len(df)} transactions.")
    X, y, df_encoded = preprocess_data(df)
    print("🛠️ Preprocessing complete. Features ready.")

    # Train and predict
    model_name = "RandomForest"  # Change to LogisticRegression or IsolationForest
    model, predictions, risk_scores = train_and_predict(X, y, model_name=model_name)

    # Export results
    df_final = export_data(df, df_encoded, predictions, risk_scores, filename="fraud_predictions.csv")

    # Plot visualizations
    plot_risk_distribution(df_final)

    print("\n🎯 Fraud detection pipeline completed successfully!")
    print("📊 Data ready for Power BI and web dashboards.\n")


if __name__ == "__main__":
    main()