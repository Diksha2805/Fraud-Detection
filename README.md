# 💳 Financial Fraud Detection System - PRO VERSION

A machine learning-based system that detects fraudulent financial transactions using synthetic data, SMOTE balancing, and classification models like Random Forest and Logistic Regression. Outputs risk scores, predictions, and exports results for dashboards.

---

## 📑 Table of Contents
- [🚀 Features](#-features)
- [🛠️ Technologies Used](#-technologies-used)
- [📦 Installation](#-installation)
- [⚙️ How It Works](#-how-it-works)
- [📤 Output Files](#-output-files)
- [📊 Visualizations](#-visualizations)
- [🔮 Future Enhancements](#-future-enhancements)
- [📄 License](#-license)

---

## 🚀 Features
- Simulates 500+ synthetic financial transactions
- Handles missing values and categorical encoding
- Applies SMOTE for class balancing
- Trains RandomForest, LogisticRegression, or IsolationForest
- Calculates fraud prediction and risk score
- Assigns RiskLevel: High / Medium / Low
- Exports results to CSV and JSON
- Generates fraud and risk visualizations

---

## 🛠️ Technologies Used
- Python (pandas, numpy, matplotlib, seaborn)
- scikit-learn (ML models, metrics)
- imbalanced-learn (SMOTE)
- JSON / CSV Export

---

## 📦 Installation

1. Clone this repo:
   ```bash
   git clone https://github.com/your-username/fraud-detection.git
   cd fraud-detection
Install requirements:

bash
Copy
Edit
pip install -r requirements.txt
Run the script:

bash
Copy
Edit
python main.py
⚙️ How It Works
generate_transaction_data() creates synthetic transactions with real-world traits.

Missing values handled and categorical columns one-hot encoded.

SMOTE balances the dataset.

Model is trained and tested (RandomForest by default).

Predictions and risk scores are generated.

High-risk transactions trigger simulated alerts.

Outputs saved as CSV and JSON, and visual plots are shown.

📤 Output Files
fraud_predictions.csv: All predictions with risk scores and levels

fraud_predictions.json: JSON version of results

Fraud_Detection_Report.pdf: PDF with visualizations

README_Fraud_Detection.pdf: This documentation as PDF

📊 Visualizations
Includes:

Risk Level vs Fraud Predictions (bar plot)

Risk Score Distribution (histogram)

See the included Fraud_Detection_Report.pdf.

🔮 Future Enhancements
Use real-world transaction data

Deploy via Flask API

Connect to Power BI dashboards

Add email/SMS alerts for high-risk detection

📄 License
This project is open-source and free to use for learning and research.

yaml
Copy
Edit
