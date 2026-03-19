## 📉 Customer Churn Prediction (Data Preprocessing & Neural Network)

A deep learning app that predicts telecom customer churn with a strong focus on data cleaning, feature engineering, and reproducible results.

⸻

## 🌐 Live Demo
- App deployed on Streamlit Cloud

⸻

## 🚀 Features
- Reads and cleans customer data: removes missing values and fixes data types.
- Encodes categorical columns and scales numeric ones using MinMaxScaler.
- Engineers robust feature pipelines for model-ready data.
- Handles class imbalance using class_weight to improve churn detection.
- Trains and tests a neural network achieving 74% accuracy and 85% churn recall.
- Supports single customer prediction and batch prediction via CSV upload.
- Includes data visualizations for churn by tenure and monthly charges.

⸻

## 🧠 Tech Stack
- Programming Language: Python
- Libraries: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
- Deep Learning: TensorFlow, Keras (Sequential Neural Network)
- App Framework: Streamlit
- Development Tools: Jupyter Notebook, VS Code

⸻

## 📊 Dataset
- Source: Telco Customer Churn dataset (WA_Fn-UseC_-Telco-Customer-Churn.csv)
- Label: "Churn" (1 = Yes, 0 = No)
- Size: 7,000+ records with contract, charges, and usage details

⸻

## 📈 Model Performance

| Metric | Score |
|---|---|
| Accuracy | 74% |
| Churn Recall | 85% |
| Churn Precision | 50% |
| F1-Score (Churn) | 0.63 |

> Optimized for recall because in a churn use case it is more costly to miss a churner than to wrongly flag a loyal customer. class_weight={0:1, 1:3} was used to handle class imbalance (73% No Churn vs 27% Churn).

⸻

## 🛠️ How It Works
- Data Loading & Preprocessing: Loads CSV, drops missing values, and corrects data types.
- Feature Engineering: Encodes binary columns (0/1), one-hot encodes service and contract details.
- Scaling: Scales tenure, MonthlyCharges, and TotalCharges using MinMaxScaler.
- Model: Splits data (80% train, 20% test), trains a neural network (2 hidden layers + dropout) with EarlyStopping and class_weight to handle imbalance.
- App Interface: Streamlit app preprocesses user input like training data and outputs churn predictions with risk statements.

⸻

## 📦 Files Included
- WA_Fn-UseC_-Telco-Customer-Churn.csv — Main dataset.
- CustomerChurnPrediction.ipynb — Analysis, preprocessing, and modeling notebook.
- app.py — Streamlit web app.
- customer_churn_model.keras — Trained neural network model.
- scaler.save — Saved MinMaxScaler for consistent preprocessing.
- requirements.txt — Python dependencies.