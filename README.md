# 💳 Credit Card Fraud Detection System

This project involves developing, training, and deploying a **machine learning model** to detect fraudulent credit card transactions. The model is trained on an imbalanced dataset using **Random Forest** and deployed as a real-time interactive app using **Streamlit** on **Hugging Face Spaces**.

---

## 📌 Project Overview

Credit card fraud detection is a crucial problem in the financial industry, as fraudulent transactions can lead to significant financial losses. This project leverages machine learning to detect such anomalies using anonymized transaction data.

---

## 🎯 Objectives

- Build an effective fraud detection system on highly **imbalanced data**

- Engineer features and preprocess data for robust model performance

- Achieve **high precision and recall**, especially for the minority (fraudulent) class

- Deploy a user-friendly **Streamlit** app for real-time predictions

---

## 🧠 Model Summary

- **Algorithm Used**: Random Forest Classifier

- **Accuracy**: 99%

- **Classes**: `0` (Genuine), `1` (Fraud)

---

## 🛠️ Tech Stack

- **Python 3.9**

- **scikit-learn**

- **Pandas / NumPy**

- **Matplotlib / Seaborn**

- **Streamlit**

- **Hugging Face Spaces (Gradio / Streamlit integration)**

---

## 🧪 Workflow

### 1. Data Preprocessing

- Used StandardScaler to scale the `Amount` feature

- Data is anonymized with PCA (V1 to V28); no feature names available

- Addressed extreme class imbalance using stratified train/test split


### 2. Model Training

- Trained a **Random Forest Classifier**

- Evaluated using:

  - **Confusion Matrix**

  - **ROC-AUC Score**

  - **Precision, Recall, F1-score**


### 3. Deployment

- Built a **Streamlit app** for end-user interaction

- Hosted on **Hugging Face Spaces** for public access

---

## ▶️ How to Run

- Ensure Python 3.13+ is available
- Install dependencies:

```bash
python3 -m pip install --user --break-system-packages -r requirements.txt
```

- Start the Streamlit app:

```bash
streamlit run app.py --server.headless true
```

- Open the provided URL to use the app. Use the Realtime tab for single predictions or upload a CSV in the Batch tab.

## ℹ️ Notes

- The repository includes a `random_forest_model.joblib`. If it fails to load due to environment mismatches, the app will automatically train a small synthetic fallback model so you can still interact with the UI.
- You can also provide a custom path to a model file via the `MODEL_PATH` environment variable.

