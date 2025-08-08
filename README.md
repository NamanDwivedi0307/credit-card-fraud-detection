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

