# 💳 Credit Card Fraud Detection System

A machine learning project to detect **fraudulent credit card transactions** from highly **imbalanced** data. The solution trains a **Random Forest classifier** and serves real-time predictions via a lightweight **Streamlit** app.

---

## Table of Contents

- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [Tech Stack](#️-tech-stack)
- [Project Files](#-project-files)
- [Workflow](#-workflow)
- [How to Run (Local)](#-how-to-run-local)
- [Model Performance](#-model-performance)
- [Notes on the Dataset](#-notes-on-the-dataset)
- [Future Improvements](#-future-improvements)
- [Author](#-author)

---

## 📌 Project Overview

Credit card fraud detection is a crucial problem in the financial industry, as fraudulent transactions can lead to significant financial losses. This project leverages machine learning to detect such anomalies using anonymized transaction data, with a focus on metrics that matter for imbalanced classification.

---

## 🎯 Key Features

- Handles **extreme class imbalance** with stratified splits
- Trains a **Random Forest Classifier** for robust fraud detection
- Evaluates using metrics suited for imbalance: **Precision / Recall / F1**, **ROC-AUC**
- Exports a trained model artifact (`.joblib`) for quick inference
- Includes a **Streamlit app** (`app.py`) for interactive, real-time predictions

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.9 | Core language |
| scikit-learn | Model training and evaluation |
| Pandas / NumPy | Data manipulation |
| Matplotlib / Seaborn | Visualization |
| Streamlit | Interactive inference app |
| joblib | Model serialization |

---

## 📁 Project Files

```
credit-card-fraud-detection/
├── app.py                      # Streamlit inference app
├── random_forest_model.joblib  # Trained model artifact
├── requirements.txt            # Python dependencies
├── .gitignore                  # Python/Jupyter ignores
└── README.md                   # Project documentation
```

---

## 🧪 Workflow

### 1. Data Preprocessing

- Scaled the `Amount` feature using `StandardScaler`
- Dataset features are anonymized PCA components (`V1` to `V28`)
- Used a stratified train/test split to preserve class distribution

### 2. Model Training

- Trained a **Random Forest Classifier** on the preprocessed data
- Evaluated using:
  - Confusion Matrix
  - ROC-AUC Score
  - Precision, Recall, F1-score (fraud class)

### 3. Deployment

- Streamlit app loads the saved `.joblib` model for real-time predictions
- Can be hosted on **Hugging Face Spaces** for public access

---

## 🚀 How to Run (Local)

### Prerequisites

- Python 3.9+
- pip

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/NamanDwivedi0307/credit-card-fraud-detection.git
cd credit-card-fraud-detection

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the Streamlit app
streamlit run app.py
```

The app will open at `http://localhost:8501`. Enter transaction feature values in the sidebar and click **Predict** to get a fraud/genuine classification.

---

## 📊 Model Performance

The Random Forest model achieves strong results on the test set:

| Metric | Value |
|---|---|
| Overall Accuracy | ~99% |
| ROC-AUC | High (imbalance-aware) |
| Key class | Fraud (minority class) |

> For imbalanced fraud detection, **Precision, Recall, and F1 for the fraud class** are more informative than overall accuracy. Consider also reporting **PR-AUC** for the minority class.

---

## 📝 Notes on the Dataset

- The dataset contains anonymized PCA features (`V1`–`V28`) plus `Amount` and `Time`
- The fraud class is heavily underrepresented (typically < 0.2% of transactions)
- Commonly sourced from the [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

---

## 🔮 Future Improvements

- Apply **SMOTE** or class weighting to improve recall on the fraud class
- Evaluate **PR-AUC** alongside ROC-AUC for a fuller picture
- Add explainability with **feature importances** or **SHAP values**
- Add a reproducible training notebook/script
- Set up CI for automated model evaluation

---

## 👤 Author

**Naman Dwivedi**

