"""
Streamlit inference app for Credit Card Fraud Detection.

Loads the pre-trained Random Forest model (random_forest_model.joblib)
and allows users to enter transaction feature values and get a
real-time Genuine / Fraud prediction.

Usage:
    streamlit run app.py
"""

import joblib
import numpy as np
import streamlit as st

MODEL_PATH = "random_forest_model.joblib"

# Feature names match the Kaggle Credit Card Fraud dataset
# (30 features: Time, V1-V28, Amount)
FEATURE_NAMES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]


LABEL_FRAUD = "Fraud 🚨"
LABEL_GENUINE = "Genuine ✅"


@st.cache_resource
def load_model():
    """Load the trained Random Forest model from disk."""
    return joblib.load(MODEL_PATH)


def main():
    st.set_page_config(
        page_title="Credit Card Fraud Detection",
        page_icon="💳",
        layout="wide",
    )

    st.title("💳 Credit Card Fraud Detection")
    st.markdown(
        "Enter the transaction feature values below and click **Predict** "
        "to determine whether the transaction is **Genuine** or **Fraudulent**."
    )

    model = load_model()

    st.sidebar.header("Transaction Features")

    feature_values = {}
    for feature in FEATURE_NAMES:
        if feature == "Time":
            feature_values[feature] = st.sidebar.number_input(
                feature,
                value=0.0,
                help="Seconds elapsed since the first transaction in the dataset.",
            )
        elif feature == "Amount":
            feature_values[feature] = st.sidebar.number_input(
                feature,
                value=0.0,
                min_value=0.0,
                help="Transaction amount (USD).",
            )
        else:
            feature_values[feature] = st.sidebar.number_input(
                feature,
                value=0.0,
                help=f"Anonymized PCA feature {feature}.",
            )

    if st.sidebar.button("Predict", type="primary"):
        input_array = np.array([[feature_values[f] for f in FEATURE_NAMES]])
        prediction = model.predict(input_array)[0]
        probability = model.predict_proba(input_array)[0]

        st.markdown("---")
        if prediction == 1:
            st.error(
                f"🚨 **Fraudulent Transaction Detected**  \n"
                f"Fraud probability: **{probability[1]:.2%}**"
            )
        else:
            st.success(
                f"✅ **Genuine Transaction**  \n"
                f"Fraud probability: **{probability[1]:.2%}**"
            )

        st.markdown("### Prediction Details")
        col1, col2 = st.columns(2)
        col1.metric("Prediction", LABEL_FRAUD if prediction == 1 else LABEL_GENUINE)
        col2.metric("Fraud Probability", f"{probability[1]:.2%}")

    st.markdown("---")
    st.caption(
        "Model: Random Forest Classifier | "
        "Dataset: Anonymized credit card transactions (PCA features V1–V28)"
    )


if __name__ == "__main__":
    main()
