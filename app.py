import os
import io
from typing import List, Optional

import numpy as np
import pandas as pd
import streamlit as st

try:
    import joblib
except Exception as import_error:
    joblib = None


MODEL_PATH = os.environ.get("MODEL_PATH", "random_forest_model.joblib")


def load_model(model_path: str):
    if joblib is None:
        raise ImportError("joblib is not available. Please ensure dependencies are installed.")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    model = joblib.load(model_path)
    return model


def get_feature_names(model) -> Optional[List[str]]:
    names = getattr(model, "feature_names_in_", None)
    if names is None:
        return None
    # Convert numpy array to list if needed
    if hasattr(names, "tolist"):
        return names.tolist()
    return list(names)


def default_feature_names(n_features: Optional[int]) -> List[str]:
    # Common for credit card dataset: Time, V1..V28, Amount
    # We fall back to a generic naming if dimensions are different
    candidate = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
    if isinstance(n_features, int) and n_features == len(candidate):
        return candidate
    # Otherwise generate generic f1..fn names
    if isinstance(n_features, int) and n_features > 0:
        return [f"f{i+1}" for i in range(n_features)]
    # Last resort: full candidate list
    return candidate


def ensure_feature_order(df: pd.DataFrame, feature_order: List[str]) -> pd.DataFrame:
    missing = [c for c in feature_order if c not in df.columns]
    if missing:
        # Add missing columns as zeros
        for c in missing:
            df[c] = 0.0
    # Extra columns will be ignored by selecting strictly feature_order
    return df[feature_order]


def render_sidebar_inputs(feature_names: List[str]) -> pd.DataFrame:
    st.sidebar.header("Input Features")
    values = {}
    for name in feature_names:
        # Heuristic ranges for nicer UI
        if name.lower() == "amount":
            values[name] = st.sidebar.number_input(name, min_value=0.0, max_value=10000.0, value=50.0, step=1.0)
        elif name.lower() == "time":
            values[name] = st.sidebar.number_input(name, min_value=0.0, max_value=200000.0, value=0.0, step=1.0)
        else:
            values[name] = st.sidebar.number_input(name, value=0.0, step=0.1, format="%.4f")
    return pd.DataFrame([values])


def predict_proba(model, X: pd.DataFrame) -> np.ndarray:
    # Some sklearn models may not implement predict_proba; fall back to decision_function
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)  # shape (n, 2)
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        # Min-max normalize to 0..1 as a proxy
        scores = np.asarray(scores).reshape(-1)
        if scores.max() == scores.min():
            probs = np.vstack([1 - 0.5 * np.ones_like(scores), 0.5 * np.ones_like(scores)]).T
        else:
            norm = (scores - scores.min()) / (scores.max() - scores.min())
            probs = np.vstack([1 - norm, norm]).T
        return probs
    # Last resort: use predict as class, construct one-hot-ish probabilities
    preds = model.predict(X)
    probs = np.vstack([1 - preds, preds]).T
    return probs


def main():
    st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳", layout="wide")
    st.title("💳 Credit Card Fraud Detection")
    st.caption("Real-time predictions using a trained Random Forest model")

    # Load model
    load_error = None
    model = None
    try:
        model = load_model(MODEL_PATH)
    except Exception as e:
        load_error = str(e)

    if load_error:
        st.warning(f"Failed to load saved model: {load_error}")
        with st.spinner("Training a fallback model for demo purposes..."):
            from sklearn.datasets import make_classification
            from sklearn.ensemble import RandomForestClassifier
            # Create synthetic, imbalanced dataset with 30 features (Time, V1..V28, Amount)
            X, y = make_classification(
                n_samples=5000,
                n_features=30,
                n_informative=10,
                n_redundant=5,
                n_repeated=0,
                n_clusters_per_class=2,
                weights=[0.995, 0.005],
                random_state=42,
            )
            feature_names = default_feature_names(30)
            X_df = pd.DataFrame(X, columns=feature_names)
            rf = RandomForestClassifier(n_estimators=200, max_depth=None, n_jobs=-1, random_state=42)
            rf.fit(X_df, y)
            model = rf
            try:
                # Attempt to save so future runs can reuse
                joblib.dump(model, MODEL_PATH)
            except Exception:
                pass
            st.success("Fallback model ready.")
    
    feature_names = get_feature_names(model)
    n_features = getattr(model, "n_features_in_", None)
    if feature_names is None:
        feature_names = default_feature_names(n_features)

    tab_realtime, tab_batch, tab_about = st.tabs(["Realtime Input", "Batch CSV Upload", "About"])

    with tab_realtime:
        sample_df = render_sidebar_inputs(feature_names)
        X = ensure_feature_order(sample_df.copy(), feature_names)
        if st.button("Predict", type="primary"):
            proba = predict_proba(model, X)
            fraud_prob = float(proba[0, 1])
            pred_label = int(fraud_prob >= 0.5)

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Fraud Probability", f"{fraud_prob:.2%}")
            with col2:
                st.metric("Predicted Class", "Fraud" if pred_label == 1 else "Genuine")

            st.progress(min(max(fraud_prob, 0.0), 1.0))

    with tab_batch:
        st.write("Upload a CSV file with columns matching the model features.")
        uploaded = st.file_uploader("Choose CSV", type=["csv"])
        if uploaded is not None:
            try:
                df = pd.read_csv(uploaded)
                Xb = ensure_feature_order(df.copy(), feature_names)
                proba = predict_proba(model, Xb)
                result = df.copy()
                result["fraud_probability"] = proba[:, 1]
                result["predicted_class"] = (result["fraud_probability"] >= 0.5).astype(int)
                st.dataframe(result.head(50), use_container_width=True)

                csv_buf = io.StringIO()
                result.to_csv(csv_buf, index=False)
                st.download_button("Download Predictions", data=csv_buf.getvalue(), file_name="predictions.csv", mime="text/csv")
            except Exception as e:
                st.error(f"Failed to score file: {e}")

    with tab_about:
        st.markdown(
            """
            - This app detects likely fraudulent transactions.
            - Inputs should align with the model's expected features. If the model exposes `feature_names_in_`, those are used; otherwise a robust default is applied.
            - Batch uploads accept CSVs and return probabilities and predicted classes.
            """
        )


if __name__ == "__main__":
    main()