import os
import joblib

model_path = os.environ.get("MODEL_PATH", "random_forest_model.joblib")

print(f"Loading model from: {model_path}")
model = joblib.load(model_path)
print("Model type:", type(model))

names = getattr(model, "feature_names_in_", None)
print("feature_names_in_:", names)
print("n_features_in_:", getattr(model, "n_features_in_", None))

print(model)