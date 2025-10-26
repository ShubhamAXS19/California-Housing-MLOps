
import numpy as np
import joblib
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_squared_error, r2_score

# Import from config.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.config import PROCESSED_DATA_DIR

MODEL_DIR = "artifacts/model"
os.makedirs(MODEL_DIR, exist_ok=True)

def train_model():
    # --- Load preprocessed data ---
    X_train = np.load(os.path.join(PROCESSED_DATA_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(PROCESSED_DATA_DIR, "y_train.npy"))
    X_test = np.load(os.path.join(PROCESSED_DATA_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(PROCESSED_DATA_DIR, "y_test.npy"))

    print("✅ Loaded preprocessed data successfully")

    # --- Initialize Model ---
    model = BayesianRidge()
    model.fit(X_train, y_train)

    # --- Evaluate on test data ---
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"✅ Model trained successfully")
    print(f"📊 Test MSE: {mse:.4f}, R²: {r2:.4f}")

    # --- Save the trained model ---
    model_path = os.path.join(MODEL_DIR, "model.joblib")
    joblib.dump(model, model_path)
    print(f"💾 Model saved at: {model_path}")

    # --- Save metrics (optional for DVC tracking) ---
    metrics = {
        "mse": float(mse),
        "r2": float(r2)
    }

    metrics_path = os.path.join(MODEL_DIR, "metrics.json")
    import json
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"📈 Metrics saved at: {metrics_path}")

if __name__ == "__main__":
    train_model()
