import os
import numpy as np
import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

# Import config
import os
import sys

# Add project root (one level up from 'src') to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(PROJECT_ROOT)

from src.config import PROCESSED_DATA_DIR

# Define directories
MODEL_PATH = "artifacts/model/model.joblib"
PREDICTIONS_DIR = "artifacts/predictions"
os.makedirs(PREDICTIONS_DIR, exist_ok=True)

def predict_on_test():
    # --- Load model ---
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully")

    # --- Load test data ---
    X_test = np.load(os.path.join(PROCESSED_DATA_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(PROCESSED_DATA_DIR, "y_test.npy"))
    print("✅ Test data loaded successfully")

    # --- Generate predictions ---
    y_pred = model.predict(X_test)
    print(f"✅ Predictions generated for {len(y_pred)} samples")

    # --- Save predictions ---
    predictions_path = os.path.join(PREDICTIONS_DIR, "predictions.csv")
    df_pred = pd.DataFrame({"actual": y_test, "predicted": y_pred})
    df_pred.to_csv(predictions_path, index=False)
    print(f"💾 Predictions saved at: {predictions_path}")

    # --- Evaluate again (sanity) ---
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"📊 Prediction Metrics - MSE: {mse:.4f}, R²: {r2:.4f}")

if __name__ == "__main__":
    predict_on_test()
