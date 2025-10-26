import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, r2_score

# Paths
MODEL_PATH = "artifacts/model/model.joblib"
PROCESSED_DATA_DIR = "data/processed"
UNCERTAINTY_DIR = "artifacts/uncertainty"
os.makedirs(UNCERTAINTY_DIR, exist_ok=True)

def predict_with_uncertainty():
    # --- Load model and data ---
    model = joblib.load(MODEL_PATH)
    X_test = np.load(os.path.join(PROCESSED_DATA_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(PROCESSED_DATA_DIR, "y_test.npy"))

    print("✅ Model and test data loaded")

    # --- Predict with uncertainty ---
    y_mean, y_std = model.predict(X_test, return_std=True)
    print(f"✅ Predictions with uncertainty computed for {len(y_mean)} samples")

    # --- Save predictions ---
    df_pred = pd.DataFrame({
        "actual": y_test,
        "predicted_mean": y_mean,
        "predicted_std": y_std
    })
    pred_path = os.path.join(UNCERTAINTY_DIR, "predictions_with_uncertainty.csv")
    df_pred.to_csv(pred_path, index=False)
    print(f"💾 Predictions with uncertainty saved at: {pred_path}")

    # --- Evaluate model performance ---
    mse = mean_squared_error(y_test, y_mean)
    r2 = r2_score(y_test, y_mean)
    print(f"📊 MSE: {mse:.4f}, R²: {r2:.4f}")

    # --- Visualize uncertainty ---
    plt.figure(figsize=(10, 6))
    plt.errorbar(range(100), y_mean[:100], yerr=y_std[:100], fmt='o', capsize=3, label="Pred ± Std")
    plt.scatter(range(100), y_test[:100], color='red', alpha=0.6, label="Actual")
    plt.legend()
    plt.title("Prediction vs Actual with Uncertainty (first 100 samples)")
    plt.xlabel("Sample Index")
    plt.ylabel("Target Value")
    plt.grid(True)
    plot_path = os.path.join(UNCERTAINTY_DIR, "prediction_uncertainty.png")
    plt.savefig(plot_path)
    print(f"📈 Uncertainty visualization saved at: {plot_path}")

if __name__ == "__main__":
    predict_with_uncertainty()
