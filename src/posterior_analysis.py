import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MODEL_PATH = "artifacts/model/model.joblib"
POSTERIOR_DIR = "artifacts/posterior"
os.makedirs(POSTERIOR_DIR, exist_ok=True)

def posterior_analysis():
    # --- Load trained model ---
    model = joblib.load(MODEL_PATH)
    print("✅ Loaded trained BayesianRidge model")

    # --- Extract posterior parameters ---
    coef_mean = model.coef_
    coef_cov = model.sigma_  # Covariance matrix
    alpha = model.alpha_
    lambda_ = model.lambda_

    print(f"📈 Posterior α (noise precision): {alpha:.4f}")
    print(f"📉 Posterior λ (weights precision): {lambda_:.4f}")
    print(f"✅ Coefficient mean shape: {coef_mean.shape}")
    print(f"✅ Covariance matrix shape: {coef_cov.shape}")

    # --- Save posterior coefficients ---
    coef_df = pd.DataFrame({
        "feature_index": range(len(coef_mean)),
        "coef_mean": coef_mean
    })
    coef_df.to_csv(os.path.join(POSTERIOR_DIR, "posterior_coefficients.csv"), index=False)
    print(f"💾 Posterior coefficients saved at: artifacts/posterior/posterior_coefficients.csv")

    # --- Optional: visualize coefficient uncertainty ---
    plt.figure(figsize=(10, 6))
    plt.errorbar(range(len(coef_mean)), coef_mean, yerr=np.sqrt(np.diag(coef_cov)), fmt="o", capsize=4)
    plt.title("Posterior Mean ± Std of Coefficients")
    plt.xlabel("Feature Index")
    plt.ylabel("Coefficient Value")
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join(POSTERIOR_DIR, "posterior_coeff_uncertainty.png")
    plt.savefig(plot_path)
    print(f"📊 Posterior uncertainty plot saved at: {plot_path}")

if __name__ == "__main__":
    posterior_analysis()
