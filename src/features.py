import yaml
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# ✅ Import config paths
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import RAW_DATA_PATH, PROCESSED_DATA_DIR, PREPROCESSOR_DIR


def load_params(params_path="params.yaml"):
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


def preprocess_and_save(params):
    # Read params
    test_size = params["split"]["test_size"]
    random_state = params["split"]["random_state"]

    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs(PREPROCESSOR_DIR, exist_ok=True)

    # Load raw data
    df = pd.read_csv(RAW_DATA_PATH)

    # Separate features and target
    X = df.drop("median_house_value", axis=1)
    y = df["median_house_value"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Identify column types
    num_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_features = X.select_dtypes(include=["object"]).columns.tolist()

    # Pipelines
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),  # fill NaNs
        ("scaler", StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_features),
        ("cat", cat_pipeline, cat_features),
    ])

    # Fit and transform
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Save outputs
    np.save(os.path.join(PROCESSED_DATA_DIR, "X_train.npy"), X_train_processed)
    np.save(os.path.join(PROCESSED_DATA_DIR, "X_test.npy"), X_test_processed)
    np.save(os.path.join(PROCESSED_DATA_DIR, "y_train.npy"), y_train.to_numpy())
    np.save(os.path.join(PROCESSED_DATA_DIR, "y_test.npy"), y_test.to_numpy())

    # Save preprocessor (includes imputer and scaler)
    joblib.dump(preprocessor, os.path.join(PREPROCESSOR_DIR, "preprocessor.joblib"))

    print("✅ Preprocessing complete. Data and preprocessor saved.")


if __name__ == "__main__":
    params = load_params("params.yaml")
    preprocess_and_save(params)
