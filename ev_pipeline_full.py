"""
Electric Vehicle Range Prediction - Complete Pipeline
=====================================================
Uses sklearn ColumnTransformer + Pipeline (fully picklable).
Usage:
  TRAIN : python ev_pipeline_full.py --mode train --data Main_file.csv
  PREDICT: python ev_pipeline_full.py --mode predict --data input.csv --output predictions.csv
"""

import argparse
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# ──────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────
TARGET      = "Electric Range"
DROP_COLS   = [
    "Unnamed: 0",
    "Clean Alternative Fuel Vehicle (CAFV) Eligibility",
    "Electric Utility", "Postal Code", "2020 Census Tract", "State",
]
FREQ_COLS   = ["City", "Model", "Vehicle Location", "County", "VIN (1-10)"]
BINARY_COLS = ["Make"]
OHE_COLS    = ["Electric Vehicle Type"]
NUM_COLS    = ["Model Year", "Legislative District", "DOL Vehicle ID"]

MODEL_PATH  = "ev_pipeline.pkl"

# ──────────────────────────────────────────────────────────────
# CUSTOM TRANSFORMERS  (picklable - defined at module level)
# ──────────────────────────────────────────────────────────────

class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Replaces each category with its relative frequency (fitted on train)."""
    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y=None):
        X = pd.DataFrame(X, columns=self.cols)
        self.freq_maps_ = {
            col: X[col].value_counts(normalize=True).to_dict()
            for col in self.cols
        }
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.cols).copy()
        for col in self.cols:
            X[col] = X[col].map(self.freq_maps_[col]).fillna(0)
        return X.values.astype(float)


class BinaryEncoder(BaseEstimator, TransformerMixin):
    """Maps each category to an integer index then expands to binary bits."""
    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y=None):
        X = pd.DataFrame(X, columns=self.cols)
        self.maps_    = {}
        self.n_bits_  = {}
        for col in self.cols:
            cats = X[col].unique()
            self.maps_[col]   = {cat: idx for idx, cat in enumerate(cats)}
            self.n_bits_[col] = max(1, int(np.ceil(np.log2(len(cats) + 1))))
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.cols)
        parts = []
        for col in self.cols:
            int_arr = X[col].map(self.maps_[col]).fillna(0).astype(int).values
            n = self.n_bits_[col]
            bits = np.column_stack([(int_arr >> b) & 1 for b in range(n)])
            parts.append(bits)
        return np.hstack(parts).astype(float)


# ──────────────────────────────────────────────────────────────
# BUILD PIPELINE
# ──────────────────────────────────────────────────────────────

def build_pipeline():
    preprocessor = ColumnTransformer(transformers=[
        ("freq",   FrequencyEncoder(cols=FREQ_COLS),                              FREQ_COLS),
        ("binary", BinaryEncoder(cols=BINARY_COLS),                               BINARY_COLS),
        ("ohe",    OneHotEncoder(handle_unknown="ignore", sparse_output=False),   OHE_COLS),
        ("num",    "passthrough",                                                  NUM_COLS),
    ], remainder="drop")

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=50, max_depth=20, random_state=42, n_jobs=-1,min_samples_leaf=1
        ))
    ])
    return pipeline


# ──────────────────────────────────────────────────────────────
# TRAIN
# ──────────────────────────────────────────────────────────────

def train(data_path):
    print("=" * 60)
    print("  TRAINING MODE")
    print("=" * 60)

    df = pd.read_csv(data_path)
    print(f"\n[LOAD]  Shape: {df.shape}")

    # Step 1 – Drop NA
    df.dropna(inplace=True)
    print(f"[STEP 1] Drop NA -> {df.shape}")

    # Step 2 – Drop unwanted columns
    df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)
    print(f"[STEP 2] Drop cols -> {df.shape}")

    # Step 3 – Split
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train = X_train.reset_index(drop=True)
    X_test  = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test  = y_test.reset_index(drop=True)
    print(f"[STEP 3] Train: {X_train.shape} | Test: {X_test.shape}")

    # Step 4-5 – Build & Fit pipeline
    pipeline = build_pipeline()
    print("\n[STEP 4-5] Fitting ColumnTransformer + RandomForest pipeline ...")
    pipeline.fit(X_train, y_train)
    print("  Done!")

    # Step 6 – Evaluate
    y_pred = pipeline.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    print(f"\n[STEP 6] Test Set Performance")
    print(f"  MAE  : {mae:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  R2   : {r2:.4f}")

    # Step 7 – Save
    joblib.dump(pipeline, MODEL_PATH)
    print(f"\n[SAVE] Pipeline saved -> {MODEL_PATH}")

    # Summary
    print("\n" + "=" * 60)
    print("  PIPELINE SUMMARY")
    print("=" * 60)
    print(f"""
  ColumnTransformer
  ├─ FrequencyEncoder  : {FREQ_COLS}
  ├─ BinaryEncoder     : {BINARY_COLS}
  ├─ OneHotEncoder     : {OHE_COLS}
  └─ Passthrough (num) : {NUM_COLS}

  RandomForestRegressor
    n_estimators = 100
    max_depth    = 10
    random_state = 42

  MAE  = {mae:.4f}
  RMSE = {rmse:.4f}
  R2   = {r2:.4f}
    """)


# ──────────────────────────────────────────────────────────────
# PREDICT
# ──────────────────────────────────────────────────────────────

def predict(data_path, output_path):
    print("=" * 60)
    print("  PREDICTION MODE")
    print("=" * 60)

    # Load model
    pipeline = joblib.load(MODEL_PATH)
    print(f"[LOAD MODEL] {MODEL_PATH}")

    # Load input CSV
    df_input = pd.read_csv(data_path)
    print(f"[LOAD DATA] {df_input.shape}")

    # Drop unwanted columns if present (won't error if absent)
    cols_to_drop = [c for c in DROP_COLS + [TARGET] if c in df_input.columns]
    df_input.drop(columns=cols_to_drop, inplace=True)

    # Predict
    preds = pipeline.predict(df_input)

    # Output
    df_out = df_input.copy()
    df_out["Predicted_Electric_Range"] = np.round(preds, 2)
    df_out.to_csv(output_path, index=False)

    print(f"\n[DONE] Predictions saved -> {output_path}")
    print(f"\n  Sample output (first 5 rows):")
    print(df_out[["Predicted_Electric_Range"]].head().to_string())


# ──────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EV Range Prediction Pipeline")
    parser.add_argument("--mode",   required=True, choices=["train", "predict"],
                        help="'train' to build model, 'predict' to run inference")
    parser.add_argument("--data",   required=True, help="Path to input CSV")
    parser.add_argument("--output", default="predictions.csv",
                        help="(predict mode) Path to save output CSV")
    args = parser.parse_args()

    if args.mode == "train":
        train(args.data)
    else:
        predict(args.data, args.output)
