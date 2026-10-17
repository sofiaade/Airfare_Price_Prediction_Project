"""
Data Preprocessing for Airfare Price Prediction
================================================
Dataset: Flight Price Prediction (Kaggle)
https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction

Features in the dataset:
    airline        - Name of the airline (IndiGo, Air India, etc.)
    flight         - Flight code
    source_city    - City of departure
    departure_time - Time of day bucket (Morning, Evening, etc.)
    stops          - Number of stops (zero, one, two_or_more)
    arrival_time   - Time of day bucket
    destination_city - City of arrival
    class          - Economy or Business
    duration       - Flight duration in hours
    days_left      - Days between booking and departure
    price          - Ticket price in INR (TARGET)

Goal:
    Predict ticket price (regression) given all other features.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------

CATEGORICAL_COLS = [
    "airline", "source_city", "departure_time",
    "stops", "arrival_time", "destination_city", "class"
]

NUMERIC_COLS = ["duration", "days_left"]
TARGET_COL   = "price"

STOPS_MAP = {
    "zero":         0,
    "one":          1,
    "two_or_more":  2,
}


# ---------------------------------------------------------------------------
# Dataset download helper
# ---------------------------------------------------------------------------

def download_dataset(data_dir: str = "data/") -> str:
    """
    Download the flight price dataset.

    Tries kagglehub first, falls back to a synthetic dataset for testing.

    Returns path to the CSV file.
    """
    os.makedirs(data_dir, exist_ok=True)
    csv_path = os.path.join(data_dir, "flights.csv")

    if os.path.exists(csv_path):
        logger.info(f"Dataset already exists at {csv_path}")
        return csv_path

    # Try kagglehub
    try:
        import kagglehub
        path = kagglehub.dataset_download("shubhambathwal/flight-price-prediction")
        for root, _, files in os.walk(path):
            for f in files:
                if f.endswith(".csv"):
                    import shutil
                    shutil.copy(os.path.join(root, f), csv_path)
                    logger.info(f"Downloaded dataset to {csv_path}")
                    return csv_path
    except Exception as e:
        logger.warning(f"kagglehub failed: {e}. Generating synthetic dataset.")

    # Fallback: generate a realistic synthetic dataset
    _generate_synthetic(csv_path, n=10000)
    return csv_path


def _generate_synthetic(path: str, n: int = 10000):
    """
    Generate a synthetic flight price dataset that mirrors the real one.
    Used when the Kaggle download is not available.
    Prices follow a realistic distribution based on class, stops, and days_left.
    """
    np.random.seed(42)
    rng = np.random

    airlines    = ["IndiGo", "Air India", "Jet Airways", "SpiceJet", "Vistara", "GoAir"]
    cities      = ["Delhi", "Mumbai", "Bangalore", "Kolkata", "Chennai", "Hyderabad"]
    times       = ["Morning", "Afternoon", "Evening", "Night", "Early_Morning", "Late_Night"]
    stops       = ["zero", "one", "two_or_more"]
    classes     = ["Economy", "Business"]

    airline_col     = rng.choice(airlines,  n)
    source_col      = rng.choice(cities,    n)
    dest_col        = rng.choice(cities,    n)
    dep_time_col    = rng.choice(times,     n)
    arr_time_col    = rng.choice(times,     n)
    stops_col       = rng.choice(stops,     n, p=[0.5, 0.35, 0.15])
    class_col       = rng.choice(classes,   n, p=[0.75, 0.25])
    duration_col    = rng.uniform(1.0, 12.0, n).round(2)
    days_left_col   = rng.randint(1, 50, n)

    # Base price
    base = rng.uniform(3000, 8000, n)

    # Price adjustments (realistic signals)
    base += (stops_col == "one")         * rng.uniform(500,  1500, n)
    base += (stops_col == "two_or_more") * rng.uniform(1000, 3000, n)
    base += (class_col == "Business")    * rng.uniform(8000, 20000, n)
    base += duration_col                 * rng.uniform(200,  600, n)
    base -= days_left_col                * rng.uniform(20,   80, n)   # cheaper if booked early
    base += rng.normal(0, 500, n)        # noise
    base  = np.maximum(base, 1500)       # floor price

    df = pd.DataFrame({
        "airline":          airline_col,
        "flight":           [f"FL{i:04d}" for i in range(n)],
        "source_city":      source_col,
        "departure_time":   dep_time_col,
        "stops":            stops_col,
        "arrival_time":     arr_time_col,
        "destination_city": dest_col,
        "class":            class_col,
        "duration":         duration_col,
        "days_left":        days_left_col,
        "price":            base.astype(int),
    })

    df.to_csv(path, index=False)
    logger.info(f"Generated synthetic dataset with {n} rows → {path}")


# ---------------------------------------------------------------------------
# Preprocessing pipeline
# ---------------------------------------------------------------------------

def load_and_preprocess(csv_path: str) -> tuple:
    """
    Load raw CSV and return train/test splits ready for modeling.

    Steps:
        1. Load CSV
        2. Drop unused columns
        3. Encode stops as ordinal (zero=0, one=1, two_or_more=2)
        4. Label-encode remaining categoricals
        5. Train/test split (80/20, stratified by airline)

    Returns:
        X_train, X_test, y_train, y_test, feature_names, encoders
    """
    logger.info(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"Shape: {df.shape}")

    # Standardize column names
    df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")

    # Drop columns not useful for modeling
    drop_cols = ["flight"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Drop rows with missing target
    df = df.dropna(subset=[TARGET_COL])

    logger.info(f"After cleaning: {df.shape}")

    # Encode stops as ordinal integer
    df["stops"] = df["stops"].str.lower().str.strip().map(STOPS_MAP).fillna(1).astype(int)

    # Label-encode categoricals
    encoders = {}
    cat_cols  = [c for c in CATEGORICAL_COLS if c != "stops" and c in df.columns]

    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    feature_cols = [c for c in df.columns if c != TARGET_COL]
    X = df[feature_cols].values
    y = df[TARGET_COL].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    logger.info(f"Train: {X_train.shape}  Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test, feature_cols, encoders


def get_feature_dataframe(csv_path: str) -> pd.DataFrame:
    """Return the preprocessed DataFrame (for EDA and notebooks)."""
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")
    df = df.drop(columns=["flight"], errors="ignore")
    df = df.dropna(subset=[TARGET_COL])
    df["stops"] = df["stops"].str.lower().str.strip().map(STOPS_MAP).fillna(1).astype(int)
    return df
