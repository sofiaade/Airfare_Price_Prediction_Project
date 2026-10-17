"""
Model Training for Airfare Price Prediction
============================================
Trains and compares four models:
    1. Linear Regression      — baseline
    2. Random Forest          — ensemble, handles non-linearity
    3. Gradient Boosting      — usually best for tabular data
    4. XGBoost                — fast gradient boosting

Evaluation metrics:
    MAE  — Mean Absolute Error (in INR — most interpretable)
    RMSE — Root Mean Squared Error (penalizes large errors)
    R²   — Coefficient of determination (1.0 = perfect)

Usage:
    python src/train.py
    python src/train.py --model xgboost --tune
"""

import argparse
import json
import logging
import os
import pickle
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

from preprocess import load_and_preprocess, download_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

def get_models() -> dict:
    """
    Return all models to train.
    Hyperparameters are reasonable defaults — tune with --tune flag.
    """
    models = {
        "linear_regression": LinearRegression(),

        "random_forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1,
        ),

        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            random_state=42,
        ),
    }

    # XGBoost is optional (install separately)
    try:
        from xgboost import XGBRegressor
        models["xgboost"] = XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0,
        )
    except ImportError:
        logger.warning("XGBoost not installed — skipping. Run: pip install xgboost")

    return models


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, X_test, y_test, model_name: str) -> dict:
    """Compute MAE, RMSE, and R² on the test set."""
    preds = model.predict(X_test)
    mae   = mean_absolute_error(y_test, preds)
    rmse  = np.sqrt(mean_squared_error(y_test, preds))
    r2    = r2_score(y_test, preds)

    logger.info(
        f"{model_name:<25} | MAE: ₹{mae:>8,.0f} | RMSE: ₹{rmse:>8,.0f} | R²: {r2:.4f}"
    )
    return {"model": model_name, "mae": mae, "rmse": rmse, "r2": r2}


def cross_validate(model, X_train, y_train, model_name: str, cv: int = 5) -> float:
    """Run k-fold cross-validation, return mean R²."""
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="r2", n_jobs=-1)
    logger.info(f"{model_name} CV R²: {scores.mean():.4f} ± {scores.std():.4f}")
    return float(scores.mean())


# ---------------------------------------------------------------------------
# Hyperparameter tuning
# ---------------------------------------------------------------------------

def tune_best_model(X_train, y_train) -> GradientBoostingRegressor:
    """
    Grid search over Gradient Boosting hyperparameters.
    Only run with --tune flag as it takes several minutes.
    """
    from sklearn.model_selection import GridSearchCV

    param_grid = {
        "n_estimators":  [100, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth":     [3, 5, 7],
        "subsample":     [0.7, 0.8, 1.0],
    }

    logger.info("Running grid search (this may take 5-10 minutes)...")
    gs = GridSearchCV(
        GradientBoostingRegressor(random_state=42),
        param_grid,
        cv=3,
        scoring="r2",
        n_jobs=-1,
        verbose=1,
    )
    gs.fit(X_train, y_train)
    logger.info(f"Best params: {gs.best_params_}")
    logger.info(f"Best CV R²:  {gs.best_score_:.4f}")
    return gs.best_estimator_


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def get_feature_importance(model, feature_names: list) -> dict:
    """Extract feature importances from tree-based models."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        pairs = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
        return {name: float(imp) for name, imp in pairs}
    return {}


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train_all(
    data_dir:   str = "data/",
    models_dir: str = "models/",
    tune:       bool = False,
    model_name: str = "all",
) -> list[dict]:
    """
    Train all models, evaluate, save best model.

    Returns list of result dicts for all models.
    """
    Path(models_dir).mkdir(parents=True, exist_ok=True)

    # Data
    csv_path = download_dataset(data_dir)
    X_train, X_test, y_train, y_test, feature_names, encoders = load_and_preprocess(csv_path)

    # Save encoders
    with open(os.path.join(models_dir, "encoders.pkl"), "wb") as f:
        pickle.dump(encoders, f)
    with open(os.path.join(models_dir, "feature_names.json"), "w") as f:
        json.dump(feature_names, f)

    # Select models to train
    all_models = get_models()
    if model_name != "all" and model_name in all_models:
        models_to_train = {model_name: all_models[model_name]}
    else:
        models_to_train = all_models

    # Tune if requested
    if tune:
        models_to_train["gradient_boosting_tuned"] = tune_best_model(X_train, y_train)

    # Train and evaluate
    logger.info("\n" + "="*65)
    logger.info("MODEL COMPARISON")
    logger.info("="*65)

    results = []
    trained_models = {}

    for name, model in models_to_train.items():
        logger.info(f"\nTraining: {name}")
        t0 = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - t0
        logger.info(f"  Trained in {elapsed:.1f}s")

        result = evaluate(model, X_test, y_test, name)
        result["train_time_s"] = elapsed
        result["feature_importance"] = get_feature_importance(model, feature_names)
        results.append(result)
        trained_models[name] = model

    # Save all models
    for name, model in trained_models.items():
        path = os.path.join(models_dir, f"{name}.pkl")
        with open(path, "wb") as f:
            pickle.dump(model, f)
        logger.info(f"Saved {name} → {path}")

    # Save results
    results_path = os.path.join(models_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print leaderboard
    logger.info("\n" + "="*65)
    logger.info("LEADERBOARD (sorted by R²)")
    logger.info("="*65)
    for r in sorted(results, key=lambda x: x["r2"], reverse=True):
        logger.info(f"  {r['model']:<30} R²={r['r2']:.4f}  MAE=₹{r['mae']:,.0f}")

    best = max(results, key=lambda x: x["r2"])
    logger.info(f"\n🏆 Best model: {best['model']} (R²={best['r2']:.4f})")

    # Save best model separately for easy loading
    best_model = trained_models[best["model"]]
    with open(os.path.join(models_dir, "best_model.pkl"), "wb") as f:
        pickle.dump(best_model, f)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   default="data/")
    parser.add_argument("--models_dir", default="models/")
    parser.add_argument("--model",      default="all",
                        help="Model to train: all, linear_regression, random_forest, gradient_boosting, xgboost")
    parser.add_argument("--tune",       action="store_true",
                        help="Run hyperparameter tuning (slow)")
    args = parser.parse_args()

    train_all(args.data_dir, args.models_dir, args.tune, args.model)
