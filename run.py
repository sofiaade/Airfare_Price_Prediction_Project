"""
Airfare Price Prediction — Full Pipeline
=========================================
Run everything in one command:
    python run.py

Steps:
    1. Download / generate dataset
    2. Run EDA (generates plots/)
    3. Train all models
    4. Print leaderboard
    5. Demo prediction

Options:
    python run.py --skip_eda        # skip plots
    python run.py --model xgboost   # train one model only
    python run.py --tune            # hyperparameter tuning (slow)
"""

import argparse
import sys
import os

sys.path.insert(0, "src")

from preprocess import download_dataset, get_feature_dataframe
from train import train_all
from eda import (
    plot_price_distribution, plot_price_by_airline,
    plot_price_by_class, plot_price_by_stops,
    plot_price_vs_duration, plot_price_vs_days_left,
    print_summary,
)
from predict import AirfarePricePredictor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_eda",  action="store_true")
    parser.add_argument("--model",     default="all")
    parser.add_argument("--tune",      action="store_true")
    parser.add_argument("--data_dir",  default="data/")
    parser.add_argument("--model_dir", default="models/")
    args = parser.parse_args()

    print("\n🛫  Airfare Price Prediction Pipeline")
    print("="*45)

    # 1. Data
    print("\n📦 Step 1: Loading dataset...")
    csv_path = download_dataset(args.data_dir)
    df = get_feature_dataframe(csv_path)
    print_summary(df)

    # 2. EDA
    if not args.skip_eda:
        print("\n📊 Step 2: Generating EDA plots → plots/")
        os.makedirs("plots/", exist_ok=True)
        plot_price_distribution(df)
        plot_price_by_airline(df)
        plot_price_by_class(df)
        plot_price_by_stops(df)
        plot_price_vs_duration(df)
        plot_price_vs_days_left(df)
        print("   Done. Open plots/ to view charts.")

    # 3. Train
    print(f"\n🤖 Step 3: Training models...")
    results = train_all(
        data_dir=args.data_dir,
        models_dir=args.model_dir,
        tune=args.tune,
        model_name=args.model,
    )

    # 4. Feature importance plot
    best = max(results, key=lambda x: x["r2"])
    if best.get("feature_importance"):
        from eda import plot_feature_importance
        plot_feature_importance(best["feature_importance"])

    # 5. Demo prediction
    print("\n✈️  Step 4: Demo prediction...")
    predictor = AirfarePricePredictor(models_dir=args.model_dir)

    examples = [
        dict(airline="IndiGo", source_city="Delhi", destination_city="Mumbai",
             flight_class="Economy", stops="zero", duration=2.17, days_left=1,
             departure_time="Evening", arrival_time="Night"),
        dict(airline="Air India", source_city="Delhi", destination_city="Mumbai",
             flight_class="Business", stops="zero", duration=2.17, days_left=30,
             departure_time="Morning", arrival_time="Morning"),
    ]

    for ex in examples:
        price = predictor.predict_single(**ex)
        print(f"   {ex['airline']} {ex['source_city']}→{ex['destination_city']} "
              f"({ex['flight_class']}, {ex['stops']} stop, {ex['days_left']}d left): "
              f"₹{price:,.0f}")

    print("\n✅ Pipeline complete!")
    print(f"   Best model: {best['model']}  R²={best['r2']:.4f}  MAE=₹{best['mae']:,.0f}")
    print(f"   Models saved to: {args.model_dir}")
    print(f"   Plots saved to:  plots/\n")


if __name__ == "__main__":
    main()
