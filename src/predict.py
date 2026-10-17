"""
Airfare Price Predictor
========================
Load the trained model and predict prices for new flights.

Usage:
    # Predict a single flight
    python src/predict.py \
        --airline IndiGo \
        --source Delhi \
        --destination Mumbai \
        --class Economy \
        --stops zero \
        --duration 2.5 \
        --days_left 15 \
        --departure_time Morning \
        --arrival_time Afternoon

    # Or import in your own code:
    from predict import AirfarePricePredictor
    predictor = AirfarePricePredictor()
    price = predictor.predict_single(
        airline="IndiGo", source_city="Delhi", destination_city="Mumbai",
        flight_class="Economy", stops="zero", duration=2.5,
        days_left=15, departure_time="Morning", arrival_time="Afternoon"
    )
    print(f"Predicted price: ₹{price:,.0f}")
"""

import argparse
import json
import logging
import os
import pickle

import numpy as np
import pandas as pd

from preprocess import STOPS_MAP

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class AirfarePricePredictor:
    """
    Wrapper around the trained model for making predictions.
    Handles encoding of categorical inputs automatically.
    """

    def __init__(self, models_dir: str = "models/"):
        self.models_dir = models_dir
        self._load()

    def _load(self):
        model_path    = os.path.join(self.models_dir, "best_model.pkl")
        encoders_path = os.path.join(self.models_dir, "encoders.pkl")
        features_path = os.path.join(self.models_dir, "feature_names.json")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"No trained model found at {model_path}. "
                f"Run: python src/train.py first."
            )

        with open(model_path, "rb")    as f: self.model    = pickle.load(f)
        with open(encoders_path, "rb") as f: self.encoders = pickle.load(f)
        with open(features_path)       as f: self.feature_names = json.load(f)

        logger.info(f"Loaded model: {type(self.model).__name__}")

    def _encode_input(self, **kwargs) -> np.ndarray:
        """
        Encode a single flight's features into the vector the model expects.
        Handles unknown categories gracefully.
        """
        row = {}

        for feat in self.feature_names:
            if feat == "stops":
                stops_str = kwargs.get("stops", "zero").lower().strip()
                row["stops"] = STOPS_MAP.get(stops_str, 1)

            elif feat in self.encoders:
                le  = self.encoders[feat]
                val = str(kwargs.get(feat, ""))
                if val in le.classes_:
                    row[feat] = le.transform([val])[0]
                else:
                    # Unknown category — use most common (index 0)
                    logger.warning(f"Unknown value '{val}' for '{feat}'. Using default.")
                    row[feat] = 0

            else:
                row[feat] = float(kwargs.get(feat, 0))

        return np.array([row[f] for f in self.feature_names]).reshape(1, -1)

    def predict_single(
        self,
        airline:          str,
        source_city:      str,
        destination_city: str,
        flight_class:     str,
        stops:            str,
        duration:         float,
        days_left:        int,
        departure_time:   str,
        arrival_time:     str,
    ) -> float:
        """
        Predict the price for a single flight.

        Args:
            airline:          Airline name (e.g., "IndiGo", "Air India")
            source_city:      Departure city (e.g., "Delhi")
            destination_city: Arrival city (e.g., "Mumbai")
            flight_class:     "Economy" or "Business"
            stops:            "zero", "one", or "two_or_more"
            duration:         Flight duration in hours
            days_left:        Days between booking and departure
            departure_time:   "Morning", "Afternoon", "Evening", "Night", etc.
            arrival_time:     Same options as departure_time

        Returns:
            Predicted price in INR (float)
        """
        X = self._encode_input(
            airline=airline,
            source_city=source_city,
            destination_city=destination_city,
            **{"class": flight_class},
            stops=stops,
            duration=duration,
            days_left=days_left,
            departure_time=departure_time,
            arrival_time=arrival_time,
        )
        return float(self.model.predict(X)[0])

    def predict_batch(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict prices for a DataFrame of flights.
        DataFrame must have the same columns as training data.
        """
        from preprocess import load_and_preprocess
        # Re-encode using saved encoders
        df = df.copy()
        df["stops"] = df["stops"].str.lower().str.strip().map(STOPS_MAP).fillna(1)

        for col, le in self.encoders.items():
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda x: le.transform([str(x)])[0] if str(x) in le.classes_ else 0
                )

        X = df[[f for f in self.feature_names if f in df.columns]].values
        return self.model.predict(X)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--airline",        default="IndiGo")
    parser.add_argument("--source",         default="Delhi")
    parser.add_argument("--destination",    default="Mumbai")
    parser.add_argument("--class_type",     default="Economy", dest="flight_class")
    parser.add_argument("--stops",          default="zero")
    parser.add_argument("--duration",       type=float, default=2.5)
    parser.add_argument("--days_left",      type=int,   default=15)
    parser.add_argument("--departure_time", default="Morning")
    parser.add_argument("--arrival_time",   default="Afternoon")
    parser.add_argument("--models_dir",     default="models/")
    args = parser.parse_args()

    predictor = AirfarePricePredictor(models_dir=args.models_dir)

    price = predictor.predict_single(
        airline          = args.airline,
        source_city      = args.source,
        destination_city = args.destination,
        flight_class     = args.flight_class,
        stops            = args.stops,
        duration         = args.duration,
        days_left        = args.days_left,
        departure_time   = args.departure_time,
        arrival_time     = args.arrival_time,
    )

    print(f"\n{'='*45}")
    print(f"  Flight: {args.source} → {args.destination}")
    print(f"  Airline: {args.airline}  |  Class: {args.flight_class}")
    print(f"  Stops: {args.stops}  |  Duration: {args.duration}h")
    print(f"  Days until departure: {args.days_left}")
    print(f"  Departure: {args.departure_time}  |  Arrival: {args.arrival_time}")
    print(f"{'='*45}")
    print(f"  💺 Predicted Price: ₹{price:,.0f}")
    print(f"{'='*45}\n")
