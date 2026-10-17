# Airfare Price Prediction

Predicting domestic Indian flight prices using machine learning. Compares Linear Regression, Random Forest, Gradient Boosting, and XGBoost on the [Flight Price Prediction dataset](https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction).

---

## Dataset

300,000+ flight records with the following features:

| Feature | Description |
|---|---|
| `airline` | Airline name (IndiGo, Air India, Vistara, etc.) |
| `source_city` | Departure city |
| `destination_city` | Arrival city |
| `departure_time` | Time of day bucket (Morning, Evening, Night, etc.) |
| `arrival_time` | Time of day bucket |
| `stops` | Non-stop, 1 stop, or 2+ stops |
| `class` | Economy or Business |
| `duration` | Flight duration in hours |
| `days_left` | Days between booking and departure |
| `price` | **Target** — ticket price in INR |

---

## Quickstart

```bash
git clone https://github.com/sofiaade/Airfare_Price_Prediction_Project
cd Airfare_Price_Prediction_Project

python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Run the full pipeline (EDA + training + demo prediction)
python run.py
```

---

## Project Structure

```
Airfare_Price_Prediction_Project/
├── src/
│   ├── preprocess.py   ← data loading, encoding, train/test split
│   ├── train.py        ← model training and comparison
│   ├── eda.py          ← exploratory analysis plots
│   └── predict.py      ← load model and predict new flights
├── notebooks/
│   └── exploration.ipynb
├── data/               ← dataset downloaded here
├── models/             ← trained models saved here
├── plots/              ← EDA charts saved here
├── run.py              ← one-command pipeline
└── requirements.txt
```

---

## Usage

**Train all models:**
```bash
python src/train.py
```

**Train one specific model:**
```bash
python src/train.py --model xgboost
```

**Predict a single flight:**
```bash
python src/predict.py \
  --airline IndiGo \
  --source Delhi \
  --destination Mumbai \
  --class_type Economy \
  --stops zero \
  --duration 2.5 \
  --days_left 10 \
  --departure_time Morning \
  --arrival_time Afternoon
```

**In Python:**
```python
from src.predict import AirfarePricePredictor

predictor = AirfarePricePredictor()
price = predictor.predict_single(
    airline="Vistara",
    source_city="Delhi",
    destination_city="Bangalore",
    flight_class="Business",
    stops="zero",
    duration=3.0,
    days_left=5,
    departure_time="Morning",
    arrival_time="Afternoon",
)
print(f"Predicted price: ₹{price:,.0f}")
```

---

## Results

| Model | MAE (₹) | RMSE (₹) | R² |
|---|---|---|---|
| Linear Regression | — | — | — |
| Random Forest | — | — | — |
| Gradient Boosting | — | — | — |
| XGBoost | — | — | — |

*Fill in after running `python run.py`*

---

## Key Findings

> Fill in after running experiments. Questions to answer:
> - Which features matter most? (class? stops? days_left?)
> - Does booking earlier actually make flights cheaper?
> - Which airline is most expensive on average?

---

## Getting the Dataset

The dataset downloads automatically via `kagglehub`. If that fails (needs Kaggle credentials), the code generates a realistic synthetic dataset so the pipeline still runs.

To use the real dataset manually:
1. Download from [Kaggle](https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction)
2. Place the CSV in `data/flights.csv`
