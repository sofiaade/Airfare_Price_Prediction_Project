"""
Exploratory Data Analysis — Airfare Price Prediction
======================================================
Generates 6 plots saved to plots/:

    1. price_distribution.png     — histogram of ticket prices
    2. price_by_airline.png       — boxplot: price per airline
    3. price_by_class.png         — Economy vs Business comparison
    4. price_by_stops.png         — how stops affect price
    5. price_vs_duration.png      — scatter: duration vs price
    6. price_vs_days_left.png     — scatter: booking lead time vs price
    7. feature_importance.png     — after training, top features

Run:
    python src/eda.py
"""

import os
import logging
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from preprocess import download_dataset, get_feature_dataframe, STOPS_MAP

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PLOTS_DIR = "plots/"
os.makedirs(PLOTS_DIR, exist_ok=True)

# Consistent color palette
PALETTE = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#3B1F2B", "#44BBA4"]


def _save(fig, name: str):
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved → {path}")


def plot_price_distribution(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df["price"], bins=60, color=PALETTE[0], edgecolor="white", alpha=0.85)
    ax.set_xlabel("Ticket Price (INR)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Airfare Prices")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x:,.0f}"))
    median = df["price"].median()
    ax.axvline(median, color=PALETTE[2], linestyle="--", linewidth=1.5, label=f"Median: ₹{median:,.0f}")
    ax.legend()
    _save(fig, "price_distribution.png")


def plot_price_by_airline(df: pd.DataFrame):
    order = df.groupby("airline")["price"].median().sort_values(ascending=False).index
    fig, ax = plt.subplots(figsize=(12, 5))
    data = [df[df["airline"] == a]["price"].values for a in order]
    bp = ax.boxplot(data, patch_artist=True, medianprops=dict(color="white", linewidth=2))
    for patch, color in zip(bp["boxes"], PALETTE * 5):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    ax.set_xticklabels(order, rotation=20, ha="right")
    ax.set_ylabel("Ticket Price (INR)")
    ax.set_title("Price Distribution by Airline")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x:,.0f}"))
    _save(fig, "price_by_airline.png")


def plot_price_by_class(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, cls, color in zip(axes, ["Economy", "Business"], [PALETTE[0], PALETTE[2]]):
        subset = df[df["class"] == cls]["price"]
        ax.hist(subset, bins=40, color=color, edgecolor="white", alpha=0.85)
        ax.set_title(f"{cls} Class")
        ax.set_xlabel("Price (INR)")
        ax.set_ylabel("Count")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x:,.0f}"))
        ax.axvline(subset.median(), color="black", linestyle="--",
                   label=f"Median: ₹{subset.median():,.0f}")
        ax.legend(fontsize=9)

    fig.suptitle("Economy vs Business Class — Price Distribution", fontsize=13)
    plt.tight_layout()
    _save(fig, "price_by_class.png")


def plot_price_by_stops(df: pd.DataFrame):
    # Map numeric stops back to labels for display
    reverse_map = {0: "Non-stop", 1: "1 Stop", 2: "2+ Stops"}
    df = df.copy()
    df["stops_label"] = df["stops"].map(reverse_map)

    order  = ["Non-stop", "1 Stop", "2+ Stops"]
    means  = [df[df["stops_label"] == s]["price"].mean() for s in order]
    medians= [df[df["stops_label"] == s]["price"].median() for s in order]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(order))
    bars = ax.bar(x, means, color=PALETTE[:3], alpha=0.85, edgecolor="white", width=0.5)
    ax.scatter(x, medians, color="black", zorder=5, s=60, label="Median")

    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                f"₹{val:,.0f}", ha="center", va="bottom", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_ylabel("Average Price (INR)")
    ax.set_title("Average Airfare by Number of Stops")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x:,.0f}"))
    ax.legend()
    _save(fig, "price_by_stops.png")


def plot_price_vs_duration(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 5))
    economy  = df[df["class"] == "Economy"]
    business = df[df["class"] == "Business"]

    ax.scatter(economy["duration"],  economy["price"],
               alpha=0.3, s=8, color=PALETTE[0], label="Economy")
    ax.scatter(business["duration"], business["price"],
               alpha=0.4, s=8, color=PALETTE[2], label="Business")

    # Trend line
    for subset, color in [(economy, PALETTE[0]), (business, PALETTE[2])]:
        z = np.polyfit(subset["duration"], subset["price"], 1)
        p = np.poly1d(z)
        xs = np.linspace(subset["duration"].min(), subset["duration"].max(), 100)
        ax.plot(xs, p(xs), color=color, linewidth=2)

    ax.set_xlabel("Flight Duration (hours)")
    ax.set_ylabel("Ticket Price (INR)")
    ax.set_title("Flight Duration vs Price")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x:,.0f}"))
    ax.legend()
    _save(fig, "price_vs_duration.png")


def plot_price_vs_days_left(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 5))

    # Bin by days_left to show trend clearly
    df = df.copy()
    df["days_bin"] = pd.cut(df["days_left"], bins=10)
    grouped = df.groupby("days_bin")["price"].mean().reset_index()
    grouped["days_mid"] = grouped["days_bin"].apply(lambda x: x.mid)

    ax.plot(grouped["days_mid"], grouped["price"],
            color=PALETTE[0], linewidth=2.5, marker="o", markersize=6)
    ax.fill_between(grouped["days_mid"], grouped["price"],
                    alpha=0.15, color=PALETTE[0])

    ax.set_xlabel("Days Left Before Departure")
    ax.set_ylabel("Average Ticket Price (INR)")
    ax.set_title("Booking Lead Time vs Price\n(Earlier booking = cheaper?)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x:,.0f}"))
    _save(fig, "price_vs_days_left.png")


def plot_feature_importance(importance_dict: dict, top_n: int = 10):
    items  = list(importance_dict.items())[:top_n]
    names  = [i[0] for i in items]
    values = [i[1] for i in items]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(names[::-1], values[::-1], color=PALETTE[0], alpha=0.85, edgecolor="white")
    ax.set_xlabel("Feature Importance")
    ax.set_title(f"Top {top_n} Most Important Features")
    for bar, val in zip(bars, values[::-1]):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=9)
    _save(fig, "feature_importance.png")


def print_summary(df: pd.DataFrame):
    print("\n" + "="*50)
    print("DATASET SUMMARY")
    print("="*50)
    print(f"  Rows:            {len(df):,}")
    print(f"  Price range:     ₹{df['price'].min():,.0f} – ₹{df['price'].max():,.0f}")
    print(f"  Median price:    ₹{df['price'].median():,.0f}")
    print(f"  Airlines:        {df['airline'].nunique()}")
    print(f"  Routes:          {df['source_city'].nunique()} cities")
    print(f"  Economy rows:    {(df['class']=='Economy').sum():,}")
    print(f"  Business rows:   {(df['class']=='Business').sum():,}")
    print("="*50 + "\n")


if __name__ == "__main__":
    csv_path = download_dataset("data/")
    df       = get_feature_dataframe(csv_path)

    print_summary(df)

    logger.info("Generating EDA plots...")
    plot_price_distribution(df)
    plot_price_by_airline(df)
    plot_price_by_class(df)
    plot_price_by_stops(df)
    plot_price_vs_duration(df)
    plot_price_vs_days_left(df)

    logger.info("EDA complete. Plots saved to plots/")
