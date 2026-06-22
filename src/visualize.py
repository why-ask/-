from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs") / ".matplotlib"))

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_DATA_DIR = Path(r"E:\project\house_price")


def currency_axis(ax: plt.Axes) -> None:
    ax.yaxis.set_major_formatter(lambda value, _: f"${value / 1000:.0f}k")


def style_axis(ax: plt.Axes) -> None:
    ax.grid(True, axis="y", color="#d8dee9", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#8f99a8")
    ax.spines["bottom"].set_color("#8f99a8")
    ax.tick_params(colors="#344054")


def save_monthly_price_trend(train: pd.DataFrame, output_path: Path) -> None:
    monthly = (
        train.assign(SaleMonth=pd.to_datetime(dict(year=train["YrSold"], month=train["MoSold"], day=1)))
        .groupby("SaleMonth")
        .agg(MedianPrice=("SalePrice", "median"), AveragePrice=("SalePrice", "mean"), Sales=("SalePrice", "size"))
        .reset_index()
        .sort_values("SaleMonth")
    )

    fig, ax = plt.subplots(figsize=(11, 5.8), dpi=140)
    ax.plot(
        monthly["SaleMonth"],
        monthly["MedianPrice"],
        color="#2166ac",
        linewidth=2.4,
        marker="o",
        markersize=4,
        label="Median sale price",
    )
    ax.plot(
        monthly["SaleMonth"],
        monthly["AveragePrice"],
        color="#b2182b",
        linewidth=1.8,
        linestyle="--",
        label="Average sale price",
    )
    ax.set_title("Historical Sale Price Trend by Month", fontsize=15, pad=14)
    ax.set_xlabel("Sale month")
    ax.set_ylabel("Sale price")
    currency_axis(ax)
    style_axis(ax)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(frameon=False, loc="upper left")

    sales_ax = ax.twinx()
    sales_ax.bar(monthly["SaleMonth"], monthly["Sales"], width=18, color="#8ecae6", alpha=0.22, label="Sales count")
    sales_ax.set_ylabel("Sales count")
    sales_ax.spines["top"].set_visible(False)
    sales_ax.spines["right"].set_color("#8f99a8")
    sales_ax.tick_params(colors="#344054")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_price_distribution(train: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.6), dpi=140)
    ax.hist(train["SalePrice"], bins=36, color="#4c78a8", edgecolor="white", alpha=0.85)
    ax.axvline(train["SalePrice"].median(), color="#f58518", linewidth=2.2, label="Median")
    ax.axvline(train["SalePrice"].mean(), color="#b279a2", linewidth=2.2, linestyle="--", label="Average")
    ax.set_title("Sale Price Distribution", fontsize=15, pad=14)
    ax.set_xlabel("Sale price")
    ax.set_ylabel("House count")
    currency_axis(ax)
    style_axis(ax)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_quality_price_boxplot(train: pd.DataFrame, output_path: Path) -> None:
    qualities = sorted(train["OverallQual"].dropna().unique())
    data = [train.loc[train["OverallQual"] == quality, "SalePrice"] for quality in qualities]

    fig, ax = plt.subplots(figsize=(10, 5.6), dpi=140)
    box = ax.boxplot(data, patch_artist=True, showfliers=False)
    ax.set_xticks(range(1, len(qualities) + 1), labels=qualities)
    for patch in box["boxes"]:
        patch.set_facecolor("#72b7b2")
        patch.set_alpha(0.75)
        patch.set_edgecolor("#2f5d62")
    for median in box["medians"]:
        median.set_color("#7a1f1f")
        median.set_linewidth(2)

    ax.set_title("Sale Price by Overall Quality", fontsize=15, pad=14)
    ax.set_xlabel("OverallQual")
    ax.set_ylabel("Sale price")
    currency_axis(ax)
    style_axis(ax)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_living_area_scatter(train: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.8), dpi=140)
    scatter = ax.scatter(
        train["GrLivArea"],
        train["SalePrice"],
        c=train["OverallQual"],
        cmap="viridis",
        alpha=0.72,
        s=28,
        linewidth=0,
    )
    ax.set_title("Sale Price vs. Above-Ground Living Area", fontsize=15, pad=14)
    ax.set_xlabel("GrLivArea")
    ax.set_ylabel("Sale price")
    currency_axis(ax)
    style_axis(ax)
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("OverallQual")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def create_summary_table(train: pd.DataFrame, output_path: Path) -> None:
    summary = pd.DataFrame(
        {
            "metric": [
                "rows",
                "median_sale_price",
                "average_sale_price",
                "min_sale_price",
                "max_sale_price",
                "sale_year_range",
            ],
            "value": [
                len(train),
                round(train["SalePrice"].median(), 2),
                round(train["SalePrice"].mean(), 2),
                round(train["SalePrice"].min(), 2),
                round(train["SalePrice"].max(), 2),
                f"{int(train['YrSold'].min())}-{int(train['YrSold'].max())}",
            ],
        }
    )
    summary.to_csv(output_path, index=False)


def build_visuals(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = data_dir / "train.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"Expected train.csv in {data_dir}")

    train = pd.read_csv(train_path, keep_default_na=False, na_values=[""])

    save_monthly_price_trend(train, output_dir / "historical_price_trend.png")
    save_price_distribution(train, output_dir / "sale_price_distribution.png")
    save_quality_price_boxplot(train, output_dir / "quality_price_boxplot.png")
    save_living_area_scatter(train, output_dir / "living_area_sale_price.png")
    create_summary_table(train, output_dir / "eda_summary.csv")

    print(f"Wrote charts to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate exploratory charts for the House Prices dataset.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="Directory containing train.csv.")
    parser.add_argument("--output-dir", default="docs/images", help="Directory for generated charts.")
    return parser.parse_args()


if __name__ == "__main__":
    build_visuals(parse_args())
