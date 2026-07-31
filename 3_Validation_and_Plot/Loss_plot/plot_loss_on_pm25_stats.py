"""Plot station loss against last-year PM2.5 mean and standard deviation."""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd


matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("loss_file", type=Path, help="Path to the station loss CSV file.")
    return parser.parse_args()


def main(loss_file):
    data_file = BASE_DIR / "data_matrix.npy"
    feature_file = BASE_DIR / "station_features.csv"
    out_png = BASE_DIR / "loss_vs_pm25_distribution.png"
    hours_per_year = 365 * 24
    max_valid = 1000.0
    vmin_quantile = 0.05
    vmax_quantile = 0.95
    axis_max_quantile = 0.99
    scatter_size = 40
    scatter_alpha = 0.8

    print("Loading loss and PM2.5 data...")
    df_loss = pd.read_csv(loss_file).dropna(subset=["Total_Loss"])
    df_loss["Station_ID"] = df_loss["Station_ID"].astype(str)

    data = np.load(data_file, mmap_mode="r")
    pm25 = np.asarray(data[:, -1, -hours_per_year:], dtype=np.float64)
    valid = (~np.isnan(pm25)) & (pm25 <= max_valid)
    pm25 = np.where(valid, pm25, np.nan)
    station_means = np.nanmean(pm25, axis=1)
    station_stds = np.nanstd(pm25, axis=1)
    df_features = pd.read_csv(feature_file, usecols=["站点编号"])
    df_stats = pd.DataFrame(
        {
            "Station_ID": df_features["站点编号"].astype(str),
            "PM25_Mean": station_means,
            "PM25_Std": station_stds,
        }
    )

    df_merged = pd.merge(df_loss, df_stats, on="Station_ID", how="inner")
    df_merged = df_merged.dropna(
        subset=["Total_Loss", "PM25_Mean", "PM25_Std"]
    )
    print(f"Plotting {len(df_merged)} stations...")

    plt.figure(figsize=(10, 8))
    vmin_loss = df_merged["Total_Loss"].quantile(vmin_quantile)
    vmax_loss = df_merged["Total_Loss"].quantile(vmax_quantile)
    scatter = plt.scatter(
        df_merged["PM25_Mean"],
        df_merged["PM25_Std"],
        c=df_merged["Total_Loss"],
        cmap="Spectral_r",
        s=scatter_size,
        alpha=scatter_alpha,
        edgecolor="w",
        linewidth=0.5,
        vmin=vmin_loss,
        vmax=vmax_loss,
    )

    cbar = plt.colorbar(scatter)
    cbar.set_label(
        "Combined Loss",
        fontsize=12,
        fontweight="bold",
    )
    cbar_ticks = np.linspace(vmin_loss, vmax_loss, 6)
    cbar_labels = [f"{value:.2f}" for value in cbar_ticks]
    cbar_labels[0] = f"{vmin_loss:.2f} (5%)"
    cbar_labels[-1] = f"{vmax_loss:.2f} (95%)"
    cbar.set_ticks(cbar_ticks)
    cbar.set_ticklabels(cbar_labels)

    x_max = df_merged["PM25_Mean"].quantile(axis_max_quantile) * 1.1
    y_max = df_merged["PM25_Std"].quantile(axis_max_quantile) * 1.1
    plt.xlim(0, x_max)
    plt.ylim(0, y_max)
    plt.title(
        "Station Loss vs. PM2.5 Distribution",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel("PM2.5 Mean (μg/m³)", fontsize=12)
    plt.ylabel("PM2.5 Standard Deviation (μg/m³)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"Saved {out_png.name}.")


if __name__ == "__main__":
    args = parse_args()
    main(args.loss_file)
