"""Plot scaled loss against last-year PM2.5 mean and standard deviation."""

import argparse
from pathlib import Path

import matplotlib
import matplotlib.colors as mcolors
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
    out_png_total = BASE_DIR / "Normalized_Total_Loss_loss_vs_pm25_distribution.png"
    out_png_base = BASE_DIR / "Normalized_Base_Loss_loss_vs_pm25_distribution.png"
    hours_per_year = 365 * 24
    max_valid = 1000.0
    vmax_quantile = 0.95
    vmin_quantile = 0.05
    axis_max_quantile = 0.99
    scatter_size = 40
    scatter_alpha = 0.8
    color_map = "Spectral_r"

    print("Loading loss and PM2.5 data...")
    df_loss = pd.read_csv(loss_file).dropna(subset=["Base_Loss"])
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
        subset=["Total_Loss", "Base_Loss", "PM25_Mean", "PM25_Std"]
    )
    df_merged["Norm_Total_Loss"] = (
        df_merged["Total_Loss"] / (df_merged["PM25_Std"] + 1e-8)
    )
    df_merged["Norm_Base_Loss"] = (
        df_merged["Base_Loss"] / (df_merged["PM25_Std"] + 1e-8)
    )
    print(f"Plotting {len(df_merged)} stations...")

    x_max = df_merged["PM25_Mean"].quantile(axis_max_quantile) * 1.1
    y_max = df_merged["PM25_Std"].quantile(axis_max_quantile) * 1.1

    plt.figure(figsize=(10, 8))
    norm_total = mcolors.Normalize(
        vmin=df_merged["Norm_Total_Loss"].quantile(vmin_quantile),
        vmax=df_merged["Norm_Total_Loss"].quantile(vmax_quantile),
    )
    scatter_total = plt.scatter(
        df_merged["PM25_Mean"],
        df_merged["PM25_Std"],
        c=df_merged["Norm_Total_Loss"],
        cmap=color_map,
        norm=norm_total,
        s=scatter_size,
        alpha=scatter_alpha,
        edgecolor="w",
        linewidth=0.5,
    )
    cbar_total = plt.colorbar(scatter_total)
    cbar_total.set_label(
        "Normalized Combined Loss (Combined_Loss / Std)",
        fontsize=12,
        fontweight="bold",
    )
    cbar_total_ticks = np.linspace(norm_total.vmin, norm_total.vmax, 6)
    cbar_total_labels = [f"{value:.3f}" for value in cbar_total_ticks]
    cbar_total_labels[0] = f"{norm_total.vmin:.3f} (5%)"
    cbar_total_labels[-1] = f"{norm_total.vmax:.3f} (95%)"
    cbar_total.set_ticks(cbar_total_ticks)
    cbar_total.set_ticklabels(cbar_total_labels)
    plt.xlim(0, x_max)
    plt.ylim(0, y_max)
    plt.title(
        "Normalized Combined Loss vs. PM2.5 Distribution",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel("PM2.5 Mean (μg/m³)", fontsize=12)
    plt.ylabel("PM2.5 Standard Deviation (μg/m³)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_png_total, dpi=300)
    plt.close()
    print(f"Saved {out_png_total.name}.")

    plt.figure(figsize=(10, 8))
    norm_base = mcolors.Normalize(
        vmin=df_merged["Norm_Base_Loss"].quantile(vmin_quantile),
        vmax=df_merged["Norm_Base_Loss"].quantile(vmax_quantile),
    )
    scatter_base = plt.scatter(
        df_merged["PM25_Mean"],
        df_merged["PM25_Std"],
        c=df_merged["Norm_Base_Loss"],
        cmap=color_map,
        norm=norm_base,
        s=scatter_size,
        alpha=scatter_alpha,
        edgecolor="w",
        linewidth=0.5,
    )
    cbar_base = plt.colorbar(scatter_base)
    cbar_base.set_label(
        "Normalized Base Loss (Base_Loss / Std)",
        fontsize=12,
        fontweight="bold",
    )
    cbar_base_ticks = np.linspace(norm_base.vmin, norm_base.vmax, 6)
    cbar_base_labels = [f"{value:.3f}" for value in cbar_base_ticks]
    cbar_base_labels[0] = f"{norm_base.vmin:.3f} (5%)"
    cbar_base_labels[-1] = f"{norm_base.vmax:.3f} (95%)"
    cbar_base.set_ticks(cbar_base_ticks)
    cbar_base.set_ticklabels(cbar_base_labels)
    plt.xlim(0, x_max)
    plt.ylim(0, y_max)
    plt.title(
        "Normalized Base Loss vs. PM2.5 Distribution",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel("PM2.5 Mean (μg/m³)", fontsize=12)
    plt.ylabel("PM2.5 Standard Deviation (μg/m³)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_png_base, dpi=300)
    plt.close()
    print(f"Saved {out_png_base.name}.")


if __name__ == "__main__":
    args = parse_args()
    main(args.loss_file)
