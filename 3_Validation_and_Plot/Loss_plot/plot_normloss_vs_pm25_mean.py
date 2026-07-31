"""Plot scaled loss against last-year PM2.5 and report correlations."""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats as sps


matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data_matrix.npy"
FEATURE_FILE = BASE_DIR / "station_features.csv"
PM25_CHANNEL = -1
VAL_HOURS = 365 * 24
MAX_VALID = 1000.0
MIN_VALID_RATIO = 0.3
AXIS_MAX_QUANTILE = 0.99
SCATTER_SIZE = 10
SCATTER_ALPHA = 0.5
SCATTER_COLOR = "#1f77b4"
HEXBIN_GRIDSIZE = 45
HEXBIN_CMAP = "YlGnBu"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("loss_file", type=Path, help="Path to the station loss CSV file.")
    return parser.parse_args()


def build_station_table(data, df_loss, station_ids, val_hours):
    pm25 = np.asarray(data[:, PM25_CHANNEL, -val_hours:], dtype=np.float64)
    valid = (~np.isnan(pm25)) & (pm25 <= MAX_VALID)
    pm25 = np.where(valid, pm25, np.nan)

    df_stats = pd.DataFrame(
        {
            "Station_ID": station_ids.astype(str),
            "PM25_Mean": np.nanmean(pm25, axis=1),
            "PM25_Std": np.nanstd(pm25, axis=1),
            "Valid_Hours": valid.sum(axis=1),
        }
    )
    df = pd.merge(df_loss, df_stats, on="Station_ID", how="inner").dropna()
    df = df[df["Valid_Hours"] >= MIN_VALID_RATIO * val_hours]
    df["Norm_Base"] = df["Base_Loss"] / df["PM25_Std"]
    df["Norm_Total"] = df["Total_Loss"] / df["PM25_Std"]
    return df


def report_correlations(df, label):
    print(f"\n{label} (N={len(df)})")
    pairs = (
        ("PM25_Mean", "Total_Loss", "Combined loss vs. PM2.5 mean"),
        ("PM25_Mean", "Base_Loss", "Base loss vs. PM2.5 mean"),
        ("PM25_Mean", "Norm_Total", "Scaled combined loss vs. PM2.5 mean"),
        ("PM25_Mean", "Norm_Base", "Scaled base loss vs. PM2.5 mean"),
        ("PM25_Std", "Norm_Total", "Scaled combined loss vs. PM2.5 std"),
        ("PM25_Std", "Norm_Base", "Scaled base loss vs. PM2.5 std"),
    )

    for x_col, y_col, description in pairs:
        pearson_r, pearson_p = sps.pearsonr(df[x_col], df[y_col])
        spearman_r, spearman_p = sps.spearmanr(df[x_col], df[y_col])
        print(
            f"{description}: Pearson r={pearson_r:+.3f}, p={pearson_p:.2e}; "
            f"Spearman rho={spearman_r:+.3f}, p={spearman_p:.2e}"
        )


def plot_norm_loss_vs_mean(df, loss_col, loss_name, y_label, out_png, density=True):
    x = df["PM25_Mean"].to_numpy()
    y = df[loss_col].to_numpy()
    pearson_r, _ = sps.pearsonr(x, y)
    spearman_r, _ = sps.spearmanr(x, y)
    x_max = np.quantile(x, AXIS_MAX_QUANTILE) * 1.1
    y_min = y.min() * 0.9
    y_max = np.quantile(y, AXIS_MAX_QUANTILE) * 1.05
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.set_axisbelow(True)

    if density:
        visible = (x >= 0) & (x <= x_max) & (y >= y_min) & (y <= y_max)
        hexbin = ax.hexbin(
            x[visible],
            y[visible],
            gridsize=HEXBIN_GRIDSIZE,
            extent=(0, x_max, y_min, y_max),
            mincnt=1,
            cmap=HEXBIN_CMAP,
            linewidths=0.15,
            edgecolors="white",
            alpha=0.85,
            zorder=1,
        )
        ax.scatter(
            x[visible],
            y[visible],
            s=7,
            color="black",
            alpha=0.18,
            linewidths=0,
            zorder=2,
        )
        cbar = fig.colorbar(hexbin, ax=ax, pad=0.015)
        cbar.set_label("Station count per bin", fontsize=11)
        cbar.ax.tick_params(labelsize=10)
    else:
        ax.scatter(
            x,
            y,
            s=SCATTER_SIZE,
            color=SCATTER_COLOR,
            alpha=SCATTER_ALPHA,
        )

    slope, intercept = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x_max, 100)
    ax.plot(
        x_line,
        slope * x_line + intercept,
        color="dimgray",
        linestyle="--",
        linewidth=2.0,
        zorder=3,
        label="Linear fit",
    )
    ax.annotate(
        f"Pearson $r$ = {pearson_r:+.3f}\nSpearman $\\rho$ = {spearman_r:+.3f}",
        xy=(0.97, 0.95),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=13,
        bbox={
            "boxstyle": "round,pad=0.4",
            "facecolor": "white",
            "edgecolor": "lightgray",
            "alpha": 0.9,
        },
    )
    ax.set_xlim(0, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title(f"{loss_name} vs. PM2.5 Mean", fontsize=14, fontweight="bold")
    ax.set_xlabel("PM2.5 Mean (μg/m³)", fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"Saved {out_png.name}.")


def main(loss_file):
    print("Loading loss and PM2.5 data...")
    df_loss = pd.read_csv(loss_file).dropna(subset=["Total_Loss"])
    df_loss["Station_ID"] = df_loss["Station_ID"].astype(str)
    station_ids = pd.read_csv(FEATURE_FILE, usecols=["站点编号"])["站点编号"]
    data = np.load(DATA_FILE, mmap_mode="r")

    df = build_station_table(data, df_loss, station_ids, VAL_HOURS)
    print(
        f"PM2.5 mean: {df['PM25_Mean'].min():.1f}-"
        f"{df['PM25_Mean'].max():.1f} μg/m³; "
        f"median={df['PM25_Mean'].median():.1f}"
    )
    report_correlations(df, f"Last year: {VAL_HOURS} hours")
    print(f"\nPlotting {len(df)} stations...")
    plot_norm_loss_vs_mean(
        df,
        "Norm_Base",
        "Normalized Base Loss",
        "Normalized Base Loss (Base_Loss / Std)",
        BASE_DIR / "Normalized_Base_Loss_vs_pm25_mean.png",
    )
    plot_norm_loss_vs_mean(
        df,
        "Norm_Total",
        "Normalized Combined Loss",
        "Normalized Combined Loss (Combined_Loss / Std)",
        BASE_DIR / "Normalized_Total_Loss_vs_pm25_mean.png",
    )



if __name__ == "__main__":
    args = parse_args()
    main(args.loss_file)
