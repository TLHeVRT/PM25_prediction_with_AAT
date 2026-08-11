"""Plot station MAE skill on a map of China."""

import argparse
from io import BytesIO
from pathlib import Path
from urllib.request import Request, urlopen

import geopandas as gpd
import matplotlib
import pandas as pd


matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
FEATURE_FILE = BASE_DIR / "station_features.csv"
OUT_FILE = BASE_DIR / "MAEskill_map.png"
CHINA_MAP_URL = "https://geo.datav.aliyun.com/areas_v3/bound/100000_full.json"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "baseskill_file", type=Path, help="Path to the station MAE skill CSV file."
    )
    return parser.parse_args()


def main(baseskill_file):
    print("Loading station data...")
    df_skill = pd.read_csv(baseskill_file, usecols=["Station_ID", "base_skill"])
    df_features = pd.read_csv(FEATURE_FILE)
    df_skill["Station_ID"] = df_skill["Station_ID"].astype(str)
    df_features["站点编号"] = df_features["站点编号"].astype(str)
    df_valid = pd.merge(
        df_features,
        df_skill,
        left_on="站点编号",
        right_on="Station_ID",
        how="inner",
        validate="one_to_one",
    ).dropna(subset=["base_skill"])

    skill_min = df_valid["base_skill"].quantile(0.05)
    skill_max = df_valid["base_skill"].quantile(0.95)

    plt.rcParams["font.family"] = "Times New Roman"
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

    request = Request(CHINA_MAP_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=20) as response:
        china_map = gpd.read_file(BytesIO(response.read()), columns=["geometry"])
    china_map.plot(
        ax=ax,
        facecolor="#F5F5F5",
        edgecolor="#666666",
        linewidth=0.5,
        zorder=1,
    )

    scatter = ax.scatter(
        df_valid["经度"],
        df_valid["纬度"],
        c=df_valid["base_skill"],
        cmap="coolwarm",
        vmin=skill_min,
        vmax=skill_max,
        s=25,
        alpha=0.85,
        edgecolors="white",
        linewidths=0.3,
        zorder=2,
    )

    cbar = plt.colorbar(scatter, ax=ax, fraction=0.035, pad=0.04)
    cbar.set_label(
        "MAE Skill",
        fontsize=12,
        fontweight="bold",
    )
    cbar_ticks = [skill_min + (skill_max - skill_min) * i / 5 for i in range(6)]
    cbar_labels = [f"{value:.2f}" for value in cbar_ticks]
    cbar_labels[0] = f"{skill_min:.2f} (5%)"
    cbar_labels[-1] = f"{skill_max:.2f} (95%)"
    cbar.set_ticks(cbar_ticks)
    cbar.set_ticklabels(cbar_labels)
    cbar.ax.tick_params(labelsize=10)

    ax.set_title(
        "Spatial Distribution of Station MAE Skill",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)
    ax.set_ylim(17, 55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.savefig(OUT_FILE, dpi=300, bbox_inches="tight", transparent=False)
    plt.close(fig)
    print(f"Saved {OUT_FILE.name}.")


if __name__ == "__main__":
    args = parse_args()
    main(args.baseskill_file)