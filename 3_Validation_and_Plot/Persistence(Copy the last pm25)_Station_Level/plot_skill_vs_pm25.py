"""Plot model skill against test-year PM2.5 mean and standard deviation."""

import numpy as np
import pandas as pd
import matplotlib
from scipy.stats import pearsonr, spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DATA_PATH = "data_matrix.npy"
MODEL_RESULTS_PATH = "station_test_losses.csv"
PERSISTENCE_RESULTS_PATH = "Persistence_test_losses.csv"
STATION_FEATURES_PATH = "station_features.csv"
BASE_FIGURE_PATH = "skill_MAE_vs_pm25_mean.png"
COMBINED_FIGURE_PATH = "skill_combined_vs_pm25_mean.png"
BASE_STD_FIGURE_PATH = "skill_MAE_vs_pm25_std.png"
COMBINED_STD_FIGURE_PATH = "skill_combined_vs_pm25_std.png"
BASE_SKILL_TABLE_PATH = "base_skill.csv"

HOURS_PER_YEAR = 365 * 24
PM25_CHANNEL = 9
WEATHER_FEATURES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_speed_100m",
    "wind_direction_100m",
    "pressure_msl",
    "cloud_cover",
]


raw_data = np.load(DATA_PATH, mmap_mode="r")
test_pm25 = np.asarray(
    raw_data[:, PM25_CHANNEL, -HOURS_PER_YEAR:], dtype=np.float32
)
valid_pm25 = (~np.isnan(test_pm25)) & (test_pm25 <= 1000)
pm25_mean = np.where(valid_pm25, test_pm25, 0).sum(
    axis=1, dtype=np.float64
) / valid_pm25.sum(axis=1)
pm25_std = np.nanstd(
    np.where(valid_pm25, test_pm25, np.nan),
    axis=1,
    dtype=np.float64,
)

weather_means = np.empty((raw_data.shape[0], len(WEATHER_FEATURES)))
weather_stds = np.empty((raw_data.shape[0], len(WEATHER_FEATURES)))
for channel, weather_feature in enumerate(WEATHER_FEATURES):
    test_weather = np.asarray(
        raw_data[:, channel, -HOURS_PER_YEAR:], dtype=np.float32
    )
    weather_means[:, channel] = np.nanmean(
        test_weather, axis=1, dtype=np.float64
    )
    weather_stds[:, channel] = np.nanstd(
        test_weather, axis=1, dtype=np.float64
    )

model_results = pd.read_csv(MODEL_RESULTS_PATH)
persistence_results = pd.read_csv(PERSISTENCE_RESULTS_PATH)
model_results["PM25_Mean"] = pm25_mean
model_results["PM25_Std"] = pm25_std
for channel, weather_feature in enumerate(WEATHER_FEATURES):
    model_results[f"{weather_feature}_Mean"] = weather_means[:, channel]
    model_results[f"{weather_feature}_Std"] = weather_stds[:, channel]

results = model_results.merge(
    persistence_results,
    on="Station_ID",
    suffixes=("_Model", "_Persistence"),
)
results["Skill_Base"] = 1 - (
    results["Base_Loss_Model"] / results["Base_Loss_Persistence"]
)
results["Skill_Combined"] = 1 - (
    results["Total_Loss_Model"] / results["Total_Loss_Persistence"]
)
results[["Station_ID", "Skill_Base"]].rename(
    columns={"Skill_Base": "base_skill"}
).to_csv(BASE_SKILL_TABLE_PATH, index=False, encoding="utf-8-sig")

station_features = pd.read_csv(STATION_FEATURES_PATH)
station_features = station_features.rename(
    columns={station_features.columns[0]: "Station_ID"}
)
static_feature_columns = station_features.columns[1:-6]
static_feature_results = results[["Station_ID", "Skill_Base"]].merge(
    station_features,
    on="Station_ID",
)

pearson_base = pearsonr(results["PM25_Mean"], results["Skill_Base"]).statistic
spearman_base = spearmanr(results["PM25_Mean"], results["Skill_Base"]).statistic
pearson_combined = pearsonr(
    results["PM25_Mean"], results["Skill_Combined"]
).statistic
spearman_combined = spearmanr(
    results["PM25_Mean"], results["Skill_Combined"]
).statistic
pearson_base_std = pearsonr(
    results["PM25_Std"], results["Skill_Base"]
).statistic
spearman_base_std = spearmanr(
    results["PM25_Std"], results["Skill_Base"]
).statistic
pearson_combined_std = pearsonr(
    results["PM25_Std"], results["Skill_Combined"]
).statistic
spearman_combined_std = spearmanr(
    results["PM25_Std"], results["Skill_Combined"]
).statistic

print(f"Skill_Base: Pearson r = {pearson_base:.6f}, Spearman ρ = {spearman_base:.6f}")
print(
    f"Skill_Combined: Pearson r = {pearson_combined:.6f}, "
    f"Spearman ρ = {spearman_combined:.6f}"
)
print(
    f"Skill_Base vs. PM25_Std: Pearson r = {pearson_base_std:.6f}, "
    f"Spearman ρ = {spearman_base_std:.6f}"
)
print(
    f"Skill_Combined vs. PM25_Std: Pearson r = {pearson_combined_std:.6f}, "
    f"Spearman ρ = {spearman_combined_std:.6f}"
)
print("\nSkill_Base vs. station static features:")
for feature in static_feature_columns:
    feature_pearson = pearsonr(
        static_feature_results[feature], static_feature_results["Skill_Base"]
    ).statistic
    feature_spearman = spearmanr(
        static_feature_results[feature], static_feature_results["Skill_Base"]
    ).statistic
    print(
        f"{feature}: Pearson r = {feature_pearson:+.6f}, "
        f"Spearman ρ = {feature_spearman:+.6f}"
    )

print("\nSkill_Base vs. last-year weather statistics:")
for weather_feature in WEATHER_FEATURES:
    for statistic in ("Mean", "Std"):
        weather_column = f"{weather_feature}_{statistic}"
        weather_pearson = pearsonr(
            results[weather_column], results["Skill_Base"]
        ).statistic
        weather_spearman = spearmanr(
            results[weather_column], results["Skill_Base"]
        ).statistic
        print(
            f"{weather_column}: Pearson r = {weather_pearson:+.6f}, "
            f"Spearman ρ = {weather_spearman:+.6f}"
        )

plt.figure(figsize=(7, 5))
hexbin = plt.hexbin(
    results["PM25_Mean"],
    results["Skill_Base"],
    gridsize=30,
    mincnt=1,
    cmap="Blues",
    linewidths=0.15,
    edgecolors="white",
    alpha=0.85,
    zorder=1,
)
colorbar = plt.colorbar(hexbin)
colorbar.set_label("Station count per bin", fontsize=11)
colorbar.ax.tick_params(labelsize=10)
plt.scatter(
    results["PM25_Mean"],
    results["Skill_Base"],
    s=7,
    color="black",
    alpha=0.18,
    linewidths=0,
    zorder=2,
)
plt.axhline(0, color="gray", linewidth=1, linestyle="--")
plt.annotate(
    f"Pearson $r$ = {pearson_base:+.3f}\n"
    f"Spearman $\\rho$ = {spearman_base:+.3f}",
    xy=(0.97, 0.95),
    xycoords="axes fraction",
    ha="right",
    va="top",
    fontsize=10,
    bbox={
        "boxstyle": "round,pad=0.25",
        "facecolor": "white",
        "edgecolor": "lightgray",
        "alpha": 0.9,
    },
)
plt.xlabel("Test-year mean PM2.5 concentration (μg/m³)")
plt.ylabel("Skill (MAE Loss)")
plt.title("MAE-loss Skill vs. Mean PM2.5 Concentration")
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(BASE_FIGURE_PATH, dpi=300)
plt.close()

plt.figure(figsize=(7, 5))
hexbin = plt.hexbin(
    results["PM25_Mean"],
    results["Skill_Combined"],
    gridsize=30,
    mincnt=1,
    cmap="Oranges",
    linewidths=0.15,
    edgecolors="white",
    alpha=0.85,
    zorder=1,
)
colorbar = plt.colorbar(hexbin)
colorbar.set_label("Station count per bin", fontsize=11)
colorbar.ax.tick_params(labelsize=10)
plt.scatter(
    results["PM25_Mean"],
    results["Skill_Combined"],
    s=7,
    color="black",
    alpha=0.18,
    linewidths=0,
    zorder=2,
)
plt.axhline(0, color="gray", linewidth=1, linestyle="--")
plt.annotate(
    f"Pearson $r$ = {pearson_combined:+.3f}\n"
    f"Spearman $\\rho$ = {spearman_combined:+.3f}",
    xy=(0.97, 0.95),
    xycoords="axes fraction",
    ha="right",
    va="top",
    fontsize=10,
    bbox={
        "boxstyle": "round,pad=0.25",
        "facecolor": "white",
        "edgecolor": "lightgray",
        "alpha": 0.9,
    },
)
plt.xlabel("Test-year mean PM2.5 concentration (μg/m³)")
plt.ylabel("Skill (Combined Loss)")
plt.title("Combined-loss Skill vs. Mean PM2.5 Concentration")
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(COMBINED_FIGURE_PATH, dpi=300)
plt.close()

plt.figure(figsize=(7, 5))
hexbin = plt.hexbin(
    results["PM25_Std"],
    results["Skill_Base"],
    gridsize=30,
    mincnt=1,
    cmap="Blues",
    linewidths=0.15,
    edgecolors="white",
    alpha=0.85,
    zorder=1,
)
colorbar = plt.colorbar(hexbin)
colorbar.set_label("Station count per bin", fontsize=11)
colorbar.ax.tick_params(labelsize=10)
plt.scatter(
    results["PM25_Std"],
    results["Skill_Base"],
    s=7,
    color="black",
    alpha=0.18,
    linewidths=0,
    zorder=2,
)
plt.axhline(0, color="gray", linewidth=1, linestyle="--")
plt.annotate(
    f"Pearson $r$ = {pearson_base_std:+.3f}\n"
    f"Spearman $\\rho$ = {spearman_base_std:+.3f}",
    xy=(0.97, 0.95),
    xycoords="axes fraction",
    ha="right",
    va="top",
    fontsize=10,
    bbox={
        "boxstyle": "round,pad=0.25",
        "facecolor": "white",
        "edgecolor": "lightgray",
        "alpha": 0.9,
    },
)
plt.xlabel("Test-year PM2.5 standard deviation (μg/m³)")
plt.ylabel("Skill (MAE Loss)")
plt.title("MAE-loss Skill vs. PM2.5 Standard Deviation")
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(BASE_STD_FIGURE_PATH, dpi=300)
plt.close()

plt.figure(figsize=(7, 5))
hexbin = plt.hexbin(
    results["PM25_Std"],
    results["Skill_Combined"],
    gridsize=30,
    mincnt=1,
    cmap="Oranges",
    linewidths=0.15,
    edgecolors="white",
    alpha=0.85,
    zorder=1,
)
colorbar = plt.colorbar(hexbin)
colorbar.set_label("Station count per bin", fontsize=11)
colorbar.ax.tick_params(labelsize=10)
plt.scatter(
    results["PM25_Std"],
    results["Skill_Combined"],
    s=7,
    color="black",
    alpha=0.18,
    linewidths=0,
    zorder=2,
)
plt.axhline(0, color="gray", linewidth=1, linestyle="--")
plt.annotate(
    f"Pearson $r$ = {pearson_combined_std:+.3f}\n"
    f"Spearman $\\rho$ = {spearman_combined_std:+.3f}",
    xy=(0.97, 0.95),
    xycoords="axes fraction",
    ha="right",
    va="top",
    fontsize=10,
    bbox={
        "boxstyle": "round,pad=0.25",
        "facecolor": "white",
        "edgecolor": "lightgray",
        "alpha": 0.9,
    },
)
plt.xlabel("Test-year PM2.5 standard deviation (μg/m³)")
plt.ylabel("Skill (Combined Loss)")
plt.title("Combined-loss Skill vs. PM2.5 Standard Deviation")
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(COMBINED_STD_FIGURE_PATH, dpi=300)
plt.close()
