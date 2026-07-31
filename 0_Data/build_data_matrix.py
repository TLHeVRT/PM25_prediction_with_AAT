"""Combine per-station PM2.5 and weather data into a 3D array matching the legacy format."""

from pathlib import Path

import numpy as np
import pandas as pd


STATIONS = Path("stations.csv")
PM25_DIR = Path("pm25")
WEATHER_DIR = Path("weather")
OUTPUT = Path("data_matrix.npy")
WEATHER_COLUMNS = [
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


stations = pd.read_csv(STATIONS, dtype={"station_id": str})
station_ids = stations["station_id"].tolist()
first_weather = pd.read_csv(WEATHER_DIR / f"{station_ids[0]}.csv", usecols=["time"])
hours = len(first_weather)
missing_before = 0
filled_count = 0

matrix = np.lib.format.open_memmap(
    OUTPUT,
    mode="w+",
    dtype=np.float16,
    shape=(len(station_ids), 10, hours),
)

for i, station_id in enumerate(station_ids):
    weather = pd.read_csv(WEATHER_DIR / f"{station_id}.csv")
    matrix[i, :9, :] = weather[WEATHER_COLUMNS].to_numpy(dtype=np.float16).T

    pm25 = pd.to_numeric(
        pd.read_csv(PM25_DIR / f"{station_id}.csv")["pm25"],
        errors="coerce",
    )
    pm25 = pm25.mask(~np.isfinite(pm25))
    # 32768 is used only to cap extremely large outliers.
    pm25 = pm25.clip(upper=32768)
    missing = pm25.isna()
    missing_before += int(missing.sum())
    groups = missing.ne(missing.shift()).cumsum()
    gap_lengths = missing.groupby(groups).transform("sum")
    interpolated = pm25.interpolate(method="linear", limit_area="inside")
    short_gaps = missing & gap_lengths.le(2) & interpolated.notna()
    filled_count += int(short_gaps.sum())
    pm25.loc[short_gaps] = interpolated.loc[short_gaps].round()
    matrix[i, 9, :] = pm25.round(1).to_numpy(dtype=np.float16)

    if (i + 1) % 50 == 0 or i + 1 == len(station_ids):
        matrix.flush()
        print(i + 1, "/", len(station_ids), flush=True)

del matrix
print("Completed:", OUTPUT, (len(station_ids), 10, hours))
print("PM2.5 missing before interpolation:", missing_before,
      "interpolated:", filled_count, "remaining missing:", missing_before - filled_count)
