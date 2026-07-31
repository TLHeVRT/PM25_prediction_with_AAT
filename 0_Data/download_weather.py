"""Download per-station ERA5 weather data from stations.csv and skip existing files."""

import os
import time
from pathlib import Path

import pandas as pd
import requests


STATIONS = Path("stations.csv")
OUTPUT_DIR = Path("weather")
URL = "https://archive-api.open-meteo.com/v1/archive"
VARIABLES = (
    "temperature_2m,relative_humidity_2m,precipitation,"
    "wind_speed_10m,wind_direction_10m,wind_speed_100m,"
    "wind_direction_100m,pressure_msl,cloud_cover"
)


def get_weather(session, latitude, longitude):
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": "2017-10-01",
        "end_date": "2022-10-01",
        "hourly": VARIABLES,
        "models": "era5",
        "timezone": "Asia/Shanghai",
        "cell_selection": "land",
    }
    for attempt in range(5):
        try:
            response = session.get(URL, params=params, timeout=120)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 429 and attempt < 4:
                time.sleep(10 * (2 ** attempt))
                continue
            print("HTTP request failed:", exc)
            return None
        except requests.exceptions.RequestException as exc:
            print("Request failed:", exc)
            return None
    return None


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    stations = pd.read_csv(STATIONS, dtype={"station_id": str})
    completed = {path.stem for path in OUTPUT_DIR.glob("*.csv")}
    stations = stations[~stations["station_id"].isin(completed)].reset_index(drop=True)
    with requests.Session() as session:
        for i, row in stations.iterrows():
            station_id = row["station_id"]
            print(i + 1, "/", len(stations), station_id, flush=True)
            data = get_weather(session, row["latitude"], row["longitude"])
            if data and "hourly" in data:
                part = OUTPUT_DIR / f"{station_id}.csv.part"
                pd.DataFrame(data["hourly"]).to_csv(part, index=False)
                os.replace(part, OUTPUT_DIR / f"{station_id}.csv")
            if i + 1 < len(stations):
                time.sleep(30 if (i + 1) % 5 == 0 else 3)


if __name__ == "__main__":
    main()
