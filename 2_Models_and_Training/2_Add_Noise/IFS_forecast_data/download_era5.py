"""Download matching per-station ERA5 weather data."""

import os
import time
from pathlib import Path

import pandas as pd
import requests


STATIONS = Path("station_features.csv")
OUTPUT_DIR = Path("era5")
URL = "https://archive-api.open-meteo.com/v1/archive"
VARIABLES = (
    "temperature_2m,relative_humidity_2m,precipitation,"
    "wind_speed_10m,wind_direction_10m,wind_speed_100m,"
    "wind_direction_100m,pressure_msl,cloud_cover"
)
START_DATE = "2024-03-14"
END_DATE = "2025-03-15"
FIRST_TIME = "2024-03-14T09:00"
LAST_TIME = "2025-03-15T08:00"


def get_weather(session, latitude, longitude):
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": VARIABLES,
        "models": "era5",
        "timezone": "Asia/Shanghai",
        "cell_selection": "land",
        "wind_speed_unit": "kmh",
        "precipitation_unit": "mm",
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
    stations = pd.read_csv(STATIONS, dtype={"站点编号": str})
    completed = {path.stem for path in OUTPUT_DIR.glob("*.csv")}
    stations = stations[~stations["站点编号"].isin(completed)].reset_index(drop=True)

    with requests.Session() as session:
        for i, row in stations.iterrows():
            station_id = row["站点编号"]
            print(i + 1, "/", len(stations), station_id, flush=True)
            data = get_weather(session, row["纬度"], row["经度"])
            if data and "hourly" in data:
                weather = pd.DataFrame(data["hourly"])
                weather = weather[
                    weather["time"].between(FIRST_TIME, LAST_TIME, inclusive="both")
                ]
                part = OUTPUT_DIR / f"{station_id}.csv.part"
                weather.to_csv(part, index=False)
                os.replace(part, OUTPUT_DIR / f"{station_id}.csv")
            if i + 1 < len(stations):
                time.sleep(30 if (i + 1) % 5 == 0 else 3)


if __name__ == "__main__":
    main()
