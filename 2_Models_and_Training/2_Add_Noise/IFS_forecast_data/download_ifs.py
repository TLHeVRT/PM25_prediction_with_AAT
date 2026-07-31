"""Download daily 48-hour ECMWF IFS forecasts for all stations."""

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm


STATIONS = Path("station_features.csv")
OUTPUT_DIR = Path("ifs")
URL = "https://single-runs-api.open-meteo.com/v1/forecast"
VARIABLES = (
    "temperature_2m,relative_humidity_2m,precipitation,"
    "wind_speed_10m,wind_direction_10m,wind_speed_100m,"
    "wind_direction_100m,pressure_msl,cloud_cover"
)
VARIABLE_NAMES = VARIABLES.split(",")
RUNS = pd.date_range("2024-03-14", periods=183, freq="48h", tz="UTC")
BATCH_SIZE = 100


def get_batch(session, stations, run):
    params = {
        "latitude": ",".join(stations["纬度"].astype(str)),
        "longitude": ",".join(stations["经度"].astype(str)),
        "hourly": VARIABLES,
        "models": "ecmwf_ifs",
        "run": run.strftime("%Y-%m-%dT%H:%M"),
        "forecast_hours": 49,
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

    with requests.Session() as session:
        request_number = 0
        for i, run in enumerate(RUNS):
            name = run.strftime("%Y%m%dT%H%MZ")
            output = OUTPUT_DIR / f"{name}.npz"
            if output.exists():
                print(i + 1, "/", len(RUNS), output.name, "exists", flush=True)
                continue

            starts = range(0, len(stations), BATCH_SIZE)
            batch_outputs = [
                OUTPUT_DIR / f"{name}.batch{batch_number:02d}.npz"
                for batch_number in range(len(starts))
            ]
            with tqdm(
                total=len(stations),
                desc=f"{i + 1}/{len(RUNS)} {name}",
                unit="station",
                mininterval=0,
                dynamic_ncols=True,
            ) as progress:
                for batch_number, start in enumerate(starts):
                    batch = stations.iloc[start:start + BATCH_SIZE]
                    batch_output = batch_outputs[batch_number]
                    if batch_output.exists():
                        progress.set_postfix_str(
                            f"batch {batch_number + 1}/{len(starts)} resumed",
                            refresh=False,
                        )
                        progress.update(len(batch))
                        continue

                    data = get_batch(session, batch, run)
                    request_number += 1
                    time.sleep(30 if request_number % 5 == 0 else 3)
                    if (
                        not data
                        or len(data) != len(batch)
                        or any("hourly" not in location for location in data)
                    ):
                        break

                    block = np.asarray(
                        [
                            np.column_stack(
                                [
                                    location["hourly"][variable][1:]
                                    for variable in VARIABLE_NAMES
                                ]
                            )
                            for location in data
                        ],
                        dtype=np.float32,
                    )
                    batch_part = OUTPUT_DIR / f"{name}.batch{batch_number:02d}.part.npz"
                    np.savez_compressed(
                        batch_part,
                        data=block,
                        grid_latitude=np.asarray(
                            [location["latitude"] for location in data]
                        ),
                        grid_longitude=np.asarray(
                            [location["longitude"] for location in data]
                        ),
                        time=np.asarray(data[0]["hourly"]["time"][1:]),
                    )
                    os.replace(batch_part, batch_output)
                    progress.set_postfix_str(
                        f"batch {batch_number + 1}/{len(starts)} saved",
                        refresh=False,
                    )
                    progress.update(len(batch))

            if not all(path.exists() for path in batch_outputs):
                continue

            blocks = []
            grid_latitude = []
            grid_longitude = []
            for batch_output in batch_outputs:
                with np.load(batch_output, allow_pickle=False) as batch:
                    blocks.append(batch["data"])
                    grid_latitude.append(batch["grid_latitude"])
                    grid_longitude.append(batch["grid_longitude"])
                    forecast_time = batch["time"]

            part = OUTPUT_DIR / f"{name}.part.npz"
            np.savez_compressed(
                part,
                data=np.concatenate(blocks),
                station_id=np.asarray(stations["站点编号"], dtype=str),
                latitude=stations["纬度"].to_numpy(),
                longitude=stations["经度"].to_numpy(),
                grid_latitude=np.concatenate(grid_latitude),
                grid_longitude=np.concatenate(grid_longitude),
                time=forecast_time,
                variables=np.asarray(VARIABLE_NAMES),
                timezone="Asia/Shanghai",
            )
            os.replace(part, output)
            for batch_output in batch_outputs:
                batch_output.unlink()
            print(i + 1, "/", len(RUNS), output.name, flush=True)


if __name__ == "__main__":
    main()
