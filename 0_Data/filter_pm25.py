"""Filter PM2.5 stations, export per-station files and stations.csv, then remove intermediate files."""

import csv
import gzip
import shutil
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


START = date(2017, 10, 1)
END = date(2022, 10, 1)
WORK_DIR = Path(".pm25_work")
RAW_DIR = WORK_DIR / "raw"
RAW_STATIONS = WORK_DIR / "stations.csv"
OUTPUT_DIR = Path("pm25")
STATIONS_OUTPUT = Path("stations.csv")


def longest_gap(mask):
    padded = np.r_[False, mask, False]
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return int((edges[1::2] - edges[::2]).max(initial=0))


def main():
    station_ids = set()
    for path in RAW_DIR.glob("china_sites_*.csv.gz"):
        with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as f:
            station_ids.update(next(csv.reader(f))[3:])
    station_ids = sorted(x for x in station_ids if x)
    station_index = {station_id: i for i, station_id in enumerate(station_ids)}
    total_hours = ((END - START).days + 1) * 24
    values = np.full((total_hours, len(station_ids)), np.nan, dtype="f4")

    day = START
    while day <= END:
        path = RAW_DIR / f"china_sites_{day:%Y%m%d}.csv.gz"
        if path.exists():
            with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as f:
                reader = csv.reader(f)
                header = next(reader)
                columns = [(i, station_index[name]) for i, name in enumerate(header) if name in station_index]
                for row in reader:
                    if len(row) < 3 or row[2] != "PM2.5":
                        continue
                    try:
                        hour = int(row[1])
                    except ValueError:
                        continue
                    time_index = (day - START).days * 24 + hour
                    for source, target in columns:
                        if source < len(row) and row[source]:
                            try:
                                values[time_index, target] = float(row[source])
                            except ValueError:
                                pass
        day += timedelta(days=1)

    missing = np.isnan(values)
    gaps = np.asarray([longest_gap(missing[:, i]) for i in range(len(station_ids))])
    retained = np.flatnonzero(gaps < 168)
    times = [
        (datetime.combine(START, datetime.min.time()) + timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M")
        for i in range(total_hours)
    ]

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir()
    for number, index in enumerate(retained, 1):
        pd.DataFrame({"time": times, "pm25": values[:, index]}).to_csv(
            OUTPUT_DIR / f"{station_ids[index]}.csv", index=False, na_rep="", float_format="%.3f"
        )
        if number % 50 == 0:
            print(number, "/", len(retained), flush=True)

    rows = None
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            with RAW_STATIONS.open("r", encoding=encoding, newline="") as f:
                rows = list(csv.DictReader(f))
            break
        except UnicodeDecodeError:
            pass
    metadata = {row["Code"]: row for row in rows}
    with STATIONS_OUTPUT.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "station_id", "station_name", "city", "longitude", "latitude",
        ])
        for index in retained:
            station_id = station_ids[index]
            row = metadata[station_id]
            writer.writerow([
                station_id, row["Station_name_Chinese"], row["City_Chinese"],
                row["Longitude"], row["Latitude"],
            ])

    shutil.rmtree(WORK_DIR)
    print("Completed:", len(retained), "stations")


if __name__ == "__main__":
    main()
