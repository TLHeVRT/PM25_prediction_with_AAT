"""Download raw daily PM2.5 files and station coordinates for filter_pm25.py."""

import gzip
import hashlib
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path


START = date(2017, 10, 1)
END = date(2022, 10, 1)
WORK_DIR = Path(".pm25_work")
RAW_DIR = WORK_DIR / "raw"
STATION_FILE = WORK_DIR / "stations.csv"
PM25_URL = "https://quotsoft.net/air/data/china_sites_{day}.csv"
STATION_URL = "https://zenodo.org/records/10911197/files/Station_information.csv?download=1"
STATION_MD5 = "f239df010925c4bb9e2419c20885cdc6"


def download_station_file():
    if STATION_FILE.exists() and hashlib.md5(STATION_FILE.read_bytes()).hexdigest() == STATION_MD5:
        return
    part = STATION_FILE.with_suffix(".part")
    urllib.request.urlretrieve(STATION_URL, part)
    if hashlib.md5(part.read_bytes()).hexdigest() != STATION_MD5:
        raise RuntimeError("Station coordinate file checksum verification failed")
    os.replace(part, STATION_FILE)


def download_day(day):
    text = day.strftime("%Y%m%d")
    output = RAW_DIR / f"china_sites_{text}.csv.gz"
    if output.exists():
        return text, "Existing"
    for attempt in range(5):
        try:
            request = urllib.request.Request(PM25_URL.format(day=text), headers={"User-Agent": "PM25-downloader/1.0"})
            with urllib.request.urlopen(request, timeout=90) as response:
                data = response.read()
            part = output.with_suffix(".part")
            with part.open("wb") as raw:
                with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as zipped:
                    zipped.write(data)
            os.replace(part, output)
            return text, "Downloaded"
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return text, "Missing from source"
        except Exception:
            pass
        time.sleep(2 ** attempt)
    return text, "Failed"


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    download_station_file()
    days = [START + timedelta(days=i) for i in range((END - START).days + 1)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(download_day, day) for day in days]
        for number, future in enumerate(as_completed(futures), 1):
            day, status = future.result()
            if number % 50 == 0 or status in {"Missing from source", "Failed"}:
                print(number, "/", len(days), day, status, flush=True)


if __name__ == "__main__":
    main()
