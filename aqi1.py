import requests
import pandas as pd
from datetime import datetime, timedelta

# ---------------- CONFIG ----------------
API_KEY = "4d21b51dc6022fac4ec7cddebfd80bd8385fd75670b41a0ea719785af3e1b484"
HEADERS = {"X-API-Key": API_KEY}

CSV_FILE = "delhi_30day_pollution_with_sources.csv"

# 🔒 FIXED DELHI STATION (historical CPCB-backed)
LOCATION_ID = 3  # Example: Anand Vihar (historical works well)

TARGET_POLLUTANTS = ["pm25", "pm10", "no2", "so2", "co"]

SOURCE_MAP = {
    "pm25": "Crop burning / Dust / Combustion",
    "pm10": "Construction / Road dust",
    "no2": "Vehicle emissions",
    "so2": "Industrial activity",
    "co": "Traffic + Biomass burning"
}
# ---------------------------------------


def fetch_sensor_hours(sensor_id, pollutant, start_iso, end_iso):
    url = f"https://api.openaq.org/v3/sensors/{sensor_id}/hours"
    params = {
        "date_from": start_iso,
        "date_to": end_iso,
        "limit": 1000
    }

    r = requests.get(url, headers=HEADERS, params=params, timeout=15)
    r.raise_for_status()

    return [
        {
            "timestamp": row["period"]["datetimeTo"]["local"],
            pollutant: row["value"]
        }
        for row in r.json().get("results", [])
    ]


def get_sensors_for_location(location_id):
    url = f"https://api.openaq.org/v3/locations/{location_id}/sensors"
    r = requests.get(url, headers=HEADERS, timeout=15)
    r.raise_for_status()
    return r.json().get("results", [])


def classify_sources(df):
    def dominant(row):
        vals = {p: row[p] for p in TARGET_POLLUTANTS if pd.notna(row[p])}
        return max(vals, key=vals.get) if vals else None

    df["dominant_pollutant"] = df.apply(dominant, axis=1)
    df["likely_source"] = df["dominant_pollutant"].map(SOURCE_MAP)
    return df


def main():
    print("📥 Fetching last 30 days historical Delhi AQ data...")

    end_dt = datetime.utcnow()
    start_dt = end_dt - timedelta(days=30)

    sensors = get_sensors_for_location(LOCATION_ID)

    sensor_map = {
        s["parameter"]["name"]: s["id"]
        for s in sensors
        if s["parameter"]["name"] in TARGET_POLLUTANTS
    }

    master_df = None

    for pollutant, sensor_id in sensor_map.items():
        print(f"  Downloading {pollutant.upper()}...")
        records = fetch_sensor_hours(
            sensor_id,
            pollutant,
            start_dt.isoformat(),
            end_dt.isoformat()
        )

        if records:
            df = pd.DataFrame(records)
            master_df = df if master_df is None else pd.merge(
                master_df, df, on="timestamp", how="outer"
            )

    master_df["timestamp"] = pd.to_datetime(master_df["timestamp"])
    master_df = master_df.sort_values("timestamp").ffill().bfill()

    master_df = classify_sources(master_df)

    master_df.to_csv(CSV_FILE, index=False)

    print(f"✅ Saved 30-day dataset → {CSV_FILE}")
    print(f"   Rows: {len(master_df)}")


if __name__ == "__main__":
    main()
