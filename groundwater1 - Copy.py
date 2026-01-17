import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

# -----------------------------
# CONFIG
# -----------------------------
CSV_FILE = "BORIVALI_DWLR_REALTIME.csv"
STATION_ID = "BOR-01"

LAT, LON = 19.23, 72.85
AQUIFER_AREA = 2.0          # sqkm
SPECIFIC_YIELD = 0.15
LAND_USE = "Urban"

np.random.seed(42)

# -----------------------------
# Time helpers
# -----------------------------
def current_hour():
    return datetime.now().replace(minute=0, second=0, microsecond=0)

def season(month):
    if month in [6, 7, 8, 9]:
        return "monsoon"
    if month in [10, 11]:
        return "post_monsoon"
    if month in [12, 1, 2]:
        return "winter"
    return "summer"

# -----------------------------
# Climate generators
# -----------------------------
def generate_rainfall(month):
    if season(month) == "monsoon":
        if np.random.rand() < 0.12:
            return np.random.uniform(5, 50)
        return np.random.uniform(0, 2)
    if season(month) == "post_monsoon":
        return np.random.uniform(0, 3)
    return np.random.uniform(0, 0.5)

def generate_temperature(ts):
    hour = ts.hour
    diurnal = 2.5 * np.sin((hour - 6) / 24 * 2 * np.pi)

    base = {
        "summer": 33,
        "monsoon": 29,
        "post_monsoon": 31,
        "winter": 26
    }[season(ts.month)]

    return base + diurnal + np.random.normal(0, 0.6)

def generate_demand(month):
    if season(month) == "summer":
        return np.random.uniform(0.004, 0.006)
    if season(month) == "monsoon":
        return np.random.uniform(0.002, 0.003)
    if season(month) == "post_monsoon":
        return np.random.uniform(0.003, 0.004)
    return np.random.uniform(0.0025, 0.0035)

# -----------------------------
# Load or initialize dataset
# -----------------------------
file_exists = Path(CSV_FILE).exists()

if file_exists:
    df = pd.read_csv(CSV_FILE, parse_dates=["timestamp"])
    last_time = df["timestamp"].iloc[-1]
    last_level = df["water_level_m"].iloc[-1]
else:
    df = pd.DataFrame()
    last_time = current_hour() - timedelta(days=365 * 3)
    last_level = 9.0

# -----------------------------
# Generate missing hourly rows
# -----------------------------
rows = []
now = current_hour()
t = last_time + timedelta(hours=1)

while t <= now:
    rain = generate_rainfall(t.month)
    temp = generate_temperature(t)
    demand = generate_demand(t.month)

    # Recharge (lagged & smoothed)
    recharge_effect = rain * 0.015
    extraction_effect = demand * 1.8

    new_level = (
        last_level
        - recharge_effect
        + extraction_effect
        + np.random.normal(0, 0.015)
    )

    new_level = float(np.clip(new_level, 2.0, 15.0))

    rows.append({
        "timestamp": t,
        "month": t.month,
        "station_id": STATION_ID,
        "latitude": LAT,
        "longitude": LON,
        "water_level_m": new_level,
        "rainfall_mm": rain,
        "demand_mcm": demand,
        "aquifer_area_sqkm": AQUIFER_AREA,
        "specific_yield": SPECIFIC_YIELD,
        "temperature_c": temp,
        "land_use_type": LAND_USE,
        "policy_active": t >= pd.Timestamp("2024-07-01")
    })

    last_level = new_level
    t += timedelta(hours=1)

# -----------------------------
# Append & save
# -----------------------------
if rows:
    df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    df.to_csv(CSV_FILE, index=False)

print(f"✅ DWLR data updated to {now}")
print(f"Total rows: {len(df)}")
