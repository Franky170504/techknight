import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

# -----------------------------
# CONFIG
# -----------------------------
CSV_FILE = "BORIVALI_DWLR_REALTIME_multi.csv"
# We will use these values to create rows for each station
STATIONS = {1: "Station_01", 2: "Station_02", 3: "Station_03", 4: "Station_04", 
            5: "Station_05"}

LAT, LON = 19.23, 72.85
AQUIFER_AREA = 2.0         
AQUIFER_AREA_M2 = AQUIFER_AREA * 1_000_000
SPECIFIC_YIELD = 0.15
LAND_USE = "Urban"

np.random.seed(42)

# -----------------------------
# Helpers (Keeping your existing logic)
# -----------------------------
def current_hour():
    return datetime.now().replace(minute=0, second=0, microsecond=0)

def season(month):
    if month in [6, 7, 8, 9]: return "monsoon"
    if month in [10, 11]: return "post_monsoon"
    if month in [12, 1, 2]: return "winter"
    return "summer"

def generate_rainfall(month):
    if season(month) == "monsoon":
        return np.random.uniform(5, 50) if np.random.rand() < 0.12 else np.random.uniform(0, 2)
    return np.random.uniform(0, 3) if season(month) == "post_monsoon" else np.random.uniform(0, 0.5)

def generate_temperature(ts):
    hour = ts.hour
    diurnal = 2.5 * np.sin((hour - 6) / 24 * 2 * np.pi)
    base = {"summer": 33, "monsoon": 29, "post_monsoon": 31, "winter": 26}[season(ts.month)]
    return base + diurnal + np.random.normal(0, 0.6)

def generate_demand(month):
    bases = {"summer": (0.004, 0.006), "monsoon": (0.002, 0.003), 
             "post_monsoon": (0.003, 0.004), "winter": (0.0025, 0.0035)}
    low, high = bases[season(month)]
    return np.random.uniform(low, high)

def get_status(level):
    if level < 5:
        return 'Safe'
    elif 5 < level < 10:
        return 'Semi-Critical'
    else:
        return 'Critical'


# -----------------------------
# Load or initialize dataset
# -----------------------------
file_exists = Path(CSV_FILE).exists()

if file_exists:
    df = pd.read_csv(CSV_FILE, parse_dates=["timestamp"])
    last_time = df["timestamp"].max()
    # Get the last known level for each station
    last_levels = df[df["timestamp"] == last_time].set_index("station_id")["water_level_m"].to_dict()
else:
    df = pd.DataFrame()
    last_time = current_hour() - timedelta(days=365 * 3)
    # Initialize all stations at 9.0 meters
    last_levels = {sid: 9.0 for sid in STATIONS.values()}

# -----------------------------
# Generate missing hourly rows for ALL stations
# -----------------------------
rows = []
now = current_hour()
t = last_time + timedelta(hours=1)

while t <= now:
    # Weather is the same for the whole area that hour
    rain = generate_rainfall(t.month)
    temp = generate_temperature(t)

    # Generate data for each station
    for s_name in STATIONS.values():
        demand = generate_demand(t.month)
        
        # Seasonal recharge effect - much stronger during monsoon
        season_name = season(t.month)
        if season_name == "monsoon":
            recharge_effect = rain * 0.08  # Much stronger recharge during monsoon
        elif season_name == "post_monsoon":
            recharge_effect = rain * 0.03  # Moderate recharge after monsoon
        else:
            recharge_effect = rain * 0.005  # Minimal recharge in dry seasons

        # Seasonal extraction effect - slightly less aggressive in wet seasons
        if season_name == "monsoon":
            extraction_effect = demand * 1.2  # Reduced extraction during monsoon
        else:
            extraction_effect = demand * 1.8  # Normal extraction in dry seasons

        # Calculate new level based on that specific station's previous level
        new_level = (
            last_levels[s_name]
            - recharge_effect
            + extraction_effect
            + np.random.normal(0, 0.015) # Random sensor noise
        )
        new_level = float(np.clip(new_level, 2.0, 15.0))
        last_levels[s_name] = new_level # Update state for next hour

        rows.append({
            "timestamp": t,
            "month": t.month,
            "station_id": s_name,
            "latitude": LAT + np.random.uniform(-0.01, 0.01), # Slight offset for realism
            "longitude": LON + np.random.uniform(-0.01, 0.01),
            "water_level_m": new_level,
            "rainfall_mm": rain,
            "demand_mcm": demand,
            "aquifer_area_sqkm": AQUIFER_AREA,
            "specific_yield": SPECIFIC_YIELD,
            "temperature_c": temp,
            "land_use_type": LAND_USE,
            "policy_active": t >= pd.Timestamp("2024-07-01")
            # "severity" : 
        })
    t += timedelta(hours=1)

# Append new rows
if rows:
    new_data = pd.DataFrame(rows)
    df = pd.concat([df, new_data], ignore_index=True)

# -----------------------------
# 🔹 RECHARGE & AVAILABILITY (Calculated per Station)
# -----------------------------
df = df.sort_values(["station_id", "timestamp"], ascending=[True, True])
# Use groupby to ensure delta_h is calculated within each station's timeline
df["delta_h"] = df.groupby("station_id")["water_level_m"].shift(1) - df["water_level_m"]
df["delta_h"] = df["delta_h"].fillna(0.0)

# Seasonal direct recharge component
def get_seasonal_recharge_factor(month):
    season_name = season(month)
    if season_name == "monsoon":
        return 2.5  # High recharge factor during monsoon
    elif season_name == "post_monsoon":
        return 1.2  # Moderate recharge after monsoon
    elif season_name == "winter":
        return 0.8  # Low recharge in winter
    else:  # summer
        return 0.3  # Very low recharge in summer

df["seasonal_recharge_factor"] = df["month"].apply(get_seasonal_recharge_factor)

# Combined recharge: delta_h based + seasonal rainfall-based
df["recharge"] = (
    SPECIFIC_YIELD *
    df["delta_h"].clip(lower=0) *
    AQUIFER_AREA_M2 *
    df["seasonal_recharge_factor"]  # Seasonal multiplier
)

# Availability (m³)
df["availability"] = df["recharge"] - (df["demand_mcm"] * 1_000_000)
df['status'] = df['water_level_m'].apply(get_status)

# Save and Sort back to chronological order for the CSV
df = df.sort_values("timestamp")
df.to_csv(CSV_FILE, index=False)

<<<<<<< HEAD
print(f"DWLR data updated for {len(STATIONS)} stations to {now}")
=======
print(f"DWLR data updated for 10 stations to {now}")
>>>>>>> 44f8e800da1efc841f87bdafc4860dc5a7e84b83
print(f"Total rows: {len(df)}")
print(df[["timestamp", "station_id", "water_level_m", "recharge"]].tail(10))