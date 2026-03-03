"""
Training Data Generation — v2
Builds TFT training dataset from database sources.
Single source of truth for feature engineering — shared with inference pipeline.

Feature set: 22 features (reduced from 30 in v1)
Removed features and rationale:
  demand_lag_24h_norm      TFT encoder already contains t-24 as sequence position.
                           Explicit scalar was a shortcut for capacity-constrained hidden_size=32.
  demand_delta_24h         Derived from lag_24h minus lag_48h — both visible in encoder sequence.
  is_monday_after_weekend  Redundant with dow_0.
  is_friday_before_weekend Redundant with dow_4.
  lag_24h_was_weekend      Model derives this from sequence + dow dummies.
  lag_168h_was_weekend     Redundant with is_weekend.
  dow_sin_temp             Ranked last (32/32) in encoder variable importance.
  day_transition_type      Fully captured by is_weekend + dow dummies.
"""

import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

required_vars = ['DB_USER', 'DB_PASSWORD', 'DB_HOST', 'DB_PORT', 'DB_NAME']
missing = [var for var in required_vars if not os.getenv(var)]
if missing:
    print(f"ERROR: Missing environment variables: {', '.join(missing)}")
    sys.exit(1)

engine = create_engine(
    f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@"
    f"{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)

BASE_DIR   = Path(__file__).parent.parent.parent.parent
OUTPUT_DIR = BASE_DIR / 'training'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BALANCE_POINT = 18      # °C — heating degree-day threshold
SCALE_FACTOR  = 10_000  # normalisation constant for demand lags


def get_version_number() -> int:
    existing = list(OUTPUT_DIR.glob('tft_training_data-v*.csv'))
    if not existing:
        return 1
    versions = []
    for f in existing:
        match = re.search(r'-v(\d+)\.csv$', f.name)
        if match:
            versions.append(int(match.group(1)))
    return max(versions, default=0) + 1


def load_from_database() -> pd.DataFrame:
    demand = pd.read_sql(text("""
        SELECT date_time AS timestamp,
               "actual_demand(MW)" AS demand
        FROM energy_demand
        ORDER BY date_time
    """), engine)
    demand['timestamp'] = pd.to_datetime(demand['timestamp'], utc=True).dt.tz_localize(None)

    weather = pd.read_sql(text("""
        SELECT date_time AS timestamp,
               "temperature_2m(°C)"       AS temperature,
               "relative_humidity_2m(%)"  AS humidity,
               "rain(mm)"                 AS rain,
               "snowfall(cm)"             AS snowfall
        FROM weather
        ORDER BY date_time
    """), engine)
    weather['timestamp'] = pd.to_datetime(weather['timestamp'], utc=True).dt.tz_localize(None)

    holidays = pd.read_sql(text("""
        SELECT date, is_public_holiday
        FROM holidays
        WHERE is_public_holiday = true
        ORDER BY date
    """), engine)
    holidays['date'] = pd.to_datetime(holidays['date'])

    df = demand.merge(weather, on='timestamp', how='inner')
    df['date'] = df['timestamp'].dt.date.astype('datetime64[ns]')
    df['is_public_holiday'] = df['date'].isin(holidays['date']).astype(float)
    df = df.drop(columns=['date'])

    return df.sort_values('timestamp').reset_index(drop=True)


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    ts    = df['timestamp']
    hour  = ts.dt.hour
    dow   = ts.dt.dayofweek
    month = ts.dt.month

    df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24)

    for i in range(7):
        df[f'dow_{i}'] = (dow == i).astype(float)

    df['month_sin'] = np.sin(2 * np.pi * (month - 1) / 12)
    df['month_cos'] = np.cos(2 * np.pi * (month - 1) / 12)

    df['is_public_holiday'] = df['is_public_holiday'].astype(float)
    df['is_weekend']        = (dow >= 5).astype(float)

    df['season'] = month.map({
        12: 'Winter', 1: 'Winter',  2: 'Winter',
        3:  'Spring', 4: 'Spring',  5: 'Spring',
        6:  'Summer', 7: 'Summer',  8: 'Summer',
        9:  'Autumn', 10: 'Autumn', 11: 'Autumn',
    })

    return df


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    temp = df['temperature']
    df['heating_demand'] = np.maximum(BALANCE_POINT - temp, 0)
    df['temp_lag_24h']   = temp.shift(24).ffill().bfill()
    # humidity, rain, snowfall pass through from DB unchanged
    return df


def add_demand_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add demand lag features (leakage-free — all values from strictly before t)."""
    demand      = df['target_demand']
    demand_mean = demand.mean()
    demand_std  = demand.std()

    lag_168h = demand.shift(168).fillna(demand_mean)

    # Weekly lag (only lag kept — 24h removed as redundant with encoder sequence)
    df['demand_lag_168h_norm'] = lag_168h / SCALE_FACTOR

    # Rolling volatility: std over 7-day window shifted by 1 to exclude demand(t)
    df['demand_rolling_std_7d'] = (
        demand.shift(1).rolling(168, min_periods=24).std().fillna(demand_std) / SCALE_FACTOR
    )

    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add interaction features."""
    df['heating_hour_cos_product'] = df['heating_demand'] * df['hour_cos']
    df['weekend_temp_interaction'] = df['is_weekend']     * df['temperature']
    return df


# ── Canonical 22-feature list (shared with train.py and inference pipeline) ──
TRAINING_FEATURES = [
    # Hourly cycle
    'hour_sin', 'hour_cos',
    # Weekly cycle — dummies
    'dow_0', 'dow_1', 'dow_2', 'dow_3', 'dow_4', 'dow_5', 'dow_6',
    # Annual cycle
    'month_sin', 'month_cos',
    # Calendar
    'is_public_holiday', 'is_weekend',
    # Weather
    'heating_demand', 'temp_lag_24h',
    'humidity', 'rain', 'snowfall',
    # Demand lags
    'demand_lag_168h_norm', 'demand_rolling_std_7d',
    # Interactions
    'heating_hour_cos_product', 'weekend_temp_interaction',
]


def select_final_columns(df: pd.DataFrame) -> pd.DataFrame:
    columns = ['timestamp', 'target_demand', 'season'] + TRAINING_FEATURES
    return df[columns]


def main():
    print("Loading data...")
    df = load_from_database()
    print(f"  Loaded {len(df):,} records")

    print("Adding features...")
    df['target_demand'] = df['demand']
    df = add_temporal_features(df)
    df = add_weather_features(df)
    df = add_demand_lag_features(df)
    df = add_interaction_features(df)

    df = select_final_columns(df)
    df = df.dropna()

    version     = get_version_number()
    output_path = OUTPUT_DIR / f'tft_training_data-v{version}.csv'
    df.to_csv(output_path, index=False)

    print(f"\nSaved: {output_path}")
    print(f"  Rows     : {len(df):,}")
    print(f"  Cols     : {len(df.columns)} ({len(TRAINING_FEATURES)} features + timestamp + target + season)")
    print(f"  Features : {len(TRAINING_FEATURES)} (v1 had 30, removed 8 redundant)")
    print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")


if __name__ == '__main__':
    main()