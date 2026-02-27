"""
Training Data Generation
Builds TFT training dataset from database sources.
Single source of truth for feature engineering — shared with inference pipeline.

Feature set: 30 features selected via Ridge regression walk-forward CV
in notebooks/02_feature_engineering.ipynb (leakage-fixed).
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

# Validate required environment variables
required_vars = ['DB_USER', 'DB_PASSWORD', 'DB_HOST', 'DB_PORT', 'DB_NAME']
missing = [var for var in required_vars if not os.getenv(var)]
if missing:
    print(f"ERROR: Missing environment variables: {', '.join(missing)}")
    sys.exit(1)

# Database connection
engine = create_engine(
    f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@"
    f"{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)

# Output directory
BASE_DIR = Path(__file__).parent.parent.parent.parent  # tftproj root
OUTPUT_DIR = BASE_DIR / 'training'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Constants
BALANCE_POINT = 18   # °C — conventional heating degree-day threshold
SCALE_FACTOR = 10_000  # normalisation constant for demand lags


def get_version_number() -> int:
    """Determine next version number."""
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
    """Load raw data from database tables."""
    demand_query = """
        SELECT date_time AS timestamp,
               "actual_demand(MW)" AS demand
        FROM energy_demand
        ORDER BY date_time
    """
    demand = pd.read_sql(text(demand_query), engine)
    demand['timestamp'] = pd.to_datetime(demand['timestamp'], utc=True).dt.tz_localize(None)

    weather_query = """
        SELECT date_time AS timestamp,
               "temperature_2m(°C)"      AS temperature,
               "relative_humidity_2m(%)"  AS humidity,
               "rain(mm)"                AS rain,
               "snowfall(cm)"            AS snowfall
        FROM weather
        ORDER BY date_time
    """
    weather = pd.read_sql(text(weather_query), engine)
    weather['timestamp'] = pd.to_datetime(weather['timestamp'], utc=True).dt.tz_localize(None)

    holidays_query = """
        SELECT date, is_public_holiday
        FROM holidays
        WHERE is_public_holiday = true
        ORDER BY date
    """
    holidays = pd.read_sql(text(holidays_query), engine)
    holidays['date'] = pd.to_datetime(holidays['date'])

    # Merge
    df = demand.merge(weather, on='timestamp', how='inner')
    df['date'] = df['timestamp'].dt.date.astype('datetime64[ns]')
    df['is_public_holiday'] = df['date'].isin(holidays['date']).astype(float)
    df = df.drop(columns=['date'])

    return df.sort_values('timestamp').reset_index(drop=True)


# -- Feature engineering (shared with prepare_data.py) -------------------------

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-based features."""
    ts = df['timestamp']
    hour = ts.dt.hour
    dow = ts.dt.dayofweek
    month = ts.dt.month

    # Cyclical hour
    df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24)

    # Day-of-week dummies (replace cyclical dow_sin/cos — dummies better in Ridge)
    for i in range(7):
        df[f'dow_{i}'] = (dow == i).astype(float)

    # Cyclical month
    df['month_sin'] = np.sin(2 * np.pi * (month - 1) / 12)
    df['month_cos'] = np.cos(2 * np.pi * (month - 1) / 12)

    # Calendar flags
    df['is_public_holiday'] = df['is_public_holiday'].astype(float)
    df['is_monday_after_weekend'] = (dow == 0).astype(float)
    df['is_friday_before_weekend'] = ((dow == 4) & (hour >= 12)).astype(float)
    df['day_transition_type'] = 0.0
    df.loc[dow == 6, 'day_transition_type'] = 1.0   # Sunday→Monday
    df.loc[dow == 4, 'day_transition_type'] = 2.0   # Friday→Saturday
    df['is_weekend'] = (dow >= 5).astype(float)

    # Season (categorical — used by TFT as time_varying_known_categorical)
    df['season'] = month.map({
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Autumn', 10: 'Autumn', 11: 'Autumn',
    })

    return df


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add weather-derived features."""
    temp = df['temperature']
    df['heating_demand'] = np.maximum(BALANCE_POINT - temp, 0)
    df['temp_lag_24h'] = temp.shift(24).ffill().bfill()
    # humidity, rain, snowfall pass through from DB — no transformation needed
    return df


def add_demand_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add demand lag features (normalised by SCALE_FACTOR).

    Leakage-free: all values are strictly from before time t.
    """
    demand = df['target_demand']
    demand_mean = demand.mean()
    demand_std = demand.std()
    dow = df['timestamp'].dt.dayofweek

    # Raw lags
    lag_24h = demand.shift(24).fillna(demand_mean)
    lag_48h = demand.shift(48).fillna(demand_mean)
    lag_168h = demand.shift(168).fillna(demand_mean)

    # Normalised lags
    df['demand_lag_24h_norm'] = lag_24h / SCALE_FACTOR
    df['demand_lag_168h_norm'] = lag_168h / SCALE_FACTOR

    # Delta: rate of change from 48h ago to 24h ago (both historical — no leakage)
    df['demand_delta_24h'] = (lag_24h - lag_48h) / SCALE_FACTOR

    # Rolling std over 7-day window, shifted by 1 to exclude demand(t)
    df['demand_rolling_std_7d'] = (
        demand.shift(1).rolling(168, min_periods=24).std().fillna(demand_std) / SCALE_FACTOR
    )

    # Lag day-type context flags
    df['lag_24h_was_weekend'] = (dow.shift(1).fillna(dow) >= 5).astype(float)
    df['lag_168h_was_weekend'] = df['is_weekend']

    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add interaction features.

    dow_sin is computed as an intermediate for dow_sin_temp but is NOT a model feature.
    """
    df['heating_hour_cos_product'] = df['heating_demand'] * df['hour_cos']
    df['weekend_temp_interaction'] = df['is_weekend'] * df['temperature']

    # dow_sin needed as intermediate only
    dow = df['timestamp'].dt.dayofweek
    dow_sin = np.sin(2 * np.pi * dow / 7)
    df['dow_sin_temp'] = dow_sin * df['temperature']

    return df


# -- Canonical feature list (used by training AND inference) -------------------

TRAINING_FEATURES = [
    # Hourly cycle
    'hour_sin', 'hour_cos',
    # Weekly cycle — dummies (replace cyclical dow_sin/cos)
    'dow_0', 'dow_1', 'dow_2', 'dow_3', 'dow_4', 'dow_5', 'dow_6',
    # Annual cycle
    'month_sin', 'month_cos',
    # Calendar
    'is_public_holiday',
    'is_monday_after_weekend', 'is_friday_before_weekend',
    'day_transition_type', 'is_weekend',
    # Weather
    'heating_demand', 'temp_lag_24h',
    'humidity', 'rain', 'snowfall',
    # Demand lags (leakage-free)
    'demand_lag_24h_norm', 'demand_lag_168h_norm',
    'demand_delta_24h', 'demand_rolling_std_7d',
    # Lag day-type context
    'lag_24h_was_weekend', 'lag_168h_was_weekend',
    # Interactions
    'heating_hour_cos_product', 'weekend_temp_interaction', 'dow_sin_temp',
]


def select_final_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Select columns for training output."""
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

    version = get_version_number()
    output_path = OUTPUT_DIR / f'tft_training_data-v{version}.csv'
    df.to_csv(output_path, index=False)

    print(f"\nSaved: {output_path}")
    print(f"  Rows: {len(df):,}")
    print(f"  Cols: {len(df.columns)} ({len(TRAINING_FEATURES)} features + timestamp + target + season)")
    print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")


if __name__ == '__main__':
    main()
