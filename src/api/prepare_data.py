"""
Dynamic Feature Engineering Pipeline
Generates features for inference — uses the exact same logic as
complete_data.py to avoid training/serving skew.

Feature set: 22 features (v3) aligned with training/train.py ALL_FEATURES.
"""

import os
import numpy as np
import pandas as pd
from datetime import timedelta
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

BALANCE_POINT = 18        # °C — must match complete_data.py
SCALE_FACTOR = 10_000     # must match complete_data.py


def get_db_engine():
    """Create PostgreSQL connection."""
    required = ['DB_USER', 'DB_PASSWORD', 'DB_HOST', 'DB_PORT', 'DB_NAME']
    if missing := [v for v in required if not os.getenv(v)]:
        raise ValueError(f"Missing environment variables: {', '.join(missing)}")
    return create_engine(
        f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@"
        f"{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )


def load_historical_data(engine, end_date: str | None = None, limit: int = 1000) -> pd.DataFrame:
    """Load raw historical data from database.

    Includes humidity, rain, snowfall — required by the 22-feature set.
    """
    date_filter = ""
    if end_date:
        for fmt in ['%d/%m/%Y', '%d-%m-%Y', '%d/%m/%Y %H:%M:%S', '%d-%m-%Y %H:%M:%S']:
            try:
                target = pd.to_datetime(end_date, format=fmt)
                date_filter = f"WHERE ed.date_time <= '{target}'"
                break
            except ValueError:
                continue
        else:
            raise ValueError(f"Invalid date: {end_date}. Use DD/MM/YYYY")

    query = f"""
        SELECT
            ed.date_time                    AS timestamp,
            ed."actual_demand(MW)"          AS target_demand,
            COALESCE(h.is_public_holiday, false) AS is_public_holiday,
            w."temperature_2m(°C)"          AS temperature,
            w."relative_humidity_2m(%)"     AS humidity,
            w."rain(mm)"                    AS rain,
            w."snowfall(cm)"               AS snowfall
        FROM energy_demand ed
        INNER JOIN weather w ON ed.date_time = w.date_time
        LEFT  JOIN holidays h ON DATE(ed.date_time) = h.date
        {date_filter}
        ORDER BY ed.date_time DESC
        LIMIT {limit}
    """
    df = pd.read_sql(text(query), engine)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_convert(None)
    return df.sort_values('timestamp').reset_index(drop=True)


# -- Feature engineering (mirrors complete_data.py exactly) --------------------

def _add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-based features — identical to complete_data.add_temporal_features."""
    ts = df['timestamp']
    hour = ts.dt.hour
    dow = ts.dt.dayofweek
    month = ts.dt.month

    # Cyclical hour
    df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24)

    # Day-of-week dummies
    for i in range(7):
        df[f'dow_{i}'] = (dow == i).astype(float)

    # Cyclical month
    df['month_sin'] = np.sin(2 * np.pi * (month - 1) / 12)
    df['month_cos'] = np.cos(2 * np.pi * (month - 1) / 12)

    # Calendar flags
    df['is_public_holiday'] = df['is_public_holiday'].astype(float)
    df['is_weekend'] = (dow >= 5).astype(float)

    # Season (categorical — used by TFT)
    df['season'] = month.map({
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Autumn', 10: 'Autumn', 11: 'Autumn',
    })

    return df


def _add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add weather-derived features — identical to complete_data.add_weather_features."""
    temp = df['temperature']
    df['heating_demand'] = np.maximum(BALANCE_POINT - temp, 0)
    df['temp_lag_24h'] = temp.shift(24).ffill().bfill()
    # humidity, rain, snowfall already in df from DB query
    return df


def _add_demand_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add demand lag features (leakage-free) — identical to complete_data.add_demand_lag_features."""
    demand = df['target_demand']
    demand_mean = demand.mean()
    demand_std = demand.std()

    lag_168h = demand.shift(168).fillna(demand_mean)

    # Weekly lag (only lag kept — 24h removed as redundant with encoder sequence)
    df['demand_lag_168h_norm'] = lag_168h / SCALE_FACTOR

    # Rolling volatility: std over 7-day window shifted by 1 to exclude demand(t)
    df['demand_rolling_std_7d'] = (
        demand.shift(1).rolling(168, min_periods=24).std().fillna(demand_std) / SCALE_FACTOR
    )

    return df


def _add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add interaction features — identical to complete_data.add_interaction_features."""
    df['heating_hour_cos_product'] = df['heating_demand'] * df['hour_cos']
    df['weekend_temp_interaction'] = df['is_weekend'] * df['temperature']
    return df


def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate all features from raw data (for historical portion)."""
    df = df.copy()
    df = _add_temporal_features(df)
    df = _add_weather_features(df)
    df = _add_demand_lag_features(df)
    df = _add_interaction_features(df)
    return df


def generate_future_features(historical_df: pd.DataFrame, hours: int) -> pd.DataFrame:
    """Generate features for future prediction hours.

    Weather values (temperature, humidity, rain, snowfall) are persisted
    from the last known observation — this is the best we can do without
    an external weather forecast API.
    """
    last_time = historical_df['timestamp'].iloc[-1]
    last_temp = float(historical_df['temperature'].iloc[-1])
    last_humid = float(historical_df['humidity'].iloc[-1])
    last_rain = float(historical_df['rain'].iloc[-1])
    last_snow = float(historical_df['snowfall'].iloc[-1])
    demand_mean = float(historical_df['target_demand'].mean())
    demand_std = float(historical_df['target_demand'].std())

    rows: list[dict] = []
    for h in range(1, hours + 1):
        ts = last_time + timedelta(hours=h)
        hour = ts.hour
        dow = ts.weekday()
        month = ts.month

        # ---- Demand lags from historical data ----
        lag_168h_idx = len(historical_df) - 168 + h - 1

        lag_168h = (
            float(historical_df.iloc[lag_168h_idx]['target_demand'])
            if 0 <= lag_168h_idx < len(historical_df)
            else demand_mean
        )

        heating = max(BALANCE_POINT - last_temp, 0.0)
        hour_cos = float(np.cos(2 * np.pi * hour / 24))

        row: dict = {
            'timestamp': ts,
            'target_demand': demand_mean,
            'temperature': last_temp,
            'humidity': last_humid,
            'rain': last_rain,
            'snowfall': last_snow,
            'is_public_holiday': 0.0,
            # Cyclical hour
            'hour_sin': float(np.sin(2 * np.pi * hour / 24)),
            'hour_cos': hour_cos,
            # Cyclical month
            'month_sin': float(np.sin(2 * np.pi * (month - 1) / 12)),
            'month_cos': float(np.cos(2 * np.pi * (month - 1) / 12)),
            # Calendar
            'is_weekend': float(dow >= 5),
            # Weather derived
            'heating_demand': heating,
            'temp_lag_24h': last_temp,
            # Demand lags (leakage-free)
            'demand_lag_168h_norm': lag_168h / SCALE_FACTOR,
            'demand_rolling_std_7d': demand_std / SCALE_FACTOR,
            # Interactions
            'heating_hour_cos_product': heating * hour_cos,
            'weekend_temp_interaction': float(dow >= 5) * last_temp,
        }

        # Day-of-week dummies
        for i in range(7):
            row[f'dow_{i}'] = float(dow == i)

        # Season
        if month in (12, 1, 2):
            row['season'] = 'Winter'
        elif month in (3, 4, 5):
            row['season'] = 'Spring'
        elif month in (6, 7, 8):
            row['season'] = 'Summer'
        else:
            row['season'] = 'Autumn'

        rows.append(row)

    return pd.DataFrame(rows)


def ensure_model_features(df: pd.DataFrame, model_features: list) -> pd.DataFrame:
    """Ensure dataframe has all features required by model."""
    for feat in model_features:
        if feat not in df.columns:
            df[feat] = 0.0
    return df
