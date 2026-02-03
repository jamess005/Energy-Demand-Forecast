"""
Training Data Generation
Builds TFT training dataset from database sources.
"""

import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Load environment variables
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

# Output directory - relative to project root
BASE_DIR = Path(__file__).parent.parent.parent.parent  # tftproj root
OUTPUT_DIR = BASE_DIR / 'training'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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
    """Load data from database tables."""
    demand_query = """
        SELECT date_time as timestamp, "actual_demand(MW)" as demand
        FROM energy_demand
        ORDER BY date_time
    """
    demand = pd.read_sql(text(demand_query), engine)
    demand['timestamp'] = pd.to_datetime(demand['timestamp'], utc=True).dt.tz_localize(None)
    
    weather_query = """
        SELECT date_time as timestamp, "temperature_2m(°C)" as temperature_2m,
               "relative_humidity_2m(%)" as relative_humidity_2m,
               "rain(mm)" as rain, "snow_depth(m)" as snow_depth,
               "snowfall(cm)" as snowfall
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
    
    df = demand.merge(weather, on='timestamp', how='inner')
    df['date'] = df['timestamp'].dt.date.astype('datetime64[ns]')
    df['is_public_holiday'] = df['date'].isin(holidays['date']).astype(int)
    df = df.drop(columns=['date'])
    
    return df.sort_values('timestamp').reset_index(drop=True)


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-based features."""
    ts = df['timestamp']
    hour = ts.dt.hour
    dow = ts.dt.dayofweek
    month = ts.dt.month
    
    # Cyclical encodings
    df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    df['dow_sin'] = np.sin(2 * np.pi * dow / 7)
    df['month_cos'] = np.cos(2 * np.pi * month / 12)
    
    # Day of week dummies
    for i in range(7):
        df[f'dow_{i}'] = (dow == i).astype(int)
    
    # Weekend/weekday
    df['is_weekend'] = (dow >= 5).astype(int)
    df['is_weekday'] = (dow < 5).astype(int)
    
    # Transition days
    df['is_monday_after_weekend'] = ((dow == 0) & (hour < 12)).astype(int)
    df['is_friday_before_weekend'] = ((dow == 4) & (hour >= 12)).astype(int)
    
    # Time of day
    df['is_early_morning'] = ((hour >= 5) & (hour < 8)).astype(int)
    df['is_morning_ramp'] = ((hour >= 6) & (hour < 10)).astype(int)
    df['is_night'] = ((hour >= 22) | (hour < 6)).astype(int)
    
    # Hour patterns
    df['is_peak_hour'] = ((hour >= 9) & (hour < 20)).astype(int)
    df['is_valley_hour'] = ((hour >= 0) & (hour < 5)).astype(int)
    df['hour_squared'] = (hour ** 2) / (24 ** 2)
    
    # Season
    df['season'] = month.map({
        12: 'winter', 1: 'winter', 2: 'winter',
        3: 'spring', 4: 'spring', 5: 'spring',
        6: 'summer', 7: 'summer', 8: 'summer',
        9: 'autumn', 10: 'autumn', 11: 'autumn'
    })
    
    # DST
    df['daylight_savings_winter'] = month.isin([11, 12, 1, 2, 3]).astype(int)
    df['daylight_savings_summer'] = month.isin([4, 5, 6, 7, 8, 9, 10]).astype(int)
    
    # Transition type
    transition_type = np.zeros(len(df))
    transition_type[(dow == 6)] = 1  # Sunday to Monday
    transition_type[(dow == 4)] = 2  # Friday to Saturday
    df['day_transition_type'] = transition_type
    
    return df


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add weather-derived features."""
    temp = df['temperature_2m']
    
    df['temperature'] = temp
    df['heating_demand'] = (18 - temp).clip(lower=0)
    df['heating_demand_sq'] = df['heating_demand'] ** 2
    df['temp_severity'] = (temp - temp.mean()).abs() / temp.std()
    df['heating_demand_log'] = np.log1p(df['heating_demand'])
    df['is_cold'] = (temp < 5).astype(int)
    df['temp_lag_24h'] = temp.shift(24).ffill().bfill()
    
    # Monday cold multiplier
    df['monday_cold_multiplier'] = (df['dow_0'] * df['heating_demand']).clip(upper=20)
    
    return df


def add_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add temperature regime classification."""
    temp = df['temperature']
    
    df['regime_0'] = (temp < 0).astype(int)
    df['regime_1'] = ((temp >= 0) & (temp < 10)).astype(int)
    df['regime_2'] = ((temp >= 10) & (temp < 20)).astype(int)
    df['regime_3'] = (temp >= 20).astype(int)
    
    return df


def add_demand_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add demand history features."""
    demand = df['demand']
    dow = df['timestamp'].dt.dayofweek
    
    # Normalize for lag features
    demand_mean = demand.mean()
    demand_std = demand.std()
    demand_norm = (demand - demand_mean) / demand_std
    
    df['demand_lag_24h_norm'] = demand_norm.shift(24).ffill().bfill()
    df['demand_lag_168h_norm'] = demand_norm.shift(168).ffill().bfill()
    df['demand_delta_24h'] = (demand_norm - demand_norm.shift(24)).ffill().bfill()
    df['demand_lag_ratio'] = (df['demand_lag_24h_norm'] / (df['demand_lag_168h_norm'].clip(lower=0.01))).clip(-5, 5)
    
    # Reliability weights
    lag_24h_valid = (~demand.shift(24).isna()).astype(float)
    lag_168h_valid = (~demand.shift(168).isna()).astype(float)
    df['lag_reliability'] = (lag_24h_valid * 0.7 + lag_168h_valid * 0.3).ffill().bfill()
    
    # Adjusted lag with weekend handling
    lag_24h = df['demand_lag_24h_norm']
    lag_168h = df['demand_lag_168h_norm']
    
    df['lag_24h_was_weekend'] = dow.shift(1).isin([5, 6]).astype(int)
    df['lag_168h_was_weekend'] = dow.shift(7).isin([5, 6]).astype(int)
    
    weekend_today = df['is_weekend']
    same_type_24h = (df['lag_24h_was_weekend'] == weekend_today).astype(float)
    
    df['demand_lag_adjusted'] = (same_type_24h * lag_24h + (1 - same_type_24h) * 0.5 * (lag_24h + lag_168h)).ffill().bfill()
    
    # Transition adjustment
    df['transition_adjustment'] = (df['is_monday_after_weekend'] * 0.1 - df['is_friday_before_weekend'] * 0.05)
    
    # Rolling stats
    df['demand_rolling_std_7d'] = (demand.rolling(168, min_periods=24).std().ffill().bfill() / demand_std)
    
    # Additional lag transforms
    df['demand_lag_24h_sq'] = df['demand_lag_24h_norm'] ** 2
    df['demand_lag_168h_sq'] = df['demand_lag_168h_norm'] ** 2
    df['demand_lag_24h_log'] = np.log1p(df['demand_lag_24h_norm'].clip(lower=0))
    
    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add feature interactions."""
    df['peak_lag_interaction'] = df['is_peak_hour'] * df['demand_lag_24h_norm']
    df['peak_heating_interaction'] = df['is_peak_hour'] * df['heating_demand']
    df['weekend_temp_interaction'] = df['is_weekend'] * df['temperature']
    df['heating_hour_cos_product'] = df['heating_demand'] * df['hour_cos']
    df['temp_lag_ratio_interaction'] = df['temperature'] * df['demand_lag_ratio']
    df['night_temp_interaction'] = df['is_night'] * df['temperature']
    df['weekend_transition_temp'] = df['day_transition_type'] * df['temperature'] * df['is_weekend']
    df['dow_sin_temp'] = df['dow_sin'] * df['temperature']
    
    return df


def select_final_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Select columns for training."""
    columns = [
        'timestamp', 'target_demand', 'season',
        'hour_sin', 'hour_cos', 'dow_sin', 'month_cos',
        'dow_0', 'dow_1', 'dow_2', 'dow_3', 'dow_4', 'dow_5', 'dow_6',
        'is_weekend', 'is_weekday', 'is_public_holiday',
        'is_monday_after_weekend', 'is_friday_before_weekend',
        'is_early_morning', 'is_morning_ramp', 'is_night',
        'day_transition_type', 'is_peak_hour', 'is_valley_hour', 'hour_squared',
        'daylight_savings_winter', 'daylight_savings_summer',
        'temperature', 'heating_demand', 'heating_demand_sq',
        'temp_severity', 'heating_demand_log', 'monday_cold_multiplier', 'is_cold',
        'regime_0', 'regime_1', 'regime_2', 'regime_3',
        'demand_lag_24h_norm', 'demand_lag_168h_norm', 'demand_delta_24h',
        'demand_lag_ratio', 'lag_reliability', 'demand_lag_adjusted',
        'transition_adjustment', 'demand_rolling_std_7d',
        'lag_24h_was_weekend', 'lag_168h_was_weekend',
        'demand_lag_24h_sq', 'demand_lag_168h_sq', 'demand_lag_24h_log', 'temp_lag_24h',
        'peak_lag_interaction', 'peak_heating_interaction', 'weekend_temp_interaction',
        'heating_hour_cos_product', 'temp_lag_ratio_interaction',
        'night_temp_interaction', 'weekend_transition_temp', 'dow_sin_temp',
    ]
    
    return df[columns]


def main():
    print("Loading data...")
    df = load_from_database()
    print(f"  Loaded {len(df):,} records")
    
    print("Adding features...")
    df['target_demand'] = df['demand']
    df = add_temporal_features(df)
    df = add_weather_features(df)
    df = add_regime_features(df)
    df = add_demand_lag_features(df)
    df = add_interaction_features(df)
    
    df = select_final_columns(df)
    df = df.dropna()
    
    version = get_version_number()
    output_path = OUTPUT_DIR / f'tft_training_data-v{version}.csv'
    df.to_csv(output_path, index=False)
    
    print(f"\nSaved: {output_path}")
    print(f"  Rows: {len(df):,}")
    print(f"  Cols: {len(df.columns)}")
    print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")


if __name__ == '__main__':
    main()
