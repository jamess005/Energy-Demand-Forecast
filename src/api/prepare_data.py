"""
Dynamic Feature Engineering Pipeline
"""

import os
import numpy as np
import pandas as pd
from datetime import timedelta
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

SCALE_FACTOR = 10_000


def get_db_engine():
    """Create PostgreSQL connection."""
    required = ['DB_USER', 'DB_PASSWORD', 'DB_HOST', 'DB_PORT', 'DB_NAME']
    if missing := [v for v in required if not os.getenv(v)]:
        raise ValueError(f"Missing environment variables: {', '.join(missing)}")
    return create_engine(
        f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@"
        f"{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )


def load_historical_data(engine, end_date: str = None, limit: int = 1000) -> pd.DataFrame:
    """Load raw historical data from database."""
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
            ed.date_time AS timestamp,
            ed."actual_demand(MW)" AS target_demand,
            ed.daylight_savings_winter,
            ed.daylight_savings_summer,
            COALESCE(h.is_public_holiday, false) AS is_public_holiday,
            w."temperature_2m(°C)" AS temperature
        FROM energy_demand ed
        INNER JOIN weather w ON ed.date_time = w.date_time
        LEFT JOIN holidays h ON DATE(ed.date_time) = h.date
        {date_filter}
        ORDER BY ed.date_time DESC
        LIMIT {limit}
    """
    df = pd.read_sql(text(query), engine)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_convert(None)
    return df.sort_values('timestamp').reset_index(drop=True)


def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate all features from raw data."""
    df = df.copy()
    
    df['hour'] = df['timestamp'].dt.hour
    df['dow'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['year'] = df['timestamp'].dt.year
    
    # Cyclic encodings
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['dow'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['dow'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * (df['month'] - 1) / 12)
    df['month_cos'] = np.cos(2 * np.pi * (df['month'] - 1) / 12)
    
    for i in range(7):
        df[f'dow_{i}'] = (df['dow'] == i).astype(float)
    
    df['is_weekend'] = (df['dow'] >= 5).astype(float)
    df['is_weekday'] = (df['dow'] < 5).astype(float)
    df['is_public_holiday'] = df['is_public_holiday'].astype(float)
    df['is_monday_after_weekend'] = (df['dow'] == 0).astype(float)
    df['is_early_morning'] = ((df['hour'] >= 0) & (df['hour'] <= 4)).astype(float)
    df['is_morning_ramp'] = ((df['hour'] >= 4) & (df['hour'] <= 10)).astype(float)
    df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 4)).astype(float)
    df['is_peak_hour'] = ((df['hour'] >= 9) & (df['hour'] <= 12)).astype(float)
    df['is_peak_period'] = df['is_peak_hour']
    df['is_valley_hour'] = ((df['hour'] == 2) | (df['hour'] == 3)).astype(float)
    df['hour_squared'] = (df['hour'] / 24) ** 2
    
    prev_dow = df['dow'].shift(24).fillna(df['dow'])
    df['day_transition_type'] = 0.0
    df.loc[(prev_dow >= 5) & (df['dow'] < 5), 'day_transition_type'] = 1.0
    df.loc[(prev_dow < 5) & (df['dow'] >= 5), 'day_transition_type'] = 2.0
    
    df['season'] = np.select(
        [df['month'].isin([12, 1, 2]), df['month'].isin([3, 4, 5]), df['month'].isin([6, 7, 8])],
        ['Winter', 'Spring', 'Summer'],
        default='Autumn'
    )
    
    # Weather features
    df['heating_demand'] = np.maximum(18 - df['temperature'], 0)
    df['heating_demand_sq'] = df['heating_demand'] ** 2
    df['heating_demand_log'] = np.log1p(df['heating_demand'])
    df['temp_severity'] = np.clip((18 - df['temperature']) / 28, 0, 1)
    df['is_cold'] = (df['temperature'] < 5).astype(float)
    df['heating_evening_impact'] = df['heating_demand'] * ((df['hour'] >= 17) & (df['hour'] <= 20)).astype(float)
    df['monday_cold_multiplier'] = (df['dow'] == 0).astype(float) * df['heating_demand']
    
    # Regime indicators
    df['regime_0'] = (df['year'] == 2019).astype(float)
    df['regime_1'] = (df['year'] == 2020).astype(float)
    df['regime_2'] = df['year'].isin([2021, 2022]).astype(float)
    df['regime_3'] = (df['year'] >= 2023).astype(float)
    
    # Demand lags
    demand_mean = df['target_demand'].mean()
    demand_std = df['target_demand'].std()
    df['demand_lag_24h'] = df['target_demand'].shift(24).fillna(demand_mean)
    df['demand_lag_168h'] = df['target_demand'].shift(168).fillna(demand_mean)
    df['demand_lag_24h_norm'] = df['demand_lag_24h'] / SCALE_FACTOR
    df['demand_lag_168h_norm'] = df['demand_lag_168h'] / SCALE_FACTOR
    df['demand_lag_ratio'] = (df['demand_lag_24h'] / df['demand_lag_168h'].replace(0, 1)).clip(0.7, 1.3)
    df['demand_rolling_std_7d'] = df['target_demand'].rolling(168, min_periods=24).std().fillna(demand_std) / SCALE_FACTOR
    
    df['lag_24h_was_weekend'] = (df['dow'].shift(24) >= 5).fillna(0).astype(float)
    df['lag_24h_was_saturday'] = (df['dow'].shift(24) == 5).fillna(0).astype(float)
    df['lag_24h_was_sunday'] = (df['dow'].shift(24) == 6).fillna(0).astype(float)
    df['lag_168h_was_weekend'] = df['is_weekend']
    
    lag_was_weekend = df['lag_24h_was_weekend']
    df['lag_reliability'] = np.where(df['is_weekend'] == lag_was_weekend, 1.0, 0.5)
    df['ignore_lag_signal'] = (
        ((df['dow'] == 0) & (lag_was_weekend == 1)) |
        ((df['dow'] == 5) & (lag_was_weekend == 0))
    ).astype(float)
    
    df['baseline_demand_norm'] = demand_mean / SCALE_FACTOR
    df['demand_vs_baseline'] = (df['target_demand'] - demand_mean) / SCALE_FACTOR
    df['demand_lag_24h_dampened'] = df['demand_lag_24h_norm'] * 0.8
    df['demand_lag_168h_dampened'] = df['demand_lag_168h_norm'] * 0.8
    df['demand_delta_24h'] = (df['target_demand'] - df['demand_lag_24h']) / SCALE_FACTOR
    df['demand_lag_adjusted'] = df['demand_lag_24h_norm']
    df['demand_lag_24h_sq'] = df['demand_lag_24h_norm'] ** 2
    df['demand_lag_168h_sq'] = df['demand_lag_168h_norm'] ** 2
    df['demand_lag_24h_log'] = np.log1p(df['demand_lag_24h_norm'])
    df['seasonal_weekly_pattern'] = df['demand_lag_168h_norm']
    
    # Interaction features
    df['peak_lag_interaction'] = df['is_peak_hour'] * df['demand_lag_24h_norm']
    df['temp_lag_ratio_interaction'] = df['temperature'] * df['demand_lag_ratio']
    df['peak_heating_interaction'] = df['is_peak_hour'] * df['heating_demand']
    df['weekend_temp_interaction'] = df['is_weekend'] * df['temperature']
    df['heating_hour_cos_product'] = df['heating_demand'] * df['hour_cos']
    df['temp_peak_interaction'] = df['temperature'] * df['is_peak_hour']
    df['temp_morning_ramp_interaction'] = df['temperature'] * df['is_morning_ramp']
    df['temp_night_interaction'] = df['temperature'] * df['is_night']
    df['monday_morning_heating'] = ((df['dow'] == 0) & (df['hour'] <= 10)).astype(float) * df['heating_demand']
    df['evening_heating_interaction'] = ((df['hour'] >= 17) & (df['hour'] <= 20)).astype(float) * df['heating_demand']
    df['weekend_transition_temp'] = df['day_transition_type'] * df['temperature']
    df['dow_sin_temp'] = df['dow_sin'] * df['temperature']
    df['temp_lag_24h'] = df['temperature'].shift(24).fillna(df['temperature'])
    
    df['expected_transition_delta'] = 0.0
    df['transition_lag_penalty'] = 0.0
    df['morning_ramp_lag_adjustment'] = 0.0
    df['demand_lag_correction'] = 0.0
    df['is_week_after_holiday'] = 0.0
    
    return df


def generate_future_features(historical_df: pd.DataFrame, hours: int) -> pd.DataFrame:
    """Generate features for future prediction hours."""
    last_time = historical_df['timestamp'].iloc[-1]
    last_temp = historical_df['temperature'].iloc[-1]
    demand_mean = historical_df['target_demand'].mean()
    
    rows = []
    for h in range(1, hours + 1):
        ts = last_time + timedelta(hours=h)
        hour, dow, month, year = ts.hour, ts.weekday(), ts.month, ts.year
        
        lag_24h_idx = len(historical_df) - 24 + h - 1
        lag_168h_idx = len(historical_df) - 168 + h - 1
        lag_24h = historical_df.iloc[lag_24h_idx]['target_demand'] if 0 <= lag_24h_idx < len(historical_df) else demand_mean
        lag_168h = historical_df.iloc[lag_168h_idx]['target_demand'] if 0 <= lag_168h_idx < len(historical_df) else demand_mean
        
        prev_dow = (ts - timedelta(hours=24)).weekday()
        heating = max(18 - last_temp, 0)
        
        row = {
            'timestamp': ts,
            'target_demand': demand_mean,
            'temperature': last_temp,
            'daylight_savings_winter': historical_df['daylight_savings_winter'].iloc[-1],
            'daylight_savings_summer': historical_df['daylight_savings_summer'].iloc[-1],
            'is_public_holiday': 0.0,
            'hour': hour, 'dow': dow, 'month': month, 'year': year,
            'hour_sin': np.sin(2 * np.pi * hour / 24),
            'hour_cos': np.cos(2 * np.pi * hour / 24),
            'dow_sin': np.sin(2 * np.pi * dow / 7),
            'dow_cos': np.cos(2 * np.pi * dow / 7),
            'month_sin': np.sin(2 * np.pi * (month - 1) / 12),
            'month_cos': np.cos(2 * np.pi * (month - 1) / 12),
            'is_weekend': float(dow >= 5),
            'is_weekday': float(dow < 5),
            'is_monday_after_weekend': float(dow == 0),
            'is_early_morning': float(0 <= hour <= 4),
            'is_morning_ramp': float(4 <= hour <= 10),
            'is_night': float(hour >= 22 or hour <= 4),
            'is_peak_hour': float(9 <= hour <= 12),
            'is_peak_period': float(9 <= hour <= 12),
            'is_valley_hour': float(hour in [2, 3]),
            'hour_squared': (hour / 24) ** 2,
            'day_transition_type': 0.0,
            'heating_demand': heating,
            'heating_demand_sq': heating ** 2,
            'heating_demand_log': np.log1p(heating),
            'temp_severity': np.clip((18 - last_temp) / 28, 0, 1),
            'is_cold': float(last_temp < 5),
            'heating_evening_impact': heating * float(17 <= hour <= 20),
            'monday_cold_multiplier': float(dow == 0) * heating,
            'regime_0': float(year == 2019),
            'regime_1': float(year == 2020),
            'regime_2': float(year in [2021, 2022]),
            'regime_3': float(year >= 2023),
            'demand_lag_24h': lag_24h,
            'demand_lag_168h': lag_168h,
            'demand_lag_24h_norm': lag_24h / SCALE_FACTOR,
            'demand_lag_168h_norm': lag_168h / SCALE_FACTOR,
            'demand_lag_ratio': min(max(lag_24h / lag_168h if lag_168h else 1.0, 0.7), 1.3),
            'demand_rolling_std_7d': historical_df['target_demand'].std() / SCALE_FACTOR,
            'lag_24h_was_weekend': float(prev_dow >= 5),
            'lag_24h_was_saturday': float(prev_dow == 5),
            'lag_24h_was_sunday': float(prev_dow == 6),
            'lag_168h_was_weekend': float(dow >= 5),
            'lag_reliability': 0.5 if ((dow == 0 and hour <= 10) or (dow == 5 and hour <= 10)) else 1.0,
            'ignore_lag_signal': float((dow == 0 and prev_dow == 6) or (dow == 5 and prev_dow == 4)),
            'baseline_demand_norm': demand_mean / SCALE_FACTOR,
            'demand_vs_baseline': 0.0,
            'demand_lag_24h_dampened': lag_24h / SCALE_FACTOR * 0.8,
            'demand_lag_168h_dampened': lag_168h / SCALE_FACTOR * 0.8,
            'demand_delta_24h': 0.0,
            'demand_lag_adjusted': lag_24h / SCALE_FACTOR,
            'demand_lag_24h_sq': (lag_24h / SCALE_FACTOR) ** 2,
            'demand_lag_168h_sq': (lag_168h / SCALE_FACTOR) ** 2,
            'demand_lag_24h_log': np.log1p(lag_24h / SCALE_FACTOR),
            'seasonal_weekly_pattern': lag_168h / SCALE_FACTOR,
            'expected_transition_delta': 0.0,
            'transition_lag_penalty': 0.0,
            'morning_ramp_lag_adjustment': 0.0,
            'demand_lag_correction': 0.0,
            'is_week_after_holiday': 0.0,
        }
        
        for i in range(7):
            row[f'dow_{i}'] = float(dow == i)
        
        if month in [12, 1, 2]:
            row['season'] = 'Winter'
        elif month in [3, 4, 5]:
            row['season'] = 'Spring'
        elif month in [6, 7, 8]:
            row['season'] = 'Summer'
        else:
            row['season'] = 'Autumn'
        
        row['peak_lag_interaction'] = row['is_peak_hour'] * row['demand_lag_24h_norm']
        row['temp_lag_ratio_interaction'] = last_temp * row['demand_lag_ratio']
        row['peak_heating_interaction'] = row['is_peak_hour'] * heating
        row['weekend_temp_interaction'] = row['is_weekend'] * last_temp
        row['heating_hour_cos_product'] = heating * row['hour_cos']
        row['temp_peak_interaction'] = last_temp * row['is_peak_hour']
        row['temp_morning_ramp_interaction'] = last_temp * row['is_morning_ramp']
        row['temp_night_interaction'] = last_temp * row['is_night']
        row['monday_morning_heating'] = float(dow == 0 and hour <= 10) * heating
        row['evening_heating_interaction'] = float(17 <= hour <= 20) * heating
        row['weekend_transition_temp'] = row['day_transition_type'] * last_temp
        row['dow_sin_temp'] = row['dow_sin'] * last_temp
        row['temp_lag_24h'] = last_temp
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def ensure_model_features(df: pd.DataFrame, model_features: list) -> pd.DataFrame:
    """Ensure dataframe has all features required by model."""
    for feat in model_features:
        if feat not in df.columns:
            df[feat] = 0.0
    return df
