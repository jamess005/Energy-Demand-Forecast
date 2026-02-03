"""
Weather Data Feed
Fetches historical hourly weather data for Berlin from Open-Meteo API.
Fetches 30 days to avoid gaps and updates existing records with latest API values.
"""
import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os
import sys
from datetime import datetime, timedelta

load_dotenv()

# Weather location - configurable via environment variables
WEATHER_LAT = float(os.getenv('WEATHER_LAT', '52.52'))  # Berlin default
WEATHER_LON = float(os.getenv('WEATHER_LON', '13.41'))  # Berlin default

def get_db_engine():
    """Create database engine from environment variables."""
    required = ['DB_USER', 'DB_PASSWORD', 'DB_HOST', 'DB_PORT', 'DB_NAME']
    if missing := [v for v in required if not os.getenv(v)]:
        raise EnvironmentError(f"Missing environment variables: {', '.join(missing)}")
    
    return create_engine(
        f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@"
        f"{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )

def add_dst_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add daylight savings time indicators."""
    is_summer = (
        ((df['date_time'].dt.month > 3) & (df['date_time'].dt.month < 10)) |
        ((df['date_time'].dt.month == 3) & (df['date_time'].dt.day >= 25)) |
        ((df['date_time'].dt.month == 10) & (df['date_time'].dt.day < 25))
    )
    df['daylight_savings_winter'] = ~is_summer
    df['daylight_savings_summer'] = is_summer
    return df

def fetch_weather_data(start_date, end_date):
    """Fetch hourly weather data from Open-Meteo."""
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    client = openmeteo_requests.Client(session=retry_session)
    
    params = {
        "latitude": WEATHER_LAT,
        "longitude": WEATHER_LON,
        "start_date": start_date.strftime('%Y-%m-%d'),
        "end_date": end_date.strftime('%Y-%m-%d'),
        "hourly": ["temperature_2m", "relative_humidity_2m", "rain", "snowfall", "snow_depth"],
        "timezone": "UTC"
    }
    
    response = client.weather_api("https://archive-api.open-meteo.com/v1/archive", params=params)[0]
    hourly = response.Hourly()
    
    timestamps = pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s"),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s"),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left"
    )
    
    df = pd.DataFrame({
        'date_time': timestamps,
        'temperature_2m(°C)': hourly.Variables(0).ValuesAsNumpy(),
        'relative_humidity_2m(%)': hourly.Variables(1).ValuesAsNumpy(),
        'rain(mm)': hourly.Variables(2).ValuesAsNumpy(),
        'snow_depth(m)': hourly.Variables(4).ValuesAsNumpy(),
        'snowfall(cm)': hourly.Variables(3).ValuesAsNumpy()
    })
    
    df = add_dst_flags(df)
    return df.dropna().reset_index(drop=True)

def main():
    engine = get_db_engine()
    
    # Fetch 30 days to avoid gaps
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=30)
    
    print(f"Fetching weather data: {start_date} to {end_date}")
    
    df = fetch_weather_data(start_date, end_date)
    
    # Ensure timezone-naive timestamps
    df['date_time'] = pd.to_datetime(df['date_time']).dt.tz_localize(None)
    
    if df.empty:
        print("No data fetched from API")
        return
    
    # Delete existing records in this date range, then insert fresh data
    min_date = df['date_time'].min()
    max_date = df['date_time'].max()
    
    with engine.begin() as conn:
        delete_query = text(
            "DELETE FROM weather WHERE date_time >= :min_date AND date_time <= :max_date"
        )
        result = conn.execute(delete_query, {'min_date': min_date, 'max_date': max_date})
        deleted_count = result.rowcount
    
    # Insert fresh data
    df.to_sql('weather', engine, if_exists='append', index=False)
    
    if deleted_count > 0:
        print(f"Updated {len(df)} weather records (replaced {deleted_count} existing)")
    else:
        print(f"Inserted {len(df)} new weather records")
    
    print(f"Date range: {min_date.date()} to {max_date.date()}")
    print(f"Temperature range: {df['temperature_2m(°C)'].min():.1f}°C to {df['temperature_2m(°C)'].max():.1f}°C")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)