import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os
import sys
import argparse

# Load environment variables
load_dotenv()

# Parse command line arguments
parser = argparse.ArgumentParser(description='Load weather CSV data to database')
parser.add_argument('--csv', type=str, help='Path to weather CSV file')
args, _ = parser.parse_known_args()

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

# Load CSV file (skip first 2 rows containing metadata)
csv_file = args.csv
if not csv_file:
    print("ERROR: Please provide CSV file path with --csv argument")
    print("Usage: python weather.py --csv /path/to/weather-data.csv")
    sys.exit(1)

try:
    df = pd.read_csv(csv_file, skiprows=2)
except FileNotFoundError:
    print(f"ERROR: CSV file not found: {csv_file}")
    sys.exit(1)

print(f"Loaded {len(df)} hourly weather records")
print(f"Columns: {df.columns.tolist()}")

# Parse timestamp (already in ISO format, hourly resolution)
df['timestamp'] = pd.to_datetime(df['time'])

# Create daylight savings indicators for model features
# CEST (summer): approximately end of March to end of October
is_summer = (
    ((df['timestamp'].dt.month > 3) & (df['timestamp'].dt.month < 10)) |
    ((df['timestamp'].dt.month == 3) & (df['timestamp'].dt.day >= 25)) |
    ((df['timestamp'].dt.month == 10) & (df['timestamp'].dt.day < 25))
)

df['daylight_savings_winter'] = ~is_summer  # CET (winter time)
df['daylight_savings_summer'] = is_summer   # CEST (summer time)

# Prepare dataframe for database insertion
weather_df = pd.DataFrame({
    'date_time': df['timestamp'],
    'temperature_2m(°C)': df['temperature_2m (°C)'],
    'relative_humidity_2m(%)': df['relative_humidity_2m (%)'],
    'rain(mm)': df['rain (mm)'],
    'snow_depth(m)': df['snow_depth (m)'],
    'snowfall(cm)': df['snowfall (cm)'],
    'daylight_savings_winter': df['daylight_savings_winter'],
    'daylight_savings_summer': df['daylight_savings_summer']
})

print(f"Prepared {len(weather_df)} records for database")
print(f"Date range: {weather_df['date_time'].min()} to {weather_df['date_time'].max()}")

# Load data to PostgreSQL database
try:
    weather_df.to_sql('weather', engine, if_exists='append', index=False)
    print(f"✓ Successfully loaded {len(weather_df)} records to weather table")
except Exception as e:
    print(f"ERROR: Database insert failed: {e}")
    sys.exit(1)