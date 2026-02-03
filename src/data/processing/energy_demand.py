import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os
import sys
import argparse

# Load environment variables
load_dotenv()

# Parse command line arguments
parser = argparse.ArgumentParser(description='Load energy demand CSV data to database')
parser.add_argument('--csv', type=str, help='Path to energy demand CSV file')
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

# Load CSV file
csv_file = args.csv
if not csv_file:
    print("ERROR: Please provide CSV file path with --csv argument")
    print("Usage: python energy_demand.py --csv /path/to/demand-data.csv")
    sys.exit(1)

try:
    df = pd.read_csv(csv_file)
except FileNotFoundError:
    print(f"ERROR: CSV file not found: {csv_file}")
    sys.exit(1)

print(f"Loaded {len(df)} raw 15-minute records")
print(f"Columns: {list(df.columns)}")

# Find the timestamp column (different CSVs have different column names)
timestamp_col = [col for col in df.columns if 'MTU' in col or 'Time' in col][0]
print(f"Using timestamp column: {timestamp_col}")

# Parse timestamp from MTU column (extract start time from range)
timestamp_str = df[timestamp_col].str.split(' - ').str[0]
timestamp_clean = timestamp_str.str.replace(r' \([A-Z]+\)$', '', regex=True)
df['timestamp'] = pd.to_datetime(timestamp_clean, format='%d/%m/%Y %H:%M')

# Create daylight savings indicators for model features
# CEST (summer): approximately end of March to end of October
is_summer = (
    ((df['timestamp'].dt.month > 3) & (df['timestamp'].dt.month < 10)) |
    ((df['timestamp'].dt.month == 3) & (df['timestamp'].dt.day >= 25)) |
    ((df['timestamp'].dt.month == 10) & (df['timestamp'].dt.day < 25))
)

df['daylight_savings_winter'] = ~is_summer  # CET (winter time)
df['daylight_savings_summer'] = is_summer   # CEST (summer time)

# Floor timestamps to hourly intervals
df['hour'] = df['timestamp'].dt.floor('h')

# Aggregate 15-minute intervals to hourly averages
hourly = df.groupby('hour').agg({
    'Actual Total Load (MW)': 'mean',
    'Day-ahead Total Load Forecast (MW)': 'mean',
    'daylight_savings_winter': 'first',
    'daylight_savings_summer': 'first'
}).reset_index()

# Rename columns to match database schema
hourly.columns = [
    'date_time',
    'actual_demand(MW)',
    'demand_forecast(MW)',
    'daylight_savings_winter',
    'daylight_savings_summer'
]

print(f"Aggregated to {len(hourly)} hourly records")
print(f"Date range: {hourly['date_time'].min()} to {hourly['date_time'].max()}")

# Load data to PostgreSQL database
try:
    hourly.to_sql('energy_demand', engine, if_exists='append', index=False)
    print(f"✓ Successfully loaded {len(hourly)} records to energy_demand table")
except Exception as e:
    print(f"ERROR: Database insert failed: {e}")
    sys.exit(1)