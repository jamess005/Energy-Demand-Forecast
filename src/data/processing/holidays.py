import http.client
import json
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os
import sys

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

# Step 1: Define years to fetch
years = [2019, 2020]

# Create complete date range for all years
start_date = pd.to_datetime(f'{min(years)}-01-01')
end_date = pd.to_datetime(f'{max(years)}-12-31')
all_dates = pd.date_range(start=start_date, end=end_date, freq='D')

# Create base DataFrame with all dates
dates_df = pd.DataFrame({'date': all_dates})
print(f"Created date range: {len(dates_df)} dates from {start_date.date()} to {end_date.date()}")

# Step 2: Fetch holiday data from API
holiday_data = []

for year in years:
    print(f"Fetching holidays for {year}...")
    
    conn = http.client.HTTPSConnection("date.nager.at")
    conn.request("GET", f"/api/v3/PublicHolidays/{year}/DE")
    res = conn.getresponse()
    data = res.read()
    
    holidays = json.loads(data.decode("utf-8"))
    
    for holiday in holidays:
        counties = holiday.get('counties', [])
        types = holiday.get('types', [])
        
        holiday_data.append({
            'date': pd.to_datetime(holiday['date']),
            'is_public_holiday': 'Public' in types,
            'is_national': holiday.get('global', False),
            'holiday_type': types[0] if types else None,
            'counties_affected': len(counties) if counties else 0
        })
    
    print(f"  Found {len(holidays)} holidays")
    conn.close()

# Create holidays DataFrame
holidays_df = pd.DataFrame(holiday_data)

# Step 3: Merge with complete date range
result_df = dates_df.merge(holidays_df, on='date', how='left')

# Fill NaN values for non-holiday dates (fix FutureWarning with infer_objects)
result_df['is_public_holiday'] = result_df['is_public_holiday'].fillna(False)
result_df['is_national'] = result_df['is_national'].fillna(False)
result_df = result_df.infer_objects(copy=False)
result_df['is_public_holiday'] = result_df['is_public_holiday'].astype(bool)
result_df['is_national'] = result_df['is_national'].astype(bool)
result_df['counties_affected'] = result_df['counties_affected'].fillna(0).astype(int)
result_df['holiday_type'] = result_df['holiday_type'].fillna('')

# Step 4: Add daylight savings indicators
is_summer = (
    ((result_df['date'].dt.month > 3) & (result_df['date'].dt.month < 10)) |
    ((result_df['date'].dt.month == 3) & (result_df['date'].dt.day >= 25)) |
    ((result_df['date'].dt.month == 10) & (result_df['date'].dt.day < 25))
)

result_df['daylight_savings_winter'] = ~is_summer
result_df['daylight_savings_summer'] = is_summer

# Normalize date to remove any time component (set to midnight)
result_df['date'] = result_df['date'].dt.normalize()

print(f"\nPrepared {len(result_df)} dates")
print(f"Date range: {result_df['date'].min()} to {result_df['date'].max()}")
print(f"Public holidays: {result_df['is_public_holiday'].sum()}")
print(f"National holidays: {result_df['is_national'].sum()}")

# Load to database
try:
    result_df.to_sql('holidays', engine, if_exists='append', index=False)
    print(f"✓ Successfully loaded {len(result_df)} dates to holidays table")
except Exception as e:
    print(f"ERROR: Database insert failed: {e}")
    sys.exit(1)