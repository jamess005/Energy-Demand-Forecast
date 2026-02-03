"""
Holidays Feed
Creates complete calendar with German public holiday flags.
Generates rows for every day of the year with holiday markers.
"""
import http.client
import json
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os
import sys

load_dotenv()

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
        ((df['date'].dt.month > 3) & (df['date'].dt.month < 10)) |
        ((df['date'].dt.month == 3) & (df['date'].dt.day >= 25)) |
        ((df['date'].dt.month == 10) & (df['date'].dt.day < 25))
    )
    df['daylight_savings_winter'] = ~is_summer
    df['daylight_savings_summer'] = is_summer
    return df

def get_existing_years(engine) -> set:
    """Get years that already exist in holidays table."""
    query = "SELECT DISTINCT EXTRACT(YEAR FROM date) as year FROM holidays"
    try:
        existing = pd.read_sql(query, engine)
        return set(int(y) for y in existing['year'])
    except:
        return set()

def fetch_holidays_for_year(year: int) -> list:
    """Fetch German public holidays for a specific year."""
    conn = http.client.HTTPSConnection("date.nager.at")
    conn.request("GET", f"/api/v3/PublicHolidays/{year}/DE")
    response = conn.getresponse()
    data = json.loads(response.read().decode("utf-8"))
    conn.close()
    
    holidays = []
    for holiday in data:
        types = holiday.get('types') or []
        counties = holiday.get('counties') or []
        holidays.append({
            'date': pd.to_datetime(holiday['date']),
            'is_public_holiday': 'Public' in types,
            'is_national': holiday.get('global', False),
            'holiday_type': types[0] if types else None,
            'counties_affected': len(counties)
        })
    
    return holidays

def create_year_calendar(year: int) -> pd.DataFrame:
    """Create complete calendar for a year with holiday markers."""
    all_dates = pd.date_range(start=f'{year}-01-01', end=f'{year}-12-31', freq='D')
    df = pd.DataFrame({'date': all_dates})
    
    holidays = fetch_holidays_for_year(year)
    holidays_df = pd.DataFrame(holidays)
    
    df = df.merge(holidays_df, on='date', how='left')
    df['is_public_holiday'] = df['is_public_holiday'].fillna(False).astype(bool)
    df['is_national'] = df['is_national'].fillna(False).astype(bool)
    df['counties_affected'] = df['counties_affected'].fillna(0).astype(int)
    df['holiday_type'] = df['holiday_type'].fillna('')
    
    df = add_dst_flags(df)
    df['date'] = df['date'].dt.normalize()
    
    return df

def main():
    engine = get_db_engine()
    
    current_year = pd.Timestamp.now().year
    target_years = [current_year, current_year + 1]
    
    existing_years = get_existing_years(engine)
    years_to_fetch = [y for y in target_years if y not in existing_years]
    
    if not years_to_fetch:
        print(f"Years {target_years} already exist in database")
        return
    
    print(f"Fetching holidays for: {years_to_fetch}")
    
    all_data = []
    for year in years_to_fetch:
        year_df = create_year_calendar(year)
        all_data.append(year_df)
        print(f"Generated {len(year_df)} dates for {year} ({year_df['is_public_holiday'].sum()} holidays)")
    
    df = pd.concat(all_data, ignore_index=True)
    
    df.to_sql('holidays', engine, if_exists='append', index=False)
    print(f"Inserted {len(df)} total dates")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)