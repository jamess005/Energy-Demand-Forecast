"""
ENTSO-E Data Feed
Fetches live electricity demand data for Germany.
Unavailability data is set to zeros (API data unreliable/corrupted).
"""
import pandas as pd
from entsoe import EntsoePandasClient
from sqlalchemy import create_engine, text
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

def add_dst_flags(df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
    """Add daylight savings time indicators."""
    is_summer = (
        ((df[timestamp_col].dt.month > 3) & (df[timestamp_col].dt.month < 10)) |
        ((df[timestamp_col].dt.month == 3) & (df[timestamp_col].dt.day >= 25)) |
        ((df[timestamp_col].dt.month == 10) & (df[timestamp_col].dt.day < 25))
    )
    df['daylight_savings_winter'] = ~is_summer
    df['daylight_savings_summer'] = is_summer
    return df

def clean_duplicates(engine, table_name):
    """Remove duplicate timestamps, keeping first occurrence."""
    print(f"  Checking for duplicates in {table_name}...")
    
    with engine.begin() as conn:
        # Find duplicates
        check_dupes = text(f"""
            SELECT date_time, COUNT(*) as count 
            FROM {table_name}
            GROUP BY date_time 
            HAVING COUNT(*) > 1 
            LIMIT 5
        """)
        dupes = pd.read_sql(check_dupes, conn)
        
        if len(dupes) > 0:
            print(f"  ⚠ Found {len(dupes)} duplicate timestamps, removing...")
            
            # Delete duplicates, keeping only first (by ctid - physical row ID)
            delete_dupes = text(f"""
                DELETE FROM {table_name} a
                USING {table_name} b
                WHERE a.ctid < b.ctid
                AND a.date_time = b.date_time
            """)
            result = conn.execute(delete_dupes)
            print(f"  ✓ Removed {result.rowcount} duplicate rows")
        else:
            print(f"  ✓ No duplicates found")

def fetch_demand_data(client, start_date, end_date):
    """Fetch actual load and day-ahead forecast from ENTSO-E."""
    country_code = 'DE_LU'
    
    print("  Fetching demand data from ENTSO-E API...")
    actual_load = client.query_load(country_code, start=start_date, end=end_date)
    
    # Reset index to convert to regular dataframe
    df = actual_load.reset_index()
    df.columns = ['date_time', 'actual_demand(MW)']
    
    # Convert to timezone-naive
    df['date_time'] = pd.to_datetime(df['date_time'])
    if df['date_time'].dt.tz is not None:
        df['date_time'] = df['date_time'].dt.tz_localize(None)
    
    # Resample to hourly (ENTSO-E sometimes returns 15-min data)
    # Use mean for sub-hourly periods, keeping hourly timestamps
    df = df.set_index('date_time')
    df = df.resample('h').mean().reset_index()
    df = df.dropna(subset=['actual_demand(MW)'])
    
    try:
        forecast = client.query_load_forecast(country_code, start=start_date, end=end_date)
        forecast_df = forecast.reset_index()
        forecast_df.columns = ['date_time', 'demand_forecast(MW)']
        forecast_df['date_time'] = pd.to_datetime(forecast_df['date_time'])
        if forecast_df['date_time'].dt.tz is not None:
            forecast_df['date_time'] = forecast_df['date_time'].dt.tz_localize(None)
        # Resample forecast to hourly too
        forecast_df = forecast_df.set_index('date_time').resample('h').mean().reset_index()
        df = df.merge(forecast_df, on='date_time', how='left')
    except:
        print("  ⚠ Could not fetch day-ahead forecast")
    
    df['demand_forecast(MW)'] = df.get('demand_forecast(MW)', 0).fillna(0)
    
    df = add_dst_flags(df, 'date_time')
    
    # Remove any duplicates from API
    df = df.drop_duplicates(subset=['date_time']).sort_values('date_time').reset_index(drop=True)
    
    print(f"  ✓ Retrieved {len(df)} hourly demand records")
    return df

def create_unavailability_data(demand_timestamps):
    """Create zero-filled unavailability data (API data is unreliable)."""
    print("  Creating zero-filled unavailability data...")
    df = pd.DataFrame({
        'date_time': demand_timestamps,
        'planned_unavailability(MW)': 0.0,
        'actual_unavailability(MW)': 0.0,
        'total_unavailability(MW)': 0.0,
        'daylight_savings_winter': False,
        'daylight_savings_summer': False
    })
    
    # Copy DST flags from demand data if available
    # (This is a simple approach - just set all to False or compute properly)
    is_summer = (
        ((df['date_time'].dt.month > 3) & (df['date_time'].dt.month < 10)) |
        ((df['date_time'].dt.month == 3) & (df['date_time'].dt.day >= 25)) |
        ((df['date_time'].dt.month == 10) & (df['date_time'].dt.day < 25))
    )
    df['daylight_savings_winter'] = ~is_summer
    df['daylight_savings_summer'] = is_summer
    
    print(f"  ✓ Created {len(df)} zero-filled records")
    return df

def safe_delete_and_insert(engine, table_name, df, date_column='date_time'):
    """Safely delete existing records in date range and insert new ones."""
    if df.empty:
        print(f"  No data to insert into {table_name}")
        return 0, 0
    
    min_date = df[date_column].min()
    max_date = df[date_column].max()
    
    # Ensure dates are timezone-naive strings
    min_date_str = min_date.strftime('%Y-%m-%d %H:%M:%S') if hasattr(min_date, 'strftime') else str(min_date)
    max_date_str = max_date.strftime('%Y-%m-%d %H:%M:%S') if hasattr(max_date, 'strftime') else str(max_date)
    
    print(f"  Updating {table_name} for range: {min_date_str} to {max_date_str}")
    
    with engine.begin() as conn:
        # Delete existing records in this range
        delete_query = text(f"""
            DELETE FROM {table_name}
            WHERE {date_column} >= :min_date
            AND {date_column} <= :max_date
        """)
        
        result = conn.execute(delete_query, {
            'min_date': min_date_str,
            'max_date': max_date_str
        })
        deleted_count = result.rowcount
    
    # Insert fresh data
    df.to_sql(table_name, engine, if_exists='append', index=False)
    
    return len(df), deleted_count

def main():
    if not os.getenv('ENTSOE_API_KEY'):
        raise EnvironmentError("ENTSOE_API_KEY not found in environment")
    
    engine = get_db_engine()
    
    print("="*70)
    print("ENTSO-E DATA FEED")
    print("="*70)
    
    # Step 1: Clean existing duplicates
    print("\n[1/4] Cleaning existing duplicates...")
    clean_duplicates(engine, 'energy_demand')
    clean_duplicates(engine, 'energy_unavailability')
    
    # Step 2: Fetch demand data
    print("\n[2/4] Fetching demand data...")
    client = EntsoePandasClient(api_key=os.getenv('ENTSOE_API_KEY'))
    
    end_date = pd.Timestamp.now(tz='UTC')
    start_date = end_date - pd.Timedelta(days=30)
    
    print(f"  Date range: {start_date.date()} to {end_date.date()}")
    
    demand_df = fetch_demand_data(client, start_date, end_date)
    
    if demand_df.empty:
        print("  ✗ No data fetched from API")
        return
    
    # Step 3: Create unavailability data (zeros)
    print("\n[3/4] Creating unavailability data...")
    unavail_df = create_unavailability_data(demand_df['date_time'])
    
    # Step 4: Update database
    print("\n[4/4] Updating database...")
    
    demand_inserted, demand_deleted = safe_delete_and_insert(engine, 'energy_demand', demand_df)
    unavail_inserted, unavail_deleted = safe_delete_and_insert(engine, 'energy_unavailability', unavail_df)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    if demand_deleted > 0:
        print(f"✓ Updated {demand_inserted} demand records (replaced {demand_deleted})")
    else:
        print(f"✓ Inserted {demand_inserted} new demand records")
    
    if unavail_deleted > 0:
        print(f"✓ Updated {unavail_inserted} unavailability records (replaced {unavail_deleted})")
    else:
        print(f"✓ Inserted {unavail_inserted} new unavailability records")
    
    print(f"\nDemand range: {demand_df['actual_demand(MW)'].min():.0f} - {demand_df['actual_demand(MW)'].max():.0f} MW")
    print(f"Date range: {demand_df['date_time'].min().date()} to {demand_df['date_time'].max().date()}")
    print("="*70)

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)