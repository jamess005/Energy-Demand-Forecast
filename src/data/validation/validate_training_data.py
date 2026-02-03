"""
TFT Training Data Validator
============================
Validates and cleans TFT training data with comprehensive quality checks.

Features:
- Smart duplicate detection (similar rows)
- Data range validation
- Temporal continuity checks
- Feature correlation analysis
- Anomaly detection
- Data leakage verification

Author: James
Project: Germany Electricity Load Forecasting
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

# =============================================================================
# CONFIGURATION
# =============================================================================

# Duplicate detection thresholds
DUPLICATE_SIMILARITY_THRESHOLD = 0.999  # How similar rows must be (0-1)
TIMESTAMP_DUPLICATE_WINDOW = pd.Timedelta(hours=1)  # Consider timestamps within this window

# Expected data ranges (for Germany electricity)
EXPECTED_RANGES = {
    'target_demand': (30000, 90000),  # MW
    'temperature': (-20, 40),  # °C
    'humidity': (0, 100),  # %
    'planned_unavailability': (0, 50000),  # MW
    'actual_unavailability': (0, 50000),  # MW
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*70}")
    print(f" {title}")
    print('='*70)


def print_status(message: str, indent: int = 0):
    """Print a status message with optional indentation."""
    prefix = "  " * indent
    print(f"{prefix}✓ {message}")


def print_warning(message: str, indent: int = 0):
    """Print a warning message."""
    prefix = "  " * indent
    print(f"{prefix}⚠ {message}")


def print_error(message: str, indent: int = 0):
    """Print an error message."""
    prefix = "  " * indent
    print(f"{prefix}✗ {message}")


# =============================================================================
# DATA LOADING
# =============================================================================

def load_training_data(filepath: Path) -> pd.DataFrame:
    """Load training data CSV file."""
    if not filepath.exists():
        print_error(f"File not found: {filepath}")
        sys.exit(1)
    
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print_status(f"Loaded {len(df):,} records from {filepath.name}")
    print_status(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print_status(f"Features: {len(df.columns)} columns")
    
    return df


# =============================================================================
# DUPLICATE DETECTION
# =============================================================================

def calculate_row_similarity(row1: pd.Series, row2: pd.Series, exclude_cols: list = None) -> float:
    """Calculate similarity between two rows (0 to 1)."""
    if exclude_cols is None:
        exclude_cols = ['timestamp']
    
    # Get numeric columns only
    numeric_cols = row1.select_dtypes(include=[np.number]).index
    numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    if len(numeric_cols) == 0:
        return 0.0
    
    # Calculate normalized differences
    differences = []
    for col in numeric_cols:
        val1, val2 = row1[col], row2[col]
        
        # Skip if both are NaN
        if pd.isna(val1) and pd.isna(val2):
            continue
        
        # Different if one is NaN
        if pd.isna(val1) or pd.isna(val2):
            differences.append(1.0)
            continue
        
        # Calculate relative difference
        max_val = max(abs(val1), abs(val2), 1e-10)
        diff = abs(val1 - val2) / max_val
        differences.append(diff)
    
    # Similarity is 1 - average difference
    if len(differences) == 0:
        return 1.0
    
    avg_diff = np.mean(differences)
    similarity = 1.0 - avg_diff
    
    return similarity


def find_duplicate_rows(df: pd.DataFrame, threshold: float = DUPLICATE_SIMILARITY_THRESHOLD) -> pd.DataFrame:
    """Find rows that are very similar to each other."""
    print_section("DUPLICATE DETECTION")
    print(f"Similarity threshold: {threshold:.1%} (higher = stricter)")
    
    duplicates = []
    
    # Group by nearby timestamps to speed up comparison
    df['timestamp_rounded'] = df['timestamp'].dt.floor('1H')
    
    for ts, group in df.groupby('timestamp_rounded'):
        if len(group) <= 1:
            continue
        
        # Compare all pairs within this time window
        indices = group.index.tolist()
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                idx1, idx2 = indices[i], indices[j]
                row1, row2 = df.loc[idx1], df.loc[idx2]
                
                similarity = calculate_row_similarity(row1, row2)
                
                if similarity >= threshold:
                    duplicates.append({
                        'index1': idx1,
                        'index2': idx2,
                        'timestamp1': row1['timestamp'],
                        'timestamp2': row2['timestamp'],
                        'similarity': similarity,
                        'demand1': row1['target_demand'],
                        'demand2': row2['target_demand']
                    })
    
    df.drop('timestamp_rounded', axis=1, inplace=True)
    
    if len(duplicates) > 0:
        dup_df = pd.DataFrame(duplicates)
        print_warning(f"Found {len(duplicates)} duplicate pairs")
        print(f"\nSample duplicates:")
        print(dup_df.head(10).to_string(index=False))
        return dup_df
    else:
        print_status("No duplicates found")
        return pd.DataFrame()


def remove_duplicates(df: pd.DataFrame, duplicates_df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows, keeping the first occurrence."""
    if len(duplicates_df) == 0:
        return df
    
    # Get all indices to remove (keep index1, remove index2)
    indices_to_remove = duplicates_df['index2'].unique()
    
    df_clean = df.drop(indices_to_remove).reset_index(drop=True)
    
    print_status(f"Removed {len(indices_to_remove)} duplicate rows")
    print_status(f"Remaining records: {len(df_clean):,}")
    
    return df_clean


# =============================================================================
# DATA VALIDATION
# =============================================================================

def check_missing_values(df: pd.DataFrame) -> bool:
    """Check for missing values."""
    print_section("MISSING VALUES CHECK")
    
    missing = df.isnull().sum()
    total_missing = missing.sum()
    
    if total_missing > 0:
        print_warning(f"Found {total_missing} missing values")
        for col in missing[missing > 0].index:
            pct = (missing[col] / len(df)) * 100
            print_error(f"{col}: {missing[col]:,} ({pct:.2f}%)", indent=1)
        return False
    else:
        print_status("No missing values")
        return True


def check_data_ranges(df: pd.DataFrame) -> bool:
    """Check if data falls within expected ranges."""
    print_section("DATA RANGE VALIDATION")
    
    all_valid = True
    
    for col, (min_val, max_val) in EXPECTED_RANGES.items():
        if col not in df.columns:
            continue
        
        series = df[col]
        
        # Check for values outside range
        below_min = (series < min_val).sum()
        above_max = (series > max_val).sum()
        
        if below_min > 0 or above_max > 0:
            print_warning(f"{col}:")
            if below_min > 0:
                print_error(f"{below_min:,} values below {min_val}", indent=1)
                print(f"  Min found: {series.min():.2f}", indent=1)
            if above_max > 0:
                print_error(f"{above_max:,} values above {max_val}", indent=1)
                print(f"  Max found: {series.max():.2f}", indent=1)
            all_valid = False
        else:
            actual_min = series.min()
            actual_max = series.max()
            print_status(f"{col}: {actual_min:.2f} to {actual_max:.2f} (within range)")
    
    return all_valid


def check_temporal_continuity(df: pd.DataFrame) -> bool:
    """Check for gaps in hourly time series."""
    print_section("TEMPORAL CONTINUITY CHECK")
    
    df_sorted = df.sort_values('timestamp').reset_index(drop=True)
    time_diffs = df_sorted['timestamp'].diff()
    
    # Find gaps (excluding first row and DST transitions)
    expected_diff = pd.Timedelta(hours=1)
    dst_diff = pd.Timedelta(hours=2)  # DST fall back
    
    gaps = time_diffs[(time_diffs > expected_diff) & (time_diffs != dst_diff)]
    
    if len(gaps) > 0:
        print_warning(f"Found {len(gaps)} gaps in time series")
        
        # Show first few gaps
        gap_indices = gaps.index[:5]
        for idx in gap_indices:
            prev_ts = df_sorted.loc[idx - 1, 'timestamp']
            curr_ts = df_sorted.loc[idx, 'timestamp']
            gap_size = time_diffs[idx]
            print_error(f"Gap: {prev_ts} -> {curr_ts} ({gap_size})", indent=1)
        
        if len(gaps) > 5:
            print(f"  ... and {len(gaps) - 5} more gaps", indent=1)
        
        return False
    else:
        dst_transitions = (time_diffs == dst_diff).sum()
        print_status(f"Continuous hourly data ({dst_transitions} DST transitions)")
        return True


def check_infinite_values(df: pd.DataFrame) -> bool:
    """Check for infinite values."""
    print_section("INFINITE VALUES CHECK")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_count = np.isinf(df[numeric_cols]).sum()
    
    total_inf = inf_count.sum()
    
    if total_inf > 0:
        print_warning(f"Found {total_inf} infinite values")
        for col in inf_count[inf_count > 0].index:
            print_error(f"{col}: {inf_count[col]} infinite values", indent=1)
        return False
    else:
        print_status("No infinite values")
        return True


def check_duplicate_timestamps(df: pd.DataFrame) -> bool:
    """Check for duplicate timestamps."""
    print_section("TIMESTAMP UNIQUENESS CHECK")
    
    n_dupes = df['timestamp'].duplicated().sum()
    
    if n_dupes > 0:
        print_error(f"Found {n_dupes} duplicate timestamps")
        
        # Show examples
        dupe_timestamps = df[df['timestamp'].duplicated(keep=False)]['timestamp'].unique()[:5]
        for ts in dupe_timestamps:
            count = (df['timestamp'] == ts).sum()
            print_warning(f"{ts}: appears {count} times", indent=1)
        
        return False
    else:
        print_status("All timestamps are unique")
        return True


def check_data_leakage(df: pd.DataFrame) -> bool:
    """Check for potential data leakage in lag features."""
    print_section("DATA LEAKAGE CHECK")
    
    # Lag features should be shifted - check they don't equal current values
    lag_checks = [
        ('demand_rolling_mean_30d', 'target_demand'),
        ('demand_rolling_mean_7d', 'target_demand'),
        ('temp_rolling_mean_30d', 'temperature'),
    ]
    
    all_ok = True
    
    for lag_col, current_col in lag_checks:
        if lag_col not in df.columns or current_col not in df.columns:
            continue
        
        # Check correlation (should not be 1.0)
        corr = df[lag_col].corr(df[current_col])
        
        # Check if any values are identical (excluding first rows where it might be NaN)
        identical = (df[lag_col] == df[current_col]).sum()
        
        if corr > 0.999 or identical > len(df) * 0.5:
            print_warning(f"{lag_col} suspicious:")
            print_error(f"Correlation with {current_col}: {corr:.4f}", indent=1)
            print_error(f"Identical values: {identical}/{len(df)}", indent=1)
            all_ok = False
        else:
            print_status(f"{lag_col}: correlation = {corr:.4f}")
    
    if all_ok:
        print_status("No data leakage detected")
    
    return all_ok


def check_target_distribution(df: pd.DataFrame) -> bool:
    """Check target variable distribution."""
    print_section("TARGET DISTRIBUTION CHECK")
    
    target = df['target_demand']
    
    print(f"  Mean:     {target.mean():,.0f} MW")
    print(f"  Median:   {target.median():,.0f} MW")
    print(f"  Std Dev:  {target.std():,.0f} MW")
    print(f"  Min:      {target.min():,.0f} MW")
    print(f"  Max:      {target.max():,.0f} MW")
    
    # Check for outliers (values > 3 std from mean)
    mean = target.mean()
    std = target.std()
    outliers = ((target < mean - 3*std) | (target > mean + 3*std)).sum()
    
    if outliers > 0:
        pct = (outliers / len(df)) * 100
        print_warning(f"{outliers:,} outliers ({pct:.2f}%) beyond 3σ")
        return False
    else:
        print_status("No extreme outliers detected")
        return True


# =============================================================================
# VALIDATION REPORT
# =============================================================================

def generate_validation_report(df: pd.DataFrame) -> dict:
    """Run all validation checks and generate report."""
    print("="*70)
    print(" DATA VALIDATION REPORT")
    print(f" {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    results = {
        'missing_values': check_missing_values(df),
        'data_ranges': check_data_ranges(df),
        'temporal_continuity': check_temporal_continuity(df),
        'infinite_values': check_infinite_values(df),
        'duplicate_timestamps': check_duplicate_timestamps(df),
        'data_leakage': check_data_leakage(df),
        'target_distribution': check_target_distribution(df),
    }
    
    # Summary
    print_section("VALIDATION SUMMARY")
    
    passed = sum(results.values())
    total = len(results)
    
    print(f"\n  Checks passed: {passed}/{total}")
    
    for check, result in results.items():
        status = "✓" if result else "✗"
        check_name = check.replace('_', ' ').title()
        print(f"  {status} {check_name}")
    
    if passed == total:
        print("\n✓ All validation checks passed!")
        print("  Data is ready for training.")
    else:
        print(f"\n⚠ {total - passed} validation check(s) failed")
        print("  Review issues before training.")
    
    return results


# =============================================================================
# MAIN VALIDATION PIPELINE
# =============================================================================

def main():
    """Main validation pipeline."""
    start_time = datetime.now()
    
    # Get input file
    if len(sys.argv) < 2:
        print("Usage: python validate_training_data.py <path_to_csv>")
        print("\nExample:")
        print("  python validate_training_data.py /path/to/tft_training_data-v1.csv")
        sys.exit(1)
    
    input_path = Path(sys.argv[1])
    
    # Load data
    df = load_training_data(input_path)
    
    # Find duplicates
    duplicates_df = find_duplicate_rows(df, threshold=DUPLICATE_SIMILARITY_THRESHOLD)
    
    # Remove duplicates if found
    if len(duplicates_df) > 0:
        response = input("\nRemove duplicates? (y/n): ").strip().lower()
        if response == 'y':
            df = remove_duplicates(df, duplicates_df)
        else:
            print("Keeping duplicates for now")
    
    # Run validation checks
    validation_results = generate_validation_report(df)
    
    # Save cleaned data if duplicates were removed
    if len(duplicates_df) > 0 and response == 'y':
        output_path = input_path.parent / f"{input_path.stem}_validated{input_path.suffix}"
        df.to_csv(output_path, index=False)
        print_status(f"\nSaved cleaned data to: {output_path}")
        
        # File size comparison
        original_size = input_path.stat().st_size / (1024 * 1024)
        new_size = output_path.stat().st_size / (1024 * 1024)
        print(f"  Original: {original_size:.2f} MB")
        print(f"  Cleaned:  {new_size:.2f} MB")
        print(f"  Saved:    {original_size - new_size:.2f} MB")
    
    # Timing
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n{'='*70}")
    print(f" Validation completed in {elapsed:.1f} seconds")
    print("="*70)


if __name__ == '__main__':
    main()