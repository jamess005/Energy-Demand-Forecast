"""
Data Pipeline Orchestrator
Runs all data feeds in sequence to populate model_ready_data table.
"""
import subprocess
import sys
from datetime import datetime

FEEDS = [
    ('holidays_feed.py', 'Holidays calendar'),
    ('weather_feed.py', 'Weather data'),
    ('entsoe_feed.py', 'Electricity demand & unavailability'),
]

def run_script(script_name: str) -> bool:
    """Execute a Python script and return success status."""
    try:
        result = subprocess.run(
            ['python3', script_name],
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout, end='')
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nFailed: {script_name}")
        print(e.stdout)
        print(e.stderr, file=sys.stderr)
        return False

def main():
    print("="*70)
    print(f"Data Pipeline | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    results = {}
    for script, description in FEEDS:
        print(f"\n[{description}]")
        results[script] = run_script(script)
    
    print("\n" + "="*70)
    success_count = sum(results.values())
    print(f"Complete: {success_count}/{len(FEEDS)} successful")
    print("="*70)
    
    return all(results.values())

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)