"""
Data Pipeline Orchestrator
Runs all data feeds in sequence, then builds training data.
"""
import subprocess
import sys
from datetime import datetime
from pathlib import Path

FEEDS = [
    ('holidays_feed.py', 'Holidays calendar'),
    ('weather_feed.py', 'Weather data'),
    ('entsoe_feed.py', 'Electricity demand & unavailability'),
]

# Training data generator (relative to project root)
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
COMPLETE_DATA_SCRIPT = PROJECT_ROOT / 'src' / 'data' / 'processing' / 'complete_data.py'

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
    
    # Build training data from database
    print(f"\n[Build training data]")
    if COMPLETE_DATA_SCRIPT.exists():
        results['complete_data.py'] = run_script(str(COMPLETE_DATA_SCRIPT))
    else:
        print(f"  Skipped: {COMPLETE_DATA_SCRIPT} not found")
    
    print("\n" + "="*70)
    success_count = sum(results.values())
    print(f"Complete: {success_count}/{len(results)} successful")
    print("="*70)
    
    return all(results.values())

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)