import pandas as pd
import sys
import os
import json

def analyze_gaps(file_path, output_json_path):
    print(f"Analyzing potentially missing trading days in {file_path} (excluding weekends and typical overnight closures)...")
    try:
        df = pd.read_csv(file_path, parse_dates=['DateTime'])
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    if df.empty:
        print("DataFrame is empty, no gaps to analyze.")
        return

    df.sort_values('DateTime', inplace=True)
    df.drop_duplicates(subset=['DateTime'], inplace=True)

    time_diffs = df['DateTime'].diff().dropna()
    
    long_gap_threshold = pd.Timedelta(hours=9)

    missing_trading_day_gaps = []
    for i, gap_duration in time_diffs.items():
        if gap_duration <= pd.Timedelta(minutes=1):
            continue

        start_time = df.loc[i - 1, 'DateTime']
        end_time = df.loc[i, 'DateTime']
        
        is_weekend_gap = False
        current_check_time = start_time + pd.Timedelta(minutes=1)
        while current_check_time < end_time:
            if current_check_time.dayofweek == 5 or current_check_time.dayofweek == 6:
                is_weekend_gap = True
                break
            current_check_time += pd.Timedelta(days=1)

        if not is_weekend_gap:
            if gap_duration > long_gap_threshold:
                missing_trading_day_gaps.append({
                    'start': start_time.isoformat(),
                    'end': end_time.isoformat(),
                    'duration': str(gap_duration)
                })

    if not missing_trading_day_gaps:
        print("No potentially missing trading days (gaps > 9 hours during weekdays) found in the data.")
    else:
        print(f"Found {len(missing_trading_day_gaps)} potentially missing trading days (gaps > 9 hours during weekdays). Details saved to {output_json_path}")
        
        # Save to JSON
        with open(output_json_path, 'w') as f:
            json.dump(missing_trading_day_gaps, f, indent=4)

    print("\nNote: This analysis excludes weekends and filters for gaps longer than 9 hours during weekdays.")
    print("It does NOT explicitly exclude official holidays, which might appear as 'missing trading days'.")
    print("A dynamic holiday calendar would be required for precise holiday exclusion.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python analyze_gaps.py <path_to_csv> <output_json_path>")
        sys.exit(1)
    
    csv_file_path = sys.argv[1]
    output_json_path = sys.argv[2]
    analyze_gaps(csv_file_path, output_json_path)