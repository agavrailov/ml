import pandas as pd
import sys
import os
import json

# Approximate US equity trading hours in UTC.
# Using 13:00-21:00 UTC covers both EDT (UTC-4, open 13:30) and EST (UTC-5, open 14:30)
# without requiring an external timezone library.
MARKET_OPEN_HOUR_UTC = 13
MARKET_CLOSE_HOUR_UTC = 21


def _gap_overlaps_trading_hours(start: pd.Timestamp, end: pd.Timestamp) -> bool:
    """Return True if the gap spans any part of regular US equity trading hours.

    Iterates over each calendar day covered by the gap and checks for overlap
    with the [MARKET_OPEN_HOUR_UTC, MARKET_CLOSE_HOUR_UTC) window.
    """
    day = start.normalize()
    while day <= end.normalize():
        session_open = day + pd.Timedelta(hours=MARKET_OPEN_HOUR_UTC)
        session_close = day + pd.Timedelta(hours=MARKET_CLOSE_HOUR_UTC)
        overlap_start = max(start, session_open)
        overlap_end = min(end, session_close)
        if overlap_end > overlap_start:
            return True
        day += pd.Timedelta(days=1)
    return False


def analyze_gaps(file_path, output_json_path):
    print(f"Analyzing potentially missing trading days in {file_path} (excluding weekends and typical overnight closures)...")
    try:
        df = pd.read_csv(file_path, parse_dates=['DateTime'])
    except Exception as e:
        print(f"Error reading CSV: {e}")
        # On read error, clear any existing gap JSON so stale gaps are not reused.
        try:
            with open(output_json_path, "w") as f:
                json.dump([], f, indent=4)
        except Exception:
            pass
        return

    if df.empty:
        print("DataFrame is empty, no gaps to analyze.")
        # Overwrite any existing output with an empty list to avoid stale gaps.
        with open(output_json_path, "w") as f:
            json.dump([], f, indent=4)
        return

    df.sort_values('DateTime', inplace=True)
    df.drop_duplicates(subset=['DateTime'], inplace=True)

    time_diffs = df['DateTime'].diff().dropna()

    long_gap_threshold = pd.Timedelta(hours=9)

    all_gaps = []
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

        if not is_weekend_gap and gap_duration > long_gap_threshold:
            all_gaps.append({
                'start': start_time.isoformat(),
                'end': end_time.isoformat(),
                'duration': str(gap_duration),
                'overlaps_trading_hours': _gap_overlaps_trading_hours(start_time, end_time),
            })

    real_gaps = [g for g in all_gaps if g['overlaps_trading_hours']]

    if not all_gaps:
        print("No weekday gaps > 9 hours found in the data.")
        with open(output_json_path, 'w') as f:
            json.dump([], f, indent=4)
    else:
        print(
            f"Found {len(all_gaps)} weekday gaps > 9 h "
            f"({len(real_gaps)} overlap trading hours "
            f"{MARKET_OPEN_HOUR_UTC}:00-{MARKET_CLOSE_HOUR_UTC}:00 UTC). "
            f"Details saved to {output_json_path}"
        )
        with open(output_json_path, 'w') as f:
            json.dump(all_gaps, f, indent=4)

    print(
        f"\nNote: All {len(all_gaps)} gaps are written to the JSON for forward-filling."
        f"\n      Only the {len(real_gaps)} that overlap trading hours indicate real data gaps."
    )


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python analyze_gaps.py <path_to_csv> <output_json_path>")
        sys.exit(1)

    csv_file_path = sys.argv[1]
    output_json_path = sys.argv[2]
    analyze_gaps(csv_file_path, output_json_path)
