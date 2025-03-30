import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_data(filepath):
    """
    Load treadmill data from a CSV file with headers.
    
    The CSV file is expected to have the following columns:
        - timestamp: UNIX time (in seconds)
        - distance: Measured distance in millimeters (mm)
        - speed: Running speed in millimeters per second (mm/sec); 0 indicates inactivity.
        
    Adjustments:
        - The distance is shifted so that the first sample is 0.
        - The timestamps are kept as datetime but can be converted to seconds relative to the start.
    
    Args:
        filepath (str): Path to the CSV file.
    
    Returns:
        pd.DataFrame: DataFrame with a datetime index.
    """
    df = pd.read_csv(filepath)
    # Convert timestamp from UNIX seconds to datetime.
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df.set_index('timestamp', inplace=True)
    # Adjust distance so that it starts at 0.
    df['distance'] = df['distance'] - df['distance'].iloc[0]
    return df

def fill_gaps(df):
    """
    Fill in time gaps by inserting rows for periods with no data.
    
    Only gaps larger than 1.5 times the median sampling interval are filled.
    Inserted rows have speed = 0 and the distance remains constant from the previous sample.
    
    Args:
        df (pd.DataFrame): Original data with columns ['distance', 'speed'].
    
    Returns:
        pd.DataFrame: DataFrame with gaps filled.
    """
    time_diffs = df.index.to_series().diff().dropna()
    if time_diffs.empty:
        return df
    median_dt = time_diffs.median()
    dt_sec = median_dt.total_seconds()
    
    filled_rows = []
    prev_time = df.index[0]
    prev_distance = df.iloc[0]['distance']
    prev_speed = df.iloc[0]['speed']
    filled_rows.append({'timestamp': prev_time, 'distance': prev_distance, 'speed': prev_speed})
    
    # Iterate through each subsequent row.
    for current_time, row in df.iloc[1:].iterrows():
        gap_sec = (current_time - prev_time).total_seconds()
        if gap_sec > 1.5 * dt_sec:
            n_missing = int(np.floor(gap_sec / dt_sec)) - 1
            for i in range(1, n_missing + 1):
                new_time = prev_time + pd.Timedelta(seconds=i * dt_sec)
                filled_rows.append({'timestamp': new_time, 'distance': prev_distance, 'speed': 0})
        filled_rows.append({'timestamp': current_time, 'distance': row['distance'], 'speed': row['speed']})
        prev_time = current_time
        prev_distance = row['distance']
    
    filled_df = pd.DataFrame(filled_rows)
    filled_df.set_index('timestamp', inplace=True)
    filled_df.sort_index(inplace=True)
    return filled_df

def compute_bouts(df):
    """
    A bout is defined as a contiguous period where 'speed' is nonzero.
    For each bout, the following are computed:
        - start_time_sec: Start time in seconds from the beginning
        - end_time_sec: End time in seconds from the beginning
        - duration_sec: Duration in seconds
        - distance_traveled (mm): Change in distance over the bout (in mm)
    
    Args:
        df (pd.DataFrame): Continuous data with columns ['distance', 'speed'].
    
    Returns:
        pd.DataFrame: Bout statistics.
    """
    bouts = []
    bout_start = None
    bout_start_distance = None
    
    for t, row in df.iterrows():
        if row['speed'] != 0:
            if bout_start is None:
                bout_start = t
                bout_start_distance = row['distance']
        else:
            if bout_start is not None:
                bout_end = t
                duration = (bout_end - bout_start).total_seconds()
                distance_traveled = row['distance'] - bout_start_distance
                bouts.append({
                    'start_time_sec': (bout_start - df.index[0]).total_seconds(),
                    'end_time_sec': (bout_end - df.index[0]).total_seconds(),
                    'duration_sec': duration,
                    'distance_traveled (mm)': distance_traveled
                })
                bout_start = None
                bout_start_distance = None
    # If a bout is still ongoing at the end, conclude it.
    if bout_start is not None:
        bout_end = df.index[-1]
        duration = (bout_end - bout_start).total_seconds()
        distance_traveled = df.iloc[-1]['distance'] - bout_start_distance
        bouts.append({
            'start_time_sec': (bout_start - df.index[0]).total_seconds(),
            'end_time_sec': (bout_end - df.index[0]).total_seconds(),
            'duration_sec': duration,
            'distance_traveled (mm)': distance_traveled
        })
    bouts_df = pd.DataFrame(bouts)
    return bouts_df

def plot_distance_over_time(df):
    """
    Plot the distance over time.
    
    The x-axis is represented in seconds from the start (first timestamp = 0).
    
    Args:
        df (pd.DataFrame): DataFrame with continuous data.
    """
    time_sec = (df.index - df.index[0]).total_seconds()
    plt.figure(figsize=(12, 4))
    plt.plot(time_sec, df['distance'], label='Distance (mm)')
    plt.xlabel('Time (sec)')
    plt.ylabel('Distance (mm)')
    plt.title('Distance Over Time (Time in sec, Distance in mm)')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_bouts(df, bouts_df):
    """
    Plot locomotion bouts over time by shading the regions of activity.
    
    The x-axis represents time in seconds from the start.
    
    Args:
        df (pd.DataFrame): Continuous data.
        bouts_df (pd.DataFrame): Bout statistics.
    """
    time_sec = (df.index - df.index[0]).total_seconds()
    plt.figure(figsize=(12, 4))
    plt.plot(time_sec, df['distance'], color='gray', label='Distance (mm)')
    
    for idx, bout in bouts_df.iterrows():
        plt.axvspan(bout['start_time_sec'], bout['end_time_sec'], color='orange', alpha=0.3,
                    label='Locomotion Bout' if idx == 0 else "")
    plt.xlabel('Time (sec)')
    plt.ylabel('Distance (mm)')
    plt.title('Locomotion Bouts Over Time (Time in sec, Distance in mm)')
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    """
    Main function:
      1. Loads the CSV data.
      2. Adjusts the data so that the distance starts at 0 and time is in seconds from the start.
      3. Displays descriptive statistics for the raw and gap-filled data.
      4. Computes running bouts.
      5. Generates plots for distance and locomotion bouts.
    """
    filepath = r"D:\jgronemeyer\240324_HFSA\data\sub-STREHAB02\ses-06\beh\20250329_165119_sub-STREHAB02_ses-06_task-widefield_treadmill_data.csv"
    
    # Load and adjust data.
    df = load_data(filepath)
    print("Raw Data Descriptive Statistics (Distance in mm, Speed in mm/sec):")
    print(df.describe())
    
    # Fill gaps where the time interval is large.
    filled_df = fill_gaps(df)
    print("\nFilled Data Descriptive Statistics (Distance in mm, Speed in mm/sec):")
    print(filled_df.describe())
    
    # Compute locomotion bouts.
    bouts_df = compute_bouts(filled_df)
    print("\nRunning Bout Statistics (Time in sec, Distance in mm):")
    print(bouts_df)
    
    # Plot the distance over time and the locomotion bouts.
    plot_distance_over_time(filled_df)
    plot_bouts(filled_df, bouts_df)

if __name__ == "__main__":
    main()

