# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 14:21:13 2024

@author: Jake Gronemeyer
"""

# %%
import pandas as pd
import os
import matplotlib.pyplot as plt

dhyana_path = r"C:\dev\dh.tif"
thor_path = r'D:\jgronemeyer\Experiment\data\habituation\sub-WH01\ses-01\func\habituation-sub-WH01_ses-01_task-widefield'

#data_path = thor_path
data_path = dhyana_path

meta_ext = '_frame_metadata.json'
thor_json_path = os.path.join(data_path, meta_ext)

df_json = pd.read_json(thor_json_path)
#df_json = pd.read_json(dhyana_json_path)

# Display the first few rows of the DataFrame
print(df_json.head())
print(df_json.columns)
plt.plot(range(len(df_json.T.index)),df_json.T['ElapsedTime-ms'].values)

#%% Extract 'TimeReceivedByCore' and 'Time' columns and plot from Pandas
time_received = pd.to_datetime(df_json.T['TimeReceivedByCore'])
time = pd.to_datetime(df_json.T['Time'])
time_received.plot()
time.plot()

#%% Create a new DataFrame for detailed plotting
time_df = pd.DataFrame({
    'Index': range(len(time_received)),
    'TimeReceivedByCore': time_received,
    'Time': time
})

#%% Plot using matplotlib

plt.figure(figsize=(10, 6))
plt.plot(time_df['Index'], time_df['TimeReceivedByCore'], label='TimeReceivedByCore')
plt.plot(time_df['Index'], time_df['Time'], label='Time')
plt.xlabel('Index')
plt.ylabel('Time')
plt.title('TimeReceivedByCore and Time over Index')
plt.legend()
plt.show()

# %%

# %% Pandas Handling
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#============= Filename ================#
path = r"D:\jgronemeyer\Experiment\data\habituation\sub-WH01\ses-06\func\WH01-thor_frame_metadata.json"
#=======================================#

# Define a threshold for divergence (e.g., 1 ms)
threshold = 1  # milliseconds

# Load the JSON Data
with open(path, 'r') as file:
    data = json.load(file)

# =======================Extract Data and Create DataFrame ================== #

p0_data = data['p0'] #p0 is a list of the frames at Position 0 \
                        #(artifact of hardware sequencing in MMCore)
df = pd.DataFrame(p0_data) # dataframe it

# Expand 'camera_metadata' into separate columns
camera_metadata_df = pd.json_normalize(df['camera_metadata'])
df = df.join(camera_metadata_df)

df['TimeReceivedByCore'] = pd.to_datetime(df['TimeReceivedByCore'], 
                                          format='%Y-%m-%d %H:%M:%S.%f') # Convert to datetime

df['runner_time_ms'] = df['runner_time_ms'].astype(float) # Convert to float

# ==================== Compute Time Intervals Between Frames ================= #

# Sort the DataFrame by 'TimeReceivedByCore' if needed
df = df.sort_values('TimeReceivedByCore').reset_index(drop=True)

df['runner_interval'] = df['runner_time_ms'].diff() # Compute differential

df['time_received_ms'] = df['TimeReceivedByCore'].astype(np.int64) / 1e6  # Convert to milliseconds
df['core_interval'] = df['time_received_ms'].diff() #Compute differential

# Compute Differences Between Intervals
df['interval_difference'] = df['runner_interval'] - df['core_interval']

# ========================== Identify Divergences ========================== #

# Identify where the absolute difference exceeds the threshold
df['divergence'] = df['interval_difference'].abs() > threshold

# ================= Plot the Intervals and Differences ====================== #
plt.figure(figsize=(12, 10))

# ----------- Runner Time Intervals and Core Time Interval Plot 1
plt.subplot(3, 1, 1)
plt.plot(df.index, df['runner_interval'], label='Runner Time Intervals', marker='o')
plt.plot(df.index, df['core_interval'], label='Core Time Intervals', marker='x')
plt.xlabel('Frame Index')
plt.ylabel('Interval (ms)')
plt.title('Intervals Between Frames')
plt.legend()
plt.grid(True)

# Highlighting divergence points
for idx in df[df['divergence']].index:
    plt.axvline(x=idx, color='red', linestyle='--', alpha=0.5)

# ----------- Difference Between Intervals Plot 2
plt.subplot(3, 1, 2)
plt.plot(df.index, df['interval_difference'], label='Interval Difference (Runner - Core)', marker='d')
plt.xlabel('Frame Index')
plt.ylabel('Interval Difference (ms)')
plt.title('Difference Between Runner and Core Intervals')
plt.axhline(y=threshold, color='red', linestyle='--', alpha=0.5, label='Threshold')
plt.axhline(y=-threshold, color='red', linestyle='--', alpha=0.5)
plt.legend()
plt.grid(True)

# Highlight divergence points
for idx in df[df['divergence']].index:
    plt.axvline(x=idx, color='red', linestyle='--', alpha=0.5)

# ----------- Cumulative Time Comparison Plot 3
plt.subplot(3, 1, 3)
df['cumulative_runner_time'] = df['runner_interval'].cumsum()
df['cumulative_core_time'] = df['core_interval'].cumsum()
plt.plot(df.index, df['cumulative_runner_time'], label='Cumulative Runner Time', marker='o')
plt.plot(df.index, df['cumulative_core_time'], label='Cumulative Core Time', marker='x')
plt.xlabel('Frame Index')
plt.ylabel('Cumulative Time (ms)')
plt.title('Cumulative Time Comparison')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Output Divergence Points if any
divergence_points = df[df['divergence']]
if not divergence_points.empty:
    print("Divergence detected at frame indices (0-based):")
    for idx, row in divergence_points.iterrows():
        print(f"Between Frame {idx} and Frame {idx+1}: Interval difference = {row['interval_difference']:.2f} ms")
else:
    print("No significant divergence detected.")


# Calculate the average framerate
total_frames = len(df)
total_time_seconds = df['runner_time_ms'].iloc[-1] / 1000  # Convert to seconds
average_framerate = total_frames / total_time_seconds

# Print the average framerate
print(f"Average framerate: {average_framerate:.2f} frames per second")

# Calculate the total time of the capture in seconds
total_time_seconds = df['runner_time_ms'].iloc[-1] / 1000  # Convert to seconds
total_time_minutes = total_time_seconds / 60  # Convert to minutes

# Print the total time
print(f"Total time of the capture: {total_time_seconds:.2f} seconds ({total_time_minutes:.2f} minutes)")
# %%
