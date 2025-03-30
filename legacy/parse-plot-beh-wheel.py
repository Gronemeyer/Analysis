# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 21:09:39 2024

@author: Jacob Gronemeyer
"""

# -*- coding: utf-8 -*-


#%% IMPORT IMPORT IMPORT IMPORT

import pandas as pd
import seaborn as sns
import numpy as np
import os
import matplotlib.pyplot as plt

#%% INITIALIZE INITIALIZE INITIALIZE INITIALIZE


DATA_DIR = r'C:\dev\sipefield-gratings\PsychoPy\data'
PROTOCOL = r'baseline'
SUBJECT = r'sub-SB03'
SESSION = r'ses-01'
BEHAVIOR = r'wheel_df.csv'

# Load the data
# Construct the file path
beh_path = os.path.join(DATA_DIR, PROTOCOL, SUBJECT, SESSION, 'beh')
anat_path = os.path.join(DATA_DIR, PROTOCOL, SUBJECT, SESSION, 'func')


# Parse the beh_path directory for a file ending with 'wheel_df.csv'
for file in os.listdir(beh_path):
    if file.endswith('wheel_df.csv'):
        file_path = os.path.join(beh_path, file)
        break
    if file.endswith('_frame_metadata.json'):
        frame_md_json = os.path.join(anat_path, file)
        break

# Create a pandas dataframe
wheel_df = pd.read_csv(f'{beh_path}\\{file}')


#%% PLOT PLOT PLOT PLOT PLOT

# Read the CSV file into a DataFrame

# Calculate the time difference in seconds
total_seconds = wheel_df['timestamp'].array[-1] - wheel_df['timestamp'][0] # Get Range
time = np.arange(0, total_seconds, 1) # create array [0,1,...12] with the total_seconds 

# Create separate plots for each variable
plt.figure(figsize=(10, 6), dpi=300)

# Plot 'speed' over time
plt.subplot(3, 1, 1)
plt.plot(wheel_df['timestamp'], wheel_df['speed'])
plt.title('Speed')
plt.xlabel('Time (secs)')
plt.ylabel('Speed')

# Plot 'distance' over time
plt.subplot(3, 1, 2)
plt.plot(wheel_df['timestamp'], wheel_df['distance'])
plt.title('Distance')
plt.xlabel('Time (secs)')
plt.ylabel('Distance')

# Plot 'direction' over time
plt.subplot(3, 1, 3)
plt.plot(wheel_df['timestamp'], wheel_df['direction'])
plt.title('Direction')
plt.xlabel('Time (secs)')
plt.ylabel('Direction')

# Adjust the layout
plt.tight_layout()

# Show the plots
plt.show()

#%% Reverse (flip) backwards data due to wrong encoder direction

wheel_df['speed'] = wheel_df['speed']*-1
wheel_df['distance'] = wheel_df['distance']*-1
