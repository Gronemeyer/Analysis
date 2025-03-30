
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


def load_wheel_data(directory) -> pd.DataFrame:

    # Parse the beh_path directory for a file ending with 'wheel_df.csv'
    path = os.path.join(os.path.dirname(directory), 'beh')
    for file in os.listdir(path):
        if file.endswith('wheeldf.csv'):
            file_path = os.path.join(path, file)
            break
    if file_path is None:
        raise FileNotFoundError(f"No file ending with 'wheeldf.csv' found in {path}.")
    df = pd.read_csv(file_path)
    # Create a pandas dataframe
    return df

def load_psychopy_data(directory):

    # Parse the beh_path directory for a file ending with '.csv'
    path = os.path.join(os.path.dirname(directory), 'beh')
    for file in os.listdir(path):
        if file.endswith('.csv'):
            file_path = os.path.join(path, file)
            break
    if file_path is None:
        raise FileNotFoundError(f"No file ending with '.csv' found in {path}.")
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)

    # Display the first few rows of the DataFrame
    print(df.head())

    # List the columns from the DataFrame
    print(df.columns)
    
    return df

def load_roi_data(directory) -> pd.DataFrame:

    # Parse the directory for a file ending with 'roi_dff_data.csv'
    path = os.path.join(os.path.dirname(directory), 'beh')
    for file in os.listdir(path):
        if file.endswith('roi_dff_data.csv'):
            file_path = os.path.join(path, file)
            break
    if file_path is None:
        raise FileNotFoundError(f"No file ending with 'roi_dff_data.csv' found in {path}.")
    # Load the CSV into a pandas DataFrame
    df = pd.read_csv(file_path)

        # Display the first few rows of the DataFrame
    print(df.head())

    # List the columns from the DataFrame
    print(df.columns)
    
    return df


def plot_data(wheel_df: pd.DataFrame, stim_df: pd.DataFrame, roi_df: pd.DataFrame):
    # Calculate the time difference in seconds
    total_seconds = wheel_df['timestamp'].array[-1] - wheel_df['timestamp'][0]  # Get Range
    time = np.arange(0, total_seconds, 1)  # create array [0,1,...12] with the total_seconds

    # Create separate plots for each variable
    plt.figure(figsize=(12, 10))#, dpi=300)
    plt.suptitle('SB03-SALINE')

    # Plot 'ROI fluorescence' over time
    plt.subplot(4, 1, 1)
    plt.plot(roi_df)
    plt.title('ROI Fluorescence (ΔF/F)')
    plt.xlabel('Time (Frames)')
    plt.ylabel('ΔF/F')
    # for start_time in stim_df['stim_grayScreen.started']:
    #     plt.axvline(x=start_time, color='red', linestyle='--', label='stim_grayScreen.started')
    # for start_time in stim_df['stim_grating.started']:
    #     plt.axvline(x=start_time, color='green', linestyle='--', label='stim_grating.started')
    

    # Plot 'speed' over time
    plt.subplot(4, 1, 2)
    plt.plot(wheel_df['timestamp'], wheel_df['speed'])
    plt.title('Speed')
    plt.xlabel('Time (secs)')
    plt.ylabel('Speed')
    for start_time in stim_df['stim_grayScreen.started']:
        plt.axvline(x=start_time, color='red', linestyle='--', label='stim_grayScreen.started')
    for start_time in stim_df['stim_grating.started']:
        plt.axvline(x=start_time, color='green', linestyle='--', label='stim_grating.started')

    # Plot 'distance' over time
    plt.subplot(4, 1, 3)
    plt.plot(wheel_df['timestamp'], wheel_df['distance'])
    plt.title('Distance')
    plt.xlabel('Time (secs)')
    plt.ylabel('Distance')
    for start_time in stim_df['stim_grayScreen.started']:
        plt.axvline(x=start_time, color='red', linestyle='--', label='stim_grayScreen.started')
    for start_time in stim_df['stim_grating.started']:
        plt.axvline(x=start_time, color='green', linestyle='--', label='stim_grating.started')

    # Plot 'direction' over time
    plt.subplot(4, 1, 4)
    plt.plot(wheel_df['timestamp'], wheel_df['direction'])
    plt.title('Direction')
    plt.xlabel('Time (secs)')
    plt.ylabel('Direction')
    for start_time in stim_df['stim_grayScreen.started']:
        plt.axvline(x=start_time, color='red', linestyle='--', label='stim_grayScreen.started')
    for start_time in stim_df['stim_grating.started']:
        plt.axvline(x=start_time, color='green', linestyle='--', label='stim_grating.started')

    # Adjust the layout
    plt.tight_layout()

    # Show the plots
    plt.show()



def main():
    path=r"C:\dev\sipefield-gratings\PsychoPy\data\low\sub-SB03\ses-01\beh"

    wheeldf=load_wheel_data(path)
    psychodf=load_psychopy_data(path)
    roidf=load_roi_data(path)

    plot_data(wheel_df=wheeldf, stim_df=psychodf, roi_df=roidf)
if __name__ == "__main__": 
    main()
# %%
