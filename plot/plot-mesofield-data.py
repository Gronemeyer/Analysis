# -*- coding: utf-8 -*-
"""
Purpose: generating mesoscopic mean fluorescence and wheel encoder speed plots
Created on Fri Jan 24 14:11:35 2025

@author: jgronemeyer

generating plots for the 241123_WH01 session 1 and session 3 data

- session 2 data was cut short (2000 frames) due to bug (now fixed) in acquisition 
"""
#%% IMPORT DEPENDENCIES
import pandas as pd
import matplotlib.pyplot as plt
import math
import os
import re
from collections import defaultdict
from datetime import datetime
import pprint

#%%
GLOBAL_FILE_PATTERNS = {
    "meso_tiff": {
        "regex": r"meso\.ome\.tiff$",
        "destination": ("meso", "tiff")
    },
    "meso_metadata": {
        "regex": r"meso\.ome\.tiff_frame_metadata\.json$",
        "destination": ("meso", "metadata")
    },
    "pupil_tiff": {
        "regex": r"pupil\.ome\.tiff$",
        "destination": ("pupil", "tiff")
    },
    "pupil_metadata": {
        "regex": r"pupil\.ome\.tiff_frame_metadata\.json$",
        "destination": ("pupil", "metadata")
    },
    "encoder_data": {
        "regex": r"encoder-data\.csv$",
        "destination": ("encoder",)
    },
    "psydat": {
        "regex": r"\.psydat$",
        "destination": ("psydat",)
    },
    "ses_config": {
        "regex": r"session_config\.json$",
        "destination": ("session_config",)
    },
    "dlc_pupil": {
        "regex": r"full\.pickle$",
        "destination": ("processed", "dlc_pupil")
    },
    "meso_trace": {
        "regex": r"meso-mean-trace\.csv$",
        "destination": ("processed", "meso_trace")
    },
}


def build_file_hierarchy(root_dir):
    """
    Creates a hierarchical database object of files within a directory hierarchy.
    """
    db = defaultdict(lambda: defaultdict(dict))

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            full_path = os.path.join(dirpath, filename)
            subject_match = re.search(r"sub-([A-Za-z0-9]+)", full_path)
            session_match = re.search(r"ses-([A-Za-z0-9]+)", full_path)
            subject = subject_match.group(1) if subject_match else "unknown"
            session = session_match.group(1) if session_match else "unknown"

            found_match = False
            for _, pattern_info in GLOBAL_FILE_PATTERNS.items():
                if re.search(pattern_info["regex"], filename):
                    dest = pattern_info["destination"]
                    # Ensure nested dicts exist if needed
                    if dest[0] not in db[subject][session]:
                        db[subject][session][dest[0]] = {}
                    if len(dest) == 2:
                        db[subject][session][dest[0]][dest[1]] = full_path
                    else:
                        db[subject][session][dest[0]] = full_path
                    found_match = True
                    break

            if not found_match:
                if 'other_func_files' not in db[subject][session]:
                    db[subject][session]['other'] = []
                record = {
                    "Filename": filename,
                    "Path": full_path,
                    "Directory": dirpath,
                    "Size (GB)": os.path.getsize(full_path) / (1024**3),
                    "Modified Date": datetime.fromtimestamp(os.path.getmtime(full_path)).strftime('%Y-%m-%d %H:%M:%S')
                }
                db[subject][session]['other'].append(record)

    return db


#%% EXPERIMENT DATA COLLECTION PROGRESS-SUMMARY DATAFRAME

def get_experiment_progress_summary(datadict):
    summary = []

    for subject, sessions in datadict.items():
        for session, categories in sessions.items():
            summary_entry = {
                'Subject': subject,
                'Session': session,
                'Mesoscopic TIFFs': len(categories.get('meso', {}).get('tiff', [])) if isinstance(categories.get('meso', {}).get('tiff'), list) else 1,
                'Pupil TIFFs': len(categories.get('pupil', {}).get('tiff', [])) if isinstance(categories.get('pupil', {}).get('tiff'), list) else 1,
                'Encoder Data Files': len(categories.get('encoder', [])) if isinstance(categories.get('encoder', []), list) else 1,
                'Other Files': len(categories.get('other', []))
            }
            summary.append(summary_entry)

    return pd.DataFrame(summary)

#summary_df.to_csv('data_summary.csv', index=False)

#%%
def load_psychopy_data(file_path):
    from psychopy.tools.filetools import fromFile
    return fromFile(file_path)
#%% IMPORT FUNCTIONS --------------------------------------------------------

def apply_filters(df, speed_col='Speed', clamp_negative=False, threshold=None,
                  smoothing='rolling_median', window_size=10, alpha=0.5):
    """
    Applies optional filtering/smoothing to a speed column in a DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing speed data.
    speed_col : str
        Name of the column with raw speed values.
    clamp_negative : bool
        If True, speeds < 0 are set to 0.
    threshold : float or None
        A value below which absolute speeds are set to 0. If None, no threshold filter is applied.
    smoothing : str
        The type of smoothing to apply. Options: 'rolling_mean', 'rolling_median', 'ewm', or None for no smoothing.
    window_size : int
        Window size for rolling operations.
    alpha : float
        Smoothing factor for exponential smoothing, between 0 and 1.
        
    Returns
    -------
    pd.DataFrame
        The DataFrame with additional 'Speed_filtered' column.
    """
    df['Speed_filtered'] = df[speed_col]

    # 1. Clamp negative speeds
    if clamp_negative:
        df['Speed_filtered'] = df['Speed_filtered'].clip(lower=0)

    # 2. Threshold near-zero speeds
    if threshold is not None:
        df.loc[df['Speed_filtered'].abs() < threshold, 'Speed_filtered'] = 0

    # 3. Apply smoothing
    if smoothing == 'rolling_mean':
        df['Speed_filtered'] = df['Speed_filtered'].rolling(window=window_size, center=True).mean()
    elif smoothing == 'rolling_median':
        df['Speed_filtered'] = df['Speed_filtered'].rolling(window=window_size, center=True).median()
    elif smoothing == 'ewm':
        df['Speed_filtered'] = df['Speed_filtered'].ewm(alpha=alpha).mean()

    # Fill any NaNs from rolling or ewm at start/end
    df['Speed_filtered'].bfill()
    df['Speed_filtered'].ffill()
    
    return df

def plot_session(
    session_name, 
    df_fluorescence, 
    df_encoder,
    df_pupil, 
    fluorescence_x='Slice', 
    fluorescence_y='Mean',
    speed_col='Speed_filtered',
    locomotion_threshold=0.001,
    downsample=10,
    x_limit=None
):
    """
    Plots a single session as a three-panel figure with x-axis in seconds:
      1) Fluorescence trace with an underlaid locomotion region
      2) Speed trace
      3) Pupil diameter trace

    The x-axis is computed from the 'runner_time_ms' column in df_fluorescence.
    Tick intervals are set to every 20 seconds.

    Parameters
    ----------
    session_name : str
        Name/title for the session (e.g. 'Session 1').
    df_fluorescence : pd.DataFrame
        DataFrame containing fluorescence data and a 'runner_time_ms' column.
    df_encoder : pd.DataFrame
        DataFrame containing the speed information and a 'runner_time_ms' column.
    df_pupil : pd.DataFrame
        DataFrame containing pupil diameter data and a 'runner_time_ms' column.
    fluorescence_x : str
        Unused in this modified version (x-axis is derived from 'runner_time_ms').
    fluorescence_y : str
        Column name for fluorescence intensities.
    speed_col : str
        Column name in df_encoder representing the speed to plot.
    locomotion_threshold : float
        Threshold for marking locomotion as active.
    downsample : int
        Downsampling factor for the speed trace.
    x_limit : tuple or None
        (min, max) for x-axis range in seconds, or None to automatically use
        0 to the total duration.
    """
    # Calculate time in seconds relative to start using df_fluorescence's runner_time_ms
    start_ms = df_fluorescence['runner_time_ms'].iloc[0]
    end_ms = df_fluorescence['runner_time_ms'].iloc[-1]
    total_time_sec = (end_ms - start_ms) / 1000.0
    
    # Create an array for x ticks at 20-second intervals
    x_ticks = list(range(0, int(total_time_sec) + 20, 20))
    
    # Create x vectors for each DataFrame (using relative time in seconds)
    x_fluo = (df_fluorescence['runner_time_ms'] - start_ms) / 1000.0
    x_encoder = (df_encoder['runner_time_ms'] - start_ms) / 1000.0
    x_pupil = (df_pupil['runner_time_ms'] - start_ms) / 1000.0

    # Identify bouts of locomotion based on threshold
    locomotion_mask = df_encoder[speed_col] > locomotion_threshold

    # Create figure
    fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(session_name)

    # -- Top subplot: Fluorescence trace with underlaid locomotion
    axs[0].plot(
        x_fluo, 
        df_fluorescence[fluorescence_y], 
        linestyle='-', 
        label='Mean Fluorescence'
    )
    axs[0].set_ylabel('Mean Fluorescence')
    axs[0].set_title(f'Fluorescence (with Locomotion Underlay) - {session_name}')
    axs[0].grid(True)
    axs[0].legend()
    axs[0].set_xlim(0, x_limit[1] if x_limit else total_time_sec)
    axs[0].set_xticks(x_ticks)

    # Underlay locomotion on top subplot
    axs[0].fill_between(
        x_encoder, 
        axs[0].get_ylim()[0], 
        axs[0].get_ylim()[1],
        where=locomotion_mask,
        color='gray',
        alpha=0.2,
        label='Locomotion'
    )

    # -- Second subplot: Speed
    axs[1].plot(
        x_encoder[::downsample], 
        df_encoder.iloc[::downsample][speed_col],
        linestyle='-',
        label='Speed (Filtered)'
    )
    axs[1].set_ylabel('Speed (m/s)')
    axs[1].set_title(f'Speed - {session_name}')
    axs[1].grid(True)
    axs[1].legend()
    if x_limit:
        axs[1].set_xlim(x_limit)
    else:
        axs[1].set_xlim(0, total_time_sec)
    axs[1].set_xticks(x_ticks)

    # -- Third subplot: Pupil Diameter
    axs[2].plot(
        x_pupil, 
        df_pupil['pupil_diameter_mm'], 
        label='Pupil Diameter (mm)', 
        color='green'
    )
    axs[2].set_xlabel('Time (s)')
    axs[2].set_ylabel('Pupil Diameter (mm)')
    axs[2].set_title(f'Pupil Diameter - {session_name}')
    axs[2].grid(True)
    axs[2].legend()
    axs[2].set_xlim(0, total_time_sec)
    axs[2].set_xticks(x_ticks)
    
    plt.tight_layout()
    return plt

# ---------------------------------------------------------------------------

# %% PUPIL DF
import pandas as pd
import numpy as np
import math
import statistics as st
import matplotlib.pyplot as plt

def euclidean_distance(coord1, coord2):
    """Calculate the Euclidean distance between two points."""
    return math.dist(coord1, coord2)

def confidence_filter_coordinates(frames_coords, frames_conf, threshold):
    """
    Apply a boolean label to coordinates based on whether 
    their confidence exceeds `threshold`.
    
    Parameters
    ----------
    frames_coords : list
        List of numpy arrays containing pupil coordinates for each frame.
    frames_conf : list
        List of numpy arrays containing confidence values corresponding 
        to the coordinates in `frames_coords`.
    threshold : float
        Confidence cutoff.

    Returns
    -------
    list
        A list of [coords, conf, labels] for each frame, where 'labels' 
        is a list of booleans (True if above threshold, else False).
    """
    thresholded = []
    for coords, conf in zip(frames_coords[1:], frames_conf[1:]):
        frame_coords, frame_conf, frame_labels = [], [], []
        # Each frame has 8 sets of pupil points 
        for i in range(8):
            point = coords[0, i, 0, :]
            cval = conf[i, 0, 0]
            label = (cval >= threshold)
            frame_coords.append(point)
            frame_conf.append(cval)
            frame_labels.append(label)
        thresholded.append([frame_coords, frame_conf, frame_labels])
    return thresholded

def process_deeplabcut_pupil_data(
    pickle_path: str,
    show_plot: bool = False,
    confidence_threshold: float = 0.1,
    pixel_to_mm: float = 53.6
) -> pd.DataFrame:
    """
    Load a DeepLabCut output pickle file and compute the pupil diameter per frame.
    
    Parameters
    ----------
    pickle_path : str
        Path to the DLC output pickle file (e.g., '*full.pickle').
    show_plot : bool, optional
        If True, displays a matplotlib plot of pupil diameter (in mm) across frames.
        Defaults to False.
    confidence_threshold : float, optional
        Minimum confidence required to include two landmarks in the diameter calculation.
        Defaults to 0.1.
    pixel_to_mm : float, optional
        Conversion factor from pixels to millimeters. Defaults to 53.6.
    
    Returns
    -------
    pd.DataFrame
        A DataFrame with one column ('pupil_diameter_mm') indexed by frame number.
        Frames for which no valid diameter could be calculated will have NaN values.
    """
    
    # 1) Load the raw dataframe from the pickle
    data = pd.read_pickle(pickle_path)
    raw_df = pd.DataFrame(data = data)

    # 2) Convert each column's 'coordinates' & 'confidence' to arrays
    frame_coordinates_array = []
    frame_confidence_array = []
    for frame_column in raw_df.columns:
        coords_list = raw_df.at['coordinates', frame_column]
        conf_list = raw_df.at['confidence', frame_column]
        frame_coordinates_array.append(np.array(coords_list))
        frame_confidence_array.append(np.array(conf_list))
    
    # 3) Filter coordinates by confidence
    labeled_frames = confidence_filter_coordinates(
        frame_coordinates_array,
        frame_confidence_array,
        confidence_threshold
    )
    
    # 4) Calculate mean pupil diameter (in pixels) per frame
    pupil_diameters = []
    for frame_data in labeled_frames:
        coords, conf, labels = frame_data
        frame_diameters = []
        
        # Pairs: (0,1), (2,3), (4,5), (6,7)
        for i in range(0, 7, 2):
            if labels[i] and labels[i+1]:
                diameter_pix = euclidean_distance(coords[i], coords[i+1])
                frame_diameters.append(diameter_pix)
        
        # If multiple diameters exist, use the average
        if len(frame_diameters) > 1:
            pupil_diameters.append(st.mean(frame_diameters))
        else:
            pupil_diameters.append(np.nan)
    
    # 5) Convert diameters to Series and interpolate missing values
    diam_series = pd.Series(pupil_diameters).interpolate()
    
    # 6) Convert from pixels to mm
    diam_series = diam_series / pixel_to_mm
    
    # 7) Optionally plot the results
    if show_plot:
        plt.figure(dpi=300)
        plt.plot(diam_series, color='blue')
        plt.xlabel('Frame')
        plt.ylabel('Pupil Diameter (mm)')
        plt.title('Pupil Diameter Over Frames')
        plt.show()
    
    # 8) Return a DataFrame with the final diameters
    result_df = pd.DataFrame({'pupil_diameter_mm': diam_series})
    return result_df


# %%

def load_camera_metadata(metadata_path):
    import json
    
    # Load the JSON Data
    with open(metadata_path, 'r') as file:
        data = json.load(file)
    
    p0_data = data['p0'] #p0 is a list of the frames at Position 0 \
                            #(artifact of hardware sequencing in MMCore)
    df = pd.DataFrame(p0_data) # dataframe it

    # Expand 'camera_metadata' into separate columns
    camera_metadata_df = pd.json_normalize(df['camera_metadata'])
    df = df.join(camera_metadata_df)
    return df
    

#%% LOAD DATA ---------------------------------------------------------------

datadict = build_file_hierarchy(r"D:\jgronemeyer\250130_STREHAB_prelim")
pprint.pprint(datadict, indent=1, compact=True)

get_experiment_progress_summary(datadict)

subject_id = 'STREHAB01'
session_id = '01'

meso_df = pd.read_csv(datadict[subject_id][session_id]['processed']['meso_trace'])

encoder_df = pd.read_csv(datadict[subject_id][session_id]['encoder'])
#%% APPLY OPTIONAL FILTERS --------------------------------------------------
encoder_df = apply_filters(encoder_df, 
                           speed_col='Speed', 
                           threshold=0.001, 
                           smoothing='ewm', 
                           window_size=5)

pupil_df = process_deeplabcut_pupil_data(pickle_path=datadict[subject_id][session_id]['processed']['dlc_pupil'],
                                         show_plot=False,
                                         confidence_threshold=0.5)


meso_metadata = load_camera_metadata(datadict[subject_id][session_id]['meso']['metadata'])
pupil_metadata = load_camera_metadata(datadict[subject_id][session_id]['pupil']['metadata'])

meso_df = meso_df.join(meso_metadata)
meso_df = meso_df.join(encoder_df['Speed']).rename(columns={'Speed': 'encoder_speed'})
pupil_df = pupil_df.join(pupil_metadata)

#%% DATA ALIGNMENT ----------------------------------------------------------
'''
These time values do not line up exactly when compared at the sub millisecond level. 
The `meso_df` has more items than the `pupil_df`. The first value of `TimeReceivedByCore` in the `pupil_df` 
should be the same as the first value in the `meso_df`. This may not always be the case, so determine 
the start point for cutoff using the larger value. The last value in the `TimeReceivedByCorein` 
the `meso_df` __always__ represents the last timepoint. Therefore, remove the frames from pupil_df that 
have a timestamp later than the last value in the `meso_df`. Then, downsample `meso_df` by matching frame 
values captured at the same time as thepupil_df`. There are no exact matches, so you will need to 
round the number conservatively
'''



#plot.savefig(svg_path, format='svg')
# %%

if __name__ == "__main__":
    datadict = build_file_hierarchy(r"D:\jgronemeyer\250130_STREHAB_prelim")
    #pprint.pprint(datadict, indent=1, compact=True)

    get_experiment_progress_summary(datadict)

    subject_id = 'STREHAB01'
    session_id = '04'

    meso_df = pd.read_csv(datadict[subject_id][session_id]['processed']['meso_trace'])

    encoder_df = pd.read_csv(datadict[subject_id][session_id]['encoder'])

    #APPLY OPTIONAL FILTERS --------------------------------------------------
    encoder_df = apply_filters(encoder_df, 
                            speed_col='Speed', 
                            threshold=0.01, 
                            smoothing='median', 
                            window_size=5)

    pupil_df = process_deeplabcut_pupil_data(pickle_path=datadict[subject_id][session_id]['processed']['dlc_pupil'],
                                            show_plot=False,
                                            confidence_threshold=0.5)


    meso_metadata = load_camera_metadata(datadict[subject_id][session_id]['meso']['metadata'])
    pupil_metadata = load_camera_metadata(datadict[subject_id][session_id]['pupil']['metadata'])

    meso_df = meso_df.join(meso_metadata)
    meso_df = meso_df.join(encoder_df['Speed']).rename(columns={'Speed': 'encoder_speed'})
    pupil_df = pupil_df.join(pupil_metadata)

    # DATA ALIGNMENT ----------------------------------------------------------

    start_ts = meso_df['runner_time_ms'].iloc[0]
    meso_end_ts = meso_df['runner_time_ms'].iloc[-1]
    print(f"Start: {start_ts}, Mesoscopic End (cutoff): {meso_end_ts}")
    pupil_df = pupil_df[pupil_df['runner_time_ms'] <= meso_end_ts].copy().reset_index(drop=True)


    # PLOT ---------------------------------------------------------------------
    save_dir=r"D:\jgronemeyer\241220_checkerboard_experiment\processed\WH01\meso"
    svg_path=f"{save_dir}/{subject_id}_{session_id}_plot.svg"

    title = f"241220 ETOH Spontaneous | {subject_id} - {session_id} | CaMKII-GCaMP8s"
    plot = plot_session(
            session_name=f"{title}",
            df_fluorescence=meso_df,
            df_encoder=meso_df,
            df_pupil=pupil_df,
            fluorescence_x='Slice', 
            fluorescence_y='Mean',
            speed_col='encoder_speed',
            locomotion_threshold=0.03,
            downsample=5, 
            x_limit=(0, len(meso_df))  # adjust as desired, or None
        )
    plot.show()
# %%
# import deeplabcut as dlc

# experiment_dir = r"D:\sbaskar\241220_etoh-checkerboard"
# config_path = r"C:\Users\SIPE_LAB\Desktop\Fa24-pilot-habituation-JG-2025-01-15\config.yaml"
# videos = [os.path.join(experiment_dir, f) for f in os.listdir(experiment_dir) if f.lower().endswith('.mp4')]
# destination_dir = os.path.join(os.path.dirname(experiment_dir), subject_id, "dlc_pupil")
# if not os.path.exists(destination_dir):
#     os.makedirs(destination_dir)
# dlc.analyze_videos(config_path, videos, videotype='mp4', shuffle=1, save_as_csv=False, destfolder=destination_dir, dynamic=(True, .5, 10))