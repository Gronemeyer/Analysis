'''
Purpose: DeepLabCut output analysis for pupil diameter 
Author: Sindhuja Baskar
Contributors: Jake Gronemeyer

This script will:
- Load a DeepLabCut output pickle file,
- Extract the pupil coordinates and confidence values
- Filter the coordinates based on confidence threshold
- Calculate the Euclidean distance between each pair of coordinates
- Calculate the mean diameter for each frame
- Convert the mean diameter to mm
- Plot the pupil diameters over frames
- Plot the individual frame coordinates

'''
#%% GET LIBRARIES

import pandas as pd
import numpy as np
import math
import statistics as st
from pathlib import Path
import matplotlib.pyplot as plt


pathlist = Path(r'C:\Users\SIPE_LAB\Desktop\Fa24-pilot-habituation-JG-2025-01-15\videos').glob(r'*full.pickle')
DATA_PATH = r"C:\Users\SIPE_LAB\Desktop\Fa24-pilot-habituation-JG-2025-01-15\videos\habituation-sub-WH01_ses-01_task-widefield_pupil.omeDLC_Resnet50_Fa24-pilot-habituationJan15shuffle1_snapshot_010_full.pickle"
SAVE_CSV = r"D:\jgronemeyer\241220_checkerboard_experiment\processed"

#%% GET FILE LIST

# 241022 Note- this does not currently list all full.pickle files SB

data_file_list = []
for file in pathlist:
    file_path = str(file)
    data_file_list.append(file_path)

#%% LOAD FUNCTIONS

def load_df_from_file(path=DATA_PATH):
    """ Load a DataFrame from a pickle file """
    unpickled_data = pd.read_pickle(path) #unpickles full.pickle file
    print('loading file...')
    new_df = pd.DataFrame(data = unpickled_data)
    return new_df

def euclidean_distance(coord1, coord2):
    """ Calculate the Euclidean distance between two points """
    return math.dist(coord1, coord2)

#%% LOAD DATA

raw_dataframe = load_df_from_file(DATA_PATH) # new_df = whatever load_df_from_file 'returns'
print(raw_dataframe)


#%% SLICE FRAME COORDINATES AND CONFIDENCE INTO LIST OF ARRAYS

# Initialize an empty list to store the coordinate and confidence arrays
frame_coordinates_array = []
frame_confidence_array = []

# Iterate through each column in the DataFrame
for frame in raw_dataframe.columns:
    # Extract the 'coordinates' row from the current column
    coordinates_list = raw_dataframe.at['coordinates', frame]
    
    # Extract the 'confidence' row from the current column
    confidence_list = raw_dataframe.at['confidence', frame]
    
    # Convert the extracted data to a numpy array
    coordinates_array = np.array(coordinates_list)
    confidence_array = np.array(confidence_list)
    
    # Append the numpy array to the list of coordinate arrays
    frame_coordinates_array.append(coordinates_array)
    
    # Append the numpy array to the list of coordinate arrays
    frame_confidence_array.append(confidence_array)

# Now frame_''_array contains the list of arrays for each frame
print(frame_coordinates_array)
print(frame_confidence_array)


#%% LABEL COORDINATES BASED ON CONFIDENCE 

def confidence_filter_coordinates(frame_coordinates_array, frame_confidence_array, threshold):
    
    #Initialize an empty list to store the threshold-labeled list of coordinates
    thresholded_frame_coordinates = []
    
    #Zip the coordinate and its confidence into a pair, skipping the first item in the list (null metadata)
    for coordinates, confidence in zip(frame_coordinates_array[1:], frame_confidence_array[1:]): 
        
        # Initialize lists to store coords, conf, and label for current frame
        frame_coords = []
        frame_conf = []
        frame_label = []
        
        # Per frame, iterate through zip pairs 
        for i in range(8):
            
            # Get coordinate and confidence value
            coord = coordinates[0,i,0,:]
            conf = confidence[i,0,0]
            
            # Assign True/False boolean label to each coordinate based on confidence criteria
            label: bool = False
            if conf >= threshold:
                label = True 
                
            # Append the filtered frame values for coords, conf, and label
            frame_coords.append(coord)
            frame_conf.append(conf)
            frame_label.append(label)
            
        # Append the list of filtered coordinates with bool label
        thresholded_frame_coordinates.append([frame_coords, frame_conf, frame_label])
                
    return thresholded_frame_coordinates

threshold = 0.1

# Cast list of thresholded coordinates to a DataFrame
labeled_frames = confidence_filter_coordinates(frame_coordinates_array, frame_confidence_array, threshold)
print(labeled_frames)

#%% AVERAGE DIAMETER FOR EACH FRAME
## TODO: This needs to be a function
# Initialize an empty list to store the averaged diameter for each frame
pupil_diameters = []
# Iterate through each array of coordinates of each frame
for frame in labeled_frames: 
    # Initialize an empty list to store the diameters for the current frame
    frame_diameters = []
    
    # In the current frame iterate through each pair of coordinates
    for i in range(0, 7, 2): # 0, 2, 4, 6 results in (x_1, y_1) paired with (x_2, y_2), (x_3, y_3) and (x_4, y_4), etc.
        # Set conditional that both coordinates must have True labels to be included in diameter calculation
        if frame[2][i] and frame[2][i+1]:
            # Calculate the Euclidean distance between each coordinate pair using our custom euclidean_distance function
            diameter = euclidean_distance(frame[0][i], frame[0][i+1])
           # Append the calculated diameter to the list of diameters for the current frame
            frame_diameters.append(diameter)
        
    # Calculate mean if current frame has more than one diameter
    if len(frame_diameters) > 1:
        mean_diameter = st.mean(frame_diameters)
        
    else:
        # If frame has less than one diameter, mean diameter is appended with NaN value
        mean_diameter = None

    # Append the mean diameter to the list of diameters for all frames
    pupil_diameters.append(mean_diameter)
    
# Convert mean pupil diameter to Pandas Series
pupil_diameters = pd.Series(pupil_diameters)
# Use Linear Interpolation to fill in mean diameter for excluded frames
pupil_diameters.interpolate()
    
# Now diameters contains the list of distances for each frame
print(pupil_diameters)

#%% CONVERT PUPIL DIAMETERS TO MM
#1mm is 53.6 pixels; use this conversion factor for converting pixels to mm
for i in range(len(pupil_diameters)):
    pupil_diameters[i] = pupil_diameters[i] / 53.6
print(pupil_diameters)

#%% PLOT PUPIL DIAMETERS

# Set the DPI for the plot
plt.figure(dpi=300)
color = 'blue'  # You can change this to any color you like

# Plot the pupil diameters with the specified color
plt.plot(pupil_diameters, color=color)
plt.xlabel('Frame')
plt.ylabel('Diameter (mm)')
plt.show()

#%%
import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Scatter(y=pupil_diameters, mode='lines', name='Pupil Diameter'))
fig.update_layout(
    title='Pupil Diameters Over Frames',
    xaxis_title='Frame',
    yaxis_title='Diameter (mm)',
    hovermode='closest'
)
fig.show()
#%% PLOT INDIVIDUAL FRAME COORDINATES

# get one frame's coordinates and cast list to a numpy array
def plot_frame_coordinates(raw_dataframe: pd.DataFrame, frame_number: int):
    one_coord_frame = raw_dataframe.iloc[8, frame_number] # get one frame's coordinates
    coords = one_coord_frame[0] # get the list of arrays from the Tuple
    coords = np.array(coords) # cast the list to a Numpy array

    # Initialize the plot
    plt.figure(dpi=300)
    color = 'red'

    # Plot the x and y coordinates of the pupil
    plt.scatter(coords[:, 0, 0], coords[:, 0, 1], color=color)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
#%%
import plotly.express as px
def plot_frame_coordinates(raw_dataframe: pd.DataFrame, frame_number: int):
    one_coord_frame = raw_dataframe.iloc[8, frame_number] # get one frame's coordinates
    coords = one_coord_frame[0] # get the list of arrays from the Tuple
    coords = np.array(coords) # cast the list to a Numpy array

    # Create a DataFrame for Plotly
    df = pd.DataFrame({
        'X': coords[:, 0, 0],
        'Y': coords[:, 0, 1]
    })

    # Create an interactive scatter plot
    fig = px.scatter(df, x='X', y='Y', title=f'Frame {frame_number} Coordinates',
                        hover_data=['X', 'Y'])
    fig.update_layout(dragmode='pan', hovermode='closest')
    fig.show()


#%% TODO

#TODO Linear interpolation or qubic spline interpolation
#TODO Integrate the timestamps for the frames
#TODO Plot labeled coordinate points on a grid defined by the pixels of the video frames
#TODO Use common interesection point as a means to calculate pupil movement