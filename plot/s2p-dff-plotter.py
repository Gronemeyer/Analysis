# %% GET LIBRARIES
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import csv

# %% DEFINE FUNCTIONS

def load_suite2p_outputs(directory_path): # this will load all Suite2p output files from the specified directory
    file_dict = {
        'roi_fluorescence': '*F.npy',
        'neuropil_fluorescence': '*Fneu.npy',
        'cell_identifier' : '*iscell.npy',
        'intermediate_outputs' : '*ops.npy',
        'roi_traces' : '*spks.npy',
        'roi_statistics' : '*stat.npy'}
    loaded_data_files = {}
    pathlist = Path(directory_path)
    for key, value in file_dict.items():
        suite2p_files = list(pathlist.glob(value))
        for file in suite2p_files:
            loaded_data_files[key] = np.load(file, allow_pickle = True)
    return loaded_data_files

def calculate_dff(raw, baseline): # this will calculate the dff values for each roi
    dff_list = []
    for row in range(len(raw)):
        dff = ((raw[row] - baseline[row])/(baseline[row]))
        dff_list.append(dff)
    return dff_list

def calculate_baseline(raw_fluorescence, percentile): # this will calculate the specified percentile along each roi's raw fluorescence
    percentile_list = []
    for row in raw_fluorescence:
        find_percentile = np.percentile(row, percentile, keepdims = True)
        percentile_list.append(find_percentile)
    return percentile_list

# %% LOAD DATA
suite2p_data_output = load_suite2p_outputs(r'C:\dev\2p-analysis\suite2p\SB03\tiff\sb03_baseline\suite2p\plane0')
    
# %% FILTER ROIs (CELLS ONLY)

# assign bool T/F based on confidence index
true_cells_only = suite2p_data_output['cell_identifier'][:,0].astype(bool)
# print(true_cells_only)

# filter ROIs and neuropil based on bool value
filtered_roi = np.array(suite2p_data_output['roi_fluorescence'][true_cells_only])
filtered_neuropil = np.array(suite2p_data_output['neuropil_fluorescence'][true_cells_only])

# %% NEUROPIL SUBTRACTION

neuropil_subtracted_roi = (filtered_roi - (filtered_neuropil)) #TODO: necessary?

#%% PLOT ROIS
# plt.plot(neuropil_subtracted_roi[2])
# plt.title('Raw Fluorescence of ROI')
# plt.xlabel('frames')
# plt.ylabel('raw fluorescence')
# plt.show()

# %% CALCULATE BASELINE FLUORESCENCE
baseline_fluorescence = calculate_baseline(filtered_roi, percentile = 10)
#print(baseline_fluorescence)

# %% CALCULATE DF/F
roi_dff = calculate_dff(filtered_roi, baseline_fluorescence)
# print(roi_dff)

# %% PLOTTING DFF 
# plt.plot(roi_dff[2])
# plt.title('Neuronal Ca2+ activity across session')
# plt.xlabel('Frames')
# plt.ylabel('df/f')
# plt.show()

# %% EXPORT TO CSV

# def export_dff_to_csv(dff_data, output_path):
#     # Convert the list of ΔF/F arrays into a DataFrame
#     dff_df = pd.DataFrame(dff_data).transpose()
    
#     # Rename columns to indicate ROI numbers
#     dff_df.columns = [f"ROI_{i}" for i in range(len(dff_data))]
    
#     # Export to CSV
#     dff_df.to_csv(output_path, index=False)
#     print(f"ΔF/F data successfully exported to {output_path}")
# # %%
# # Specify the output file path
# output_csv_path = r'C:\dev\2p-analysis\suite2p\SB03\tiff\sb03_baseline\suite2p\plane0\roi_dff_data.csv'

# # Export the ΔF/F data
# export_dff_to_csv(roi_dff, output_csv_path)

# %% CALCULATE AVERAGE DFF
def calculate_average_roi(dff_data):
    """
    Calculate the average ΔF/F across all ROIs at each timepoint and export to CSV.

    Parameters:
    dff_data (list of numpy arrays): A list where each element is a numpy array of ΔF/F values for an ROI.
    output_path (str): The file path where the average data will be saved.
    """
    # Convert the list of ΔF/F arrays into a DataFrame
    dff_df = pd.DataFrame(dff_data).transpose()
    
    # Calculate the mean across all ROIs for each timepoint
    average_df = pd.DataFrame(dff_df.mean(axis=1), columns=["Average_ROI"])

    return average_df

# Calculate the average ΔF/F values
average_df = calculate_average_roi(roi_dff)

# Export the average values to a CSV
average_output_csv_path = r'C:\dev\2p-analysis\suite2p\SB03_analysis\sb03_baseline_average_roi_dff_data.csv'
average_df.to_csv(average_output_csv_path, index=False)
print(f"Average ROI ΔF/F data successfully exported to {average_output_csv_path}")

# Plot the average ΔF/F values
plt.figure(figsize=(10, 5))
plt.plot(average_df)
plt.suptitle('SB03-BASELINE')
plt.title("Average ΔF/F Across All ROIs")
plt.xlabel("Time (Frames)")
plt.ylabel("Average ΔF/F")
plt.grid()
plt.show()

# %%
