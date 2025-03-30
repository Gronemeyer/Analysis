# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 14:21:13 2024

@author: Jake Gronemeyer
"""

#%%
import pandas as pd

#%%
# Define the path to the CSV file
path = r'E:\Areas\jgronemeyer\Camkii-gcamp8\gs18\ses-3\beh\sub-gs18_ses-03_20240910_192922.csv'

# Load the CSV file into a pandas DataFrame
df = pd.read_csv(path)

# Display the first few rows of the DataFrame
print(df.head())

# List the columns from the DataFrame
print(df.columns)


#%%
import matplotlib.pyplot as plt

# Plot stim_grayScreen.started and stim_grating.started as vertical lines
plt.figure(figsize=(10, 6))

# Scatter plot for 'thisRow.t'
plt.scatter(df.index, df['thisRow.t'], label='thisRow.t')

# Plot vertical lines for 'stim_grayScreen.started'
for start_time in df['stim_grayScreen.started']:
    plt.axvline(x=start_time, color='r', linestyle='--', label='stim_grayScreen.started')

# Plot vertical lines for 'stim_grating.started'
for start_time in df['stim_grating.started']:
    plt.axvline(x=start_time, color='g', linestyle='--', label='stim_grating.started')

plt.xlabel('Index')
plt.ylabel('Time')
plt.title('Scatter Plot of thisRow.t with Vertical Lines for stim_grayScreen.started and stim_grating.started')
plt.legend()
plt.show()

#%% PLOTLY INTERACTIVE PLOT
import plotly.express as px
import plotly.graph_objects as go

# Create an interactive scatter plot for 'thisRow.t'
fig = px.scatter(df, x='thisRow.t', title='Scatter Plot of thisRow.t')
fig.update_layout(xaxis_title='thisRow.t', yaxis_title='na')
#fig.show()

# Create an interactive plot with vertical lines for 'stim_grayScreen.started' and 'stim_grating.started'
fig = go.Figure()

# Add scatter plot for 'thisRow.t'
fig.add_trace(go.Scatter(x=df['thisRow.t'], mode='markers', name='thisRow.t'))

# Add vertical lines for 'stim_grayScreen.started'
for start_time in df['stim_grayScreen.started']:
    fig.add_vline(x=start_time, line=dict(color='red', dash='dash'), name='stim_grayScreen.started')

# Add vertical lines for 'stim_grating.started'
for start_time in df['stim_grating.started']:
    fig.add_vline(x=start_time, line=dict(color='green', dash='dash'), name='stim_grating.started')

fig.update_layout(title='Visual stim presentation timepoints',
                  xaxis_title='Time')
fig.show()


# %% BOKEH INTERACTIVE PLOT
from bokeh.plotting import figure, show
from bokeh.models import Span
from bokeh.io import output_notebook
from bokeh.models import Legend, LegendItem

# Ensure the output is displayed in the notebook
output_notebook()

# Create a new plot
p = figure(title='Visual stim presentation timepoints', x_axis_label='Time', y_axis_label='na')

# Add scatter plot for 'thisRow.t'
p.scatter(df['thisRow.t'], df.index, legend_label='thisRow.t')

# Add vertical lines for 'stim_grayScreen.started'
for start_time in df['stim_grayScreen.started']:
    vline = Span(location=start_time, dimension='height', line_color='red', line_dash='dashed', line_width=2)
    p.add_layout(vline)

# Add vertical lines for 'stim_grating.started'
for start_time in df['stim_grating.started']:
    vline = Span(location=start_time, dimension='height', line_color='green', line_dash='dashed', line_width=2)
    p.add_layout(vline)

# Show the plot
show(p)
# %%
thor_json_path = r'F:\test_006\default_protocol-sub-default_subject_ses-default_session_task-default_task_004\_frame_metadata.json'
dhyana_json_path = r"F:\test_006\default_protocol-sub-default_subject_ses-default_session_task-default_task_005\_frame_metadata.json"

df_json = pd.read_json(dhyana_json_path)

# Display the first few rows of the DataFrame
print(df_json.head())

print(df_json.columns)

plt.plot(range(len(df_json.T.index)),df_json.T['ElapsedTime-ms'].values)

# %%

# Extract 'TimeReceivedByCore' and 'Time' columns
time_received = pd.to_datetime(df_json['TimeReceivedByCore'])
time = pd.to_datetime(df_json['Time'])

# Create a new DataFrame for plotting
time_df = pd.DataFrame({
    'Index': range(len(time_received)),
    #'TimeReceivedByCore': time_received,
    'Time': time
})

#%%
# Plot using matplotlib
plt.figure(figsize=(10, 6))
plt.plot(time_df['Index'], time_df['TimeReceivedByCore'], label='TimeReceivedByCore')
plt.plot(time_df['Index'], time_df['Time'], label='Time')
plt.xlabel('Index')
plt.ylabel('Time')
plt.title('TimeReceivedByCore and Time over Index')
plt.legend()
plt.show()

# Plot using Plotly
fig = go.Figure()
fig.add_trace(go.Scatter(x=time_df['Index'], y=time_df['TimeReceivedByCore'], mode='lines', name='TimeReceivedByCore'))
fig.add_trace(go.Scatter(x=time_df['Index'], y=time_df['Time'], mode='lines', name='Time'))
fig.update_layout(title='TimeReceivedByCore and Time over Index',
                  xaxis_title='Index',
                  yaxis_title='Time')
fig.show()

#%%
# Plot using Bokeh
p = figure(title='TimeReceivedByCore and Time over Index', x_axis_label='Index', y_axis_label='Time', x_range=(0, len(time_received)))
p.line(time_df['Index'], time_df['TimeReceivedByCore'], legend_label='TimeReceivedByCore', line_width=2)
p.line(time_df['Index'], time_df['Time'], legend_label='Time', line_width=2, color='green')
show(p)

# %%
# Calculate the total duration of the capture using the range of the 'ElapsedTime-ms' column
elapsed_time_ms = df_json['ElapsedTime-ms']
total_duration_seconds = (elapsed_time_ms.max() - elapsed_time_ms.min()) / 1000

# Print the result in seconds
print(f"Total duration of the capture: {total_duration_seconds} seconds")
# %%
