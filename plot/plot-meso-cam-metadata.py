# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 14:21:13 2024

@author: Jake Gronemeyer
"""

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