# -*- coding: utf-8 -*-
"""
Author : Jake Gronemeyer 
Date : 2024-07-25

Description: 
    Testing the parameters for Rotary Encoders to get
    accurate distance and speed based on Encoder Counts Per Revolution (CPR)

Reference Documents:
    https://www.cuidevices.com/blog/what-is-encoder-ppr-cpr-and-lpr



TODO: Threading (like in psychopy) to collect data only when wheel moves
        and time threads
"""
import serial
import time
import pandas as pd

# Constants
WHEEL_DIAMETER = 0.1  # in meters, example value
ENCODER_CPR = 1200   # encoder counts per revolution
SAMPLE_WINDOW = 0.05  # sample window in seconds, matching the Arduino sample window
PORT = 'COM4'

# Set up the serial port connection to arduino
arduino = serial.Serial(port=PORT, baudrate=57600, timeout=1)

# Initialize DataFrame to store encoder data

#### Shared variable for clicks (raw value received from encoder)
# clicks_lock = threading.Lock() #thread locked for synchronization
# shared_clicks = 0 #cross-thread accessible variable


def read_encoder():
    global shared_clicks
    while True:
        try:
            data = arduino.readline().decode('utf-8').strip()
            if data:
                with clicks_lock:
                    shared_clicks = int(data)
        except ValueError:
            pass
        
    
#NOTE: time_interval value should use PsychoPy's core.Clock() 
def calculate_metrics(clicks, time_interval):
    rotations = clicks / ENCODER_CPR
    distance = rotations * (3.1416 * WHEEL_DIAMETER)
    speed = distance / time_interval  # m/s
    return speed, distance

def determine_direction(clicks):
    if clicks == 0:
        return 0  # Stationary
    elif clicks > 0:
        return 1  # Forward
    else:
        return 2  # Backward

def save_data(timestamp, speed, distance, direction):
    global encoder_data
    new_data = pd.DataFrame([[timestamp, speed, distance, direction]], columns=encoder_data.columns)
    encoder_data = pd.concat([encoder_data, new_data], ignore_index=True)
    
    
def main():
    # Start the encoder reading thread

    total_distance = 0.0001
    prev_time = time.time()
    while True:
        try:
            current_time = time.time()
            time_interval = current_time - prev_time
            data = arduino.readline().decode('utf-8').strip()
            if data:
                clicks = int(data)
                speed, distance = calculate_metrics(clicks, time_interval)
                total_distance += distance
                direction = determine_direction(clicks)
            
                timestamp = current_time
                save_data(timestamp, speed, total_distance, direction)
                
                #comment out/in for debugging
                print(f"Time: {timestamp:.2f}s, Speed: {speed:.2f} m/s, Total Distance: {total_distance:.2f} m, Direction: {direction}")
            
                prev_time = current_time
        except ValueError:
            arduino.close()

if __name__ == "__main__":
    main()

