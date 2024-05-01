import os
import numpy as np
import pandas as pd

basedir = 'normal flight (altitude mode)'

wanted_msgs = {"vehicle_local_position" : ["timestamp", "x", "y", "z", "vx", "vy", "vz", "ax", "ay", "az"], "battery_status" : ['timestamp', 'voltage_v', 'voltage_filtered_v', 'current_a', 'current_filtered_a', 'current_average_a', 'discharged_mah', 'remaining'], "vehicle_attitude" : ["timestamp", "q[0]", "q[1]", "q[2]", "q[3]"], "vehicle_status" : ["timestamp", "arming_state", "nav_state", "vehicle_type",], "vehicle_global_position" : ['timestamp', 'lat', 'lon', 'alt', 'alt_ellipsoid'], "actuator_motors" : ['timestamp','control[0]', 'control[1]', 'control[2]', 'control[3]'], "actuator_outputs" : ["timestamp", "noutputs", "output[0]", "output[1]", "output[2]", "output[3]", "output[4]", "output[5]", "output[6]", "output[7]", "output[8]", "output[9]"], "sensor_baro": ['timestamp', 'temperature']}
file_data = pd.DataFrame()
for filename in os.listdir(basedir):
    if "setpoint" in filename :
        continue
    for potential in wanted_msgs.keys() :
        if potential in filename :
            current_file = pd.read_csv(os.path.join(basedir,filename))
            print(filename)
            file_data = pd.concat([file_data,current_file[wanted_msgs[potential]]], ignore_index=True)
            print("\n")
            print("\n")

# Write the combined data to a new CSV file
file_data.to_csv("magic_file.csv", index=False)
