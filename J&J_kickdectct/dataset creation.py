import pandas as pd
import numpy as np

# Define the parameters for the dataset
num_rows = 200  # Adjust based on the number of intervals you want
depth_ft = np.linspace(0, 5000, num_rows)  # Generate 200 depth intervals from 0 to 5000 ft
pore_pressure_psi_per_ft = np.random.uniform(0, 1, num_rows) * depth_ft  # Random pore pressure
fracture_pressure_psi = np.random.uniform(1, 2, num_rows) * depth_ft  # Random fracture pressure
ccs_mpa = np.random.uniform(20, 30, num_rows)  # Random CCS
friction_angle = np.random.uniform(20, 40, num_rows)  # Random friction angle
north_ft = np.linspace(0, 200, num_rows)  # Uniformly increasing north coordinates
east_ft = np.linspace(0, 200, num_rows)  # Uniformly increasing east coordinates
dip = np.linspace(0, 90, num_rows)  # Uniformly increasing dip angles
azimuth = np.linspace(0, 360, num_rows)  # Uniformly increasing azimuth angles

# Increase pore pressures at depths greater than 2000 ft
pore_pressure_psi_per_ft[depth_ft > 2000] *= 3.5  # Increase pore pressure by a factor for depths > 2000 ft

# Determine kick or no kick based on pore pressure and fracture pressure
classes = []
for i in range(num_rows):
    if pore_pressure_psi_per_ft[i] > fracture_pressure_psi[i]:
        classes.append(1)  # 'KICK' is represented as 1
    else:
        classes.append(0)  # 'NO KICK' is represented as 0

# Create the DataFrame
data = pd.DataFrame({
    'Depth_ft': depth_ft,
    'Pore_pressure_psi_per_ft': pore_pressure_psi_per_ft,
    'Fracture_pressure_psi': fracture_pressure_psi,
    'CCS_Mpa': ccs_mpa,
    'Friction_angle': friction_angle,
    'North_ft': north_ft,
    'East_ft': east_ft,
    'Dip': dip,  # Ensure all arrays are of the same length
    'Azimuth': azimuth,
    'Class': classes
})

# Save the DataFrame to a CSV file
data.to_csv('Openlab_wellA.csv', index=False)
