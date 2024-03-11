import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from windrose import WindroseAxes
import yaml
from scipy.interpolate import interp1d

"""
This example creates the .yaml file needed to compute wind resources of the site based on multi-year raw wind data.
"""

TEST_NAME = '01_energy_resource_Humboldt'
NSECTOR = 180

# Directory manager
this_dir = os.getcwd()
example_out_dir = os.path.join(this_dir, "examples_out")
os.makedirs(example_out_dir, exist_ok=True)

# Create TEST directory
out_dir = os.path.join(this_dir, example_out_dir, TEST_NAME)
os.makedirs(out_dir, exist_ok=True)

# Load wind data
df = pd.read_excel(os.path.join(this_dir, "..", "data", "energy_resources", "Humboldt", "row_data.xlsx"))

# Calculate mean wind components
u = -np.mean((df['u1'], df['u2'], df['u3']), axis=0)
v = -np.mean((df['v1'], df['v2'], df['v3']), axis=0)

# Convert to speed and direction
speed_100 = np.sqrt(u**2 + v**2)  # Speed at 100m

# Use the power law to extrapolate to 150m (hub height) using alpha=0.105
speed_150 = speed_100 * (150/100) ** 0.105
direction = (np.arctan2(u, v) * 180 / np.pi) % 360

# Wind rose plot setup
plt.rcParams["font.family"] = "Times New Roman"

# Create and display the wind rose plot
ax = WindroseAxes.from_ax()
ax.bar(direction, speed_150, normed=True,
       bins=np.arange(3, 25, 2), opening=1.0, edgecolor='black', nsector=NSECTOR)
plt.show()
ax.set_legend()
plt.savefig(os.path.join(out_dir, "wind_rose.pdf"))

table = ax._info['table']
direction_bins = ax._info['dir']
speed_bins = ax._info['bins']

new_directions = np.linspace(0, 360, NSECTOR, endpoint=False)
new_table = np.zeros((table.shape[0], NSECTOR))
for i in range(table.shape[0]):
    interp_func = interp1d(direction_bins, table[i, :], kind='linear', fill_value="extrapolate")

    # Interpolate frequencies for the new directional array
    new_table[i] = interp_func(new_directions)

# Frequency for each directional bin
directional_frequency = np.sum(new_table, axis=0)/100


# Frequency of speed for each directional bin
speed_frequency_per_direction_bin = np.zeros((len(new_directions), len(speed_bins) - 1))
for i in range(len(new_directions)):
    speed_frequency_per_direction_bin[i, :] = new_table[:, i]/np.sum(new_table[:, i])


direction_bins_py = [float(value) for value in new_directions]
speed_bins_py = [float(value) for value in speed_bins[:-1]]
# Preparing the data for YAML
yaml_data = {
    "name": "Humboldt wind data",
    "wind_resource": {
        "wind_direction": direction_bins_py,
        "wind_speed": speed_bins_py,  # Exclude the last bin edge
        "sector_probability": {
            "data": directional_frequency.tolist(),
            "dims": ["wind_direction"]
        },
        "probability": {
            "data": [row.tolist() for row in speed_frequency_per_direction_bin],  # Converting each row to a list
            "dims": ["wind_direction", "wind_speed"]
        },
        "turbulence_intensity": {
            "data": 0.06,
            "dims": []
        }
    }
}

# Save to a .yaml file
file_path = os.path.join(out_dir, f"wind_resource_Humboldt_nsector={NSECTOR}.yaml")  # Adjust the path as needed
with open(file_path, 'w') as file:
    yaml.dump(yaml_data, file, default_flow_style=None, width=float("inf"))