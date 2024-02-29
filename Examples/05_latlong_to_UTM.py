import numpy as np
from pyproj import Proj
import os
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from trtle.farmpy import Farm

TEST_NAME = "05_latlong_to_UTM"
this_dir = os.getcwd()
example_out_dir = os.path.join(this_dir, "Examples", "examples_out")
os.makedirs(example_out_dir, exist_ok=True)
out_dir = os.path.join(this_dir, example_out_dir, TEST_NAME)
os.makedirs(out_dir, exist_ok=True)

latlong_file = os.path.join("data", "layouts", "Humboldt_OptionC", "SW_latlong.csv")
latlong = pd.read_csv(latlong_file)
plt.figure()
plt.scatter(latlong['long'], latlong['lat'])

P = Proj(proj='utm', zone=10, ellps='WGS84', preserve_units=True)
x, y = P(latlong['long'], latlong['lat'])
plt.figure()
plt.scatter(x, y)
df = pd.DataFrame(np.c_[x, y], columns=['boundary_x', 'boundary_y'])
boundary_file_path = os.path.join(out_dir, 'Humboldt_C_SW.csv')
df.to_csv(boundary_file_path, index=False)

farm = Farm()
farm.farm_boundaries(boundary_file_path)
farm.polygon_area()
print(farm.area)