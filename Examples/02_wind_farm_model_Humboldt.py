import numpy as np
from trtle.farmpy import Farm
import os
import matplotlib.pyplot as plt
from data.turbines.iea15mw.iea15mw import IEA15MW
"""
This example creates a wind farm via importing the layout from an external file and computes the wake effects and AEP
excluding watch circle computation.
"""

TEST_NAME = "02_wind_farm_model_Humboldt_exc_wc"
WIND_RESOURCE_FILE_PATH = os.path.join("data", "energy_resources", "Humboldt", "wind_resource_Humboldt_nsector=180.yaml")
LAYOUT_FILE_PATH = os.path.join("data", "layouts", "_Humboldt_ex.csv")

# Directory manager
this_dir = os.getcwd()
example_out_dir = os.path.join(this_dir, "Examples", "examples_out")
os.makedirs(example_out_dir, exist_ok=True)
out_dir = os.path.join(this_dir, example_out_dir, TEST_NAME)
os.makedirs(out_dir, exist_ok=True)

farm = Farm()
farm.load_layout_from_file(LAYOUT_FILE_PATH)
farm.WTG = IEA15MW()
farm.populate_turbine_keys(msrs=np.zeros(len(farm.layout_x) + 1400),
                           tbls=np.zeros(len(farm.layout_x) + 1400),
                           moris=np.zeros(len(farm.layout_x)))
farm.complex_site(WIND_RESOURCE_FILE_PATH)
aep_without_wake, aep_with_wake, wake_effects = farm.wake_model()

print('Total power: %f GWh'%aep_with_wake)
print('total wake loss:',wake_effects)

# Visualization
wsp = 9.0
wdir = 360-20
flow_map = farm.sim_res.flow_map(grid=None, # defaults to HorizontalGrid(resolution=500, extend=0.2), see below
                                 wd=wdir,
                                 ws=wsp)
plt.figure()
flow_map.plot_wake_map(levels=10, cmap='jet', plot_colorbar=False, plot_windturbines=False, ax=None)
plt.axis('equal')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.title('Wake map for'+ f' {wdir} deg and {wsp} m/s')
plt.savefig(os.path.join(out_dir, f"wake_map_wd{wdir}deg_ws{wsp}mps.pdf"))