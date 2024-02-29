import yaml
import os
from trtle.farmpy import Farm
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

"""
In this example, a wind farm is created based on a boundary file, a list of farm-level properties and the turbine used.
(taking into consideration mooring line spread)
"""

TEST_NAME = '03_wf_standard_spacing'
WIND_RESOURCE_FILE_PATH = os.path.join("data", "energy_resources", "Humboldt", "wind_resource_Humboldt_nsector=180.yaml")
# Directory manager
this_dir = os.getcwd()
example_out_dir = os.path.join(this_dir, "Examples", "examples_out")
os.makedirs(example_out_dir, exist_ok=True)

# Create TEST directory
out_dir = os.path.join(this_dir, example_out_dir, TEST_NAME)
os.makedirs(out_dir, exist_ok=True)

layout_properties_file = os.path.join(this_dir,
                                      "data",
                                      "layout_input_files",
                                      "Humboldt_SW_sq_eq_standard_OPT.yaml")

# Load initial layout properties
with open(layout_properties_file, 'r') as file:
    layout_properties = yaml.safe_load(file)

farm = Farm()
farm.create_layout(layout_type="standard", layout_properties=layout_properties)
farm.complex_site(WIND_RESOURCE_FILE_PATH)
aep_without_wake, aep_with_wake, wake_effects = farm.wake_model()
print('Total power: %f GWh'%aep_with_wake)
print('total wake loss:',wake_effects)

# wake visualization
wsp = 9.0
wdir = 360-5
flow_map = farm.sim_res.flow_map(grid=None, # defaults to HorizontalGrid(resolution=500, extend=0.2), see below
                                 wd=wdir,
                                 ws=wsp)
plt.figure(figsize=(6, 6))
flow_map.plot_wake_map(levels=10, cmap='jet', plot_colorbar=False, plot_windturbines=False, ax=None)
plt.axis('equal')
plt.xlabel("Easting [m]")
plt.ylabel("Northing [m]")
plt.title('Wake map for'+ f' {wdir} deg and {wsp} m/s')
# plt.savefig(os.path.join(out_dir, f"Humboldt_C_NE_sq_eq_wake_map_wd{wdir}deg_ws{wsp}mps_OPT.pdf"))


# layout visualization
# mooring line spread radius
th = np.arange(0, 2.1 * np.pi, np.deg2rad(5))
mpl.rcParams['font.family'] = 'Times New Roman'
plt.figure(figsize=(6, 6))

for turbine_id, turbine in farm.turbines.items():
    moorspreadx = turbine['msr'] * np.cos(th) + turbine['x']
    moorspready = turbine['msr'] * np.sin(th) + turbine['y']
    if turbine['ID']==0:
        plt.plot(moorspreadx, moorspready, '--k', label='mooring spread circle', linewidth=0.2)
    else:
        plt.plot(moorspreadx, moorspready, '--k', linewidth=0.2)

plt.grid("True")
plt.gca().set_axisbelow(True)
plt.axis("equal")
plt.plot(farm.oboundary_x, farm.oboundary_y, label="farm boundary")
plt.scatter(farm.layout_x, farm.layout_y, label="farm layout", color="black")
plt.plot()
plt.legend()
plt.xlabel("Easting [m]")
plt.ylabel("Northing [m]")
# plt.savefig(os.path.join(out_dir, f"Humboldt_C_NE_sq_eq_standard_spacing_cap_{farm.capacity:.2f}"
#                                   f"_SX{farm.spacing_x}_SY{farm.spacing_y}_ori{farm.orient:.2f}.pdf"))
