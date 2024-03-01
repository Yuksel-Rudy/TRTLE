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

TEST_NAME = '03_wf_honeymooring'
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
                                      "Humboldt_NE_sq_eq_honeymooring_OPT.yaml")

# Load initial layout properties
with open(layout_properties_file, 'r') as file:
    layout_properties = yaml.safe_load(file)

farm = Farm()
farm.create_layout(layout_type="honeymooring", layout_properties=layout_properties)
farm.complex_site(WIND_RESOURCE_FILE_PATH)

# Adjusting turbines' location manually (move a bit to the west)
farm.layout_x += 200
farm.update_turbine_loc()

aep_without_wake, aep_with_wake, wake_effects = farm.wake_model()
print('Total power: %f GWh'%aep_with_wake)
print('total wake loss:',wake_effects)

# wake visualization
wsp = 9.0
wdir = 360-5
flow_map = farm.sim_res.flow_map(grid=None, # defaults to HorizontalGrid(resolution=500, extend=0.2), see below
                                 wd=wdir,
                                 ws=wsp)
plt.figure(figsize=(6, 9))
flow_map.plot_wake_map(levels=10, cmap='jet', plot_colorbar=False, plot_windturbines=False, ax=None)
plt.axis('equal')
plt.xlabel("Easting [m]")
plt.ylabel("Northing [m]")
plt.title('Wake map for'+ f' {wdir} deg and {wsp} m/s')


# layout visualization

# mooring orientation
N_m = 3  # number of mooring lines
for i, _ in enumerate(farm.turbines):
    farm.add_update_turbine_keys(i, "mori", 180 + 90 + layout_properties["farm properties"]["orientation"])

# anchor visualization
Ax = np.zeros([3, len(farm.layout_x)])
Ay = np.zeros([3, len(farm.layout_x)])
# Anchor Positions
for i, turbine in enumerate(farm.turbines.values()):
    for j in range(N_m):
        Ax[j, i] = turbine["x"] + turbine["msr"] * np.cos(
            np.deg2rad(turbine["mori"] + 360/N_m * j))
        Ay[j, i] = turbine["y"] + turbine["msr"] * np.sin(
            np.deg2rad(turbine["mori"] + 360/N_m * j))

        plt.plot(Ax[j, i], Ay[j, i], 'og', label="anchor" if i == 0 and j == 0 else None, markersize=1.0)
        plt.plot([Ax[j, i], turbine["x"]], [Ay[j, i], turbine["y"]], '-k', label="cable" if i == 0 and j == 0 else None, linewidth=0.5)

th = np.arange(0, 2.1 * np.pi, np.deg2rad(5))
mpl.rcParams['font.family'] = 'Times New Roman'

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
plt.savefig(os.path.join(out_dir, f"Humboldt_C_NE_sq_eq_honeymooring_spacing_cap_{farm.capacity:.2f}"
                                  f"_SX{farm.spacing_x}_SY{farm.spacing_y}_ori{farm.orient:.2f}_reversed.pdf"))
