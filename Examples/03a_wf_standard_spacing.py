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

this_dir = os.getcwd()
TEST_NAME = '03_wf_standard_spacing'
# Directory manager
example_out_dir = os.path.join(this_dir, "examples_out")
os.makedirs(example_out_dir, exist_ok=True)

# Create TEST directory
out_dir = os.path.join(this_dir, example_out_dir, TEST_NAME)
os.makedirs(out_dir, exist_ok=True)

layout_properties_file = os.path.join(this_dir,
                                      "input_files",
                                      "Humboldt_NE_sq_eq_standard_OPT.yaml")

# Load initial layout properties
with open(layout_properties_file, 'r') as file:
    layout_properties = yaml.safe_load(file)

farm = Farm()
farm.create_layout(layout_type="standard", layout_properties=layout_properties)
farm.complex_site()
aep_without_wake, aep_with_wake, wake_effects = farm.wake_model()
print(f"AEP: {aep_with_wake:.2f} GWh")
print(f"total wake loss:{wake_effects:.2f}%")

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

# mooring orientation
N_m = 3  # number of mooring lines
for i, _ in enumerate(farm.turbines):
    farm.add_update_turbine_keys(i, "mori", 90.0)

farm.anchor_position(N_m)
for i, turbine in enumerate(farm.turbines.values()):
    for j in range(N_m):
        plt.plot(turbine[f"anchor{j}_x"], turbine[f"anchor{j}_y"], 'og', markersize=1.0)
        plt.plot([turbine[f"anchor{j}_x"], turbine["x"]], [turbine[f"anchor{j}_y"], turbine["y"]], '-k',
                 label="mooring line" if i == 0 and j == 0 else None, linewidth=1.0)
shared_anchor_dict = farm.anchor_count(N_m)

# layout visualization
# mooring line spread radius
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
plt.savefig(os.path.join(out_dir, f"Humboldt_C_NE_sq_eq_standard_spacing_cap_{farm.capacity:.2f}"
                                  f"_SX{farm.spacing_x}_SY{farm.spacing_y}_ori{farm.orient:.2f}.pdf"))
