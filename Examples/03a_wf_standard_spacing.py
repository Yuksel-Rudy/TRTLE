import yaml
import os
from trtle.farmpy import Farm
import matplotlib.pyplot as plt

"""
In this example, a wind farm is created based on a boundary file, a list of farm-level properties and the turbine used.
(taking into consideration mooring line spread)
"""

TEST_NAME = '03_wf_fixed_spacing'

# Directory manager
this_dir = os.getcwd()
example_out_dir = os.path.join(this_dir, "Examples", "examples_out")
os.makedirs(example_out_dir, exist_ok=True)

# Create TEST directory
out_dir = os.path.join(this_dir, example_out_dir, TEST_NAME)
os.makedirs(out_dir, exist_ok=True)

layout_properties_file = os.path.join(this_dir, "data", "layout_input_files", "Humboldt_SW_standard_spacing.yaml")
farm = Farm()
farm.create_layout(layout_type="standard", layout_properties_file=layout_properties_file)
plt.figure()
plt.grid()
plt.axis("equal")
plt.scatter(farm.boundary_x, farm.boundary_y, label="farm boundary")
plt.scatter(farm.layout_x, farm.layout_y, label="farm layout")
plt.legend()
plt.savefig(os.path.join(out_dir, f"Humboldt_SW_standard_spacing_cap_{farm.capacity:.2f}"
                                  f"_SX{farm.spacing_x}_SY{farm.spacing_y}_ori{farm.orient:.2f}.pdf"))
