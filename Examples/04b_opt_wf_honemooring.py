import yaml
import os
from trtle.farmpy import Farm
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import optuna
from optuna.visualization import plot_rank


def objective(trial):
    farm_properties["capacity"] = 1000
    Orient = trial.suggest_float('Orient', 0, 45)
    MoorRa = trial.suggest_float('MoorRa', 780, 1400)
    farm_properties["orientation"] = Orient
    farm_properties["mooring line spread radius"] = MoorRa

    try:
        farm.honeymooring_layout(farm_properties=farm_properties)
        farm.complex_site(WIND_RESOURCE_FILE_PATH)
        aep_without_wake, aep_with_wake, wake_effects = farm.wake_model()
    except Exception as e:
        print(f'prune this trial due to error: {e}')
        raise optuna.TrialPruned()

    return wake_effects

"""
In this example, a wind farm is created based on a boundary file, a list of farm-level properties and the turbine used.
(taking into consideration mooring line spread)
"""

TEST_NAME = '04_opt_wf_standard_spacing'
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
                                      "Humboldt_NE_sq_eq_honeymooring.yaml")

# Load initial layout properties
with open(layout_properties_file, 'r') as file:
    layout_properties = yaml.safe_load(file)

farm = Farm()
farm.create_layout(layout_type="honeymooring", layout_properties=layout_properties)
farm_properties = layout_properties["farm properties"]

study_name = "04a_groupB"
study = optuna.create_study(study_name=study_name,
                            direction="minimize",
                            sampler=optuna.samplers.TPESampler(n_startup_trials=50, multivariate=True, group=True))
study.optimize(objective, n_trials=1000, gc_after_trial=True)

fig = plot_rank(study)