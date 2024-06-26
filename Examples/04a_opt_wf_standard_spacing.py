import yaml
import os
from trtle.farmpy import Farm
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import optuna
from optuna.visualization import plot_rank


def objective(trial):
    SpaceX = trial.suggest_float('SpaceX', 4, 12)
    SpaceY = trial.suggest_float('SpaceY', 4, 12)
    Orient = trial.suggest_float('Orient', 0, 45)
    SkewFa = trial.suggest_float('SkewFa', -1, 1)
    # MoorRa = trial.suggest_float('MoorRa', 780, 1400)
    farm_properties["Dspacingx"] = SpaceX
    farm_properties["Dspacingy"] = SpaceY
    farm_properties["orientation"] = Orient
    farm_properties["skew factor"] = SkewFa
    # farm_properties["mooring line spread radius"] = MoorRa

    try:
        farm.standard_layout(farm_properties=farm_properties)
        farm.complex_site()
        aep_without_wake, aep_with_wake, wake_effects = farm.wake_model()
    except Exception as e:
        print(f'prune this trial due to error: {e}')
        raise optuna.TrialPruned()

    return wake_effects

"""
In this example, 
- farm-level properties are optimized.
"""
this_dir = os.getcwd()

TEST_NAME = '04_opt_wf_standard_spacing'
# Directory manager

example_out_dir = os.path.join(this_dir, "examples_out")
os.makedirs(example_out_dir, exist_ok=True)

# Create TEST directory
out_dir = os.path.join(this_dir, example_out_dir, TEST_NAME)
os.makedirs(out_dir, exist_ok=True)

layout_properties_file = os.path.join(this_dir,
                                      "input_files",
                                      "Humboldt_NE_sq_eq_standard.yaml")

# Load initial layout properties
with open(layout_properties_file, 'r') as file:
    layout_properties = yaml.safe_load(file)

farm = Farm()
farm.create_layout(layout_type="standard", layout_properties=layout_properties)
farm_properties = layout_properties["farm properties"]

study_name = "04a_groupA"
study = optuna.create_study(study_name=study_name,
                            direction="minimize",
                            sampler=optuna.samplers.TPESampler(n_startup_trials=50, multivariate=True, group=True))
study.optimize(objective, n_trials=1000, gc_after_trial=True)

fig = plot_rank(study)