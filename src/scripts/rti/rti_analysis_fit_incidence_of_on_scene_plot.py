
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import squarify

from tlo.analysis.utils import (
    extract_params,
    extract_params_from_json,
    extract_results,
    get_scenario_info,
    get_scenario_outputs,
    load_pickled_dataframes,
    summarize,
)

outputspath = Path('./outputs/rmjlra2@ucl.ac.uk/')
# 0) Find results_folder associated with a given batch_file and get most recent
results_folder = Path('outputs/rmjlra2@ucl.ac.uk/rti_analysis_fit_incidence_of_on_scene-2022-01-06T124740Z')
# look at one log (so can decide what to extract)
log = load_pickled_dataframes(results_folder)

# get basic information about the results
info = get_scenario_info(results_folder)

# 1) Extract the parameters that have varied over the set of simulations
params = extract_params(results_folder)
extracted_incidence_of_death_on_scene = extract_results(results_folder,
                                                        module="tlo.methods.rti",
                                                        key="summary_1m",
                                                        column="incidence of death on scene per 100,000",
                                                        index="date"
                                                        )
on_scene_inc_death = summarize(extracted_incidence_of_death_on_scene, only_mean=True)
on_scene_inc_death = on_scene_inc_death * 100000
