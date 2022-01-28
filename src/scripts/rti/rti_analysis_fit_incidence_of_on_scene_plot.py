
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
results_folder = get_scenario_outputs('rti_analysis_fit_incidence_of_on_scene.py', outputspath)[-1]
# look at one log (so can decide what to extract)
log = load_pickled_dataframes(results_folder)

# get basic information about the results
info = get_scenario_info(results_folder)

# 1) Extract the parameters that have varied over the set of simulations
params = extract_params(results_folder)
extracted_incidence_of_death_on_scene = extract_results(results_folder,
                                                        module="tlo.methods.rti",
                                                        key="summary_1m",
                                                        column="incidence of prehospital death per 100,000",
                                                        index="date"
                                                        )
on_scene_inc_death = summarize(extracted_incidence_of_death_on_scene, only_mean=True).mean()
target_on_scene_inc_death = 6
closest_est_found = min(on_scene_inc_death, key=lambda x: abs(x - target_on_scene_inc_death))
best_fit_idx = np.where(on_scene_inc_death == closest_est_found)[0][0]
plt.bar(np.arange(len(on_scene_inc_death)), on_scene_inc_death, label='Model estimates', color='lightsalmon')
plt.bar(best_fit_idx, on_scene_inc_death[best_fit_idx], color='peachpuff', label='best fit found')
plt.axhline(target_on_scene_inc_death, color='gold', label='NRSC estimate')
plt.xticks(np.arange(len(on_scene_inc_death)), params['value'].round(3))
plt.ylabel('Incidence of on scene mortality per 100,000')
plt.xlabel('% of crashes that result in on-scene mortality')
plt.legend()
plt.title("Calibration of the model's on scene mortality to\n"
          "The National Road Safety Council's (NRSC) estimate")
plt.savefig("C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/FinalPaperOutput/DALYs_vs_hsb.png",
            bbox_inches='tight')

extracted_incidence_of_death_on_scene_person_years = \
    extract_results(results_folder,
                    module="tlo.methods.rti",
                    key="summary_1m",
                    column="incidence of prehospital death per 100,000",
                    index="date"
                    )
other_on_scene_inc_death = summarize(extracted_incidence_of_death_on_scene_person_years, only_mean=True).mean()
extracted_incidence_of_death_on_scene_person_years = \
    extract_results(results_folder,
                    module="tlo.methods.rti",
                    key="summary_1m",
                    column="incidence of prehospital death per 100,000",
                    index="date"
                    )
