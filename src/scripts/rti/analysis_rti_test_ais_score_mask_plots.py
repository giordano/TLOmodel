
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from tlo.analysis.utils import (
    extract_params,
    extract_results,
    get_scenario_info,
    get_scenario_outputs,
    load_pickled_dataframes,
    summarize,
)
# analysis_rti_test_ais_score_mask-2022-04-25T112307Z

outputspath = Path('./outputs/rmjlra2@ucl.ac.uk')

results_folder = get_scenario_outputs('analysis_rti_test_ais_score_mask.py', outputspath)[- 1]

info = get_scenario_info(results_folder)

# 1) Extract the parameters that have varied over the set of simulations
params = extract_params(results_folder)

extracted_incidence_of_RTI = extract_results(results_folder,
                                             module="tlo.methods.rti",
                                             key="summary_1m",
                                             column="incidence of rti per 100,000",
                                             index="date"
                                             )
extracted_incidence_of_RTI_death = extract_results(results_folder,
                                                   module="tlo.methods.rti",
                                                   key="summary_1m",
                                                   column="incidence of rti death per 100,000",
                                                   index="date"
                                                   )

mean_incidence_of_RTI = summarize(extracted_incidence_of_RTI, only_mean=True).mean()
scale_to_gbd = np.divide(954.2, mean_incidence_of_RTI)
mean_incidence_of_RTI_death = summarize(extracted_incidence_of_RTI_death, only_mean=True).mean()
scaled_inc_death = np.multiply(mean_incidence_of_RTI_death, scale_to_gbd)
plt.bar(np.arange(len(scaled_inc_death)), scaled_inc_death)
plt.ylabel('Incidence of death per 100,000 p.y.')
plt.xticks(np.arange(len(scaled_inc_death)), params.loc[params['module_param'] == 'RTI:no_med_death_ais_mask', 'value'])
plt.xlabel('AIS score above which to consider mortality without medical care')
plt.savefig("C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/CalibrateDeathNoMed/AIS_mask.png")
