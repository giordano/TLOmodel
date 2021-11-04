"""This file uses the results of the batch file to make some summary statistics.
The results of the bachrun were put into the 'outputs' results_folder
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json

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

# %% Analyse results of runs when doing a sweep of a single parameter:

# 0) Find results_folder associated with a given batch_file and get most recent
results_folder = get_scenario_outputs('rti_calibrate_shock.py.py', outputspath)[- 1]

# look at one log (so can decide what to extract)
log = load_pickled_dataframes(results_folder)

# get basic information about the results
info = get_scenario_info(results_folder)

# 1) Extract the parameters that have varied over the set of simulations
params = extract_params(results_folder)
# params = extract_params_from_json(results_folder, 'rti_incidence_parameterisation.py', 'RTI', 'base_rate_injrti')
# 2) Extract a specific log series for all runs:
extracted = extract_results(results_folder,
                            module="tlo.methods.rti",
                            key="summary_1m",
                            column="percent_in_shock",
                            index="date")
extracted_incidence_of_death = extract_results(results_folder,
                                               module="tlo.methods.rti",
                                               key="summary_1m",
                                               column="incidence of rti death per 100,000",
                                               index="date"
                                               )
extracted_incidence_of_RTI = extract_results(results_folder,
                                             module="tlo.methods.rti",
                                             key="summary_1m",
                                             column="incidence of rti per 100,000",
                                             index="date"
                                             )
# 3) Get summary of the results for that log-element
in_hospital_mortality = summarize(extracted)
incidence_of_death = summarize(extracted_incidence_of_death)
incidence_of_RTI = summarize(extracted_incidence_of_RTI)
percentage_in_shock = summarize(extracted)
# If only interested in the means
in_hospital_mortality_onlymeans = summarize(extracted, only_mean=True)
# get per parameter summaries
mean_percentage_in_shock = percentage_in_shock.mean()
mean_incidence_of_death = incidence_of_death.mean()
mean_incidence_of_rti = incidence_of_RTI.mean()
# get upper and lower estimates
mean_percent_in_shock_lower = mean_percentage_in_shock.loc[:, "lower"]
mean_percent_in_shock_upper = mean_percentage_in_shock.loc[:, "upper"]
lower_upper = np.array(list(zip(
    mean_percent_in_shock_lower.to_list(),
    mean_percent_in_shock_upper.to_list()
))).transpose()
# find the values that fall within our accepted range of incidence based on results of the GBD study

per_param_average_in_shock = mean_percentage_in_shock[:, 'mean'].values
expected_percent_in_shock = 56 / 8026
yerr = abs(lower_upper - per_param_average_in_shock)
xvals = range(info['number_of_draws'])
colors = ['lightsteelblue'] * len(xvals)
best_fit_found = min(per_param_average_in_shock, key=lambda x: abs(x - expected_percent_in_shock))
best_fit_index = np.where(per_param_average_in_shock == best_fit_found)
colors[best_fit_index[0][0]] = 'gold'
print(f"best fitting parameter value = {params.loc[best_fit_index[0][0]]}")
print(f"Resulting percentage of those with RTI in shock = {best_fit_found}")
print(f"Resulting incidence of death = {mean_incidence_of_death[best_fit_index[0][0]]['mean']}")
print(f"Resulting incidence of RTI = {mean_incidence_of_rti[best_fit_index[0][0]]['mean']}")
fig, ax = plt.subplots()
ax.bar(
    x=xvals,
    height=per_param_average_in_shock,
    yerr=yerr,
    color=colors,
)
ax.set_xticks(xvals)
ax.set_xticklabels(np.round(params['value'].to_list(), 3), rotation=90)
plt.xlabel('prob_bleeding_leads_to_shock')
plt.ylabel('Percent in shock')
plt.title('Calibration of the onset of shock for those with RTIs')
lowest_value_of_param = params['value'].min()
highest_value_of_param = params['value'].max()
plt.savefig(f"C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/Calibration/percent_in_shock"
            f"{lowest_value_of_param}_{highest_value_of_param}.png", bbox_inches='tight')
