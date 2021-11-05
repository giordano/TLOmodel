"""This file uses the results of the batch file to make some summary statistics.
The results of the bachrun were put into the 'outputs' results_folder
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tlo.analysis.utils import (
    extract_params,
    extract_results,
    get_scenario_info,
    get_scenario_outputs,
    load_pickled_dataframes,
    summarize,
)

outputspath = Path('./outputs/rmjlra2@ucl.ac.uk/')

# %% Analyse results of runs when doing a sweep of a single parameter:

# 0) Find results_folder associated with a given batch_file and get most recent
results_folder = get_scenario_outputs('rti_hsb_effect_on_inc_death.py', outputspath)[-1]

# look at one log (so can decide what to extract)
log = load_pickled_dataframes(results_folder)

# get basic information about the results
info = get_scenario_info(results_folder)

# 1) Extract the parameters that have varied over the set of simulations
params = extract_params(results_folder)

# 2) Extract a specific log series for all runs:
extracted_hsb = extract_results(results_folder,
                                module="tlo.methods.rti",
                                key="summary_1m",
                                column="percent sought healthcare",
                                index="date")
extracted_inc_death = extract_results(results_folder,
                                      module="tlo.methods.rti",
                                      key="summary_1m",
                                      column="incidence of rti death per 100,000",
                                      index="date")


# 3) Get summary of the results for that log-element
prop_sought_healthcare = summarize(extracted_hsb)
inc_death = summarize(extracted_inc_death)
# If only interested in the means
prop_sought_healthcare_onlymeans = summarize(extracted_hsb, only_mean=True)
inc_death_only_means = summarize(extracted_inc_death, only_mean=True)
# get per parameter summaries
mean_overall = prop_sought_healthcare.mean()
mean_inc_overall = inc_death.mean()
inc_mean_upper = mean_inc_overall.loc[:, 'upper']
inc_mean_lower = mean_inc_overall.loc[:, 'lower']
lower_upper_inc = np.array(list(zip(
    inc_mean_lower.to_list(),
    inc_mean_upper.to_list()
))).transpose()
# get upper and lower estimates
prop_sought_healthcare_lower = mean_overall.loc[:, "lower"]
prop_sought_healthcare_upper = mean_overall.loc[:, "upper"]
lower_upper = np.array(list(zip(
    prop_sought_healthcare_lower.to_list(),
    prop_sought_healthcare_upper.to_list()
))).transpose()
# name of parmaeter that varies
param_name = 'RTI:rt_emergency_care_ISS_score_cut_off'
# find the values that fall within our accepted range of health seeking behaviour based on results of Zafar et al
# doi: 10.1016/j.ijsu.2018.02.034
expected_hsb_upper = 0.85
expected_hsb_lower = 0.6533
per_param_average_hsb = mean_overall[:, 'mean'].values
per_param_average_inc = mean_inc_overall[:, 'mean'].values
yerr = abs(lower_upper - per_param_average_hsb)
yerr_inc = abs(lower_upper_inc - per_param_average_inc)
in_accepted_range = np.where((per_param_average_hsb > expected_hsb_lower) &
                             (per_param_average_hsb < expected_hsb_upper))
xvals = range(info['number_of_draws'])
colors = ['lightsteelblue' if i not in in_accepted_range[0] else 'lightsalmon' for i in xvals]
xlabels = [
    round(params.loc[(params.module_param == param_name)][['value']].loc[draw].value, 3)
    for draw in range(info['number_of_draws'])
]
fig, ax = plt.subplots()
ax.bar(
    x=xvals,
    height=mean_overall[:, 'mean'].values,
    yerr=yerr,
    color=colors
)
ax.set_xticks(xvals)
ax.set_xticklabels(xlabels)
plt.xlabel(param_name)
plt.ylabel('Percentage sought healthcare')
plt.title('Calibration of the ISS score which determines\nautomatic health care seeking')
plt.show()

plt.clf()
# WHO_est_in_death = 35
# best_fit_found = min(per_param_average_inc, key=lambda x: abs(x - WHO_est_in_death))
# best_fit_index = np.where(per_param_average_inc == best_fit_found)

plt.bar(x=xvals,
        height=mean_inc_overall[:, 'mean'].values,
        yerr=yerr_inc,
        color=colors)
plt.xticks(xvals, xlabels)
plt.xlabel(param_name)
plt.ylabel('Incidence of death per 100,000 person years')
plt.title('Effect of health seeking behaviour on the incidence of death')
plt.savefig(f"C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/Calibration/effect_of_hsb_on_inc_death.png",
            bbox_inches='tight')
