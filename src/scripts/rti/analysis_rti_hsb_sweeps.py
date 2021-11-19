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
multiple_injury_results_folder = get_scenario_outputs('rti_hsb_effect_on_inc_death.py', outputspath)[-1]
single_injury_results_folder = get_scenario_outputs('rti_single_injury_hsb_sweep.py.py', outputspath)[-1]
# look at one log (so can decide what to extract)
log = load_pickled_dataframes(single_injury_results_folder)

# get basic information about the results
info = get_scenario_info(single_injury_results_folder)

# 1) Extract the parameters that have varied over the set of simulations
params = extract_params(single_injury_results_folder)
param_name = 'RTI:rt_emergency_care_ISS_score_cut_off'
xvals = range(info['number_of_draws'])
# 2) Extract a specific log series for all runs:
single_extracted_hsb = extract_results(single_injury_results_folder,
                                       module="tlo.methods.rti",
                                       key="summary_1m",
                                       column="percent sought healthcare",
                                       index="date")
sing_prop_sought_healthcare = summarize(single_extracted_hsb)
sing_prop_sought_healthcare_onlymeans = summarize(single_extracted_hsb, only_mean=True)
sing_mean_hsb = sing_prop_sought_healthcare.mean()
# get upper and lower estimates
prop_sought_healthcare_lower = sing_mean_hsb.loc[:, "lower"]
prop_sought_healthcare_upper = sing_mean_hsb.loc[:, "upper"]
lower_upper = np.array(list(zip(
    prop_sought_healthcare_lower.to_list(),
    prop_sought_healthcare_upper.to_list()
))).transpose()
# name of parmaeter that varies
# find the values that fall within our accepted range of health seeking behaviour based on results of Zafar et al
# doi: 10.1016/j.ijsu.2018.02.034
expected_hsb_upper = 0.85
expected_hsb_lower = 0.6533
midpoint_hsb = (expected_hsb_upper + expected_hsb_lower) / 2
sing_per_param_average_hsb = sing_mean_hsb[:, 'mean'].values
sing_hsb_yerr = abs(lower_upper - sing_per_param_average_hsb)
in_accepted_range = np.where((sing_per_param_average_hsb > expected_hsb_lower) &
                             (sing_per_param_average_hsb < expected_hsb_upper))

closest_to_hsb_midpoint = min(sing_per_param_average_hsb, key=lambda x: abs(x - midpoint_hsb))
index_of_midpoint = np.where(sing_per_param_average_hsb == closest_to_hsb_midpoint)
colors = ['lightsteelblue' if i not in in_accepted_range[0] else 'lightsalmon' for i in xvals]
colors[index_of_midpoint[0][0]] = 'gold'
xlabels = [
    round(params.loc[(params.module_param == param_name)][['value']].loc[draw].value, 3)
    for draw in range(info['number_of_draws'])
]
fig, ax = plt.subplots()
ax.bar(
    x=xvals,
    height=sing_mean_hsb[:, 'mean'].values,
    yerr=sing_hsb_yerr,
    color=colors
)
ax.set_xticks(xvals)
ax.set_xticklabels(xlabels)
plt.xlabel(param_name)
plt.ylabel('Percentage sought healthcare')
plt.title('Calibration of the ISS score which determines\nautomatic health care seeking, single injury model')
plt.savefig(f"C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/SingleVsMultipleInjury/BatchResults/hsb_sweeps/"
            f"ParameterSpaceForHSB_single.png", bbox_inches='tight')
plt.clf()
sing_extracted_inc_death = extract_results(single_injury_results_folder,
                                           module="tlo.methods.rti",
                                           key="summary_1m",
                                           column="incidence of rti death per 100,000",
                                           index="date")
sing_inc_death = summarize(sing_extracted_inc_death)
sing_inc_death_only_means = summarize(sing_extracted_inc_death, only_mean=True)
sing_mean_inc_death_overall = sing_inc_death.mean()
sing_inc_death_mean_upper = sing_mean_inc_death_overall.loc[:, 'upper']
sing_inc_death_mean_lower = sing_mean_inc_death_overall.loc[:, 'lower']
sing_lower_upper_inc_death = np.array(list(zip(
    sing_inc_death_mean_lower.to_list(),
    sing_inc_death_mean_upper.to_list()
))).transpose()
sing_per_param_average_inc_death = sing_mean_inc_death_overall[:, 'mean'].values
sing_yerr_inc_death = abs(sing_lower_upper_inc_death - sing_per_param_average_inc_death)

WHO_est_in_death = 35
sing_best_fit_inc_death_found = min(sing_per_param_average_inc_death[in_accepted_range],
                                    key=lambda x: abs(x - WHO_est_in_death))
best_fit_death_index = np.where(sing_per_param_average_inc_death == sing_best_fit_inc_death_found)

plt.bar(x=xvals,
        height=sing_mean_inc_death_overall[:, 'mean'].values,
        yerr=sing_yerr_inc_death,
        color=colors)
for idx, inc in enumerate(sing_per_param_average_inc_death):
    plt.text(idx - 0.5, sing_inc_death_mean_upper.to_list()[idx] + 5,
             f"{np.round(sing_mean_inc_death_overall[:, 'mean'][idx], 2)}")
plt.xticks(xvals, xlabels)
plt.xlabel(param_name)
plt.ylabel('Incidence of death per 100,000 person years')
plt.title('Effect of health seeking behaviour on the incidence of death')
plt.savefig(f"C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/SingleVsMultipleInjury/BatchResults/hsb_sweeps/"
            f"effect_of_hsb_on_incidence_of_death_single.png", bbox_inches='tight')
sing_extracted_inc = extract_results(single_injury_results_folder,
                                     module="tlo.methods.rti",
                                     key="summary_1m",
                                     column="incidence of rti per 100,000",
                                     index="date")

# 3) Get summary of the results for that log-element
sing_inc = summarize(sing_extracted_inc)
# If only interested in the means
sing_inc_only_means = summarize(sing_extracted_inc, only_mean=True)
# get per parameter summaries
sing_mean_inc_overall = sing_inc.mean()
sing_inc_mean_upper = sing_mean_inc_overall.loc[:, 'upper']
sing_inc_mean_lower = sing_mean_inc_overall.loc[:, 'lower']
sing_lower_upper_inc = np.array(list(zip(
    sing_inc_mean_lower.to_list(),
    sing_inc_mean_upper.to_list()
))).transpose()
sing_per_param_average_inc = sing_mean_inc_overall[:, 'mean'].values
yerr_inc = abs(sing_lower_upper_inc - sing_per_param_average_inc)
GBD_est_inc = 954.2
sing_best_fit_found_for_inc = min(sing_per_param_average_inc[in_accepted_range], key=lambda x: abs(x - GBD_est_inc))
best_fit_for_inc_index = np.where(sing_per_param_average_inc == sing_best_fit_found_for_inc)
plt.clf()
plt.bar(x=xvals,
        height=sing_mean_inc_overall[:, 'mean'].values,
        yerr=yerr_inc,
        color=colors)
for idx, inc in enumerate(sing_per_param_average_inc):
    plt.text(idx - 0.5, sing_inc_mean_upper.to_list()[idx] + 5, f"{np.round(sing_mean_inc_overall[:, 'mean'][idx], 1)}")
plt.xticks(xvals, xlabels)
plt.xlabel(param_name)
plt.ylabel('Incidence of RTI per 100,000 person years')
plt.title('Effect of health seeking behaviour on the incidence of RTI')
plt.savefig(f"C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/SingleVsMultipleInjury/BatchResults/hsb_sweeps/"
            f"effect_of_hsb_on_incidence_of_rti_single.png", bbox_inches='tight')
plt.clf()
sing_percent_crashed_that_are_fatal = extract_results(single_injury_results_folder,
                                                      module="tlo.methods.rti",
                                                      key="summary_1m",
                                                      column="percent of crashes that are fatal",
                                                      index="date")

# 3) Get summary of the results for that log-element
sing_percent_fatal = summarize(sing_percent_crashed_that_are_fatal)
# If only interested in the means
sing_percent_fatal_only_means = summarize(sing_percent_crashed_that_are_fatal, only_mean=True)
# get per parameter summaries
sing_mean_percent_fatal_overall = sing_percent_fatal.mean()
sing_perc_fatal_mean_lower = sing_mean_percent_fatal_overall.loc[:, 'lower']
sing_perc_fatal_mean_upper = sing_mean_percent_fatal_overall.loc[:, 'upper']
sing_lower_upper_perc_fatal = np.array(list(zip(
    sing_perc_fatal_mean_lower.to_list(),
    sing_perc_fatal_mean_upper.to_list()
))).transpose()

sing_per_param_perc_fatal = sing_mean_percent_fatal_overall[:, 'mean'].values
sing_yerr_perc_fatal = abs(sing_lower_upper_perc_fatal - sing_per_param_perc_fatal)
plt.bar(x=xvals,
        height=sing_mean_percent_fatal_overall[:, 'mean'].values,
        yerr=sing_yerr_perc_fatal,
        color=colors)
for idx, inc in enumerate(sing_per_param_perc_fatal):
    plt.text(idx - 0.5, sing_perc_fatal_mean_upper.to_list()[idx] + 0.02,
             f"{np.round(sing_mean_percent_fatal_overall[:, 'mean'][idx], 3)}")
plt.xticks(xvals, xlabels)
plt.xlabel(param_name)
plt.ylabel('Percent Fatal')
plt.title('Effect of health seeking behaviour on the percent of crashed that are fatal')
plt.savefig(f"C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/SingleVsMultipleInjury/BatchResults/hsb_sweeps/"
            f"effect_of_hsb_on_percent_fatal_single.png", bbox_inches='tight')
plt.clf()
# Plot summary for best fits of different categories
# create a function that creates an infographic

# scale normalise the results so that each incidence is equal to the gbd incidence
sing_scale_to_match_GBD = np.divide(GBD_est_inc, sing_mean_inc_overall[:, 'mean'].values)
sing_scaled_incidences = sing_mean_inc_overall[:, 'mean'].values * sing_scale_to_match_GBD
sing_rescaled_incidence_of_death = sing_mean_inc_death_overall[:, 'mean'].values * sing_scale_to_match_GBD
sing_best_fitting_scaled = min(sing_rescaled_incidence_of_death, key=lambda x: abs(x - WHO_est_in_death))
sing_best_fitting_scaled_index = np.where(sing_rescaled_incidence_of_death == sing_best_fitting_scaled)[0][0]
sing_colors_for_inc = ['lightsteelblue'] * len(sing_scaled_incidences)
sing_colors_for_inc[sing_best_fitting_scaled_index] = 'gold'
sing_colors_for_inc_death = ['lightsalmon'] * len(sing_rescaled_incidence_of_death)
sing_colors_for_inc_death[sing_best_fitting_scaled_index] = 'gold'

plt.clf()
fig = plt.figure(constrained_layout=True)
# Use GridSpec for customising layout
gs = fig.add_gridspec(nrows=2, ncols=1)
# Add an empty axes that occupied the whole first row
x_vals = params.loc[params['module_param'] == 'RTI:rt_emergency_care_ISS_score_cut_off', 'value']

ax1 = fig.add_subplot(gs[0, 0])
ax1.bar(np.arange(len(sing_scaled_incidences)), sing_scaled_incidences, color=sing_colors_for_inc, width=0.4)
for idx, val in enumerate(sing_scaled_incidences):
    ax1.text(idx, val, f"{np.round(val, 2)}", rotation=90)
ax1.set_title('Incidence of RTI')
ax1.set_xticks(np.arange(len(x_vals)))
ax1.set_xticklabels(x_vals)
ax1.set_ylabel('Incidence per 100,000 p.y.')
ax2 = fig.add_subplot(gs[1, 0])
ax2.set_title('Incidence of Death')
ax2.bar(np.arange(len(sing_rescaled_incidence_of_death)), sing_rescaled_incidence_of_death,
        color=sing_colors_for_inc_death, width=0.4)
for idx, val in enumerate(sing_rescaled_incidence_of_death):
    ax2.text(idx, val, f"{np.round(val, 2)}", rotation=90)
ax2.set_xticks(np.arange(len(x_vals)))
ax2.set_xticklabels(x_vals)
ax2.set_ylabel('Incidence of death \nper 100,000 p.y.')
ax2.set_xlabel('Emergency care ISS cut off score')
plt.savefig(f"C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/SingleVsMultipleInjury/BatchResults/hsb_sweeps/"
            f"scaled_incidences_single.png", bbox_inches='tight')
print('Best ISS cut off score is ')
print(params['value'].to_list()[sing_best_fitting_scaled_index])
print('scale factor for current incidence of rti is ')
print(sing_scale_to_match_GBD[sing_best_fitting_scaled_index])
plt.clf()

fig, ax1 = plt.subplots()
ax1.set_xlabel('Model runs')
ax1.set_ylabel('Incidence of death\nper 100,000 p.y.')
ax1.bar(x_vals, sing_rescaled_incidence_of_death, width=0.4, color='lightsalmon',
        label='Incidence of death')
ax1.set_xticks(np.add(x_vals.to_list(), 0.2))
ax1.set_xticklabels(x_vals.to_list())
ax1.set_ylim([0, 120])
# Adding Twin Axes

ax2 = ax1.twinx()

ax2.set_ylabel('Percent sought care')
ax2.bar(np.add(x_vals.to_list(), 0.4), sing_mean_hsb[:, 'mean'], width=0.4, color='lightsteelblue',
        label='HSB')
ax2.set_ylim([0, 1.1])
ax1.set_title("The model's predicted incidence of death for \nvarying levels of health seeking behaviour")
# Show plot
ax1.legend(loc='upper left')
ax2.legend()
plt.savefig(f"C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/SingleVsMultipleInjury/BatchResults/hsb_sweeps/"
            f"hsb_inc_and_death.png", bbox_inches='tight')
