"""This file uses the results of the batch file to make some summary statistics.
The results of the bachrun were put into the 'outputs' results_folder
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import squarify

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
multiple_extracted_hsb = extract_results(multiple_injury_results_folder,
                                         module="tlo.methods.rti",
                                         key="summary_1m",
                                         column="percent sought healthcare",
                                         index="date")
sing_prop_sought_healthcare = summarize(single_extracted_hsb)
sing_prop_sought_healthcare_onlymeans = summarize(single_extracted_hsb, only_mean=True)
sing_mean_hsb = sing_prop_sought_healthcare.mean()
mult_prop_sought_healthcare = summarize(multiple_extracted_hsb)
mult_prop_sought_healthcare_onlymeans = summarize(multiple_extracted_hsb, only_mean=True)
mult_mean_hsb = mult_prop_sought_healthcare.mean()
# get upper and lower estimates
sing_prop_sought_healthcare_lower = sing_mean_hsb.loc[:, "lower"]
sing_prop_sought_healthcare_upper = sing_mean_hsb.loc[:, "upper"]
sing_lower_upper = np.array(list(zip(
    sing_prop_sought_healthcare_lower.to_list(),
    sing_prop_sought_healthcare_upper.to_list()
))).transpose()
mult_prop_sought_healthcare_lower = mult_mean_hsb.loc[:, "lower"]
mult_prop_sought_healthcare_upper = mult_mean_hsb.loc[:, "upper"]
mult_lower_upper = np.array(list(zip(
    mult_prop_sought_healthcare_lower.to_list(),
    mult_prop_sought_healthcare_upper.to_list()
))).transpose()
# name of parmaeter that varies
# find the values that fall within our accepted range of health seeking behaviour based on results of Zafar et al
# doi: 10.1016/j.ijsu.2018.02.034
expected_hsb_upper = 0.85
expected_hsb_lower = 0.6533
midpoint_hsb = (expected_hsb_upper + expected_hsb_lower) / 2
sing_per_param_average_hsb = sing_mean_hsb[:, 'mean'].values
sing_hsb_yerr = abs(sing_lower_upper - sing_per_param_average_hsb)
sing_in_accepted_range = np.where((sing_per_param_average_hsb > expected_hsb_lower) &
                                  (sing_per_param_average_hsb < expected_hsb_upper))
mult_per_param_average_hsb = mult_mean_hsb[:, 'mean'].values
mult_hsb_yerr = abs(mult_lower_upper - mult_per_param_average_hsb)
mult_in_accepted_range = np.where((mult_per_param_average_hsb > expected_hsb_lower) &
                                  (mult_per_param_average_hsb < expected_hsb_upper))
sing_closest_to_hsb_midpoint = min(sing_per_param_average_hsb, key=lambda x: abs(x - midpoint_hsb))
mult_closest_to_hsb_midpoint = min(mult_per_param_average_hsb, key=lambda x: abs(x - midpoint_hsb))

sing_index_of_midpoint = np.where(sing_per_param_average_hsb == sing_closest_to_hsb_midpoint)
mult_index_of_midpoint = np.where(mult_per_param_average_hsb == mult_closest_to_hsb_midpoint)

sing_colors = ['lightsteelblue' if i not in sing_in_accepted_range[0] else 'lightsalmon' for i in xvals]
sing_colors[sing_in_accepted_range[0][0]] = 'gold'
mult_colors = ['lightsteelblue' if i not in mult_in_accepted_range[0] else 'lightsalmon' for i in xvals]
mult_colors[mult_in_accepted_range[0][0]] = 'gold'
xlabels = [
    round(params.loc[(params.module_param == param_name)][['value']].loc[draw].value, 3)
    for draw in range(info['number_of_draws'])
]
fig, ax = plt.subplots()
ax.bar(
    x=xvals,
    height=sing_mean_hsb[:, 'mean'].values,
    yerr=sing_hsb_yerr,
    color=sing_colors
)
ax.set_xticks(xvals)
ax.set_xticklabels(xlabels)
plt.xlabel(param_name)
plt.ylabel('Percentage sought healthcare')
plt.title('Calibration of the ISS score which determines\nautomatic health care seeking, single injury model')
plt.savefig(f"C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/SingleVsMultipleInjury/BatchResults/hsb_sweeps/"
            f"ParameterSpaceForHSB_single.png", bbox_inches='tight')
plt.clf()
fig, ax = plt.subplots()
ax.bar(
    x=xvals,
    height=mult_mean_hsb[:, 'mean'].values,
    yerr=mult_hsb_yerr,
    color=mult_colors
)
ax.set_xticks(xvals)
ax.set_xticklabels(xlabels)
plt.xlabel(param_name)
plt.ylabel('Percentage sought healthcare')
plt.title('Calibration of the ISS score which determines\nautomatic health care seeking, multiple injury model')
plt.savefig(f"C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/SingleVsMultipleInjury/BatchResults/hsb_sweeps/"
            f"ParameterSpaceForHSB_multiple.png", bbox_inches='tight')
plt.clf()
sing_extracted_inc_death = extract_results(single_injury_results_folder,
                                           module="tlo.methods.rti",
                                           key="summary_1m",
                                           column="incidence of rti death per 100,000",
                                           index="date")
mult_extracted_inc_death = extract_results(multiple_injury_results_folder,
                                           module="tlo.methods.rti",
                                           key="summary_1m",
                                           column="incidence of rti death per 100,000",
                                           index="date")
sing_inc_death = summarize(sing_extracted_inc_death)
mult_inc_death = summarize(mult_extracted_inc_death)

sing_inc_death_only_means = summarize(sing_extracted_inc_death, only_mean=True)
mult_inc_death_only_means = summarize(mult_extracted_inc_death, only_mean=True)

sing_mean_inc_death_overall = sing_inc_death.mean()
mult_mean_inc_death_overall = mult_inc_death.mean()

sing_inc_death_mean_upper = sing_mean_inc_death_overall.loc[:, 'upper']
mult_inc_death_mean_upper = mult_mean_inc_death_overall.loc[:, 'upper']

sing_inc_death_mean_lower = sing_mean_inc_death_overall.loc[:, 'lower']
mult_inc_death_mean_lower = mult_mean_inc_death_overall.loc[:, 'lower']

sing_lower_upper_inc_death = np.array(list(zip(
    sing_inc_death_mean_lower.to_list(),
    sing_inc_death_mean_upper.to_list()
))).transpose()

mult_lower_upper_inc_death = np.array(list(zip(
    mult_inc_death_mean_lower.to_list(),
    mult_inc_death_mean_upper.to_list()
))).transpose()
sing_per_param_average_inc_death = sing_mean_inc_death_overall[:, 'mean'].values
sing_yerr_inc_death = abs(sing_lower_upper_inc_death - sing_per_param_average_inc_death)
mult_per_param_average_inc_death = mult_mean_inc_death_overall[:, 'mean'].values
mult_yerr_inc_death = abs(mult_lower_upper_inc_death - mult_per_param_average_inc_death)
WHO_est_in_death = 35
sing_best_fit_inc_death_found = min(sing_per_param_average_inc_death[sing_in_accepted_range],
                                    key=lambda x: abs(x - WHO_est_in_death))
mult_best_fit_inc_death_found = min(mult_per_param_average_inc_death[mult_in_accepted_range],
                                    key=lambda x: abs(x - WHO_est_in_death))
sing_best_fit_death_index = np.where(sing_per_param_average_inc_death == sing_best_fit_inc_death_found)
mult_best_fit_death_index = np.where(mult_per_param_average_inc_death == mult_best_fit_inc_death_found)

plt.bar(x=xvals,
        height=sing_mean_inc_death_overall[:, 'mean'].values,
        yerr=sing_yerr_inc_death,
        color=sing_colors)
for idx, inc in enumerate(sing_per_param_average_inc_death):
    plt.text(idx - 0.5, sing_inc_death_mean_upper.to_list()[idx] + 5,
             f"{np.round(sing_mean_inc_death_overall[:, 'mean'][idx], 2)}")
plt.xticks(xvals, xlabels)
plt.xlabel(param_name)
plt.ylabel('Incidence of death per 100,000 person years')
plt.title('Effect of health seeking behaviour on the incidence of death')
plt.savefig(f"C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/SingleVsMultipleInjury/BatchResults/hsb_sweeps/"
            f"effect_of_hsb_on_incidence_of_death_single.png", bbox_inches='tight')

plt.clf()
plt.bar(x=xvals,
        height=mult_mean_inc_death_overall[:, 'mean'].values,
        yerr=mult_yerr_inc_death,
        color=mult_colors)
for idx, inc in enumerate(mult_per_param_average_inc_death):
    plt.text(idx - 0.5, mult_inc_death_mean_upper.to_list()[idx] + 5,
             f"{np.round(mult_mean_inc_death_overall[:, 'mean'][idx], 2)}")
plt.xticks(xvals, xlabels)
plt.xlabel(param_name)
plt.ylabel('Incidence of death per 100,000 person years')
plt.title('Effect of health seeking behaviour on the incidence of death')
plt.savefig(f"C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/SingleVsMultipleInjury/BatchResults/hsb_sweeps/"
            f"effect_of_hsb_on_incidence_of_death_multiple.png", bbox_inches='tight')

sing_extracted_inc = extract_results(single_injury_results_folder,
                                     module="tlo.methods.rti",
                                     key="summary_1m",
                                     column="incidence of rti per 100,000",
                                     index="date")
mult_extracted_inc = extract_results(multiple_injury_results_folder,
                                     module="tlo.methods.rti",
                                     key="summary_1m",
                                     column="incidence of rti per 100,000",
                                     index="date")

# 3) Get summary of the results for that log-element
sing_inc = summarize(sing_extracted_inc)
mult_inc = summarize(mult_extracted_inc)
# If only interested in the means
sing_inc_only_means = summarize(sing_extracted_inc, only_mean=True)
mult_inc_only_means = summarize(mult_extracted_inc, only_mean=True)

# get per parameter summaries
sing_mean_inc_overall = sing_inc.mean()
sing_inc_mean_upper = sing_mean_inc_overall.loc[:, 'upper']
sing_inc_mean_lower = sing_mean_inc_overall.loc[:, 'lower']
sing_lower_upper_inc = np.array(list(zip(
    sing_inc_mean_lower.to_list(),
    sing_inc_mean_upper.to_list()
))).transpose()
mult_mean_inc_overall = mult_inc.mean()
mult_inc_mean_upper = mult_mean_inc_overall.loc[:, 'upper']
mult_inc_mean_lower = mult_mean_inc_overall.loc[:, 'lower']
mult_lower_upper_inc = np.array(list(zip(
    mult_inc_mean_lower.to_list(),
    mult_inc_mean_upper.to_list()
))).transpose()

sing_per_param_average_inc = sing_mean_inc_overall[:, 'mean'].values
mult_per_param_average_inc = mult_mean_inc_overall[:, 'mean'].values

sing_yerr_inc = abs(sing_lower_upper_inc - sing_per_param_average_inc)
mult_yerr_inc = abs(mult_lower_upper_inc - mult_per_param_average_inc)

GBD_est_inc = 954.2
sing_best_fit_found_for_inc = min(sing_per_param_average_inc[sing_in_accepted_range],
                                  key=lambda x: abs(x - GBD_est_inc))
mult_best_fit_found_for_inc = min(mult_per_param_average_inc[mult_in_accepted_range],
                                  key=lambda x: abs(x - GBD_est_inc))
sing_best_fit_for_inc_index = np.where(sing_per_param_average_inc == sing_best_fit_found_for_inc)
mult_best_fit_for_inc_index = np.where(mult_per_param_average_inc == mult_best_fit_found_for_inc)

plt.clf()
plt.bar(x=xvals,
        height=sing_mean_inc_overall[:, 'mean'].values,
        yerr=sing_yerr_inc,
        color=sing_colors)
for idx, inc in enumerate(sing_per_param_average_inc):
    plt.text(idx - 0.5, sing_inc_mean_upper.to_list()[idx] + 5, f"{np.round(sing_mean_inc_overall[:, 'mean'][idx], 1)}")
plt.xticks(xvals, xlabels)
plt.xlabel(param_name)
plt.ylabel('Incidence of RTI per 100,000 person years')
plt.title('Effect of health seeking behaviour on the incidence of RTI')
plt.savefig(f"C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/SingleVsMultipleInjury/BatchResults/hsb_sweeps/"
            f"effect_of_hsb_on_incidence_of_rti_single.png", bbox_inches='tight')
plt.clf()

plt.bar(x=xvals,
        height=mult_mean_inc_overall[:, 'mean'].values,
        yerr=mult_yerr_inc,
        color=mult_colors)
for idx, inc in enumerate(mult_per_param_average_inc):
    plt.text(idx - 0.5, mult_inc_mean_upper.to_list()[idx] + 5, f"{np.round(mult_mean_inc_overall[:, 'mean'][idx], 1)}")
plt.xticks(xvals, xlabels)
plt.xlabel(param_name)
plt.ylabel('Incidence of RTI per 100,000 person years')
plt.title('Effect of health seeking behaviour on the incidence of RTI')
plt.savefig(f"C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/SingleVsMultipleInjury/BatchResults/hsb_sweeps/"
            f"effect_of_hsb_on_incidence_of_rti_multiple.png", bbox_inches='tight')
plt.clf()
sing_percent_crashed_that_are_fatal = extract_results(single_injury_results_folder,
                                                      module="tlo.methods.rti",
                                                      key="summary_1m",
                                                      column="percent of crashes that are fatal",
                                                      index="date")
mult_percent_crashed_that_are_fatal = extract_results(multiple_injury_results_folder,
                                                      module="tlo.methods.rti",
                                                      key="summary_1m",
                                                      column="percent of crashes that are fatal",
                                                      index="date")
# 3) Get summary of the results for that log-element
sing_percent_fatal = summarize(sing_percent_crashed_that_are_fatal)
mult_percent_fatal = summarize(mult_percent_crashed_that_are_fatal)

# If only interested in the means
sing_percent_fatal_only_means = summarize(sing_percent_crashed_that_are_fatal, only_mean=True)
mult_percent_fatal_only_means = summarize(mult_percent_crashed_that_are_fatal, only_mean=True)

# get per parameter summaries
sing_mean_percent_fatal_overall = sing_percent_fatal.mean()
sing_perc_fatal_mean_lower = sing_mean_percent_fatal_overall.loc[:, 'lower']
sing_perc_fatal_mean_upper = sing_mean_percent_fatal_overall.loc[:, 'upper']
sing_lower_upper_perc_fatal = np.array(list(zip(
    sing_perc_fatal_mean_lower.to_list(),
    sing_perc_fatal_mean_upper.to_list()
))).transpose()
mult_mean_percent_fatal_overall = mult_percent_fatal.mean()
mult_perc_fatal_mean_lower = mult_mean_percent_fatal_overall.loc[:, 'lower']
mult_perc_fatal_mean_upper = mult_mean_percent_fatal_overall.loc[:, 'upper']
mult_lower_upper_perc_fatal = np.array(list(zip(
    mult_perc_fatal_mean_lower.to_list(),
    mult_perc_fatal_mean_upper.to_list()
))).transpose()

sing_per_param_perc_fatal = sing_mean_percent_fatal_overall[:, 'mean'].values
sing_yerr_perc_fatal = abs(sing_lower_upper_perc_fatal - sing_per_param_perc_fatal)
plt.bar(x=xvals,
        height=sing_mean_percent_fatal_overall[:, 'mean'].values,
        yerr=sing_yerr_perc_fatal,
        color=sing_colors)
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

mult_per_param_perc_fatal = mult_mean_percent_fatal_overall[:, 'mean'].values
mult_yerr_perc_fatal = abs(mult_lower_upper_perc_fatal - mult_per_param_perc_fatal)
plt.bar(x=xvals,
        height=mult_mean_percent_fatal_overall[:, 'mean'].values,
        yerr=mult_yerr_perc_fatal,
        color=mult_colors)
for idx, inc in enumerate(mult_per_param_perc_fatal):
    plt.text(idx - 0.5, mult_perc_fatal_mean_upper.to_list()[idx] + 0.02,
             f"{np.round(mult_mean_percent_fatal_overall[:, 'mean'][idx], 3)}")
plt.xticks(xvals, xlabels)
plt.xlabel(param_name)
plt.ylabel('Percent Fatal')
plt.title('Effect of health seeking behaviour on the percent of crashed that are fatal')
plt.savefig(f"C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/SingleVsMultipleInjury/BatchResults/hsb_sweeps/"
            f"effect_of_hsb_on_percent_fatal_multiple.png", bbox_inches='tight')
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

mult_scale_to_match_GBD = np.divide(GBD_est_inc, mult_mean_inc_overall[:, 'mean'].values)
mult_scaled_incidences = mult_mean_inc_overall[:, 'mean'].values * mult_scale_to_match_GBD
mult_rescaled_incidence_of_death = mult_mean_inc_death_overall[:, 'mean'].values * mult_scale_to_match_GBD
mult_best_fitting_scaled = min(mult_rescaled_incidence_of_death, key=lambda x: abs(x - WHO_est_in_death))
mult_best_fitting_scaled_index = np.where(mult_rescaled_incidence_of_death == mult_best_fitting_scaled)[0][0]
mult_colors_for_inc = ['lightsteelblue'] * len(mult_scaled_incidences)
mult_colors_for_inc[mult_best_fitting_scaled_index] = 'gold'
mult_colors_for_inc_death = ['lightsalmon'] * len(mult_rescaled_incidence_of_death)
mult_colors_for_inc_death[mult_best_fitting_scaled_index] = 'gold'

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

ax1 = fig.add_subplot(gs[0, 0])
ax1.bar(np.arange(len(mult_scaled_incidences)), mult_scaled_incidences, color=mult_colors_for_inc, width=0.4)
for idx, val in enumerate(mult_scaled_incidences):
    ax1.text(idx, val, f"{np.round(val, 2)}", rotation=90)
ax1.set_title('Incidence of RTI')
ax1.set_xticks(np.arange(len(x_vals)))
ax1.set_xticklabels(x_vals)
ax1.set_ylabel('Incidence per 100,000 p.y.')
ax2 = fig.add_subplot(gs[1, 0])
ax2.set_title('Incidence of Death')
ax2.bar(np.arange(len(mult_rescaled_incidence_of_death)), mult_rescaled_incidence_of_death,
        color=mult_colors_for_inc_death, width=0.4)
for idx, val in enumerate(mult_rescaled_incidence_of_death):
    ax2.text(idx, val, f"{np.round(val, 2)}", rotation=90)
ax2.set_xticks(np.arange(len(x_vals)))
ax2.set_xticklabels(x_vals)
ax2.set_ylabel('Incidence of death \nper 100,000 p.y.')
ax2.set_xlabel('Emergency care ISS cut off score')
plt.savefig(f"C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/SingleVsMultipleInjury/BatchResults/hsb_sweeps/"
            f"scaled_incidences_multiple.png", bbox_inches='tight')
print('Best ISS cut off score is ')
print(params['value'].to_list()[mult_best_fitting_scaled_index])
print('scale factor for current incidence of rti is ')
print(mult_scale_to_match_GBD[mult_best_fitting_scaled_index])
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
            f"hsb_inc_and_death_single.png", bbox_inches='tight')
plt.clf()
fig, ax1 = plt.subplots()
ax1.set_xlabel('Model runs')
ax1.set_ylabel('Incidence of death\nper 100,000 p.y.')
ax1.bar(x_vals, mult_rescaled_incidence_of_death, width=0.4, color='lightsalmon',
        label='Incidence of death')
ax1.set_xticks(np.add(x_vals.to_list(), 0.2))
ax1.set_xticklabels(x_vals.to_list())
ax1.set_ylim([0, 120])
# Adding Twin Axes

ax2 = ax1.twinx()

ax2.set_ylabel('Percent sought care')
ax2.bar(np.add(x_vals.to_list(), 0.4), mult_mean_hsb[:, 'mean'], width=0.4, color='lightsteelblue',
        label='HSB')
ax2.set_ylim([0, 1.1])
ax1.set_title("The model's predicted incidence of death for \nvarying levels of health seeking behaviour")
# Show plot
ax1.legend(loc='upper left')
ax2.legend()
plt.savefig(f"C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/SingleVsMultipleInjury/BatchResults/hsb_sweeps/"
            f"hsb_inc_and_death_multiple.png", bbox_inches='tight')
print('breakpoint')
def extract_yll_yld(results_folder):
    yll = pd.DataFrame()
    yld = pd.DataFrame()
    info = get_scenario_info(results_folder)
    for draw in range(info['number_of_draws']):
        yll_this_draw = []
        yld_this_draw = []
        for run in range(info['runs_per_draw']):
            try:
                yll_df: pd.DataFrame = \
                    load_pickled_dataframes(
                        results_folder, draw, run, "tlo.methods.healthburden"
                        )["tlo.methods.healthburden"]
                yll_df = yll_df['yll_by_causes_of_death_stacked']
                yll_df = yll_df.groupby('year').sum()
                rti_columns = [col for col in yll_df.columns if 'RTI' in col]
                yll_df['yll_rti'] = [0.0] * len(yll_df)
                for col in rti_columns:
                    yll_df['yll_rti'] += yll_df[col]
                sim_start_year = min(sing_inc.index.year)
                sim_end_year = max(sing_inc.index.year)
                sim_year_range = pd.Index(np.arange(sim_start_year, sim_end_year + 1))
                pop_size_df: pd.DataFrame = \
                    load_pickled_dataframes(
                        results_folder, draw, run, "tlo.methods.demography"
                    )["tlo.methods.demography"]
                pop_size_df = pop_size_df['population']
                pop_size_df['year'] = pop_size_df['date'].dt.year
                pop_size_df = pop_size_df.loc[pop_size_df['year'].isin(sim_year_range)]
                scaling_df = pd.DataFrame({'total': pop_size_df['total']})
                data = pd.read_csv("resources/demography/ResourceFile_Pop_Annual_WPP.csv")
                Data_Pop = data.groupby(by="Year")["Count"].sum()
                Data_Pop = Data_Pop.loc[sim_year_range]
                scaling_df['pred_pop_size'] = Data_Pop.to_list()
                scaling_df['scale_for_each_year'] = scaling_df['pred_pop_size'] / scaling_df['total']
                scaling_df.index = sim_year_range
                yll_df = yll_df.loc[sim_year_range]
                yll_df['scaled_yll'] = yll_df['yll_rti'] * scaling_df['scale_for_each_year']
                total_yll = yll_df['scaled_yll'].sum()
                yll_this_draw.append(total_yll)
                yld_df: pd.DataFrame = \
                    load_pickled_dataframes(results_folder, draw, run, "tlo.methods.rti")["tlo.methods.rti"]
                yld_df = yld_df['rti_health_burden_per_day']
                yld_df['year'] = yld_df['date'].dt.year
                yld_df = yld_df.groupby('year').sum()
                yld_df['total_daily_healthburden'] = \
                    [sum(daly_weights) for daly_weights in yld_df['daly_weights'].to_list()]
                yld_df['scaled_healthburden'] = yld_df['total_daily_healthburden'] * \
                                                scaling_df['scale_for_each_year'] / 365
                total_yld = yld_df['scaled_healthburden'].sum()
                yld_this_draw.append(total_yld)
            except KeyError:
                yll_this_draw.append(np.mean(yll_this_draw))
                yld_this_draw.append(np.mean(yld_this_draw))
        yll[str(draw)] = yll_this_draw
        yld[str(draw)] = yld_this_draw
    return yll, yld

sing_yll, sing_yld = extract_yll_yld(single_injury_results_folder)
mult_yll, mult_yld = extract_yll_yld(multiple_injury_results_folder)
gbd_dates = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019]
gbd_yld_estimate_2010_2019 = [17201.73, 16689.13, 18429.77, 17780.11, 20462.97, 19805.86, 21169.19, 19100.62,
                              23081.26, 22055.06]
gbd_yll_estimate_2010_2019 = [103892.353, 107353.63, 107015.04, 106125.14, 105933.16, 106551.59, 106424.49,
                              105551.97, 108052.59, 109301.18]
gbd_dalys_estimate_2010_2019 = np.add(gbd_yld_estimate_2010_2019, gbd_yll_estimate_2010_2019)
gbd_data = pd.DataFrame(data={'yld': gbd_yld_estimate_2010_2019, 'yll': gbd_yll_estimate_2010_2019,
                              'dalys': gbd_dalys_estimate_2010_2019},
                        index=gbd_dates)
sing_mean_yll = sing_yll.mean()
sing_mean_yll = sing_mean_yll * sing_scale_to_match_GBD
sing_mean_yld = sing_yld.mean()
sing_mean_yld = sing_mean_yld * sing_scale_to_match_GBD
sing_mean_daly = np.add(sing_mean_yld.values, sing_mean_yll.values)
mult_mean_yll = mult_yll.mean()
mult_mean_yll = mult_mean_yll * mult_scale_to_match_GBD
mult_mean_yld = mult_yld.mean()
mult_mean_yld = mult_mean_yld * mult_scale_to_match_GBD
mult_mean_daly = np.add(mult_mean_yld.values, mult_mean_yll.values)
plt.clf()
fig = plt.figure(constrained_layout=True)
# Use GridSpec for customising layout
gs = fig.add_gridspec(nrows=2, ncols=1)
ax1 = fig.add_subplot(gs[0, 0])
ax1.barh(np.arange(len(sing_mean_yll)) + 1, sing_mean_yld, color='steelblue', label='YLD')
ax1.barh(np.arange(len(sing_mean_yll)) + 1, sing_mean_yll, color='lightskyblue', label='YLL', left=sing_mean_yld)
ax1.vlines(gbd_data['dalys'].sum(), 0.5, 11.5, colors='b', linestyle='dashed', label='GBD estimate')
ax1.set_ylabel('Draw number')
ax1.set_xlabel('DALYs')
ax1.set_yticks(np.arange(len(sing_mean_yll)) + 1)
ax1.legend()
ax1.set_title('Single injury model')
ax2 = fig.add_subplot(gs[1, 0])
ax2.barh(np.arange(len(mult_mean_yld)) + 1, mult_mean_yld, color='coral', label='YLD')
ax2.barh(np.arange(len(mult_mean_yll)) + 1, mult_mean_yll, color='darksalmon', label='YLL', left=mult_mean_yld)
ax2.vlines(gbd_data['dalys'].sum(), 0.5, 11.5, colors='r', linestyle='dashed', label='GBD estimate')
ax2.set_ylabel('Draw number')
ax2.set_yticks(np.arange(len(mult_mean_yld)) + 1)
ax2.legend()
ax2.set_xlabel('DALYs')
ax2.set_title('Multiple injury model')
plt.savefig(f"C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/SingleVsMultipleInjury/BatchResults/hsb_sweeps/"
            f"DALYs.png", bbox_inches='tight')
plt.clf()
fig, ax1 = plt.subplots()
ax1.set_xlabel('Model runs')
ax1.set_ylabel('DALYs')
ax1.bar(x_vals, mult_mean_daly, width=0.4, color='lightsalmon',
        label='DALYs')
ax1.set_xticks(np.add(x_vals.to_list(), 0.2))
ax1.set_xticklabels(x_vals.to_list())
# ax1.set_ylim([0, 120])
# Adding Twin Axes

ax2 = ax1.twinx()

ax2.set_ylabel('Percent sought care')
ax2.bar(np.add(x_vals.to_list(), 0.4), mult_mean_hsb[:, 'mean'], width=0.4, color='lightsteelblue',
        label='HSB')
ax2.set_ylim([0, 1.1])
ax1.set_title("The model's predicted number of DALYs for \nvarying levels of health seeking behaviour")
# Show plot
ax1.legend(loc='upper left')
ax2.legend()
plt.savefig(f"C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/SingleVsMultipleInjury/BatchResults/hsb_sweeps/"
            f"hsb_DALYs_multiple.png", bbox_inches='tight')
plt.clf()
fig, ax1 = plt.subplots()
ax1.set_xlabel('Model runs')
ax1.set_ylabel('DALYs')
ax1.bar(x_vals, sing_mean_daly, width=0.4, color='lightsalmon',
        label='DALYs')
ax1.set_xticks(np.add(x_vals.to_list(), 0.2))
ax1.set_xticklabels(x_vals.to_list())
# Adding Twin Axes

ax2 = ax1.twinx()

ax2.set_ylabel('Percent sought care')
ax2.bar(np.add(x_vals.to_list(), 0.4), mult_mean_hsb[:, 'mean'], width=0.4, color='lightsteelblue',
        label='HSB')
ax2.set_ylim([0, 1.1])
ax1.set_title("The model's predicted number of DALYs for \nvarying levels of health seeking behaviour")
# Show plot
ax1.legend(loc='upper left')
ax2.legend()
plt.savefig(f"C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/SingleVsMultipleInjury/BatchResults/hsb_sweeps/"
            f"hsb_DALYs_single.png", bbox_inches='tight')
plt.clf()
fig, ax1 = plt.subplots()
ax1.set_xlabel('Model runs')
ax1.set_ylabel('DALYs')
ax1.bar(x_vals, mult_mean_daly, width=0.4, color='lightsalmon',
        label='DALYs')
ax1.set_xticks(np.add(x_vals.to_list(), 0.2))
ax1.set_xticklabels(x_vals.to_list())
# ax1.set_ylim([0, 120])
# Adding Twin Axes

ax2 = ax1.twinx()

ax2.set_ylabel('Percent sought care')
ax2.bar(np.add(x_vals.to_list(), 0.4), mult_mean_hsb[:, 'mean'], width=0.4, color='lightsteelblue',
        label='HSB')
ax2.axhline(y=expected_hsb_lower, color='steelblue', linestyle='dashed', label='lower HSB\nboundary')
ax2.axhline(y=expected_hsb_upper, color='lightskyblue', linestyle='dashed', label='upper HSB\nboundary')
ax2.set_ylim([0, 1.1])
ax1.set_title("The model's predicted number of DALYs for \nvarying levels of health seeking behaviour")
# Show plot
ax1.legend(loc='upper left')
ax2.legend()
plt.savefig(f"C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/SingleVsMultipleInjury/BatchResults/hsb_sweeps/"
            f"hsb_DALYs_multiple_with_hsb_bound.png", bbox_inches='tight')
plt.clf()
fig, ax1 = plt.subplots()
ax1.set_xlabel('Model runs')
ax1.set_ylabel('DALYs')
ax1.bar(x_vals, sing_mean_daly, width=0.4, color='lightsalmon',
        label='DALYs')
ax1.set_xticks(np.add(x_vals.to_list(), 0.2))
ax1.set_xticklabels(x_vals.to_list())
# ax1.set_ylim([0, 120])
# Adding Twin Axes

ax2 = ax1.twinx()

ax2.set_ylabel('Percent sought care')
ax2.bar(np.add(x_vals.to_list(), 0.4), mult_mean_hsb[:, 'mean'], width=0.4, color='lightsteelblue',
        label='HSB')
ax2.axhline(y=expected_hsb_lower, color='steelblue', linestyle='dashed', label='lower HSB\nboundary')
ax2.axhline(y=expected_hsb_upper, color='lightskyblue', linestyle='dashed', label='upper HSB\nboundary')
ax2.set_ylim([0, 1.1])
ax1.set_title("The model's predicted number of DALYs for \nvarying levels of health seeking behaviour")
# Show plot
ax1.legend(loc='upper left')
ax2.legend()
plt.savefig(f"C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/SingleVsMultipleInjury/BatchResults/hsb_sweeps/"
            f"hsb_DALYs_single_with_hsb_bound.png", bbox_inches='tight')
plt.clf()
fig, ax1 = plt.subplots()
ax1.set_xlabel('Model runs')
ax1.set_ylabel('Incidence of death\nper 100,000 p.y.')
ax1.bar(x_vals, mult_rescaled_incidence_of_death, width=0.4, color='lightsalmon',
        label='Incidence of Death')
ax1.set_xticks(np.add(x_vals.to_list(), 0.2))
ax1.set_xticklabels(x_vals.to_list())
ax1.set_ylim([0, 120])
# Adding Twin Axes

ax2 = ax1.twinx()

ax2.set_ylabel('Percent sought care')
ax2.bar(np.add(x_vals.to_list(), 0.4), mult_mean_hsb[:, 'mean'], width=0.4, color='lightsteelblue',
        label='HSB')
ax2.axhline(y=expected_hsb_lower, color='steelblue', linestyle='dashed', label='lower HSB\nboundary')
ax2.axhline(y=expected_hsb_upper, color='lightskyblue', linestyle='dashed', label='upper HSB\nboundary')
ax2.set_ylim([0, 1.1])
ax1.set_title("The model's predicted incidence of death for \nvarying levels of health seeking behaviour")
# Show plot
ax1.legend(loc='upper left')
ax2.legend()
plt.savefig(f"C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/SingleVsMultipleInjury/BatchResults/hsb_sweeps/"
            f"hsb_inc_death_multiple_with_hsb_bound.png", bbox_inches='tight')
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
ax2.bar(np.add(x_vals.to_list(), 0.4), mult_mean_hsb[:, 'mean'], width=0.4, color='lightsteelblue',
        label='HSB')
ax2.axhline(y=expected_hsb_lower, color='steelblue', linestyle='dashed', label='lower HSB\nboundary')
ax2.axhline(y=expected_hsb_upper, color='lightskyblue', linestyle='dashed', label='upper HSB\nboundary')
ax2.set_ylim([0, 1.1])
ax1.set_title("The model's predicted incidence of death for \nvarying levels of health seeking behaviour")
# Show plot
ax1.legend(loc='upper left')
ax2.legend()
plt.savefig(f"C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/SingleVsMultipleInjury/BatchResults/hsb_sweeps/"
            f"hsb_incidence_of_death_single_with_hsb_bound.png", bbox_inches='tight')
plt.clf()
gbd_incidence_death = 12.1

average_n_inj_per_draws = []
for draw in range(info['number_of_draws']):
    ave_n_inj_this_draw = []
    for run in range(info['runs_per_draw']):
        try:
            df: pd.DataFrame = \
                load_pickled_dataframes(multiple_injury_results_folder, draw, run, "tlo.methods.rti")["tlo.methods.rti"]
            df = df['Injury_information']
            total_n_injuries = sum(df.sum(axis=0)['Number_of_injuries'])
            injuries_per_person = total_n_injuries / len(df.sum(axis=0)['Number_of_injuries'])
            ave_n_inj_this_draw.append(injuries_per_person)
        except KeyError:
            ave_n_inj_this_draw.append(np.mean(ave_n_inj_this_draw))
    average_n_inj_per_draws.append(np.mean(ave_n_inj_this_draw))

gbd_incidence_death = 12.1
plt.clf()
for index in mult_in_accepted_range[0]:
    sing_scaled_inc_death = sing_rescaled_incidence_of_death[index]
    mult_scaled_inc_death = mult_rescaled_incidence_of_death[index]
    gbd_results = [GBD_est_inc, gbd_incidence_death, GBD_est_inc]
    single_results = [sing_scaled_incidences[index], sing_scaled_inc_death, sing_scaled_incidences[index]]
    mult_results = [mult_scaled_incidences[index], mult_scaled_inc_death,
                    mult_scaled_incidences[index] * average_n_inj_per_draws[index]]
    plt.bar(np.arange(3), gbd_results, width=0.25, color='gold', label='GBD')
    plt.bar(np.arange(3) + 0.25, single_results, width=0.25, color='lightsteelblue', label='Single')
    plt.bar(np.arange(3) + 0.5, mult_results, width=0.25,
            color='lightsalmon', label='Multiple')
    plt.xticks(np.arange(3) + 0.25, ['Incidence\nof\nRTI', 'Incidence\nof\ndeath', 'Incidence\nof\ninjuries'])
    for idx, val in enumerate(gbd_results):
        plt.text(np.arange(3)[idx] - 0.125, gbd_results[idx] + 10, f"{np.round(val, 2)}", fontdict={'fontsize': 9},
                 rotation=45)
    for idx, val in enumerate(single_results):
        plt.text(np.arange(3)[idx] + 0.25 - 0.125, single_results[idx] + 10, f"{np.round(val, 2)}",
                 fontdict={'fontsize': 9}, rotation=45)
    for idx, val in enumerate(mult_results):
        plt.text(np.arange(3)[idx] + 0.5 - 0.125, mult_results[idx] + 10, f"{np.round(val, 2)}",
                 fontdict={'fontsize': 9},
                 rotation=45)
    plt.legend()
    plt.title('Comparing the incidence of RTI, RTI death and injuries\nfor the GBD study, single injury model and\n'
              'multiple injury model')
    plt.ylabel('Incidence per \n 100,000 person years')
    plt.savefig(f"C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/SingleVsMultipleInjury/BatchResults/"
                f"IncidenceSummary_ISS_cut_off_is_{x_vals[index]}.png", bbox_inches='tight')
    plt.clf()

mult_incidence_of_death = mult_rescaled_incidence_of_death[mult_in_accepted_range[0]]
mult_dalys = mult_mean_daly[mult_in_accepted_range[0]]
mult_percent_sought_care = mult_mean_hsb[:, 'mean'][mult_in_accepted_range[0]]
sing_incidence_of_death = sing_rescaled_incidence_of_death[sing_in_accepted_range[0]]
sing_dalys = sing_mean_daly[sing_in_accepted_range[0]]
sing_percent_sought_care = sing_mean_hsb[:, 'mean'][sing_in_accepted_range[0]]
plt.clf()
fig = plt.figure(constrained_layout=True)
# Use GridSpec for customising layout
gs = fig.add_gridspec(nrows=2, ncols=1)
ax1 = fig.add_subplot(gs[0, 0])
ax1.bar(np.arange(len(mult_in_accepted_range[0])), mult_incidence_of_death, color='lightsteelblue',
        label='Incidence of\ndeath', width=0.4)
ax1.set_xticks(np.arange(len(mult_in_accepted_range[0])) + 0.2)
ax1.set_xticklabels([str(np.round(percent * 100, 2)) + "%" for percent in mult_percent_sought_care])
ax1.set_ylabel('Incidence of death\n per 100,000 p.y.')
ax1.set_title('Multiple injury model')
ax1.set_ylim([0, 50])
ax2 = ax1.twinx()
ax2.set_ylabel('DALYs')
ax2.bar(np.arange(len(mult_in_accepted_range[0])) + 0.4, mult_dalys, width=0.4, color='lightsalmon',
        label='DALYs')
ax2.set_ylim([0, max(mult_dalys) * 1.4])
ax1.legend(loc='upper left')
ax2.legend()
ax3 = fig.add_subplot(gs[1, 0])
ax3.bar(np.arange(len(sing_in_accepted_range[0])), sing_incidence_of_death, color='steelblue',
        label='Incidence of\ndeath', width=0.4)
ax3.set_xticks(np.arange(len(sing_in_accepted_range[0])) + 0.2)
ax3.set_xticklabels([str(np.round(percent * 100, 2)) + "%" for percent in sing_percent_sought_care])
ax3.set_ylabel('Incidence of death\n per 100,000 p.y.')
ax3.set_title('Single injury model')
ax3.set_ylim([0, 50])
ax4 = ax3.twinx()
ax4.set_ylabel('DALYs')
ax4.bar(np.arange(len(mult_in_accepted_range[0])) + 0.4, mult_dalys, width=0.4, color='darksalmon',
        label='DALYs')
ax4.set_ylim([0, max(mult_dalys) * 1.4])
ax3.legend(loc='upper left')
ax4.legend()
ax3.set_xlabel('Percent sought care in run')
plt.savefig(f"C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/SingleVsMultipleInjury/BatchResults/hsb_sweeps/"
            f"Inaccepted_bounds_summary.png", bbox_inches='tight')
plt.clf()
percent_increase_in_deaths = np.divide(mult_rescaled_incidence_of_death, sing_rescaled_incidence_of_death)
percent_increase_in_deaths = percent_increase_in_deaths * 100
percent_increase_in_deaths = percent_increase_in_deaths[mult_in_accepted_range[0]] - 100
plt.bar(np.arange(3), percent_increase_in_deaths, color='lightsteelblue')
for idx, percent in enumerate(percent_increase_in_deaths):
    plt.text(np.arange(3)[idx], percent_increase_in_deaths[idx] + 0.1,
             f"{np.round(percent_increase_in_deaths[idx], 2)}")
plt.ylabel('Percentage')
plt.xticks(np.arange(3), x_vals[mult_in_accepted_range[0]])
plt.xlabel('ISS cut off score')
plt.title('Percentage increase in deaths due to considering multiple injuries')
plt.savefig(f"C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/SingleVsMultipleInjury/BatchResults/hsb_sweeps/"
            f"In_accepted_bounds_percent_increase_deaths.png", bbox_inches='tight')
plt.clf()
percent_increase_in_dalys = np.divide(mult_dalys, sing_dalys)
percent_increase_in_dalys = percent_increase_in_dalys * 100
percent_increase_in_dalys = percent_increase_in_dalys - 100
plt.bar(np.arange(3), percent_increase_in_dalys, color='lightsteelblue')
for idx, percent in enumerate(percent_increase_in_dalys):
    plt.text(np.arange(3)[idx], percent_increase_in_dalys[idx] + 0.1,
             f"{np.round(percent_increase_in_dalys[idx], 2)}")
plt.ylabel('Percentage')
plt.xticks(np.arange(3), x_vals[mult_in_accepted_range[0]])
plt.xlabel('ISS cut off score')
plt.title('Percentage increase in DALYs due to considering multiple injuries')
plt.savefig(f"C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/SingleVsMultipleInjury/BatchResults/hsb_sweeps/"
            f"In_accepted_bounds_percent_increase_dalys.png", bbox_inches='tight')
results_df = pd.DataFrame()
results_df['sing_inc_death_unscaled'] = sing_mean_inc_death_overall[:, 'mean'].values
results_df['sing_inc_death_scaled'] = sing_rescaled_incidence_of_death
results_df['sing_inc_unscaled'] = sing_mean_inc_overall[:, 'mean'].values
results_df['sing_inc_scaled'] = sing_scaled_incidences
results_df['sing_percent_hsb'] = sing_mean_hsb[:, 'mean'].values
results_df['sing_yll_unscaled'] = sing_yll.mean().values
results_df['sing_yll_scaled'] = sing_mean_yll.values
results_df['sing_yld_unscaled'] = sing_yld.mean().values
results_df['sing_yld_scaled'] = sing_mean_yld.values
results_df['sing_dalys_unscaled'] = results_df['sing_yll_unscaled'] + results_df['sing_yld_unscaled']
results_df['sing_dalys_scaled'] = results_df['sing_yll_scaled'] + results_df['sing_yld_scaled']
results_df['sing_within_hsb_bounds'] = [False] * len(results_df)
results_df.loc[sing_in_accepted_range[0], 'sing_within_hsb_bounds'] = True
results_df['mult_inc_death_unscaled'] = mult_mean_inc_death_overall[:, 'mean'].values
results_df['mult_inc_death_scaled'] = mult_rescaled_incidence_of_death
results_df['mult_inc_unscaled'] = mult_mean_inc_overall[:, 'mean'].values
results_df['mult_inc_scaled'] = mult_scaled_incidences
results_df['mult_percent_hsb'] = mult_mean_hsb[:, 'mean'].values
results_df['mult_yll_unscaled'] = mult_yll.mean().values
results_df['mult_yll_scaled'] = mult_mean_yll.values
results_df['mult_yld_unscaled'] = mult_yld.mean().values
results_df['mult_yld_scaled'] = mult_mean_yld.values
results_df['mult_dalys_unscaled'] = results_df['mult_yll_unscaled'] + results_df['mult_yld_unscaled']
results_df['mult_dalys_scaled'] = results_df['mult_yll_scaled'] + results_df['mult_yld_scaled']
results_df['mult_within_hsb_bounds'] = [False] * len(results_df)
results_df.loc[sing_in_accepted_range[0], 'mult_within_hsb_bounds'] = True
results_df.to_csv(f"C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/SingleVsMultipleInjury/BatchResults/"
                  f"hsb_sweeps/results.csv")
plt.clf()
fig = plt.figure(constrained_layout=True)
# Use GridSpec for customising layout
gs = fig.add_gridspec(nrows=2, ncols=1)
ax1 = fig.add_subplot(gs[0, 0])
ax1.barh(np.arange(len(sing_rescaled_incidence_of_death)) + 1, sing_rescaled_incidence_of_death,
         color='steelblue', label='Model')
ax1.vlines(12.1, 0.5, 11.5, colors='b', linestyle='dashed', label='GBD estimate')
ax1.set_ylabel('Draw number')
ax1.set_xlabel('Incidence of Death per 100,000 p.y.')
ax1.set_yticks(np.arange(len(sing_mean_yll)) + 1)
ax1.legend()
ax1.set_title('Single injury model')
ax2 = fig.add_subplot(gs[1, 0])
ax2.barh(np.arange(len(mult_rescaled_incidence_of_death)) + 1, mult_rescaled_incidence_of_death, color='coral',
         label='Model')
ax2.vlines(12.1, 0.5, 11.5, colors='r', linestyle='dashed', label='GBD estimate')
ax2.set_ylabel('Draw number')
ax2.set_yticks(np.arange(len(mult_mean_yld)) + 1)
ax2.legend()
ax2.set_xlabel('Incidence of Death per 100,000 p.y.')
ax2.set_title('Multiple injury model')
plt.savefig(f"C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/SingleVsMultipleInjury/BatchResults/hsb_sweeps/"
            f"Incidence_of_deaths.png", bbox_inches='tight')
plt.clf()
gbd_data = pd.read_csv('resources/gbd/ResourceFile_Deaths_And_DALYS_GBD2019.csv')
gbd_data = gbd_data.loc[gbd_data['measure_name'] == 'DALYs (Disability-Adjusted Life Years)']
gbd_data = gbd_data.loc[gbd_data['Year'].isin(range(2010, 2020))]
gbd_data = gbd_data.groupby('cause_name').sum()
gbd_data = gbd_data.nlargest(11, 'GBD_Est')
old_order_names = gbd_data.index
old_order_values = gbd_data['GBD_Est'].values
gbd_data = gbd_data.nlargest(10, 'GBD_Est')
gbd_data.loc['Road injuries'] = [805800, 0, mult_dalys.mean(), 0, 0]
gbd_data = gbd_data.sort_values('GBD_Est', ascending=False)
new_order_names = gbd_data.index
new_order_values = gbd_data['GBD_Est'].values
new_order_colors = ['lightsalmon'] * len(old_order_names)
new_rti_index = np.where((new_order_names == 'Road injuries'))
new_order_colors[new_rti_index[0][0]] = 'gold'

old_rti_index = np.where((old_order_names == 'Road injuries'))
old_order_colors = ['lightsteelblue'] * len(old_order_names)
old_order_colors[old_rti_index[0][0]] = 'gold'
fig = plt.figure(constrained_layout=True)
# Use GridSpec for customising layout
gs = fig.add_gridspec(nrows=2, ncols=1)
ax1 = fig.add_subplot(gs[0, 0])
ax1.barh(np.arange(len(old_order_names)), old_order_values, color=old_order_colors)
ax1.set_yticks(np.arange(len(old_order_names)))
ax1.set_yticklabels(old_order_names)
ax1.set_title('GBD ranked total DALYs')
ax2 = fig.add_subplot(gs[1, 0])
ax2.barh(np.arange(len(new_order_names)), new_order_values, color=new_order_colors)
ax2.set_xlabel('Total DALYs 2010-2019')
ax2.set_yticks(np.arange(len(new_order_names)))
ax2.set_yticklabels(new_order_names)
ax2.set_title('New ranked total DALYs')
plt.savefig(f"C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/SingleVsMultipleInjury/BatchResults/hsb_sweeps/"
            f"New_daly_rankings.png", bbox_inches='tight')

old_order_names = ['HIV/AIDS', 'Neonatal\ndisorders', 'Lower\nrespiratory\ninfections', 'Malaria',
                   'Diarrheal\ndiseases', 'Tuberculosis', 'Congenital\nbirth\ndefects', 'Meningitis',
                   'Malnutrition', 'Stroke', 'Road injuries']
old_order_colors = ['linen', 'navajowhite', 'khaki', 'yellow', 'grey', 'rosybrown', 'red', 'lightsalmon', 'peachpuff',
                    'lightsteelblue', 'seagreen']
new_order_names = ['HIV/AIDS', 'Neonatal\ndisorders', 'Lower\nrespiratory\ninfections', 'Malaria',
                   'Diarrheal\ndiseases', 'Tuberculosis', 'Congenital\nbirth defects', 'Road injuries', 'Meningitis',
                   'Malnutrition', 'Stroke']
new_order_colors = ['linen', 'navajowhite', 'khaki', 'yellow', 'grey', 'rosybrown', 'red', 'seagreen', 'lightsalmon',
                    'peachpuff', 'lightsteelblue']
fig = plt.figure(constrained_layout=True)
# Use GridSpec for customising layout
gs = fig.add_gridspec(nrows=2, ncols=1)
ax1 = fig.add_subplot(gs[0, 0])
squarify.plot(old_order_values, label=old_order_names, color=old_order_colors)
ax1.axis('off')
ax1.set_title('Total DALYs predicted from 2010-2019 by condition, GBD study')
ax2 = fig.add_subplot(gs[1, 0])
squarify.plot(new_order_values, label=new_order_names, color=new_order_colors)
ax2.axis('off')
ax2.set_title('Total DALYs predicted from 2010-2019 by condition with RTI model')
plt.savefig(f"C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/SingleVsMultipleInjury/BatchResults/hsb_sweeps/"
            f"New_daly_rankings_Tree_diagram.png", bbox_inches='tight')
