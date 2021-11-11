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
param_name = 'RTI:rt_emergency_care_ISS_score_cut_off'
xvals = range(info['number_of_draws'])

# 2) Extract a specific log series for all runs:
extracted_hsb = extract_results(results_folder,
                                module="tlo.methods.rti",
                                key="summary_1m",
                                column="percent sought healthcare",
                                index="date")
prop_sought_healthcare = summarize(extracted_hsb)
prop_sought_healthcare_onlymeans = summarize(extracted_hsb, only_mean=True)
mean_overall = prop_sought_healthcare.mean()
# get upper and lower estimates
prop_sought_healthcare_lower = mean_overall.loc[:, "lower"]
prop_sought_healthcare_upper = mean_overall.loc[:, "upper"]
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
per_param_average_hsb = mean_overall[:, 'mean'].values
yerr = abs(lower_upper - per_param_average_hsb)
in_accepted_range = np.where((per_param_average_hsb > expected_hsb_lower) &
                             (per_param_average_hsb < expected_hsb_upper))

closest_to_hsb_midpoint = min(per_param_average_hsb, key=lambda x: abs(x - midpoint_hsb))
index_of_midpoint = np.where(per_param_average_hsb == closest_to_hsb_midpoint)
colors = ['lightsteelblue' if i not in in_accepted_range[0] else 'lightsalmon' for i in xvals]
colors[index_of_midpoint[0][0]] = 'gold'
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
plt.savefig(f"C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/Calibration/ParameterSpaceForHSB.png",
            bbox_inches='tight')
plt.clf()
extracted_inc_death = extract_results(results_folder,
                                      module="tlo.methods.rti",
                                      key="summary_1m",
                                      column="incidence of rti death per 100,000",
                                      index="date")
inc_death = summarize(extracted_inc_death)
inc_death_only_means = summarize(extracted_inc_death, only_mean=True)
mean_inc_death_overall = inc_death.mean()
inc_death_mean_upper = mean_inc_death_overall.loc[:, 'upper']
inc_death_mean_lower = mean_inc_death_overall.loc[:, 'lower']
lower_upper_inc_death = np.array(list(zip(
    inc_death_mean_lower.to_list(),
    inc_death_mean_upper.to_list()
))).transpose()
per_param_average_inc_death = mean_inc_death_overall[:, 'mean'].values
yerr_inc_death = abs(lower_upper_inc_death - per_param_average_inc_death)

WHO_est_in_death = 35
best_fit_inc_death_found = min(per_param_average_inc_death[in_accepted_range], key=lambda x: abs(x - WHO_est_in_death))
best_fit_death_index = np.where(per_param_average_inc_death == best_fit_inc_death_found)

plt.bar(x=xvals,
        height=mean_inc_death_overall[:, 'mean'].values,
        yerr=yerr_inc_death,
        color=colors)
for idx, inc in enumerate(per_param_average_inc_death):
    plt.text(idx - 0.5, inc_death_mean_upper.to_list()[idx] + 5,
             f"{np.round(mean_inc_death_overall[:, 'mean'][idx], 2)}")
plt.xticks(xvals, xlabels)
plt.xlabel(param_name)
plt.ylabel('Incidence of death per 100,000 person years')
plt.title('Effect of health seeking behaviour on the incidence of death')
plt.savefig(f"C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/Calibration/effect_of_hsb_on_inc_death.png",
            bbox_inches='tight')
extracted_inc = extract_results(results_folder,
                                module="tlo.methods.rti",
                                key="summary_1m",
                                column="incidence of rti per 100,000",
                                index="date")

# 3) Get summary of the results for that log-element
inc = summarize(extracted_inc)
# If only interested in the means
inc_only_means = summarize(extracted_inc, only_mean=True)
# get per parameter summaries
mean_inc_overall = inc.mean()
inc_mean_upper = mean_inc_overall.loc[:, 'upper']
inc_mean_lower = mean_inc_overall.loc[:, 'lower']
lower_upper_inc = np.array(list(zip(
    inc_mean_lower.to_list(),
    inc_mean_upper.to_list()
))).transpose()
per_param_average_inc = mean_inc_overall[:, 'mean'].values
yerr_inc = abs(lower_upper_inc - per_param_average_inc)
GBD_est_inc = 954.2
best_fit_found_for_inc = min(per_param_average_inc[in_accepted_range], key=lambda x: abs(x - GBD_est_inc))
best_fit_for_inc_index = np.where(per_param_average_inc == best_fit_found_for_inc)
plt.clf()
plt.bar(x=xvals,
        height=mean_inc_overall[:, 'mean'].values,
        yerr=yerr_inc,
        color=colors)
for idx, inc in enumerate(per_param_average_inc):
    plt.text(idx - 0.5, inc_mean_upper.to_list()[idx] + 5, f"{np.round(mean_inc_overall[:, 'mean'][idx], 1)}")
plt.xticks(xvals, xlabels)
plt.xlabel(param_name)
plt.ylabel('Incidence of RTI per 100,000 person years')
plt.title('Effect of health seeking behaviour on the incidence of RTI')
plt.savefig(f"C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/Calibration/effect_of_hsb_on_inc.png",
            bbox_inches='tight')
plt.clf()
percent_crashed_that_are_fatal = extract_results(results_folder,
                                                 module="tlo.methods.rti",
                                                 key="summary_1m",
                                                 column="percent of crashes that are fatal",
                                                 index="date")

# 3) Get summary of the results for that log-element
percent_fatal = summarize(percent_crashed_that_are_fatal)
# If only interested in the means
percent_fatal_only_means = summarize(percent_crashed_that_are_fatal, only_mean=True)
# get per parameter summaries
mean_percent_fatal_overall = percent_fatal.mean()
perc_fatal_mean_lower = mean_percent_fatal_overall.loc[:, 'lower']
perc_fatal_mean_upper = mean_percent_fatal_overall.loc[:, 'upper']
lower_upper_perc_fatal = np.array(list(zip(
    perc_fatal_mean_lower.to_list(),
    perc_fatal_mean_upper.to_list()
))).transpose()

per_param_perc_fatal = mean_percent_fatal_overall[:, 'mean'].values
yerr_perc_fatal = abs(lower_upper_perc_fatal - per_param_perc_fatal)
plt.bar(x=xvals,
        height=mean_percent_fatal_overall[:, 'mean'].values,
        yerr=yerr_perc_fatal,
        color=colors)
for idx, inc in enumerate(per_param_perc_fatal):
    plt.text(idx - 0.5, perc_fatal_mean_upper.to_list()[idx] + 0.02,
             f"{np.round(mean_percent_fatal_overall[:, 'mean'][idx], 3)}")
plt.xticks(xvals, xlabels)
plt.xlabel(param_name)
plt.ylabel('Percent Fatal')
plt.title('Effect of health seeking behaviour on the percent of crashed that are fatal')
plt.savefig(f"C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/Calibration/effect_of_hsb_on_percent_fatal.png",
            bbox_inches='tight')
plt.clf()
# Plot summary for best fits of different categories
# create a function that creates an infographic
def create_infographic(data_index, detail):
    """A function that creates an infographic from the model output for certain outputs"""
    # get relevant information for this run
    extracted_inc_this_run = extract_results(results_folder,
                                             module="tlo.methods.rti",
                                             key="summary_1m",
                                             column="incidence of rti per 100,000",
                                             index="date")
    inc = summarize(extracted_inc_this_run)
    mean_inc = inc[data_index]['mean']
    inc_upper = inc[data_index]['upper']
    inc_lower = inc[data_index]['lower']
    extracted_inc_death_this_run = extract_results(results_folder,
                                                   module="tlo.methods.rti",
                                                   key="summary_1m",
                                                   column="incidence of rti death per 100,000",
                                                   index="date")
    inc_death = summarize(extracted_inc_death_this_run)
    mean_inc_death = inc_death[data_index]['mean']
    inc_death_upper = inc_death[data_index]['upper']
    inc_death_lower = inc_death[data_index]['lower']
    extracted_hsb_this_run = extract_results(results_folder,
                                             module="tlo.methods.rti",
                                             key="summary_1m",
                                             column="percent sought healthcare",
                                             index="date")
    percent_sought_care_this_run = summarize(extracted_hsb_this_run)
    mean_percent_sought_care = percent_sought_care_this_run[data_index]['mean']
    fig = plt.figure(constrained_layout=True)
    # Use GridSpec for customising layout
    gs = fig.add_gridspec(nrows=2, ncols=2)
    # Add an empty axes that occupied the whole first row
    ax1 = fig.add_subplot(gs[0, 0:2])
    ax1.plot(mean_inc.index, mean_inc, color='lightsteelblue', label='Incidence of \nRTI', zorder=2)
    ax1.fill_between(mean_inc.index, inc_lower, inc_upper, color='lightsteelblue', alpha=0.5, zorder=1, label='95% C.I')
    ax1.plot(mean_inc_death.index, mean_inc_death, color='lightsalmon', label='Incidence of \ndeath',
             zorder=4)
    ax1.fill_between(mean_inc_death.index, inc_death_lower, inc_death_upper, color='lightsalmon', alpha=0.5, zorder=3,
                     label='95% C.I')
    ax1.legend(fontsize=5, bbox_to_anchor=(0.9, 1), loc='upper left', borderaxespad=0.)
    ax1.set_title('Incidence of RTI and death over time')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Incidence per 100,000 \nperson years')
    # Add two empty axes that occupied the remaining grid
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.bar(np.arange(2), [mean_inc.mean(), mean_inc_death.mean()], color=['lightsteelblue', 'lightsalmon'])

    ax2.set_title(f"Mean incidence of RTI = {np.round(mean_inc.mean(), 2)}.\n "
                  f"Mean incidence of death = {np.round(mean_inc_death.mean(), 2)}")
    ax2.set_xticks(np.arange(2))
    ax2.set_xticklabels(['Incidence\nof\nRTI', 'Incidence\nof\ndeath'])
    ax2.set_ylabel('Incidence per 100,000 \nperson years')
    ax3 = fig.add_subplot(gs[1, 1])
    data = [mean_percent_sought_care.mean(), 1 - mean_percent_sought_care.mean()]
    ax3.pie(data, labels=['Sought\ncare', "Didn't\nseek\ncare"], colors=['thistle', 'peachpuff'], autopct='%1.1f%%',
            startangle=90)
    ax3.set_title('Health seeking\nbehaviour for RTI')
    plt.savefig(f"C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/Calibration/optimizing_hsb_for_" + detail +
                ".png", bbox_inches='tight')

create_infographic(best_fit_death_index[0][0],
                   f"best_fitting_inc_death_iss_cutoff={params['value'][best_fit_death_index[0][0]]}")
create_infographic(index_of_midpoint[0][0],
                   f"midpoint_of_hsb_range_iss_cutoff={params['value'][index_of_midpoint[0][0]]}")
create_infographic(best_fit_for_inc_index[0][0],
                   f"best_fitting_inc_iss_cutoff={params['value'][best_fit_for_inc_index[0][0]]}")
for param_value in in_accepted_range[0]:
    create_infographic(param_value, f"summary_ISS_cut_off={param_value}")
# scale normalise the results so that each incidence is equal to the gbd incidence
scale_to_match_GBD = np.divide(GBD_est_inc, mean_inc_overall[:, 'mean'].values)
scaled_incidences = mean_inc_overall[:, 'mean'].values * scale_to_match_GBD
rescaled_incidence_of_death = mean_inc_death_overall[:, 'mean'].values * scale_to_match_GBD
best_fitting_scaled = min(rescaled_incidence_of_death, key=lambda x: abs(x - WHO_est_in_death))
best_fitting_scaled_index = np.where(rescaled_incidence_of_death == best_fitting_scaled)[0][0]
colors_for_inc = ['lightsteelblue'] * len(scaled_incidences)
colors_for_inc[best_fitting_scaled_index] = 'gold'
colors_for_inc_death = ['lightsalmon'] * len(rescaled_incidence_of_death)
colors_for_inc_death[best_fitting_scaled_index] = 'gold'

plt.clf()
fig = plt.figure(constrained_layout=True)
# Use GridSpec for customising layout
gs = fig.add_gridspec(nrows=2, ncols=1)
# Add an empty axes that occupied the whole first row
ax1 = fig.add_subplot(gs[0, 0])
ax1.bar(np.arange(len(scaled_incidences)), scaled_incidences, color=colors_for_inc, width=0.4)
for idx, val in enumerate(scaled_incidences):
    ax1.text(idx, val, f"{np.round(val, 2)}", rotation=90)
ax1.set_title('Incidence of RTI')
ax1.set_xticks(np.arange(len(params)))
ax1.set_xticklabels(params['value'].to_list())
ax1.set_ylabel('Incidence per 100,000 p.y.')
ax2 = fig.add_subplot(gs[1, 0])
ax2.set_title('Incidence of Death')
ax2.bar(np.arange(len(rescaled_incidence_of_death)), rescaled_incidence_of_death, color=colors_for_inc_death, width=0.4)
for idx, val in enumerate(rescaled_incidence_of_death):
    ax2.text(idx, val, f"{np.round(val, 2)}", rotation=90)
ax2.set_xticks(np.arange(len(params)))
ax2.set_xticklabels(params['value'].to_list())
ax2.set_ylabel('Incidence of death \nper 100,000 p.y.')
ax2.set_xlabel('Emergency care ISS cut off score')
plt.savefig(f"C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/Calibration/scaled_incidences.png",
            bbox_inches='tight')
print('Best ISS cut off score is ')
print(params['value'].to_list()[best_fitting_scaled_index])
print('scale factor for current incidence of rti is ')
print(scale_to_match_GBD[best_fitting_scaled_index])
plt.clf()

x_vals = params['value'][in_accepted_range[0]]
fig, ax1 = plt.subplots()
ax1.set_xlabel('Model runs')
ax1.set_ylabel('Incidence of death\nper 100,000 p.y.')
ax1.bar(x_vals, rescaled_incidence_of_death[in_accepted_range[0]], width=0.4, color='lightsalmon',
        label='Incidence of death')
ax1.set_xticks(x_vals + 0.2)
ax1.set_xticklabels(x_vals)
ax1.set_ylim([0, 50])
# Adding Twin Axes

ax2 = ax1.twinx()

ax2.set_ylabel('Percent sought care')
ax2.bar(x_vals + 0.4, mean_overall[:, 'mean'][in_accepted_range[0]], width=0.4, color='lightsteelblue',
        label='HSB')
ax2.set_ylim([0, 1])
ax1.set_title("The model's predicted incidence of death for \nvarying levels of health seeking behaviour")
# Show plot
ax1.legend(loc='upper left')
ax2.legend()
plt.savefig(f"C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/Calibration/HSB_and_inc_death.png",
            bbox_inches='tight')
