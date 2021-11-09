"""This file uses the results of the batch file to make some summary statistics.
The results of the bachrun were put into the 'outputs' results_folder
"""
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from src.scripts.rti.rti_create_graphs import create_rti_graphs, rti_format_data_from_azure_runs
from tlo.analysis.utils import (
    extract_params,
    extract_results,
    get_scenario_info,
    get_scenario_outputs,
    load_pickled_dataframes,
    summarize,
)

outputspath = Path('./outputs/rmjlra2@ucl.ac.uk')
# outputspath = Path('./outputs')
# %% Analyse results of runs when doing a sweep of a single parameter:
results_folder = get_scenario_outputs('rti_single_vs_multiple_injury.py', outputspath)[-1]
log = load_pickled_dataframes(results_folder)

# get basic information about the results
info = get_scenario_info(results_folder)
xvals = range(info['number_of_draws'])
params = extract_params(results_folder)
# 2) Extract a series for all runs:
people_in_rti_incidence = extract_results(results_folder, module="tlo.methods.rti", key="summary_1m",
                                          column="incidence of rti per 100,000", index="date")
deaths_from_rti_incidence = extract_results(results_folder, module="tlo.methods.rti", key="summary_1m",
                                            column="incidence of rti death per 100,000", index="date")
cols = pd.MultiIndex.from_product(
        [range(info['number_of_draws']), range(info['runs_per_draw'])],
        names=["draw", "run"]
    )
results = pd.DataFrame(columns=cols)
average_n_inj_per_draws = []
for draw in range(info['number_of_draws']):
    ave_n_inj_this_draw = []
    for run in range(info['runs_per_draw']):
        df: pd.DataFrame = \
            load_pickled_dataframes(results_folder, draw, run, "tlo.methods.rti")["tlo.methods.rti"]
        df = df['Injury_information']
        total_n_injuries = sum(df.sum(axis=0)['Number_of_injuries'])
        injuries_per_person = total_n_injuries / len(df.sum(axis=0)['Number_of_injuries'])
        ave_n_inj_this_draw.append(injuries_per_person)
    average_n_inj_per_draws.append(np.mean(ave_n_inj_this_draw))
incidence_of_rti = summarize(people_in_rti_incidence)
incidence_of_death = summarize(deaths_from_rti_incidence)

mean_inc_rti_single = incidence_of_rti[0, 'mean'].mean()
mean_inc_rti_mult = incidence_of_rti[1, 'mean'].mean()
mean_inc_death_single = incidence_of_death[0, 'mean'].mean()
mean_inc_death_mult = incidence_of_death[1, 'mean'].mean()
mean_inc_inj_single = mean_inc_rti_single * average_n_inj_per_draws[0]
mean_inj_inj_mult = mean_inc_rti_mult * average_n_inj_per_draws[1]
gbd_inc_rti = 954.2
gbd_inc_death = 12.1
gbd_inc_rti_inj = 954.2
gbd_results = [gbd_inc_rti, gbd_inc_death, gbd_inc_rti_inj]
single_results = [mean_inc_rti_single, mean_inc_death_single, mean_inc_inj_single]
mult_results = [mean_inc_rti_mult, mean_inc_death_mult, mean_inj_inj_mult]
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
    plt.text(np.arange(3)[idx] + 0.5 - 0.125, mult_results[idx] + 10, f"{np.round(val, 2)}", fontdict={'fontsize': 9},
             rotation=45)
plt.legend()
plt.title('Comparing the incidence of RTI, RTI death and injuries\nfor the GBD study, single injury model and\n'
          'multiple injury model')
plt.ylabel('Incidence per \n 100,000 person years')
plt.savefig('C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/SingleVsMultipleInjury/BatchResults/'
            'IncidenceSummary.png', bbox_inches='tight')
