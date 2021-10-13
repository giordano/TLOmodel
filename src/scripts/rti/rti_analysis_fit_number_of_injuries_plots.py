"""This file uses the results of the batch file to make some summary statistics.
The results of the bachrun were put into the 'outputs' results_folder
"""
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tlo.analysis.utils import (
    extract_params,
    extract_results,
    get_grid,
    get_scenario_info,
    get_scenario_outputs,
    load_pickled_dataframes,
    summarize,
)

# create a function that extracts results in the same way as the utils function, but allows failed
# runs to pass




outputspath = Path('./outputs/')

# %% Analyse results of runs when doing a sweep of a single parameter:

# 0) Find results_folder associated with a given batch_file and get most recent
results_folder = get_scenario_outputs('rti_analysis_fit_number_of_injuries.py', outputspath)[-1]

# look at one log (so can decide what to extract)
log = load_pickled_dataframes(results_folder)

# get basic information about the results
info = get_scenario_info(results_folder)

# 1) Extract the parameters that have varied over the set of simulations
params = extract_params(results_folder)

# 2) Extract a series for all runs:
n_inj_per_person = extract_results(results_folder, module="tlo.methods.rti", key="number_of_injuries_in_hospital",
                                   column="number_of_injuries", index="date")
deaths_from_rti_incidence = extract_results(results_folder, module="tlo.methods.rti", key="summary_1m",
                                            column="incidence of rti death per 100,000", index="date")
percent_inhospital_mortality = extract_results(results_folder, module="tlo.methods.rti", key="summary_1m",
                                                   column="percentage died after med")
# 3) Get summary of the results for that log-element (only mean and the value at then of the simulation)
incidence_ninj = summarize(n_inj_per_person, only_mean=True).mean(axis=0)
incidence_ninj.name = 'z'
death_incidence = summarize(deaths_from_rti_incidence, only_mean=True).mean(axis=0)
death_incidence.name = 'z'
inhospital_mortality_results = pd.Series([percent_inhospital_mortality[0].mean().mean() for i in
                                         range(0, info['number_of_draws'])])
inhospital_mortality_results.name = 'z'
parameter_names = ['RTI:prob_death_iss_less_than_9', 'RTI:base_rate_injrti']
# 4) Create a heatmap for incidence of RTI:
filtered_params = params.loc[params['module_param'].isin(parameter_names)]
inc_grid = get_grid(filtered_params, incidence_results)
in_hospital_mortality_grid = get_grid(filtered_params, inhospital_mortality_results)
scaling_factors = [in_hospital_mortality_grid['RTI:prob_death_iss_less_than_9'].tolist()[i][0] / (102 / 11650) for i in
                   range(0, len(in_hospital_mortality_grid['RTI:prob_death_iss_less_than_9'].tolist()))]
in_hospital_mortality_grid['scale factor'] = scaling_factors
inc_grid['scale factor'] = scaling_factors
fig1, ax1 = plt.subplots()
c1 = ax1.pcolormesh(
    inc_grid['RTI:base_rate_injrti'],
    inc_grid['scale factor'],
    inc_grid['z'],
    cmap='Greys',
    shading='flat'
)
fig1.colorbar(c1, ax=ax1)
plt.xlabel('RTI:base_rate_injrti')
plt.ylabel('in-hospital mortality scale factor')
plt.title('Incidence of RTI')
plt.show()
plt.clf()
fig2, ax2 = plt.subplots()
c2 = ax2.pcolormesh(
    in_hospital_mortality_grid['RTI:base_rate_injrti'],
    in_hospital_mortality_grid['scale factor'],
    in_hospital_mortality_grid['z'],
    cmap='Greys',
    shading='flat'
)
fig2.colorbar(c2, ax=ax2)
plt.xlabel('RTI:base_rate_injrti')
plt.ylabel('in-hospital mortality scale factor')
plt.title('In-hospital mortality')
plt.show()
