"""This file uses the results of the batch file to make some summary statistics.
The results of the bachrun were put into the 'outputs' results_folder
"""
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

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
results_folder = get_scenario_outputs('rti_single_vs_multiple_injury.py', outputspath)[0]
log = load_pickled_dataframes(results_folder)

# get basic information about the results
info = get_scenario_info(results_folder)
xvals = range(info['number_of_draws'])

# 2) Extract a series for all runs:
people_in_rti_incidence = extract_results(results_folder, module="tlo.methods.rti", key="summary_1m",
                                          column="incidence of rti per 100,000", index="date")
deaths_from_rti_incidence = extract_results(results_folder, module="tlo.methods.rti", key="summary_1m",
                                            column="incidence of rti death per 100,000", index="date")
incidence_of_injuries = extract_results(results_folder, module="tlo.methods.rti", key="Inj_category_incidence",
                                        column="tot_inc_injuries", index="date")

