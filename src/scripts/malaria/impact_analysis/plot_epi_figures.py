"""This file uses the results of the scenario runs to generate plots

*1 Epi outputs (incidence and mortality)

"""

import datetime
from pathlib import Path

import lacroix
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from tlo.analysis.utils import (
    compare_number_of_deaths,
    extract_params,
    extract_results,
    get_scenario_info,
    get_scenario_outputs,
    load_pickled_dataframes,
    summarize,
    make_age_grp_lookup,
    make_age_grp_types,
)

outputspath = Path("./outputs")

# Find results_folder associated with a given batch_file (and get most recent [-1])
results_folder = get_scenario_outputs("effect_of_treatment_packages.py", outputspath)[-1]

# Declare path for output graphs from this script
make_graph_file_name = lambda stub: results_folder / f"{stub}.png"  # noqa: E731

# look at one log (so can decide what to extract)
log = load_pickled_dataframes(results_folder)

# get basic information about the results
scenario_info = get_scenario_info(results_folder)

# Extract the parameters that have varied over the set of simulations
params = extract_params(results_folder)

# colour scheme
berry = lacroix.colorList('CranRaspberry')  # ['#F2B9B8', '#DF7878', '#E40035', '#009A90', '#0054A4', '#001563']
baseline_colour = berry[5]  # '#001563'
sc1_colour = berry[3]  # '#009A90'
sc2_colour = berry[2]  # '#E40035'

# -----------------------------------------------------------------------------------------
# %% Epi outputs
# -----------------------------------------------------------------------------------------


# ---------------------------------- HIV ---------------------------------- #

# HIV incidence
hiv_inc = summarize(
    extract_results(
        results_folder,
        module="tlo.methods.hiv",
        key="summary_inc_and_prev_for_adults_and_children_and_fsw",
        column="hiv_adult_inc_1549",
        index="date",
        do_scaling=False
    ),
    only_mean=False, collapse_columns=True
)

# TB incidence
tb_inc = summarize(
    extract_results(
        results_folder,
        module="tlo.methods.tb",
        key="tb_incidence",
        column="num_new_active_tb",
        index="date",
        do_scaling=True
    ),
    only_mean=False, collapse_columns=True
)
# scale to get rate per 100,000
tb_inc = tb_inc / 15_000_000 * 100_000

# malaria incidence
mal_inc = summarize(
    extract_results(
        results_folder,
        module="tlo.methods.malaria",
        key="incidence",
        column="inc_clin_counter",
        index="date",
        do_scaling=False
    ),
    only_mean=False, collapse_columns=True
)

# ---------------------------------- PLOTS ---------------------------------- #
plt.style.use('ggplot')

font = {'family': 'sans-serif',
        'color': 'black',
        'weight': 'bold',
        'size': 11,
        }

fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3,
                                    constrained_layout=True,
                                    figsize=(5, 8))
fig.suptitle('')

# select mean values for plotting
mean_hiv_inc = hiv_inc.iloc[:, hiv_inc.columns.get_level_values(1) == 'mean']
mean_tb_inc = tb_inc.iloc[:, tb_inc.columns.get_level_values(1) == 'mean']
mean_mal_inc = mal_inc.iloc[:, mal_inc.columns.get_level_values(1) == 'mean']


fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3,
                                    constrained_layout=True,
                                    figsize=(8, 3))
fig.suptitle('')

ax1.plot(mean_hiv_inc)
ax1.set(title='HIV',
        ylabel='HIV Incidence, per capita')
ax1.set_xticklabels([])
ax1.legend(labels=['baseline', '-hiv', '-tb', '-malaria', '-all 3'],
           loc='upper left')

# TB incidence
ax2.plot(mean_tb_inc)
ax2.set(title='TB',
        ylabel='TB Incidence, per capita')
ax2.set_xticklabels([])

# Malaria incidence
ax3.plot(mean_mal_inc)
ax3.set(title='Malaria',
        ylabel='Malaria Incidence, per 100,000')

ax3.set_xticklabels([])

plt.show()
