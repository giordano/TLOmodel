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
def extract_results_for_irregular_logs(results_folder: Path,
                                       module: str,
                                       key: str,
                                       column: str = None,
                                       index: str = None,
                                       custom_generate_series=None,
                                       do_scaling: bool = False,
                                       ) -> pd.DataFrame:
    """Utility function to unpack results

    Produces a dataframe that summaries one series from the log, with column multi-index for the draw/run. If an 'index'
    component of the log_element is provided, the dataframe uses that index (but note that this will only work if the
    index is the same in each run).
    Optionally, instead of a series that exists in the dataframe already, a function can be provided that, when applied
    to the dataframe indicated, yields a new pd.Series.
    Optionally, with `do_scaling`, each element is multiplied by the the scaling_factor recorded in the simulation
    (if available)
    """

    # get number of draws and numbers of runs
    info = get_scenario_info(results_folder)

    cols = pd.MultiIndex.from_product(
        [range(info['number_of_draws']), range(info['runs_per_draw'])],
        names=["draw", "run"]
    )

    def get_multiplier(_draw, _run):
        """Helper function to get the multiplier from the simulation, if it's specified and do_scaling=True"""
        if not do_scaling:
            return 1.0
        else:
            try:
                return load_pickled_dataframes(results_folder, _draw, _run, 'tlo.methods.demography'
                                               )['tlo.methods.demography']['scaling_factor']['scaling_factor'].values[0]
            except KeyError:
                return 1.0

    if custom_generate_series is None:

        assert column is not None, "Must specify which column to extract"

        results_index = None
        if index is not None:
            # extract the index from the first log, and use this ensure that all other are exactly the same.
            filename = f"{module}.pickle"
            df: pd.DataFrame = load_pickled_dataframes(results_folder, draw=0, run=0, name=filename)[module][key]
            results_index = df[index]

        results = pd.DataFrame(columns=cols)
        for draw in range(info['number_of_draws']):
            for run in range(info['runs_per_draw']):

                try:
                    df: pd.DataFrame = load_pickled_dataframes(results_folder, draw, run, module)[module][key]
                    results[draw, run] = df[column] * get_multiplier(draw, run)

                    # if index is not None:
                    #     idx = df[index]
                    #     assert idx.equals(results_index), "Indexes are not the same between runs"

                except KeyError:
                    results[draw, run] = np.nan

        # if 'index' is provided, set this to be the index of the results
        if index is not None:
            results.index = results_index

        return results

    else:
        # A custom commaand to generate a series has been provided.
        # No other arguements should be provided.
        assert index is None, "Cannot specify an index if using custom_generate_series"
        assert column is None, "Cannot specify a column if using custom_generate_series"

        # Collect results and then use pd.concat as indicies may be different betweeen runs
        res = dict()
        for draw in range(info['number_of_draws']):
            for run in range(info['runs_per_draw']):
                df: pd.DataFrame = load_pickled_dataframes(results_folder, draw, run, module)[module][key]
                output_from_eval = custom_generate_series(df)
                assert pd.Series == type(output_from_eval), 'Custom command does not generate a pd.Series'
                res[f"{draw}_{run}"] = output_from_eval * get_multiplier(draw, run)
        results = pd.concat(res.values(), axis=1).fillna(0)
        results.columns = cols

        return results

studies_tested = ['Madubueze et al.', 'Sanyang et al.', 'Qi et al. 2006', 'Ganveer & Tiwani', 'Thani & Kehinde',
                  'Akinpea et al.']

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
x_ticks = [f"Parameter \ndistribution {i + 1}" for i in range(0, len(params))]
# 2) Extract a series for all runs:
n_inj_per_person = extract_results_for_irregular_logs(results_folder, module="tlo.methods.rti", key="number_of_injuries_in_hospital",
                                                      column="number_of_injuries", index="date")
deaths_from_rti_incidence = extract_results(results_folder, module="tlo.methods.rti", key="summary_1m",
                                            column="incidence of rti death per 100,000", index="date")
percent_inhospital_mortality = extract_results(results_folder, module="tlo.methods.rti", key="summary_1m",
                                                   column="percentage died after med")
# 3) Get summary of the results for that log-element (only mean and the value at then of the simulation)
average_ninj = summarize(n_inj_per_person, only_mean=True).mean(axis=0)
average_ninj.name = 'z'
# average_ninj.index = studies_tested
death_incidence = summarize(deaths_from_rti_incidence, only_mean=True).mean(axis=0)
death_incidence.name = 'z'
# death_incidence.index = studies_tested
inhospital_mortality_results = pd.Series([percent_inhospital_mortality[0].mean().mean() for i in
                                         range(0, info['number_of_draws'])])
inhospital_mortality_results.name = 'z'
# inhospital_mortality_results.index = studies_tested
average_n_inj_in_kch = 7057 / 4776
best_fit_found = min(average_ninj, key = lambda x: abs(x - average_n_inj_in_kch))
best_fit_index = np.where(average_ninj == best_fit_found)
colors = ['lightsalmon' for i in average_ninj]
colors[best_fit_index[0][0]] = 'gold'
# plot number of injuries
plt.bar(np.arange(len(average_ninj)), average_ninj, color=colors)
plt.xticks(np.arange(len(average_ninj)), x_ticks, rotation=90)
plt.title('Average number of injuries of people in the health system, \nfor fitted negative exponential distribution')
plt.ylabel('Average number of injuries')
plt.savefig("C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/Calibration/number_of_injuries/"
            "ninj_fit_by_hand.png", bbox_inches='tight')
# plot the incidence of death
plt.bar(np.arange(len(death_incidence)), death_incidence, color=colors)
plt.xticks(np.arange(len(death_incidence)), x_ticks, rotation=90)
plt.title('Incidence of death, \nfor fitted negative exponential distribution')
plt.ylabel('Incidence of death per 100,000')
plt.savefig("C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/Calibration/number_of_injuries/"
            "incidence_of_death_fit_by_hand.png", bbox_inches='tight')
print('Best fitting distribution:')
print(params.values[best_fit_index[0][0]])
