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

def extract_results_number_of_injuries(results_folder: Path,
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

    cols = pd.Index(range(info['number_of_draws']))


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
            ave_in_draw = []
            for run in range(info['runs_per_draw']):
                try:
                    df: pd.DataFrame = load_pickled_dataframes(results_folder, draw, run, module)[module][key]
                    ave_in_draw.append(sum(df[column].sum()))

                    # if index is not None:
                    #     idx = df[index]
                    #     assert idx.equals(results_index), "Indexes are not the same between runs
                except KeyError:
                    results[draw] = np.nan
            results[draw] = ave_in_draw

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

outputspath = Path('./outputs/rmjlra2@ucl.ac.uk/')

# %% Analyse results of runs when doing a sweep of a single parameter:

# 0) Find results_folder associated with a given batch_file and get most recent
results_folder = get_scenario_outputs('rti_analysis_fit_number_of_injuries.py', outputspath)[-1]

# look at one log (so can decide what to extract)
log = load_pickled_dataframes(results_folder)

# get basic information about the results
info = get_scenario_info(results_folder)

# 1) Extract the parameters that have varied over the set of simulations
params = extract_params(results_folder)
search_range_lower = 1 - params.loc[params['module_param'] == 'RTI:number_of_injured_body_regions_distribution',
                                    'value'][0][1][0]
search_range_upper = 1 - params.iloc[-2]['value']
x_ticks = [f"Parameter \ndistribution {i + 1}" for i in range(0, len(params))]
# 2) Extract a series for all runs:
n_inj_overall = extract_results_number_of_injuries(results_folder, module="tlo.methods.rti", key='Injury_information',
                                                   column='Number_of_injuries')
# n_inj_overall.index = ['draw_1', 'draw_2', 'draw_3']
average_n_inj_per_draw = n_inj_overall.mean()
n_people_in_rti = extract_results(results_folder, module="tlo.methods.rti", key='summary_1m', column='number involved in a rti')
average_n_people_per_draw = summarize(n_people_in_rti, only_mean=True).sum()
overall_average_n_inj_per_person = average_n_inj_per_draw / average_n_people_per_draw
n_inj_per_person_in_hos = extract_results_for_irregular_logs(results_folder, module="tlo.methods.rti",
                                                             key="number_of_injuries_in_hospital",
                                                             column="number_of_injuries", index="date")
deaths_from_rti_incidence = extract_results(results_folder, module="tlo.methods.rti", key="summary_1m",
                                            column="incidence of rti death per 100,000", index="date")
rti_inc = extract_results(results_folder, module="tlo.methods.rti", key="summary_1m",
                          column="incidence of rti per 100,000", index="date")
extracted_hsb = extract_results(results_folder,
                                module="tlo.methods.rti",
                                key="summary_1m",
                                column="percent sought healthcare",
                                index="date")
percent_inhospital_mortality = extract_results(results_folder, module="tlo.methods.rti", key="summary_1m",
                                               column="percentage died after med")
prop_sought_healthcare_onlymeans = summarize(extracted_hsb, only_mean=True)
prop_sought_healthcare_onlymeans = prop_sought_healthcare_onlymeans.transpose()
prop_sought_healthcare_onlymeans = prop_sought_healthcare_onlymeans.mean(axis=1)
percent_inhospital_mortality_means = summarize(percent_inhospital_mortality, only_mean=True)
percent_inhospital_mortality_means = percent_inhospital_mortality_means.transpose()
percent_inhospital_mortality_means = percent_inhospital_mortality_means.mean(axis=1)
# percent_inhospital_mortality = extract_results(results_folder, module="tlo.methods.rti", key="summary_1m",
#                                                    column="percentage died after med")
# 3) Get summary of the results for that log-element (only mean and the value at then of the simulation)
average_ninj_in_hos = summarize(n_inj_per_person_in_hos, only_mean=True).mean(axis=0)
average_ninj_in_hos.name = 'z'
# average_ninj.index = studies_tested
death_incidence = summarize(deaths_from_rti_incidence, only_mean=True).mean(axis=0)
death_incidence.name = 'z'
rti_incidence = summarize(rti_inc, only_mean=True).mean(axis=0)
# death_incidence.index = studies_tested
# inhospital_mortality_results = pd.Series([percent_inhospital_mortality[0].mean().mean() for i in
#                                          range(0, info['number_of_draws'])])
# inhospital_mortality_results.name = 'z'
# inhospital_mortality_results.index = studies_tested
idxs = []
cut_off_scores_df = params.loc[params['module_param'] == 'RTI:rt_emergency_care_ISS_score_cut_off']
cut_off_scores = cut_off_scores_df['value'].unique()
for score in cut_off_scores:
    idxs.append(cut_off_scores_df.loc[cut_off_scores_df['value'] == score].index)

average_n_inj_in_kch = 7057 / 4776
best_fitting_distribution_per_score = []
best_fitting_ave_n = []
best_fit_dist_df = pd.DataFrame(columns=['best_fitting_dist', 'ave_n_inj_in_hospital', 'unscaled_inc_death', 'inc_rti',
                                         'mean_perc_hsb', 'percent_in_hos_mort'])
for n, idx in enumerate(idxs):
    best_fit_found = min(average_ninj_in_hos[idx], key=lambda x: abs(x - average_n_inj_in_kch))
    best_fitting_ave_n.append(best_fit_found)
    best_fit_index = np.where(average_ninj_in_hos == best_fit_found)
    params_in_run = params.loc[best_fit_index]
    best_fitting_distribution_per_score.append(
        params_in_run.loc[
            params_in_run['module_param'] == 'RTI:number_of_injured_body_regions_distribution'
            ]['value'].values[0]
    )
    inc_death = death_incidence.loc[best_fit_index].values[0]
    inc_rti = rti_incidence.loc[best_fit_index].values[0]
    mean_perc_hsb = prop_sought_healthcare_onlymeans.loc[best_fit_index].values[0]
    inhospital_mortality = percent_inhospital_mortality_means.loc[best_fit_index].to_list()[0]
    ISS_scores = params.loc[params['module_param'] == 'RTI:rt_emergency_care_ISS_score_cut_off', 'value'].unique()
    best_fit_dist_df.loc['ISS_cut_off_' + str(ISS_scores[n])] = \
        [
            params_in_run.loc[
                params_in_run['module_param'] == 'RTI:number_of_injured_body_regions_distribution'
                ]['value'].values[0],
            best_fit_found,
            inc_death,
            inc_rti,
            mean_perc_hsb,
            inhospital_mortality
        ]
best_fit_dist_df['scale_for_inc'] = 954.2 / best_fit_dist_df['inc_rti']
best_fit_dist_df['scaled_inc'] = best_fit_dist_df['inc_rti'] * best_fit_dist_df['scale_for_inc']
best_fit_dist_df['scaled_inc_death'] = best_fit_dist_df['scale_for_inc'] * best_fit_dist_df['unscaled_inc_death']
best_fit_dist_df.to_csv('C:/Users/Robbie Manning Smith/Desktop/rti_n_inj_dist_per_hsb.csv')
colors = ['lightsalmon' for i in average_ninj_in_hos]
colors[best_fit_index[0][0]] = 'gold'
# plot number of injuries
plt.bar(np.arange(len(average_ninj_in_hos)), average_ninj_in_hos, color=colors)
plt.xticks(np.arange(len(average_ninj_in_hos)), x_ticks, rotation=90)
plt.title('Average number of injuries of people in the health system, \nfor fitted negative exponential distribution')
plt.ylabel('Average number of injuries')
plt.savefig(f"C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/Calibration/number_of_injuries/"
            f"ninj_fit_by_hand_{search_range_lower}_{search_range_upper}.png", bbox_inches='tight')
# plot the incidence of death
plt.bar(np.arange(len(death_incidence)), death_incidence, color=colors)
plt.xticks(np.arange(len(death_incidence)), x_ticks, rotation=90)
plt.title('Incidence of death, \nfor fitted negative exponential distribution')
plt.ylabel('Incidence of death per 100,000')
plt.savefig(f"C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/Calibration/number_of_injuries/"
            f"incidence_of_death_fit_by_hand_{search_range_lower}_{search_range_upper}.png", bbox_inches='tight')
print('Best fitting distribution:')
print(params.values[best_fit_index[0][0]])
print('Average number of injuries')
print(average_ninj_in_hos.values[best_fit_index[0][0]])
print('Incidence of death')
print(death_incidence.values[best_fit_index[0][0]])
print('Overall n inj per person')
print(overall_average_n_inj_per_person[best_fit_index[0][0]])

plt.bar(np.arange(3),
        [overall_average_n_inj_per_person[best_fit_index[0][0]],
         average_ninj_in_hos.values[best_fit_index[0][0]],
         average_n_inj_in_kch], color='lightsalmon')
plt.xticks(np.arange(3), ['Mean number\n of injuries \nper person',
                          'Mean number\n of injuries \nper person\n in hospital',
                          'Mean number\n of injuries \nper person\n in KCH'])
plt.ylabel('Average number of injuries per person')
plt.title('Average number of injuries produced by best fitting distribution')
plt.savefig(f"C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/Calibration/number_of_injuries/"
            f"Av_n_inj_fit_by_hand_{search_range_lower}_{search_range_upper}.png", bbox_inches='tight')
