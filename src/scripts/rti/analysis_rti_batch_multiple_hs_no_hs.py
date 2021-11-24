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
hs_injury_results_folder = get_scenario_outputs('rti_hsb_effect_on_inc_death.py', outputspath)[-1]
no_hs_injury_results_folder = get_scenario_outputs('rti_hsb_sweep_no_hs.py', outputspath)[-1]
hs_extracted_inc_death = extract_results(hs_injury_results_folder,
                                         module="tlo.methods.rti",
                                         key="summary_1m",
                                         column="incidence of rti death per 100,000",
                                         index="date")
no_hs_extracted_inc_death = extract_results(no_hs_injury_results_folder,
                                            module="tlo.methods.rti",
                                            key="summary_1m",
                                            column="incidence of rti death per 100,000",
                                            index="date")
hs_extracted_inc = extract_results(hs_injury_results_folder,
                                   module="tlo.methods.rti",
                                   key="summary_1m",
                                   column="incidence of rti per 100,000",
                                   index="date")
no_hs_extracted_inc = extract_results(no_hs_injury_results_folder,
                                      module="tlo.methods.rti",
                                      key="summary_1m",
                                      column="incidence of rti per 100,000",
                                      index="date")
hs_inc = summarize(hs_extracted_inc)
hs_inc = hs_inc.mean()

no_hs_inc = summarize(no_hs_extracted_inc)
no_hs_inc = no_hs_inc.mean()
GBD_est_inc = 954.2

hs_inc_death = summarize(hs_extracted_inc_death)
no_hs_inc_death = summarize(no_hs_extracted_inc_death)
hs_scale_to_match_GBD = np.divide(GBD_est_inc, hs_inc[:, 'mean'].values)
no_hs_scale_to_match_GBD = np.divide(GBD_est_inc, no_hs_inc[:, 'mean'].values)
expected_hsb_upper = 0.85
expected_hsb_lower = 0.6533
hs_extracted_hsb = extract_results(hs_injury_results_folder,
                                   module="tlo.methods.rti",
                                   key="summary_1m",
                                   column="percent sought healthcare",
                                   index="date")
hs_per_param_average_hsb = summarize(hs_extracted_hsb)
hs_per_param_average_hsb = hs_per_param_average_hsb.mean()
hs_per_param_average_hsb = hs_per_param_average_hsb[:, 'mean'].values

hs_in_accepted_range = np.where((hs_per_param_average_hsb > expected_hsb_lower) &
                                (hs_per_param_average_hsb < expected_hsb_upper))

hs_mean_inc_death_overall = hs_inc_death.mean()
no_hs_mean_inc_death_overall = no_hs_inc_death.mean()

hs_inc_death_mean_upper = hs_mean_inc_death_overall.loc[:, 'upper']
no_hs_inc_death_mean_upper = no_hs_mean_inc_death_overall.loc[:, 'upper']

hs_inc_death_mean_lower = hs_mean_inc_death_overall.loc[:, 'lower']
no_hs_inc_death_mean_lower = no_hs_mean_inc_death_overall.loc[:, 'lower']

hs_lower_upper_inc_death = np.array(list(zip(
    hs_inc_death_mean_lower.to_list(),
    hs_inc_death_mean_upper.to_list()
))).transpose()

no_hs_lower_upper_inc_death = np.array(list(zip(
    no_hs_inc_death_mean_lower.to_list(),
    no_hs_inc_death_mean_upper.to_list()
))).transpose()
hs_per_param_average_inc_death = hs_mean_inc_death_overall[:, 'mean'].values
hs_per_param_average_inc_death = hs_per_param_average_inc_death * hs_scale_to_match_GBD
hs_yerr_inc_death = abs(hs_lower_upper_inc_death - hs_per_param_average_inc_death)
no_hs_per_param_average_inc_death = no_hs_mean_inc_death_overall[:, 'mean'].values
no_hs_per_param_average_inc_death = no_hs_per_param_average_inc_death * no_hs_scale_to_match_GBD
no_hs_yerr_inc_death = abs(no_hs_lower_upper_inc_death - no_hs_per_param_average_inc_death)
info = get_scenario_info(hs_injury_results_folder)
# filter results by the runs which produced acceptable health seeking behaviour
xvals = range(info['number_of_draws'])
xvals = [xvals[i] for i in hs_in_accepted_range[0]]
params = extract_params(hs_injury_results_folder)
params = params.loc[hs_in_accepted_range[0], 'value'].to_list()
param_name = 'RTI:rt_emergency_care_ISS_score_cut_off'
mean_inc_death_no_hs = no_hs_mean_inc_death_overall[:, 'mean'].mean()
mean_inc_death_hs = hs_mean_inc_death_overall[:, 'mean'][hs_in_accepted_range[0]].mean()
plt.clf()
plt.bar(np.arange(2), [mean_inc_death_hs, mean_inc_death_no_hs], color=['lightsteelblue', 'lightsalmon'])
plt.xticks(np.arange(2), ['With health\nsystem', 'Without health\nsystem'])
plt.ylabel('Incidence of death per 100,000')
for idx, inc in enumerate([mean_inc_death_hs, mean_inc_death_no_hs]):
    plt.text(idx - 0.06, [mean_inc_death_hs, mean_inc_death_no_hs][idx] + 5,
             f"{np.round([mean_inc_death_hs, mean_inc_death_no_hs][idx], 2)}")
    plt.ylim([0, mean_inc_death_no_hs + 20])
plt.title('RTI incidence of death with and without the health system')
plt.savefig(f"C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/SingleVsMultipleInjury/BatchResults/hsb_sweeps/"
            f"health_system_vs_no_health_system_inc_death.png", bbox_inches='tight')

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
                sim_start_year = min(yll_df.index)
                sim_end_year = max(yll_df.index) - 1
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


hs_yll, hs_yld = extract_yll_yld(hs_injury_results_folder)
no_hs_yll, no_hs_yld = extract_yll_yld(no_hs_injury_results_folder)
hs_dalys = hs_yll + hs_yld
hs_dalys = hs_dalys.mean()
hs_dalys = hs_dalys[hs_in_accepted_range[0]].mean()
no_hs_dalys = no_hs_yll + no_hs_yld
no_hs_dalys = no_hs_dalys.mean()
no_hs_dalys = no_hs_dalys.mean()
plt.clf()
plt.bar(np.arange(2), [hs_dalys, no_hs_dalys], color=['lightsteelblue', 'lightsalmon'])
plt.xticks(np.arange(2), ['With health\nsystem', 'Without health\nsystem'])
plt.ylabel('Total DALYs 2010-2019')
for idx, inc in enumerate([hs_dalys, no_hs_dalys]):
    plt.text(idx - 0.15, [hs_dalys, no_hs_dalys][idx] + 100000,
             f"{np.round([hs_dalys, no_hs_dalys][idx], 2)}")
plt.ylim([0, no_hs_dalys + 1000000])
plt.title('RTI DALYs with and without the health system')
plt.savefig(f"C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/SingleVsMultipleInjury/BatchResults/hsb_sweeps/"
            f"health_system_vs_no_health_system_DALYs.png", bbox_inches='tight')

percent_increase_in_deaths = mean_inc_death_no_hs / mean_inc_death_hs * 100
percent_increase_in_dalys = no_hs_dalys / hs_dalys * 100
plt.clf()
plt.bar(np.arange(2), [percent_increase_in_deaths, percent_increase_in_dalys], color=['lightsteelblue', 'lightsalmon'])
plt.xticks(np.arange(2), ['Percent increase\nin deaths', 'Percent increase\nin DALYs'])
plt.ylabel('Percentage')
for idx, inc in enumerate([percent_increase_in_deaths, percent_increase_in_dalys]):
    plt.text(idx - 0.1, [percent_increase_in_deaths, percent_increase_in_dalys][idx] + 10,
             f"{np.round([percent_increase_in_deaths, percent_increase_in_dalys][idx], 2)}%")
plt.ylim([0, percent_increase_in_deaths + 50])
plt.title('Percentage increase in DALYs caused by blocking the health system')
plt.savefig(f"C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/SingleVsMultipleInjury/BatchResults/hsb_sweeps/"
            f"health_system_vs_no_health_system_percent_reduction.png", bbox_inches='tight')
