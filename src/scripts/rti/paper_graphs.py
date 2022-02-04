from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import squarify

from tlo.analysis.utils import (
    extract_params,
    extract_params_from_json,
    extract_results,
    get_scenario_info,
    get_scenario_outputs,
    load_pickled_dataframes,
    summarize,
)

# Figure 1: GBD number of injuries
gbd_data = pd.read_csv("C:/Users/Robbie Manning Smith/Desktop/gbddata/gbd_data.csv")
# get data in years of study
gbd_data = gbd_data.loc[gbd_data['year'] > 2009]
# get male data
male_data = gbd_data.loc[gbd_data['sex'] == 'Male']
# get number of RTIs
male_data = male_data.loc[male_data['measure'] == 'Incidence']
# get predicted number of males in RTIs
male_rtis = male_data['val'].to_list()
# get female data
female_data = gbd_data.loc[gbd_data['sex'] == 'Female']
# get number of RTIs
female_data = female_data.loc[female_data['measure'] == 'Incidence']
# get predicted number of males in RTIs
female_rtis = female_data['val'].to_list()
total_rtis = np.add(male_rtis, female_rtis)
plt.plot(male_data['year'], total_rtis, color='m', label='Total')
plt.plot(male_data['year'], male_rtis, color='lightsalmon', label='Males')
plt.plot(female_data['year'], female_rtis, color='lightsteelblue', label='Females')
plt.xticks(male_data['year'])
plt.xlabel('Year')
plt.ylabel('Number of RTIs')
plt.legend()
plt.savefig("C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/FinalPaperOutput/Figure_1.png",
            bbox_inches='tight')
# Figure 3, the calibration of the model
plt.clf()
fig = plt.figure(constrained_layout=True)
# Use GridSpec for customising layout
gs = fig.add_gridspec(nrows=3, ncols=2)
outputspath = Path('./outputs/rmjlra2@ucl.ac.uk/')

results_folder = get_scenario_outputs('rti_analysis_full_calibrated.py', outputspath)[- 3]
gbd_inc_data = pd.read_csv("C:/Users/Robbie Manning Smith/Desktop/gbddata/all_countries_inc_data.csv")
gbd_inc_data = gbd_inc_data.loc[gbd_inc_data['location'] == 'Malawi']
gbd_inc_data = gbd_inc_data.loc[gbd_inc_data['measure'] == 'Incidence']
GBD_mean_inc = gbd_inc_data.val.mean()
gbd_inc_upper = gbd_inc_data.upper.mean()
gbd_inc_lower = gbd_inc_data.lower.mean()
extracted_incidence_of_RTI = extract_results(results_folder,
                                             module="tlo.methods.rti",
                                             key="summary_1m",
                                             column="incidence of rti per 100,000",
                                             index="date"
                                             )
mean_incidence_of_RTI = summarize(extracted_incidence_of_RTI, only_mean=True).mean()
mean_inc_lower = summarize(extracted_incidence_of_RTI).mean().loc[:, "lower"]
mean_inc_upper = summarize(extracted_incidence_of_RTI).mean().loc[:, "upper"]
lower_upper = np.array(list(zip(
    mean_inc_lower.to_list(),
    mean_inc_upper.to_list()
))).transpose()
closest_est_found = min(mean_incidence_of_RTI, key=lambda x: abs(x - GBD_mean_inc))
best_fit_idx = np.where(mean_incidence_of_RTI == closest_est_found)[0][0]
best_fit_found = mean_incidence_of_RTI[best_fit_idx]
data = [GBD_mean_inc, best_fit_found]
gbd_inc_yerr = [gbd_inc_lower, gbd_inc_upper]
model_inc_yerr = [lower_upper[0][best_fit_idx], lower_upper[1][best_fit_idx]]
yerr = [[gbd_inc_yerr[0], model_inc_yerr[0]], [gbd_inc_yerr[1], model_inc_yerr[1]]]
ax1 = fig.add_subplot(gs[0, 0])

ax1.bar(np.arange(len(data)), data, color=['lightsalmon', 'darksalmon'], yerr=yerr)
ax1.set_xticks(np.arange(len(data)))
ax1.set_xticklabels(['GBD', 'Model'])
ax1.set_ylabel('Inc. of RTI per\n100,000 p.y.')
# TODO: calibration of age and gender demographics
ax2 = fig.add_subplot(gs[0, 1])
gbd_proportion_male = sum(male_rtis) / (sum(male_rtis) + sum(female_rtis))
model_proportion_male = 0.6
extracted_perc_related_to_alc = extract_results(results_folder,
                                                module="tlo.methods.rti",
                                                key="rti_demography",
                                                column="percent_related_to_alcohol",
                                                index="date"
                                                )
mean_perc_alc = summarize(extracted_perc_related_to_alc, only_mean=True).mean().mean()
kch_alc_perc = 0.25
data_based_ests = [gbd_proportion_male, kch_alc_perc]
model_outputs = [model_proportion_male, mean_perc_alc]
ax2.bar(np.arange(len(data_based_ests)), data_based_ests, color='lightsteelblue', width=0.4,
        label='calibration data')
ax2.bar(np.arange(len(model_outputs)) + 0.4, model_outputs, color='steelblue', width=0.4, label='model output')
ax2.set_xticks(np.arange(len(data)) + 0.2)
ax2.set_xticklabels(['Prop. male', 'Prop. alc'])
ax2.set_ylabel('Proportion')
ax2.legend(prop={'size': 6})
# Calibration of on-scene mortality
results_folder = get_scenario_outputs('rti_analysis_fit_incidence_of_on_scene.py', outputspath)[-1]
# look at one log (so can decide what to extract)
log = load_pickled_dataframes(results_folder)

# get basic information about the results
info = get_scenario_info(results_folder)

# 1) Extract the parameters that have varied over the set of simulations
params = extract_params(results_folder)
extracted_incidence_of_death_on_scene = extract_results(results_folder,
                                                        module="tlo.methods.rti",
                                                        key="summary_1m",
                                                        column="incidence of prehospital death per 100,000",
                                                        index="date"
                                                        )
incidence_of_rti = extract_results(results_folder,
                                   module="tlo.methods.rti",
                                   key="summary_1m",
                                   column="incidence of rti per 100,000",
                                   index="date"
                                   )
inc_rti = summarize(incidence_of_rti, only_mean=True).mean()
scale_to_gbd = np.divide(954.2, inc_rti)
on_scene_inc_death = summarize(extracted_incidence_of_death_on_scene, only_mean=True).mean() * scale_to_gbd
target_on_scene_inc_death = 6
closest_est_found = min(on_scene_inc_death, key=lambda x: abs(x - target_on_scene_inc_death))
best_fit_idx = np.where(on_scene_inc_death == closest_est_found)[0][0]
ax3 = fig.add_subplot(gs[1, 0])
ax3.bar(np.arange(len(on_scene_inc_death)), on_scene_inc_death, label='Model estimates', color='plum')
ax3.bar(best_fit_idx, on_scene_inc_death[best_fit_idx], color='violet', label='best fit found')
ax3.axhline(target_on_scene_inc_death, color='fuchsia', label='NRSC estimate')
ax3.set_xticks(np.arange(len(on_scene_inc_death)))
ax3.set_xticklabels(params['value'].round(4), rotation=45, fontsize=6)
ax3.set_ylabel('Inc. on scene mort. per \n100,000 p.y.')
ax3.set_xlabel('% fatal on scene')
ax3.legend(prop={'size': 6})

results_folder = get_scenario_outputs('rti_analysis_full_calibrated.py', outputspath)[- 3]
info = get_scenario_info(results_folder)
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
results_folder = get_scenario_outputs('rti_analysis_fit_number_of_injuries.py', outputspath)[-1]

# look at one log (so can decide what to extract)
log = load_pickled_dataframes(results_folder)

# get basic information about the results
info = get_scenario_info(results_folder)

# 1) Extract the parameters that have varied over the set of simulations
params = extract_params(results_folder)
search_range_lower = 1 - params.loc[params['module_param'] == 'RTI:number_of_injured_body_regions_distribution',
                                    'value'][0][1][0]
search_range_upper = 1 - params.iloc[-2]['value'][1][0]
x_ticks = [f"Parameter \ndistribution {i + 1}" for i in range(0, len(params))]
# 2) Extract a series for all runs:
n_inj_overall = extract_results_number_of_injuries(results_folder, module="tlo.methods.rti", key='Injury_information',
                                                   column='Number_of_injuries')
n_inj_overall.index = ['draw_1', 'draw_2', 'draw_3']
average_n_inj_per_draw = n_inj_overall.mean()
n_people_in_rti = extract_results(results_folder, module="tlo.methods.rti", key='summary_1m',
                                  column='number involved in a rti')
average_n_people_per_draw = summarize(n_people_in_rti, only_mean=True).sum()
overall_average_n_inj_per_person = average_n_inj_per_draw / average_n_people_per_draw
n_inj_per_person_in_hos = extract_results_for_irregular_logs(results_folder, module="tlo.methods.rti",
                                                             key="number_of_injuries_in_hospital",
                                                             column="number_of_injuries", index="date")
idxs = []
cut_off_scores_df = params.loc[params['module_param'] == 'RTI:rt_emergency_care_ISS_score_cut_off']
cut_off_scores = cut_off_scores_df['value'].unique()
for score in cut_off_scores:
    idxs.append(cut_off_scores_df.loc[cut_off_scores_df['value'] == score].index)

average_n_inj_in_kch = 7057 / 4776
best_fitting_distribution_per_score = []
best_fitting_ave_n = []

average_ninj_in_hos = summarize(n_inj_per_person_in_hos, only_mean=True).mean(axis=0)

for n, idx in enumerate(idxs):
    best_fit_found = min(average_ninj_in_hos[idx], key=lambda x: abs(x - average_n_inj_in_kch))
    best_fitting_ave_n.append(best_fit_found)
    best_fit_index = np.where(average_ninj_in_hos == best_fit_found)
ax4 = fig.add_subplot(gs[1, 1])
ax4.bar(np.arange(2),
        [average_ninj_in_hos.values[best_fit_index[0][0]],
         average_n_inj_in_kch], color=['teal', 'lightseagreen'])
ax4.set_xticks(np.arange(2))
ax4.set_xticklabels(['Model', 'KCH'])
ax4.set_ylabel('Ave. number of\n injuries')
outputspath = Path('./outputs/rmjlra2@ucl.ac.uk/')

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
# %% Analyse results of runs when doing a sweep of a single parameter:

# 0) Find results_folder associated with a given batch_file and get most recent
# results_folder = Path('outputs/rmjlra2@ucl.ac.uk/rti_analysis_full_calibrated-2021-12-09T140232Z')
results_folder = get_scenario_outputs('rti_analysis_full_calibrated.py', outputspath)[- 3]
# look at one log (so can decide what to extract)
log = load_pickled_dataframes(results_folder)

# get basic information about the results
info = get_scenario_info(results_folder)

# 1) Extract the parameters that have varied over the set of simulations
params = extract_params(results_folder)
# get main paper results, incidence of RTI, incidence of death and DALYs
extracted_incidence_of_death = extract_results(results_folder,
                                               module="tlo.methods.rti",
                                               key="summary_1m",
                                               column="incidence of rti death per 100,000",
                                               index="date"
                                               )
extracted_incidence_of_RTI = extract_results(results_folder,
                                             module="tlo.methods.rti",
                                             key="summary_1m",
                                             column="incidence of rti per 100,000",
                                             index="date"
                                             )
yll, yld = extract_yll_yld(results_folder)
mean_incidence_of_death = summarize(extracted_incidence_of_death, only_mean=True).mean()
mean_incidence_of_RTI = summarize(extracted_incidence_of_RTI, only_mean=True).mean()
# mean yll, yld per draw
mean_yll = yll.mean()
mean_yld = yld.mean()
# Get calibration results
extracted_perc_in_hos_death = extract_results(results_folder,
                                              module="tlo.methods.rti",
                                              key="summary_1m",
                                              column="percentage died after med",
                                              index="date")
extracted_hsb = extract_results(results_folder,
                                module="tlo.methods.rti",
                                key="summary_1m",
                                column="percent sought healthcare",
                                index="date")
in_hospital_mortality = summarize(extracted_perc_in_hos_death, only_mean=True).mean()
mean_hsb = summarize(extracted_hsb, only_mean=True).mean()
# check calibration of number of injuries per person in hospital
average_n_inj_in_hospital_per_draws = []
for draw in range(info['number_of_draws']):
    ave_n_inj_in_hospital_this_draw = []
    for run in range(info['runs_per_draw']):
        try:
            df: pd.DataFrame = \
                load_pickled_dataframes(results_folder, draw, run, "tlo.methods.rti")["tlo.methods.rti"]
            df = df['number_of_injuries_in_hospital']
            total_n_injuries = sum(df['number_of_injuries'])
            injuries_per_person_in_hos = total_n_injuries / len(df['number_of_injuries'])
            ave_n_inj_in_hospital_this_draw.append(injuries_per_person_in_hos)
        except KeyError:
            pass
    average_n_inj_in_hospital_per_draws.append(np.mean(ave_n_inj_in_hospital_this_draw))
# create a dataframe with all the results
results_df = pd.DataFrame(columns=['ISS cutoff score', 'In hospital mortality scale', 'Ninj dist', 'inc',
                                   'inc_death', 'DALYs', 'YLL', 'YLD', 'HSB', 'In_hos_mort', 'In_hos_inj'])
results_df['ISS cutoff score'] = params.loc[params['module_param'] == 'RTI:rt_emergency_care_ISS_score_cut_off',
                                            'value'].tolist()
results_df['In hospital mortality scale'] = np.divide(
    params.loc[params['module_param'] == 'RTI:prob_death_iss_less_than_9', 'value'].tolist(), (102 / 11650)
)
results_df['Ninj dist'] = params.loc[params['module_param'] == 'RTI:number_of_injured_body_regions_distribution',
                                     'value'].tolist()
results_df['inc'] = mean_incidence_of_RTI
results_df['inc_death'] = mean_incidence_of_death
results_df['DALYs'] = list(np.add(mean_yll, mean_yld))
results_df['YLL'] = list(mean_yll)
results_df['YLD'] = list(mean_yld)
results_df['HSB'] = mean_hsb
results_df['In_hos_mort'] = in_hospital_mortality
results_df['In_hos_inj'] = average_n_inj_in_hospital_per_draws
gbd_inc = 954.2
scale_for_inc = np.divide(gbd_inc, results_df['inc'])
results_df['scaled_inc'] = results_df['inc'] * scale_for_inc
results_df['scaled_inc_death'] = results_df['inc_death'] * scale_for_inc
results_df['scaled_DALYs'] = results_df['DALYs'] * scale_for_inc
# plot the calibration of the model incidence of rti, hsb, number of in
expected_hsb_upper = 0.85
expected_hsb_lower = 0.6533
average_n_inj_in_kch = 7057 / 4776
expected_in_hos_mort = 144 / 7416
hsb_in_accepted_range = np.where((results_df['HSB'] > expected_hsb_lower) & (results_df['HSB'] < expected_hsb_upper))
hsb_colors = ['seagreen' if i not in hsb_in_accepted_range[0] else 'darkseagreen' for i in results_df['HSB'].index]
ax5 = fig.add_subplot(gs[2, 0])
ax5.bar(np.arange(len(results_df)), results_df['HSB'], color=hsb_colors, label='proportion sought care')
ax5.axhline(expected_hsb_upper, color='g', label='Upper HSB bound', linestyle='dashed')
ax5.axhline(expected_hsb_lower, color='g', label='Upper HSB bound', linestyle='dashed')
ax5.set_xticks(results_df.index)
ax5.set_xticklabels(results_df['ISS cutoff score'])
ax5.set_ylabel('Proportion')
ax5.legend(loc='lower left', prop={'size': 6})
ax5.set_xlabel('ISS cut-off score')

results_folder = get_scenario_outputs('rti_in_hospital_mortality_calibration.py', outputspath)[- 1]

# look at one log (so can decide what to extract)
log = load_pickled_dataframes(results_folder)

# get basic information about the results
info = get_scenario_info(results_folder)

# 1) Extract the parameters that have varied over the set of simulations
params = extract_params(results_folder)
idxs = []
cut_off_scores_df = params.loc[params['module_param'] == 'RTI:rt_emergency_care_ISS_score_cut_off']
cut_off_scores = cut_off_scores_df['value'].unique()
for score in cut_off_scores:
    idxs.append(cut_off_scores_df.loc[cut_off_scores_df['value'] == score].index)
# params = extract_params_from_json(results_folder, 'rti_incidence_parameterisation.py', 'RTI', 'base_rate_injrti')
# 2) Extract a specific log series for all runs:
extracted_perc_in_hos_death = extract_results(results_folder,
                                              module="tlo.methods.rti",
                                              key="summary_1m",
                                              column="percentage died after med",
                                              index="date")
extracted_incidence_of_death = extract_results(results_folder,
                                               module="tlo.methods.rti",
                                               key="summary_1m",
                                               column="incidence of rti death per 100,000",
                                               index="date"
                                               )
extracted_incidence_of_RTI = extract_results(results_folder,
                                             module="tlo.methods.rti",
                                             key="summary_1m",
                                             column="incidence of rti per 100,000",
                                             index="date"
                                             )
# 3) Get summary of the results for that log-element
in_hospital_mortality = summarize(extracted_perc_in_hos_death)
incidence_of_death = summarize(extracted_incidence_of_death)
incidence_of_RTI = summarize(extracted_incidence_of_RTI)

# If only interested in the means
in_hospital_mortality_onlymeans = summarize(extracted_perc_in_hos_death, only_mean=True)
mean_in_hospital_mortality_per_draw = in_hospital_mortality_onlymeans.mean()
# get per parameter summaries
mean_in_hospital_mortality_overall = in_hospital_mortality.mean()
mean_incidence_of_death = incidence_of_death.mean()
# get upper and lower estimates
mean_in_hospital_mortality_lower = mean_in_hospital_mortality_overall.loc[:, "lower"]
mean_in_hospital_mortality_upper = mean_in_hospital_mortality_overall.loc[:, "upper"]
lower_upper = np.array(list(zip(
    mean_in_hospital_mortality_lower.to_list(),
    mean_in_hospital_mortality_upper.to_list()
))).transpose()
# find the values that fall within our accepted range of incidence based on results of the GBD study

mean_incidence_of_rti = incidence_of_RTI.mean()
expected_in_hospital_mortality = 144 / 7416
mean_in_hos_mort = in_hospital_mortality_onlymeans.mean()

sample_from_one_run_params = params.loc[idxs[3]]
scales_in_runs = np.divide(
    sample_from_one_run_params.loc[sample_from_one_run_params['module_param'] == 'RTI:prob_death_iss_less_than_9',
                                   'value'],
    (102 / 11650)
)
ax6 = fig.add_subplot(gs[2, 1])

ax6.bar(np.arange(len(mean_in_hos_mort[idxs[3]])), mean_in_hos_mort[idxs[3]], color='lightcoral', label='model')
ax6.set_xticks(np.arange(len(scales_in_runs)))
ax6.set_xticklabels([np.round(val, 3) for val in scales_in_runs], rotation=45, fontsize=6)
ax6.axhline(expected_in_hospital_mortality, color='indianred', label='Expected in-hospital mortality',
            linestyle='dashed')
ax6.set_xlabel('Scale-factor')
ax6.set_ylabel('In-hospital mortality')
ax6.legend(loc='lower right', prop={'size': 6})
ax1.set_title('a)', loc='left')
ax2.set_title('b)', loc='left')
ax3.set_title('c)', loc='left')
ax4.set_title('e)', loc='left')
ax5.set_title('f)', loc='left')
ax6.set_title('g)', loc='left')
plt.savefig("C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/FinalPaperOutput/Figure_3.png",
            bbox_inches='tight')
gbd_age_gender_data = \
    pd.read_csv("C:/Users/Robbie Manning Smith/Desktop/gbddata/age_and_gender_data.csv")
# incidence data
gbd_age_gender_data = gbd_age_gender_data.loc[gbd_age_gender_data['measure'] == 'Incidence']
gender_info = gbd_age_gender_data.groupby('sex').sum()
prop_male = \
    gender_info.loc['Male', 'val'] / (gender_info.loc['Male', 'val'] + gender_info.loc['Female', 'val'])
age_info = gbd_age_gender_data.groupby('age').sum()
age_info = age_info.reindex(index=['1 to 4', '5 to 9', '10 to 14', '15 to 19', '20 to 24', '25 to 29', '30 to 34',
                                   '35 to 39', '40 to 44', '45 to 49', '50 to 54', '55 to 59', '60 to 64', '65 to 69',
                                   '70 to 74', '75 to 79', '80 to 84', '85 to 89', '90 to 94', '95 plus'])
age_info['proportion'] = age_info['val'] / sum(age_info['val'])
plt.clf()
plt.bar(np.arange(len(age_info.index)), age_info.proportion)
plt.xticks(np.arange(len(age_info.index)), age_info.index, rotation=45)
plt.show()
outputspath = Path('./outputs/rmjlra2@ucl.ac.uk/')
results_folder = get_scenario_outputs('rti_analysis_full_calibrated.py', outputspath)[- 3]
info = get_scenario_info(results_folder)
ages_in_sim = []
for draw in range(info['number_of_draws']):
    age_this_draw = []
    for run in range(info['runs_per_draw']):
        try:
            df: pd.DataFrame = \
                load_pickled_dataframes(results_folder, draw, run, "tlo.methods.rti")["tlo.methods.rti"]
            df = df['rti_demography']
            age_of_injured_this_sim = df['age'].values.tolist()
            age_of_injured_this_sim = [age for age_list in age_of_injured_this_sim for age in age_list]
            age_this_draw.append(age_of_injured_this_sim)
        except KeyError:
            pass
    ages_in_sim.append(age_this_draw)


def age_breakdown(age_array):
    """
    A function which breaks down an array of ages into specific age ranges
    :param age_array:
    :return:
    """
    # Breakdown the age data into boundaries 0-5, 6-10, 11-15, 16-20 etc...
    zero_to_five = len([i for i in age_array if i < 6])
    six_to_ten = len([i for i in age_array if 6 <= i < 11])
    eleven_to_fifteen = len([i for i in age_array if 11 <= i < 16])
    sixteen_to_twenty = len([i for i in age_array if 16 <= i < 21])
    twenty1_to_twenty5 = len([i for i in age_array if 21 <= i < 26])
    twenty6_to_thirty = len([i for i in age_array if 26 <= i < 31])
    thirty1_to_thirty5 = len([i for i in age_array if 31 <= i < 36])
    thirty6_to_forty = len([i for i in age_array if 36 <= i < 41])
    forty1_to_forty5 = len([i for i in age_array if 41 <= i < 46])
    forty6_to_fifty = len([i for i in age_array if 46 <= i < 51])
    fifty1_to_fifty5 = len([i for i in age_array if 51 <= i < 56])
    fifty6_to_sixty = len([i for i in age_array if 56 <= i < 61])
    sixty1_to_sixty5 = len([i for i in age_array if 61 <= i < 66])
    sixty6_to_seventy = len([i for i in age_array if 66 <= i < 71])
    seventy1_to_seventy5 = len([i for i in age_array if 71 <= i < 76])
    seventy6_to_eighty = len([i for i in age_array if 76 <= i < 81])
    eighty1_to_eighty5 = len([i for i in age_array if 81 <= i < 86])
    eighty6_to_ninety = len([i for i in age_array if 86 <= i < 91])
    ninety_to_ninety5 = len([i for i in age_array if 90 <= i < 95])
    ninety5_plus = len([i for i in age_array if i >= 95])
    return [zero_to_five, six_to_ten, eleven_to_fifteen, sixteen_to_twenty, twenty1_to_twenty5, twenty6_to_thirty,
            thirty1_to_thirty5, thirty6_to_forty, forty1_to_forty5, forty6_to_fifty, fifty1_to_fifty5, fifty6_to_sixty,
            sixty1_to_sixty5, sixty6_to_seventy, seventy1_to_seventy5, seventy6_to_eighty, eighty1_to_eighty5,
            eighty6_to_ninety, ninety_to_ninety5, ninety5_plus]


counts_in_sim = []
for age_list in ages_in_sim:
    for sim_list in age_list:
        age_counts = age_breakdown(sim_list)
        counts_in_sim.append(age_counts)

ave_age_distribution = [float(sum(col)) / len(col) for col in zip(*counts_in_sim)]
ave_age_distribution = list(np.divide(ave_age_distribution, sum(ave_age_distribution)))
plt.clf()
plt.bar(np.arange(len(age_info.index)), age_info.proportion, width=0.4, color='lightsteelblue', label='GBD')
plt.bar(np.arange(len(age_info.index)) + 0.4, ave_age_distribution, width=0.4, color='lightsalmon', label='Model')
plt.xticks(np.arange(len(age_info.index)) + 0.2, age_info.index, rotation=45)
plt.legend()
plt.show()
