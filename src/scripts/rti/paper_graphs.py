from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import squarify

from tlo.analysis.utils import (
    extract_params,
    extract_results,
    get_scenario_info,
    get_scenario_outputs,
    load_pickled_dataframes,
    summarize,
)

# HS most recent run rti_analysis_full_calibrated-2022-04-27T132823Z

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
fig = plt.figure(constrained_layout=True, figsize=[6.4 * 2.5, 4.8 * 2.5])
# Use GridSpec for customising layout
gs = fig.add_gridspec(nrows=4, ncols=2)
outputspath = Path('./outputs/rmjlra2@ucl.ac.uk/')

results_folder = get_scenario_outputs('rti_analysis_full_calibrated.py', outputspath)[- 4]
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
ax2 = fig.add_subplot(gs[0, 1])

results_folder = get_scenario_outputs('rti_analysis_calibrate_demographics.py', outputspath)[- 1]

extracted_n_male = extract_results(results_folder,
                                   module="tlo.methods.rti",
                                   key="rti_demography",
                                   column="males_in_rti",
                                   index="date"
                                   )
extracted_n_female = extract_results(results_folder,
                                     module="tlo.methods.rti",
                                     key="rti_demography",
                                     column="females_in_rti",
                                     index="date"
                                     )
params = extract_params(results_folder)

n_male = summarize(extracted_n_male).sum()
av_total_n_male = n_male[:, 'mean']
av_total_n_male_upper = n_male[:, 'upper']
av_total_n_male_lower = n_male[:, 'lower']
n_female = summarize(extracted_n_female).sum()
av_total_n_female = n_female[:, 'mean']
av_total_n_female_upper = n_female[:, 'upper']
av_total_n_female_lower = n_female[:, 'lower']
perc_male = np.divide(av_total_n_male.tolist(), np.add(av_total_n_male.tolist(), av_total_n_female.tolist()))
perc_male_upper = np.divide(av_total_n_male_upper.tolist(),
                            np.add(av_total_n_male_upper.tolist(), av_total_n_female_upper.tolist()))
perc_male_lower = np.divide(av_total_n_male_lower.tolist(),
                            np.add(av_total_n_male_lower.tolist(), av_total_n_female_lower.tolist()))
gbd_proportion_male = sum(male_rtis) / (sum(male_rtis) + sum(female_rtis))

closest_est_found = min(perc_male, key=lambda x: abs(x - gbd_proportion_male))
best_fit_idx = np.where(perc_male == closest_est_found)[0][0]
standard_error_perc_male = np.sqrt((perc_male[best_fit_idx] * (1 - perc_male[best_fit_idx])) /
                                   (av_total_n_male[best_fit_idx] + av_total_n_female[best_fit_idx]))
gbd_standard_error = np.sqrt((gbd_proportion_male * (1 - gbd_proportion_male)) / (sum(male_rtis) + sum(female_rtis)))
yerr = [1.96 * gbd_standard_error, 1.96 * standard_error_perc_male]
best_fit_found = perc_male[best_fit_idx]
ax2.bar(np.arange(2), [gbd_proportion_male, best_fit_found], color=['steelblue', 'lightsteelblue'], yerr=yerr)
ax2.set_xticks(np.arange(2))
ax2.set_xticklabels(['GBD', 'Model'])
ax2.set_ylabel('Proportion male')
# plot age distribution
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
ages_in_sim = []
info = get_scenario_info(results_folder)

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
age_standard_dist = [np.std(np.divide(col, sum(col))) for col in zip(*counts_in_sim)]
age_upper = np.add(ave_age_distribution,
                   np.divide(np.multiply(1.96, np.sqrt(age_standard_dist)),
                             np.sqrt(sum([float(sum(col)) / len(col) for col in zip(*counts_in_sim)]))))
age_lower = np.add(ave_age_distribution,
                   np.divide(np.multiply(- 1.96, np.sqrt(age_standard_dist)),
                             np.sqrt(sum([float(sum(col)) / len(col) for col in zip(*counts_in_sim)]))))
ax3 = fig.add_subplot(gs[1, 0])
age_info['upper_prop'] = (age_info['upper'] / age_info['val']) * age_info['proportion']
age_info['lower_prop'] = (age_info['lower'] / age_info['val']) * age_info['proportion']

ax3.bar(np.arange(len(age_info.index)), age_info.proportion, width=0.4, color='lightsteelblue', label='GBD',
        yerr=[age_info.lower_prop, age_info.upper_prop])
ax3.bar(np.arange(len(age_info.index)) + 0.4, ave_age_distribution, width=0.4, color='lightsalmon', label='Model',
        yerr=[age_lower, age_upper])
ax3.set_xticks(np.arange(len(age_info.index)) + 0.2)
ax3.set_xticklabels(age_info.index, rotation=45)
ax3.legend()
ax3.set_ylabel("Proportion")
ax3.set_xlabel("Age groups")
# plot alcohol demographics
results_folder = get_scenario_outputs('rti_analysis_full_calibrated.py', outputspath)[- 4]
extracted_perc_related_to_alc = extract_results(results_folder,
                                                module="tlo.methods.rti",
                                                key="rti_demography",
                                                column="percent_related_to_alcohol",
                                                index="date"
                                                )
mean_perc_alc = summarize(extracted_perc_related_to_alc, only_mean=True).mean().mean()
upper_perc_alcohol = summarize(extracted_perc_related_to_alc).mean()[:, 'upper'].mean()
lower_perc_alcohol = summarize(extracted_perc_related_to_alc).mean()[:, 'lower'].mean()
kch_alc_perc = 0.249
ax4 = fig.add_subplot(gs[1, 1])
ax4.bar(np.arange(len([kch_alc_perc, mean_perc_alc])), [kch_alc_perc, mean_perc_alc],
        color=['royalblue', 'midnightblue'], yerr=[[0, lower_perc_alcohol],
                                                   [0, upper_perc_alcohol]])
ax4.set_xticks(np.arange(len([kch_alc_perc, mean_perc_alc])))
ax4.set_xticklabels(['KCH', 'Model'])
ax4.set_ylabel('Proportion of crashes involving alcohol')
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
on_scene_inc_upper = summarize(extracted_incidence_of_death_on_scene).mean()[:, 'upper'] * scale_to_gbd
on_scene_inc_lower = summarize(extracted_incidence_of_death_on_scene).mean()[:, 'lower'] * scale_to_gbd
target_on_scene_inc_death = 6
closest_est_found = min(on_scene_inc_death, key=lambda x: abs(x - target_on_scene_inc_death))
best_fit_idx = np.where(on_scene_inc_death == closest_est_found)[0][0]
ax5 = fig.add_subplot(gs[2, 0])
ax5.bar(np.arange(2), [target_on_scene_inc_death, on_scene_inc_death[best_fit_idx]], color=['plum', 'violet'])
ax5.vlines(x=1, ymin=on_scene_inc_lower[best_fit_idx], ymax=on_scene_inc_upper[best_fit_idx], color='black')
ax5.set_xticks(np.arange(2))
ax5.set_xticklabels(['Police data', 'Model'])
ax5.set_ylabel('Inc. on scene mort. per \n100,000 p.y.')

results_folder = get_scenario_outputs('rti_analysis_full_calibrated.py', outputspath)[- 4]
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
ninj_upper = summarize(n_inj_per_person_in_hos).mean()[:, 'upper']
ninj_lower = summarize(n_inj_per_person_in_hos).mean()[:, 'lower']
for n, idx in enumerate(idxs):
    best_fit_found = min(average_ninj_in_hos[idx], key=lambda x: abs(x - average_n_inj_in_kch))
    best_fitting_ave_n.append(best_fit_found)
    best_fit_index = np.where(average_ninj_in_hos == best_fit_found)
ax6 = fig.add_subplot(gs[2, 1])
ax6.bar(np.arange(2),
        [average_n_inj_in_kch,
         average_ninj_in_hos.values[best_fit_index[0][0]]],
        color=['lightseagreen', 'teal'])
plt.vlines([1], ymin=ninj_lower[best_fit_index[0][0]], ymax=ninj_upper[best_fit_index[0][0]], color='black')
ax6.set_xticks(np.arange(2))
ax6.set_xticklabels(['KCH', 'Model'])
ax6.set_ylabel('Ave. number of\n injuries')
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
results_folder = get_scenario_outputs('rti_analysis_full_calibrated.py', outputspath)[- 4]
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
hsb_upper = summarize(extracted_hsb).mean()[:, 'upper'].iloc[hsb_in_accepted_range]
hsb_lower = summarize(extracted_hsb).mean()[:, 'lower'].iloc[hsb_in_accepted_range]
ave_percent_admitted_per_draw = []
for draw in range(info['number_of_draws']):
    ave_n_inpatient_days_per_run = []
    for run in range(info['runs_per_draw']):
        try:
            df: pd.DataFrame = \
                load_pickled_dataframes(
                    results_folder, draw, run, "tlo.methods.healthsystem"
                )["tlo.methods.healthsystem"]
            df = df['HSI_Event']
            for person in df.index:
                # Get the number of inpatient days per person, if there is a key error when trying to access inpatient days it
                # means that this patient didn't require any so append (0)
                try:
                    ave_n_inpatient_days_per_run.append(df.loc[person, 'Number_By_Appt_Type_Code']['InpatientDays'])
                except KeyError:
                    ave_n_inpatient_days_per_run.append(0)
        except KeyError:
            pass
    inpatient_day_counts = np.unique(ave_n_inpatient_days_per_run, return_counts=True)[1]
    ave_percent_admitted_per_draw.append(inpatient_day_counts[0] / sum(inpatient_day_counts))


ax7 = fig.add_subplot(gs[3, 0])
ax7.bar(np.arange(len(results_df.iloc[hsb_in_accepted_range])), results_df.iloc[hsb_in_accepted_range]['HSB'],
        color='darkseagreen', label='proportion sought care')
ax7.vlines(x=np.arange(len(results_df.iloc[hsb_in_accepted_range])), ymin=hsb_lower, ymax=hsb_upper, color='black')
ax7.axhline(expected_hsb_upper, color='g', label='Upper HSB bound', linestyle='dashed')
ax7.axhline(expected_hsb_lower, color='g', label='Upper HSB bound', linestyle='dashed')
ax7.set_xticks(np.arange(len(results_df.iloc[hsb_in_accepted_range])))
ax7.set_xticklabels(results_df.iloc[hsb_in_accepted_range]['ISS cutoff score'])
ax7.set_ylabel('Proportion')
ax7.legend(loc='lower left')
ax7.set_xlabel('ISS cut-off score')

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
closest_est_found = min(mean_in_hos_mort[idxs[3]], key=lambda x: abs(x - expected_in_hospital_mortality))
best_fit_idx = np.where(mean_in_hos_mort[idxs[3]] == closest_est_found)[0][0]
best_fit_found = mean_in_hos_mort[idxs[3]].to_list()[best_fit_idx]
in_hos_upper = summarize(extracted_perc_in_hos_death).mean()[:, 'upper'].iloc[best_fit_idx]
in_hos_lower = summarize(extracted_perc_in_hos_death).mean()[:, 'lower'].iloc[best_fit_idx]
ax8 = fig.add_subplot(gs[3, 1])

ax8.bar(np.arange(2), [expected_in_hospital_mortality, best_fit_found], color=['indianred', 'lightcoral'])
ax8.vlines(x=2, ymin=in_hos_lower, ymax=in_hos_upper, color='black')
ax8.set_xticks(np.arange(2))
ax8.set_xticklabels(['Tanzanian national average', 'Model'])
ax8.set_ylabel('Percent mortality')
ax1.set_title('a)       Incidence of RTI', loc='left')
ax2.set_title('b)       Proportion of RTIs involving males', loc='left')
ax3.set_title('c)       Age distribution of RTIs', loc='left')
ax4.set_title('d)       Proportion of RTIs involving alcohol', loc='left')
ax5.set_title('e)       Incidence of on scene mortality per 100,000 persons', loc='left')
ax6.set_title('f)       Average number of injuries of those with RTIs in hospital', loc='left')
ax7.set_title('g)       Proportion of those with RTIs who seek care', loc='left')
ax8.set_title('h)       Proportion of fatalities in those who receive healthcare', loc='left')
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

outputspath = Path('./outputs/rmjlra2@ucl.ac.uk/')
results_folder = get_scenario_outputs('rti_analysis_full_calibrated.py', outputspath)[- 4]
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
plt.savefig("C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/FinalPaperOutput/age_distribution.png",
            bbox_inches='tight')
# Figure 4
plt.clf()
fig, ax1 = plt.subplots()
extracted_incidence_of_death = extract_results(results_folder,
                                               module="tlo.methods.rti",
                                               key="summary_1m",
                                               column="incidence of rti death per 100,000",
                                               index="date"
                                               )
inc_of_death = summarize(extracted_incidence_of_death, only_mean=True).mean()
ax1.set_xlabel('rt_emergency_care_ISS_score_cut_off')
ax1.set_ylabel('Incidence of death\nper 100,000 p.y.')
ax1.bar(results_df.index, inc_of_death, width=0.4, color='lightsalmon',
        label='Incidence of death')
ax1.set_xticks(np.add(results_df.index, 0.2))
ax1.set_xticklabels(results_df['ISS cutoff score'])
ax1.set_ylim([0, 80])
# Adding Twin Axes

ax2 = ax1.twinx()

ax2.set_ylabel('Proportion sought care')
ax2.bar(np.add(results_df.index, 0.4), results_df['HSB'], width=0.4, color='lightsteelblue',
        label='HSB')
ax2.axhline(y=expected_hsb_lower, color='steelblue', linestyle='dashed', label='lower HSB\nboundary')
ax2.axhline(y=expected_hsb_upper, color='lightskyblue', linestyle='dashed', label='upper HSB\nboundary')
ax2.set_ylim([0, 1.1])
ax1.set_title("The model's predicted incidence of death for \nvarying levels of health seeking behaviour")
# Show plot
ax1.legend(loc='upper left')
ax2.legend()
plt.savefig("C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/FinalPaperOutput/inc_death_vs_hsb.png",
            bbox_inches='tight')
plt.clf()
# Figure 5
plt.clf()
fig, ax1 = plt.subplots()
ax1.set_xlabel('rt_emergency_care_ISS_score_cut_off')
ax1.set_ylabel('DALYs')
ax1.bar(results_df.index, results_df['DALYs'], width=0.4, color='lightsalmon',
        label='DALYs')
ax1.set_xticks(np.add(results_df.index, 0.2))
ax1.set_xticklabels(results_df['ISS cutoff score'])
ax1.set_ylim([0, 4000000])
# Adding Twin Axes

ax2 = ax1.twinx()

ax2.set_ylabel('Proportion sought care')
ax2.bar(np.add(results_df.index, 0.4), results_df['HSB'], width=0.4, color='lightsteelblue',
        label='HSB')
ax2.axhline(y=expected_hsb_lower, color='steelblue', linestyle='dashed', label='lower HSB\nboundary')
ax2.axhline(y=expected_hsb_upper, color='lightskyblue', linestyle='dashed', label='upper HSB\nboundary')
ax2.set_ylim([0, 1.1])
ax1.set_title("The model's predicted DALYs between 2010-2019 for \nvarying levels of health seeking behaviour")
# Show plot
ax1.legend(loc='upper left')
ax2.legend()
plt.savefig("C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/FinalPaperOutput/DALYs_vs_hsb.png",
            bbox_inches='tight')
plt.clf()
# Figure 6
single_results_folder = get_scenario_outputs('full_calibrated_single_injury.py', outputspath)[- 1]


# get main paper results, incidence of RTI, incidence of death and DALYs
sing_extracted_incidence_of_death = extract_results(single_results_folder,
                                                    module="tlo.methods.rti",
                                                    key="summary_1m",
                                                    column="incidence of rti death per 100,000",
                                                    index="date"
                                                    )
sing_extracted_incidence_of_RTI = extract_results(single_results_folder,
                                                  module="tlo.methods.rti",
                                                  key="summary_1m",
                                                  column="incidence of rti per 100,000",
                                                  index="date"
                                                  )
sing_yll, sing_yld = extract_yll_yld(single_results_folder)
sing_mean_incidence_of_death = summarize(sing_extracted_incidence_of_death, only_mean=True).mean()
sing_mean_incidence_of_RTI = summarize(sing_extracted_incidence_of_RTI, only_mean=True).mean()
sing_scale_for_inc = np.divide(gbd_inc, sing_mean_incidence_of_RTI)
sing_scaled_inc_death = sing_mean_incidence_of_death * sing_scale_for_inc
sing_scaled_inc_death = list(sing_scaled_inc_death)
sing_dalys = sing_yld.mean() + sing_yll.mean()
sing_dalys = list(sing_dalys)
mult_dalys = results_df['DALYs']
sing_scaled_incidences = sing_scale_for_inc * sing_mean_incidence_of_RTI
average_n_inj_per_draws = []
for draw in range(info['number_of_draws']):
    ave_n_inj_this_draw = []
    for run in range(info['runs_per_draw']):
        try:
            df: pd.DataFrame = \
                load_pickled_dataframes(results_folder, draw, run, "tlo.methods.rti")["tlo.methods.rti"]
            df = df['Injury_information']
            total_n_injuries = sum(df.sum(axis=0)['Number_of_injuries'])
            injuries_per_person = total_n_injuries / len(df.sum(axis=0)['Number_of_injuries'])
            ave_n_inj_this_draw.append(injuries_per_person)
        except KeyError:
            ave_n_inj_this_draw.append(np.mean(ave_n_inj_this_draw))
    average_n_inj_per_draws.append(np.mean(ave_n_inj_this_draw))
gbd_dates = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019]
gbd_yld_estimate_2010_2019 = [17201.73, 16689.13, 18429.77, 17780.11, 20462.97, 19805.86, 21169.19, 19100.62,
                              23081.26, 22055.06]
gbd_yll_estimate_2010_2019 = [103892.353, 107353.63, 107015.04, 106125.14, 105933.16, 106551.59, 106424.49,
                              105551.97, 108052.59, 109301.18]
gbd_dalys_estimate_2010_2019 = np.add(gbd_yld_estimate_2010_2019, gbd_yll_estimate_2010_2019)
gbd_data = pd.DataFrame(data={'yld': gbd_yld_estimate_2010_2019, 'yll': gbd_yll_estimate_2010_2019,
                              'dalys': gbd_dalys_estimate_2010_2019},
                        index=gbd_dates)
for index, value in enumerate(hsb_in_accepted_range[0]):
    fig, ax1 = plt.subplots()
    sing_inc_death = sing_mean_incidence_of_death[index]
    sing_inc_rti = sing_mean_incidence_of_RTI[index]
    scale_for_sing = 954.2 / sing_inc_rti
    sing_scaled_inc_death = scale_for_sing * sing_inc_death
    mult_unscaled_inc_death = results_df['inc_death'][value]
    mult_scaled_inc_death = results_df['scaled_inc_death'][value]
    dalys = [gbd_data['dalys'].sum(), sing_dalys[index], mult_dalys[value]]
    gbd_results = [954.2, 12.1, 954.2]
    single_results = [sing_mean_incidence_of_RTI[index], sing_inc_death, sing_mean_incidence_of_RTI[index]]
    mult_results = [results_df['inc'][value], mult_unscaled_inc_death,
                    results_df['inc'][value] * average_n_inj_per_draws[value]]
    ax1.bar(np.arange(3), gbd_results, width=0.25, color='gold', label='GBD')
    ax1.bar(np.arange(3) + 0.25, single_results, width=0.25, color='lightsteelblue', label='Single')
    ax1.bar(np.arange(3) + 0.5, mult_results, width=0.25,
            color='lightsalmon', label='Multiple')
    ax1.set_xticks(np.arange(4) + 0.25)
    ax1.set_xticklabels(['Incidence\nof\nRTI', 'Incidence\nof\ndeath', 'Incidence\nof\ninjuries', 'DALYs'])
    for idx, val in enumerate(gbd_results):
        ax1.text(np.arange(3)[idx] - 0.125, gbd_results[idx] + 10, f"{np.round(val, 2)}", fontdict={'fontsize': 9},
                 rotation=45)
    for idx, val in enumerate(single_results):
        ax1.text(np.arange(3)[idx] + 0.25 - 0.125, single_results[idx] + 10, f"{np.round(val, 2)}",
                 fontdict={'fontsize': 9}, rotation=45)
    for idx, val in enumerate(mult_results):
        ax1.text(np.arange(3)[idx] + 0.5 - 0.125, mult_results[idx] + 10, f"{np.round(val, 2)}",
                 fontdict={'fontsize': 9},
                 rotation=45)
    ax1.set_ylim([0, 1800])
    ax1.legend(loc='upper left')
    ax1.set_title('Comparing the incidence of RTI, RTI death and injuries\nfor the GBD study, single injury model and\n'
                  'multiple injury model')
    ax1.set_ylabel('Incidence per \n 100,000 person years')
    ax1.axvline(2.75, color='black', linestyle='solid')
    ax2 = ax1.twinx()
    ax2.bar([3, 3.25, 3.5], dalys, width=0.25, color=['gold', 'lightsteelblue', 'lightsalmon'])
    dalys_x_loc = [3, 3.25, 3.5]
    for idx, val in enumerate(dalys):
        ax2.text(dalys_x_loc[idx] - 0.05, dalys[idx] + 100000, f"{np.round(val, 2)}",
                 fontdict={'fontsize': 9},
                 rotation=90)
    ax2.set_ylabel('Total DALYs 2010-2019')
    ax2.set_ylim([0, 4500000])
    plt.savefig(f"C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/FinalPaperOutput/"
                f"IncidenceSummary_ISS_cut_off_is_{value + 1}.png", bbox_inches='tight')
    plt.clf()
    # scaled results
    fig, ax1 = plt.subplots()
    scaled_dalys = [gbd_data['dalys'].sum(), sing_dalys[index] * sing_scale_for_inc[index],
                    results_df.loc[int(index), 'scaled_DALYs']]
    single_results = [sing_scaled_incidences[index], sing_scaled_inc_death, sing_scaled_incidences[index]]
    mult_results = [results_df['scaled_inc'][value], results_df['scaled_inc_death'][value],
                    results_df['scaled_inc'][value] * average_n_inj_per_draws[value]]
    ax1.bar(np.arange(3), gbd_results, width=0.25, color='gold', label='GBD')
    ax1.bar(np.arange(3) + 0.25, single_results, width=0.25, color='lightsteelblue', label='Single')
    ax1.bar(np.arange(3) + 0.5, mult_results, width=0.25,
            color='lightsalmon', label='Multiple')
    ax1.set_xticks(np.arange(4) + 0.25)
    ax1.set_xticklabels(['Incidence\nof\nRTI', 'Incidence\nof\ndeath', 'Incidence\nof\ninjuries', 'DALYs'])
    for idx, val in enumerate(gbd_results):
        ax1.text(np.arange(3)[idx] - 0.125, gbd_results[idx] + 10, f"{np.round(val, 2)}", fontdict={'fontsize': 9},
                 rotation=45)
    for idx, val in enumerate(single_results):
        ax1.text(np.arange(3)[idx] + 0.25 - 0.125, single_results[idx] + 10, f"{np.round(val, 2)}",
                 fontdict={'fontsize': 9}, rotation=45)
    for idx, val in enumerate(mult_results):
        ax1.text(np.arange(3)[idx] + 0.5 - 0.125, mult_results[idx] + 10, f"{np.round(val, 2)}",
                 fontdict={'fontsize': 9},
                 rotation=45)
    ax1.set_ylim([0, 1800])
    ax1.legend(loc='upper left')
    ax1.set_title('Comparing the incidence of RTI, RTI death and injuries\nfor the GBD study, single injury model and\n'
                  'multiple injury model')
    ax1.set_ylabel('Incidence per \n 100,000 person years')
    ax1.axvline(2.75, color='black', linestyle='solid')
    ax2 = ax1.twinx()
    ax2.bar([3, 3.25, 3.5], scaled_dalys, width=0.25, color=['gold', 'lightsteelblue', 'lightsalmon'])
    dalys_x_loc = [3, 3.25, 3.5]
    for idx, val in enumerate(scaled_dalys):
        ax2.text(dalys_x_loc[idx] - 0.05, scaled_dalys[idx] + 100000, f"{np.round(val, 2)}",
                 fontdict={'fontsize': 9},
                 rotation=90)
    ax2.set_ylabel('Total DALYs 2010-2019')
    ax2.set_ylim([0, 4500000])
    plt.savefig(f"C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/FinalPaperOutput/"
                f"SCALED_IncidenceSummary_ISS_cut_off_is_{value + 1}.png", bbox_inches='tight')
    plt.clf()
# Figure 7
no_hs_results = get_scenario_outputs('rti_analysis_full_calibrated_no_hs.py', outputspath)[- 1]
no_hs_extracted_incidence_of_death = extract_results(no_hs_results,
                                                     module="tlo.methods.rti",
                                                     key="summary_1m",
                                                     column="incidence of rti death per 100,000",
                                                     index="date"
                                                     )
no_hs_extracted_incidence_of_RTI = extract_results(no_hs_results,
                                                   module="tlo.methods.rti",
                                                   key="summary_1m",
                                                   column="incidence of rti per 100,000",
                                                   index="date"
                                                   )
no_hs_yll, no_hs_yld = extract_yll_yld(no_hs_results)
no_hs_mean_inc_rti = summarize(no_hs_extracted_incidence_of_RTI, only_mean=True).mean()
no_hs_mean_inc_death = summarize(no_hs_extracted_incidence_of_death, only_mean=True).mean()
scale_for_no_hs = np.divide(gbd_inc, no_hs_mean_inc_rti)
no_hs_scaled_inc = no_hs_mean_inc_rti * scale_for_no_hs
no_hs_scaled_inc_death = no_hs_mean_inc_death * scale_for_no_hs
no_hs_dalys = no_hs_yll.mean() + no_hs_yld.mean()
no_hs_scaled_dalys = np.multiply(list(no_hs_dalys), list(scale_for_no_hs))
mean_no_hs_inc_death = no_hs_scaled_inc_death.mean()
mean_no_hs_dalys = np.mean(no_hs_scaled_dalys)
mult_inc_death = mean_incidence_of_death

plt.clf()
plt.bar([0, 1], [mult_dalys[hsb_in_accepted_range[0]].mean(), mean_no_hs_dalys],
        color=['lightsteelblue', 'lightsalmon'])
plt.xticks([0, 1], ['With health\nsystem', 'Without health\nsystem'])
plt.ylabel('DALYs')
plt.title('Number of DALYs with and without the health system')
for idx, val in enumerate([mult_dalys[hsb_in_accepted_range[0]].mean(), mean_no_hs_dalys]):
    plt.text(idx - 0.15, val + 500000, f"{np.round(val, 2)}")
plt.ylim([0, 14000000])
plt.savefig(f"C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/FinalPaperOutput/"
            f"hs_no_hs_inc_dalys.png", bbox_inches='tight')
plt.clf()
plt.clf()
plt.bar([0, 1], [mult_inc_death[hsb_in_accepted_range[0]].mean(), mean_no_hs_inc_death],
        color=['lightsteelblue', 'lightsalmon'])
plt.xticks([0, 1], ['With health\nsystem', 'Without health\nsystem'])
plt.ylabel('Incidence per 100,000 p.y.')
plt.title('Incidence of death with and without the health system')
for idx, val in enumerate([mult_inc_death[hsb_in_accepted_range[0]].mean(), mean_no_hs_inc_death]):
    plt.text(idx - 0.08, val + 5, f"{np.round(val, 2)}")
plt.ylim([0, 200])
plt.savefig(f"C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/FinalPaperOutput/"
            f"hs_no_hs_inc_death.png", bbox_inches='tight')
plt.clf()
# Figure 8

gbd_data = pd.read_csv('resources/gbd/ResourceFile_Deaths_And_DALYS_GBD2019.csv')
gbd_data = gbd_data.loc[gbd_data['measure_name'] == 'DALYs (Disability-Adjusted Life Years)']
gbd_data = gbd_data.loc[gbd_data['Year'].isin(range(2010, 2020))]
gbd_data = gbd_data.groupby('cause_name').sum()
gbd_data = gbd_data.nlargest(11, 'GBD_Est')
old_order_names = gbd_data.index
old_order_values = gbd_data['GBD_Est'].values
gbd_data = gbd_data.nlargest(10, 'GBD_Est')
gbd_data.loc['Road injuries'] = [805800, 0, results_df.iloc[hsb_in_accepted_range[0]].DALYs.mean(), 0, 0]
gbd_data = gbd_data.sort_values('GBD_Est', ascending=False)
new_order_names = gbd_data.index
new_order_values = gbd_data['GBD_Est'].values

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
plt.savefig("C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/FinalPaperOutput/DALY_rankings.png",
            bbox_inches='tight')
plt.clf()


results_folder = get_scenario_outputs('rti_analysis_full_calibrated.py', outputspath)[- 4]
# look at one log (so can decide what to extract)
log = load_pickled_dataframes(results_folder)

# get basic information about the results
info = get_scenario_info(results_folder)

# 1) Extract the parameters that have varied over the set of simulations
params = extract_params(results_folder)
deaths_each_draw = []
for draw in range(info['number_of_draws']):
    deaths_this_draw = []
    for run in range(info['runs_per_draw']):

        df: pd.DataFrame = \
            load_pickled_dataframes(results_folder, draw, run, 'tlo.methods.demography')['tlo.methods.demography']
        df = df['death']
        df = df.drop(df.loc[df['cause'] == 'Other'].index)
        deaths_this_draw.append(df.cause.values.tolist())

    deaths_each_draw.append(deaths_this_draw)
death_dist_per_draw = []
for draw in deaths_each_draw:
    counts_this_draw = []
    for run in draw:
        counts_this_draw.append(np.unique(run, return_counts=True)[1])
    mean_dist_this_draw = [sum(i) for i in zip(*counts_this_draw)]
    death_dist_per_draw.append(list(np.divide(mean_dist_this_draw, sum(mean_dist_this_draw))))
