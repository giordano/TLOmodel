
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
results_folder = get_scenario_outputs('rti_analysis_full_calibrated.py', outputspath)[- 1]

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
# plot the calibration of the model incidence of rti, hsb, number of injuries and in-hospital mortality
expected_hsb_upper = 0.85
expected_hsb_lower = 0.6533
average_n_inj_in_kch = 7057 / 4776
expected_in_hos_mort = (182 + 38) / (3840 + 1227 + 182 + 38)
hsb_in_accepted_range = np.where((results_df['HSB'] > expected_hsb_lower) & (results_df['HSB'] < expected_hsb_upper))
hsb_colors = ['lightsalmon' if i not in hsb_in_accepted_range[0] else 'darksalmon' for i in results_df['HSB'].index]
inc_colors = ['lightsteelblue' if i not in hsb_in_accepted_range[0] else 'steelblue' for i in results_df['HSB'].index]
in_hos_inj_colors = ['darkseagreen' if i not in hsb_in_accepted_range[0] else 'mediumseagreen' for i in
                     results_df['HSB'].index]
in_hos_inj_mort_colors = ['plum' if i not in hsb_in_accepted_range[0] else 'violet' for i in results_df['HSB'].index]
fig = plt.figure(constrained_layout=True)
# Use GridSpec for customising layout
gs = fig.add_gridspec(nrows=2, ncols=2)
ax1 = fig.add_subplot(gs[0, 0])
ax1.bar(np.arange(len(results_df)), results_df['HSB'], color=hsb_colors, label='Model output')
ax1.axhline(expected_hsb_upper, color='r', label='Upper HSB bound', linestyle='dashed')
ax1.axhline(expected_hsb_lower, color='r', label='Upper HSB bound', linestyle='dashed')
ax1.set_xticks(results_df.index)
ax1.set_xticklabels(results_df['ISS cutoff score'])
ax1.set_ylabel('Percent')
ax1.set_title('Percent sought care')
ax1.legend(loc='lower left')
ax1.set_xlabel('ISS cut-off score')
ax2 = fig.add_subplot(gs[0, 1])
ax2.bar(np.arange(len(results_df)), results_df['inc'], color=inc_colors, label='Model output')
ax2.axhline(gbd_inc, color='b', label='GBD estimate', linestyle='dashed')
ax2.set_ylabel('Incidence per 100,000 p.y.')
ax2.set_xticks(results_df.index)
ax2.set_xticklabels(results_df['ISS cutoff score'])
ax2.legend()
ax2.set_title('Incidence of RTI')
ax2.set_xlabel('ISS cut-off score')
ax3 = fig.add_subplot(gs[1, 0])
ax3.bar(np.arange(len(results_df)), results_df['In_hos_inj'], color=in_hos_inj_colors, label='Model output')
ax3.axhline(average_n_inj_in_kch, color='g', label='Expected injuries p.p.', linestyle='dashed')
ax3.set_xticks(results_df.index)
ax3.set_xticklabels(results_df['ISS cutoff score'])
ax3.set_ylabel('Injuries per person')
ax3.set_title('Average number of injuries per person')
ax3.legend(loc='lower left')
ax3.set_xlabel('ISS cut-off score')
ax4 = fig.add_subplot(gs[1, 1])
ax4.bar(np.arange(len(results_df)), results_df['In_hos_mort'], color=in_hos_inj_mort_colors, label='Model output')
ax4.axhline(expected_in_hos_mort, color='m', label='Expected in-hospital\nmortality', linestyle='dashed')
ax4.set_xticks(results_df.index)
ax4.set_xticklabels(results_df['ISS cutoff score'])
ax4.set_ylabel('Percent')
ax4.set_title('Percent in-hospital mortality')
ax4.legend(loc='lower left')
ax4.set_xlabel('ISS cut-off score')
plt.savefig("C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/FinalPaperOutput/calibration.png",
            bbox_inches='tight')
plt.clf()
# plot the results of the model, the incidence, incidence of death, DALYs and case fatality ratio
fig = plt.figure(constrained_layout=True)
# Use GridSpec for customising layout
gs = fig.add_gridspec(nrows=2, ncols=2)
ax1 = fig.add_subplot(gs[0, 0])
ax1.bar(np.arange(len(results_df)), results_df['inc'], color=inc_colors)
ax1.axhline(gbd_inc, color='b', linestyle='dashed')
ax1.set_xticks(results_df.index)
ax1.set_xticklabels(results_df['ISS cutoff score'])
ax1.set_ylabel('Incidence per 100,000 p.y.')
ax1.set_title('Incidence of RTI')
ax1.set_xlabel('ISS cut-off score')
ax2 = fig.add_subplot(gs[0, 1])
ax2.bar(np.arange(len(results_df)), results_df['inc_death'], color=hsb_colors)
ax2.set_xticks(results_df.index)
ax2.set_xticklabels(results_df['ISS cutoff score'])
ax2.set_ylabel('Incidence per 100,000 p.y.')
ax2.set_title('Incidence of death')
ax2.set_xlabel('ISS cut-off score')
for idx, val in enumerate(results_df['inc_death']):
    ax2.text(idx, val + 3, f"{np.round(val, 2)}", rotation=90)
ax2.set_ylim([0, max(results_df['inc_death']) + 20])
ax3 = fig.add_subplot(gs[1, 0])
ax3.bar(np.arange(len(results_df)), results_df['DALYs'], color=in_hos_inj_colors)
ax3.set_xticks(results_df.index)
ax3.set_xticklabels(results_df['ISS cutoff score'])
ax3.set_ylabel('DALYs')
ax3.set_title('DALYs')
ax3.set_xlabel('ISS cut-off score')
for idx, val in enumerate(results_df['DALYs']):
    ax3.text(idx, val / 3, f"{np.round(val, 2)}", rotation=90)
ax3.set_ylim([0, max(results_df['DALYs']) + 100000])
ax4 = fig.add_subplot(gs[1, 1])
ax4.bar(np.arange(len(results_df)), np.divide(results_df['inc_death'], results_df['inc']),
        color=in_hos_inj_mort_colors)
ax4.set_xticks(results_df.index)
ax4.set_xticklabels(results_df['ISS cutoff score'])
ax4.set_ylabel('CFR')
ax4.set_title('Case fatality ratio')
ax4.set_xlabel('ISS cut-off score')
for idx, val in enumerate(np.divide(results_df['inc_death'], results_df['inc'])):
    ax4.text(idx, val + 0.05, f"{np.round(val, 2)}", rotation=90)
ax4.set_ylim([0, 0.4])
plt.savefig("C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/FinalPaperOutput/results.png",
            bbox_inches='tight')
plt.clf()
fig, ax1 = plt.subplots()
ax1.set_xlabel('Model runs')
ax1.set_ylabel('Incidence of death\nper 100,000 p.y.')
ax1.bar(results_df.index, results_df['inc_death'], width=0.4, color='lightsalmon',
        label='Incidence of death')
ax1.set_xticks(np.add(results_df.index, 0.2))
ax1.set_xticklabels(results_df['ISS cutoff score'])
ax1.set_ylim([0, 80])
# Adding Twin Axes

ax2 = ax1.twinx()

ax2.set_ylabel('Percent sought care')
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
fig, ax1 = plt.subplots()
ax1.set_xlabel('Model runs')
ax1.set_ylabel('DALYs')
ax1.bar(results_df.index, results_df['DALYs'], width=0.4, color='lightsalmon',
        label='DALYs')
ax1.set_xticks(np.add(results_df.index, 0.2))
ax1.set_xticklabels(results_df['ISS cutoff score'])
ax1.set_ylim([0, 4000000])
# Adding Twin Axes

ax2 = ax1.twinx()

ax2.set_ylabel('Percent sought care')
ax2.bar(np.add(results_df.index, 0.4), results_df['HSB'], width=0.4, color='lightsteelblue',
        label='HSB')
ax2.axhline(y=expected_hsb_lower, color='steelblue', linestyle='dashed', label='lower HSB\nboundary')
ax2.axhline(y=expected_hsb_upper, color='lightskyblue', linestyle='dashed', label='upper HSB\nboundary')
ax2.set_ylim([0, 1.1])
ax1.set_title("The model's predicted incidence of death for \nvarying levels of health seeking behaviour")
# Show plot
ax1.legend(loc='upper left')
ax2.legend()
plt.savefig("C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/FinalPaperOutput/DALYs_vs_hsb.png",
            bbox_inches='tight')
plt.clf()
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
new_order_colors = ['lightsalmon'] * len(old_order_names)
new_rti_index = np.where((new_order_names == 'Road injuries'))
new_order_colors[new_rti_index[0][0]] = 'gold'

new_order_colors = ['lightsalmon'] * len(old_order_names)
new_rti_index = np.where([val == 'Road injuries' for val in new_order_names])
new_order_colors[new_rti_index[0][0]] = 'gold'

old_rti_index = np.where([val == 'Road injuries' for val in old_order_names])
old_order_colors = ['lightsteelblue'] * len(old_order_names)
old_order_colors[old_rti_index[0][0]] = 'gold'
fig = plt.figure(constrained_layout=True)
# Use GridSpec for customising layout
gs = fig.add_gridspec(nrows=2, ncols=1)
ax1 = fig.add_subplot(gs[0, 0])
ax1.barh(np.arange(len(old_order_names)), old_order_values, color=old_order_colors)
ax1.set_yticks(np.arange(len(old_order_names)))
ax1.set_yticklabels(old_order_names)
ax1.set_title('GBD ranked total DALYs')
ax2 = fig.add_subplot(gs[1, 0])
ax2.barh(np.arange(len(new_order_names)), new_order_values, color=new_order_colors)
ax2.set_xlabel('Total DALYs 2010-2019')
ax2.set_yticks(np.arange(len(new_order_names)))
ax2.set_yticklabels(new_order_names)
ax2.set_title('New ranked total DALYs')
plt.savefig(f"C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/FinalPaperOutput/"
            f"New_DALY_rankings.png", bbox_inches='tight')
# create a comparison between multiple injury model run and single injury model run
# 0) Find results_folder associated with a given batch_file and get most recent
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
