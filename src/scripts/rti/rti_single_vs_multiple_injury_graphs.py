"""This file uses the results of the batch file to make some summary statistics.
The results of the bachrun were put into the 'outputs' results_folder
"""
from pathlib import Path
from src.scripts.rti.rti_create_graphs import age_breakdown
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
                                          column="incidence of rti per 100,000", index="date", do_scaling=False)
deaths_from_rti_incidence = extract_results(results_folder, module="tlo.methods.rti", key="summary_1m",
                                            column="incidence of rti death per 100,000", index="date", do_scaling=False)
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
# produce graphs on general calibration

# check calibration of number of injuries per person in hospital
average_n_inj_in_hospital_per_draws = []
for draw in range(info['number_of_draws']):
    ave_n_inj_in_hospital_this_draw = []
    for run in range(info['runs_per_draw']):
        df: pd.DataFrame = \
            load_pickled_dataframes(results_folder, draw, run, "tlo.methods.rti")["tlo.methods.rti"]
        df = df['number_of_injuries_in_hospital']
        total_n_injuries = sum(df['number_of_injuries'])
        injuries_per_person_in_hos = total_n_injuries / len(df['number_of_injuries'])
        ave_n_inj_in_hospital_this_draw.append(injuries_per_person_in_hos)
    average_n_inj_in_hospital_per_draws.append(np.mean(ave_n_inj_in_hospital_this_draw))
average_inj_in_hos_mult = average_n_inj_in_hospital_per_draws[-1]
average_inj_per_person_general_mult = average_n_inj_per_draws[1]
average_n_inj_per_person_kamuzu = 7057 / 4776
plt.clf()
plt.bar(np.arange(3), [average_n_inj_per_person_kamuzu, average_inj_in_hos_mult, average_inj_per_person_general_mult],
        color=['lightsteelblue', 'lightsalmon', 'peachpuff'])
plt.xticks(np.arange(3), ['Average\ninjuries p.p.\n reported in\n Kamuzu',
                          'Average\ninjuries p.p.\n in model\nhealthsystem',
                          'Average\ninjuries p.p.\n general'])
plt.ylabel('Average number of\ninjuries per person')
plt.title('The average number of injuries per person \nin the model compared to data from KCH')
plt.savefig('C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/SingleVsMultipleInjury/BatchResults/'
            'NinjSummary.png', bbox_inches='tight')
extracted_hsb = extract_results(results_folder,
                                module="tlo.methods.rti",
                                key="summary_1m",
                                column="percent sought healthcare",
                                index="date")
prop_sought_healthcare = summarize(extracted_hsb)
mean_percent_sought_care_multiple = prop_sought_healthcare[1, 'mean'].mean()
plt.clf()
plt.pie([mean_percent_sought_care_multiple, 1 - mean_percent_sought_care_multiple],
        labels=['Sought\ncare', "Didn't\nseek\ncare"], colors=['thistle', 'peachpuff'], autopct='%1.1f%%',
        startangle=90)
plt.title('Multiple injury health seeking behaviour')
plt.savefig('C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/SingleVsMultipleInjury/BatchResults/'
            'HSB.png', bbox_inches='tight')

average_percent_male_per_draw = []
for draw in range(info['number_of_draws']):
    ave_percent_male_this_draw = []
    for run in range(info['runs_per_draw']):
        df: pd.DataFrame = \
            load_pickled_dataframes(results_folder, draw, run, "tlo.methods.rti")["tlo.methods.rti"]
        df = df['rti_demography']
        males = df['males_in_rti'].sum()
        females = df['females_in_rti'].sum()
        total = males + females
        percent_male = males / total
        ave_percent_male_this_draw.append(percent_male)
    average_percent_male_per_draw.append(np.mean(ave_percent_male_this_draw))
mult_percent_male = average_percent_male_per_draw[-1]
plt.clf()
plt.pie([mult_percent_male, 1 - mult_percent_male], labels=['male', 'female'], colors=['lemonchiffon', 'palegreen'],
        autopct='%1.1f%%', startangle=90)
plt.title('Multiple injury gender demographics of those in RTI')
plt.savefig('C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/SingleVsMultipleInjury/BatchResults/'
            'Gender.png', bbox_inches='tight')
age_range_per_draw = []
for draw in range(info['number_of_draws']):
    age_range_this_draw = []
    for run in range(info['runs_per_draw']):
        df: pd.DataFrame = \
            load_pickled_dataframes(results_folder, draw, run, "tlo.methods.rti")["tlo.methods.rti"]
        df = df['rti_demography']
        ages = df['age'].sum()
        for age in ages:
            age_range_this_draw.append(age)
    age_range_per_draw.append(age_range_this_draw)
labels = ['0-5', '6-10', '11-15', '16-20', '21-25', '26-30', '31-35', '36-40', '41-45', '46-50', '51-55', '56-60',
          '61-65', '66-70', '71-75', '76-80', '81-85', '86-90', '90+']
ages_in_draw = age_breakdown(age_range_per_draw[-1])
plt.clf()
plt.bar(np.arange(len(labels)), np.divide(ages_in_draw, sum(ages_in_draw)), color='lightsteelblue')
plt.xticks(np.arange(len(labels)), labels, rotation=90)
plt.ylabel('Percentage')
plt.xlabel('Age group')
plt.title('Multiple injury age demographics of those in RTI')
plt.savefig('C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/SingleVsMultipleInjury/BatchResults/'
            'Age.png', bbox_inches='tight')
shock_results = extract_results(results_folder,
                                module="tlo.methods.rti",
                                key="Percent_of_shock_in_rti",
                                column="Percent_of_shock_in_rti",
                                index="date")
shock_results = summarize(shock_results)
mean_percent_in_shock = shock_results[1, 'mean'].mean()
expected_percent_in_shock = 56 / 8026
plt.clf()
plt.bar(np.arange(2), [expected_percent_in_shock, mean_percent_in_shock], color=['lightsteelblue', 'lightsalmon'])
plt.xticks(np.arange(2), ['% in shock\nElaju et al.', '% in shock model'])
plt.ylabel('Percent')
plt.title('Multiple injury percent of RTI in shock')
plt.savefig('C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/SingleVsMultipleInjury/BatchResults/'
            'Percent_in_shock.png', bbox_inches='tight')
in_hospital_mortality = extract_results(results_folder,
                                        module="tlo.methods.rti",
                                        key="summary_1m",
                                        column="percentage died after med",
                                        index="date")
in_hospital_mortality = summarize(in_hospital_mortality)
mean_in_hospital_mortality = in_hospital_mortality[1, 'mean'].mean()
expected_in_hospital_mortality = (182 + 38) / (3840 + 1227 + 182 + 38)
plt.clf()
plt.bar(np.arange(2), [expected_in_hospital_mortality, mean_in_hospital_mortality],
        color=['lightsteelblue', 'lightsalmon'])
plt.xticks(np.arange(2), ['KCH\nin-hospital\nmortality', 'Model\nin-hospital\nmortality'])
plt.ylabel('Percent')
plt.title('Multiple injury percent in-hospital mortality')
plt.savefig('C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/SingleVsMultipleInjury/BatchResults/'
            'In_hospital_mortality.png', bbox_inches='tight')
rti_causes_of_death = ['RTI_death_without_med', 'RTI_death_with_med', 'RTI_unavailable_med', 'RTI_imm_death',
                       'RTI_death_shock']
per_draw_percent_cause_of_death = []
for draw in range(info['number_of_draws']):
    this_draw_percent_cause_of_death = []
    for run in range(info['runs_per_draw']):
        df: pd.DataFrame = \
            load_pickled_dataframes(results_folder, draw, run, "tlo.methods.demography")["tlo.methods.demography"]
        df = df['death']
        cause_of_death_distribution = []
        for cause in rti_causes_of_death:
            cause_of_death_distribution.append(len(df.loc[df['cause'] == cause]))
        this_draw_percent_cause_of_death.append(list(np.divide(cause_of_death_distribution,
                                                               sum(cause_of_death_distribution))))
    average_percent_by_cause = [float(sum(col)) / len(col) for col in zip(*this_draw_percent_cause_of_death)]
    per_draw_percent_cause_of_death.append(average_percent_by_cause)
labels = ['No med', 'With med', 'Unavailable\nmed', 'Death on\nscene', 'Shock']
plt.clf()
plt.bar(np.arange(len(rti_causes_of_death)), per_draw_percent_cause_of_death[0], color='lightsalmon', width=0.4,
        label='single')
plt.bar(np.arange(len(rti_causes_of_death)) + 0.4, per_draw_percent_cause_of_death[1], color='lightsteelblue',
        width=0.4, label='multiple')
plt.xticks(np.arange(len(rti_causes_of_death)) + 0.2, labels)
plt.legend()
plt.ylabel('Percent')
plt.title('Comparing deaths by cause, single vs multiple')
plt.savefig('C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/SingleVsMultipleInjury/BatchResults/'
            'Cause_of_deaths.png', bbox_inches='tight')
plt.clf()
plt.bar(np.arange(2), [gbd_inc_rti_inj, gbd_inc_rti_inj * average_n_inj_per_draws[1]],
        color=['lightsteelblue', 'lightsalmon'])
plt.xticks(np.arange(2), ['GBD', 'Model'])
plt.ylabel('Incidence per 100,000\nperson years')
plt.title('The incidence of injury due to RTI in the population')
plt.savefig('C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/SingleVsMultipleInjury/BatchResults/'
            'scaled_inc_injuries.png', bbox_inches='tight')
plt.clf()
extracted_dalys = extract_results(results_folder,
                                  module="tlo.methods.healthburden",
                                  key="dalys",
                                  column="Transport Injuries",
                                  index="date",
                                  do_scaling=True)
dalys = summarize(extracted_dalys)
mean_dalys_mult = extracted_dalys.sum()[1].mean()
mean_dalys_single = extracted_dalys.sum()[0].mean()
plt.bar(np.arange(2), [mean_dalys_single, mean_dalys_mult], color=['lightsteelblue', 'lightsalmon'])
plt.xticks(np.arange(2), ['Single injury', 'Multiple injury'])
plt.ylabel('DALYs')
plt.title('The DALYs produced in the single injury and\nmultiple injury form of the model')
plt.savefig('C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/SingleVsMultipleInjury/BatchResults/'
            'dalys.png', bbox_inches='tight')
plt.clf()
plt.plot(incidence_of_rti.index, incidence_of_rti[0, 'mean'], color='blue', label='RTI')
plt.plot(incidence_of_death.index, incidence_of_death[0, 'mean'], color='red', label='Death')
plt.fill_between(incidence_of_rti.index, incidence_of_rti[0, 'lower'], incidence_of_rti[0, 'upper'], color='blue',
                 alpha=0.5, label='95% C.I.')
plt.fill_between(incidence_of_death.index, incidence_of_death[0, 'lower'], incidence_of_death[0, 'upper'], color='red',
                 alpha=0.5, label='95% C.I.')
plt.ylabel('Incidence per 100,000 p.y.')
plt.xlabel('Simulation time')
plt.legend()
plt.title('The incidence of RTI and death for the single injury model run')
plt.savefig('C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/SingleVsMultipleInjury/BatchResults/'
            'single_model_run.png', bbox_inches='tight')
plt.clf()
plt.plot(incidence_of_rti.index, incidence_of_rti[1, 'mean'], color='blue', label='RTI')
plt.plot(incidence_of_death.index, incidence_of_death[1, 'mean'], color='red', label='Death')
plt.fill_between(incidence_of_rti.index, incidence_of_rti[1, 'lower'], incidence_of_rti[1, 'upper'], color='blue',
                 alpha=0.5, label='95% C.I.')
plt.fill_between(incidence_of_death.index, incidence_of_death[1, 'lower'], incidence_of_death[1, 'upper'], color='red',
                 alpha=0.5, label='95% C.I.')
plt.ylabel('Incidence per 100,000 p.y.')
plt.xlabel('Simulation time')
plt.legend()
plt.title('The incidence of RTI and death for the multiple injury model run')
plt.savefig('C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/SingleVsMultipleInjury/BatchResults/'
            'multiple_model_run.png', bbox_inches='tight')
gbd_dates = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019]
gbd_yld_estimate_2010_2019 = [17201.73, 16689.13, 18429.77, 17780.11, 20462.97, 19805.86, 21169.19, 19100.62, 23081.26,
                              22055.06]
gbd_yll_estimate_2010_2019 = [103892.353, 107353.63, 107015.04, 106125.14, 105933.16, 106551.59, 106424.49, 105551.97,
                              108052.59, 109301.18]
gbd_dalys_estimate_2010_2019 = np.add(gbd_yld_estimate_2010_2019, gbd_yll_estimate_2010_2019)
gbd_data = pd.DataFrame(data={'yld': gbd_yld_estimate_2010_2019,
                              'yll': gbd_yll_estimate_2010_2019,
                              'dalys': gbd_dalys_estimate_2010_2019},
                        index=gbd_dates)
sing_yll = []
mult_yll = []
sing_yld = []
mult_yld = []
for draw in range(info['number_of_draws']):
    for run in range(info['runs_per_draw']):
        yll_df: pd.DataFrame = \
            load_pickled_dataframes(results_folder, draw, run, "tlo.methods.healthburden")["tlo.methods.healthburden"]
        yll_df = yll_df['yll_by_causes_of_death_stacked']
        yll_df = yll_df.groupby('year').sum()
        rti_columns = [col for col in yll_df.columns if 'RTI' in col]
        yll_df['yll_rti'] = [0.0] * len(yll_df)
        for col in rti_columns:
            yll_df['yll_rti'] += yll_df[col]
        sim_start_year = min(incidence_of_rti.index.year)
        sim_end_year = max(incidence_of_rti.index.year)
        sim_year_range = pd.Index(np.arange(sim_start_year, sim_end_year + 1))
        pop_size_df: pd.DataFrame = \
            load_pickled_dataframes(results_folder, draw, run, "tlo.methods.demography")["tlo.methods.demography"]
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
        if draw == 0:
            sing_yll.append(total_yll)
        else:
            mult_yll.append(total_yll)
        yld_df: pd.DataFrame = \
            load_pickled_dataframes(results_folder, draw, run, "tlo.methods.rti")["tlo.methods.rti"]
        yld_df = yld_df['rti_health_burden_per_day']
        yld_df['year'] = yld_df['date'].dt.year
        yld_df = yld_df.groupby('year').sum()
        yld_df['total_daily_healthburden'] = [sum(daly_weights) for daly_weights in yld_df['daly_weights'].to_list()]
        yld_df['scaled_healthburden'] = yld_df['total_daily_healthburden'] * scaling_df['scale_for_each_year'] / 365
        total_yld = yld_df['scaled_healthburden'].sum()
        if draw == 0:
            sing_yld.append(total_yld)
        else:
            mult_yld.append(total_yld)
sing_dalys = np.add(sing_yld, sing_yll)
mult_dalys = np.add(mult_yld, mult_yll)
plt.clf()
plt.bar(np.arange(2), [gbd_data['dalys'].sum(), np.mean(sing_dalys)], color=['lightsteelblue', 'lightsalmon'])
plt.xticks(np.arange(2), ['GBD', 'Model'])
plt.ylabel('DALYs')
plt.title('The DALYs caused by RTI estimated by the GBD study and the single injury model run')
plt.savefig('C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/SingleVsMultipleInjury/BatchResults/'
            'single_model_DALYs.png', bbox_inches='tight')
plt.clf()
plt.bar(np.arange(2), [gbd_data['dalys'].sum(), np.mean(mult_dalys)], color=['lightsteelblue', 'lightsalmon'])
plt.xticks(np.arange(2), ['GBD', 'Model'])
plt.ylabel('DALYs')
plt.title('The DALYs caused by RTI estimated by the GBD study and the multiple injury model run')
plt.savefig('C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/SingleVsMultipleInjury/BatchResults/'
            'multiple_model_DALYs.png', bbox_inches='tight')
plt.clf()
gbd_yld = gbd_data['yld'].sum()
gbd_yll = gbd_data['yll'].sum()
plt.barh([1], gbd_yld, color='steelblue', label='YLD')
plt.barh([1], gbd_yll, color='lightskyblue', label='YLL', left=gbd_yld)
plt.barh([0], sing_yld, color='darksalmon', label='YLD')
plt.barh([0], sing_yll, color='coral', label='YLL', left=sing_yld)
plt.yticks(np.arange(2), ['Model', 'GBD'])
plt.xlabel('DALYs')
plt.legend()
plt.title('DALYs predicted by the GBD study and model broken down\n into YLL and YLD')
plt.savefig('C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/SingleVsMultipleInjury/BatchResults/'
            'single_model_DALYs_breakdown.png', bbox_inches='tight')
plt.clf()
gbd_yld = gbd_data['yld'].sum()
gbd_yll = gbd_data['yll'].sum()
plt.barh([1], gbd_yld, color='steelblue', label='YLD')
plt.barh([1], gbd_yll, color='lightskyblue', label='YLL', left=gbd_yld)
plt.barh([0], mult_yld, color='darksalmon', label='YLD')
plt.barh([0], mult_yll, color='coral', label='YLL', left=mult_yld)
plt.yticks(np.arange(2), ['Model', 'GBD'])
plt.xlabel('DALYs')
plt.legend()
plt.title('DALYs predicted by the GBD study and model broken down\n into YLL and YLD')
plt.savefig('C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/SingleVsMultipleInjury/BatchResults/'
            'multiple_model_DALYs_breakdown.png', bbox_inches='tight')
plt.clf()
