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
                                          column="incidence of rti per 100,000", index="date")
deaths_from_rti_incidence = extract_results(results_folder, module="tlo.methods.rti", key="summary_1m",
                                            column="incidence of rti death per 100,000", index="date")
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
