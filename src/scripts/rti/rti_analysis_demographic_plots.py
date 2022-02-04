
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

results_folder = get_scenario_outputs('rti_analysis_calibrate_demographics.py', outputspath)[- 1]

info = get_scenario_info(results_folder)

# 1) Extract the parameters that have varied over the set of simulations
params = extract_params(results_folder)

extracted_incidence_of_RTI = extract_results(results_folder,
                                             module="tlo.methods.rti",
                                             key="summary_1m",
                                             column="incidence of rti per 100,000",
                                             index="date"
                                             )

mean_incidence_of_RTI = summarize(extracted_incidence_of_RTI, only_mean=True).mean()
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

ave_age_distribution = [float(sum(col)) / len(col) for col in zip(*counts_in_sim)]
ave_age_distribution = list(np.divide(ave_age_distribution, sum(ave_age_distribution)))
plt.clf()
plt.bar(np.arange(len(age_info.index)), age_info.proportion, width=0.4, color='lightsteelblue', label='GBD')
plt.bar(np.arange(len(age_info.index)) + 0.4, ave_age_distribution, width=0.4, color='lightsalmon', label='Model')
plt.xticks(np.arange(len(age_info.index)) + 0.2, age_info.index, rotation=45)
plt.legend()
plt.ylabel("Proportion")
plt.xlabel("Age groups")
plt.savefig("C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/FinalPaperOutput/age_distribution.png",
            bbox_inches='tight')
plt.clf()
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
av_total_n_male = summarize(extracted_n_male, only_mean=True).sum()
av_total_n_female = summarize(extracted_n_female, only_mean=True).sum()
perc_male = np.divide(av_total_n_male.tolist(), np.add(av_total_n_male.tolist(), av_total_n_female.tolist()))
