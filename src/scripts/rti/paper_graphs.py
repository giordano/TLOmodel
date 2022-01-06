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
data = pd.read_csv("C:/Users/Robbie Manning Smith/Desktop/gbddata/gbd_data.csv")
# get data in years of study
data = data.loc[data['year'] > 2009]
# get male data
male_data = data.loc[data['sex'] == 'Male']
# get number of RTIs
male_data = male_data.loc[male_data['measure'] == 'Incidence']
# get predicted number of males in RTIs
male_rtis = male_data['val'].to_list()
# get female data
female_data = data.loc[data['sex'] == 'Female']
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
