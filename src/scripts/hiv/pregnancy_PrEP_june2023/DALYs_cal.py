"""This file uses the results of the scenario runs to generate plots

*1 DALYs averted and HCW time required

"""

import os
import datetime
from pathlib import Path
import pickle
import lacroix
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import statsmodels.api as sm

from tlo.analysis.utils import (
    compare_number_of_deaths,
    extract_params,
    extract_results,
    get_scenario_info,
    get_scenario_outputs,
    load_pickled_dataframes,
    summarize,
)
from tlo import Date


# Set the working directory
os.chdir('/Users/wenjiazhang/Documents/MSc_HDA/Summer/TLOmodel/')
resourcefilepath = Path("./resources")
outputspath = Path("./outputs/wz2016@ic.ac.uk/")

# Define paths for different scenarios
results0 = get_scenario_outputs("batch_prep_run-2023-08-18T231546Z.py", outputspath)[-1]
results1 = get_scenario_outputs("batch_prep_run-2023-08-18T231546Z.py", outputspath)[-1]
results2 = get_scenario_outputs("batch_prep_run-2023-08-18T231546Z.py", outputspath)[-1]

berry = lacroix.colorList('CranRaspberry')  # ['#F2B9B8', '#DF7878', '#E40035', '#009A90', '#0054A4', '#001563']
baseline_colour = berry[5]  # '#001563'
sc1_colour = berry[3]  # '#009A90'
sc2_colour = berry[2]  # '#E40035'

# %%:  ---------------------------------- DALYS ---------------------------------- #
TARGET_PERIOD = (Date(2023, 1, 1), Date(2036, 1, 1))

def num_dalys_by_cause(_df):
    """Return total number of DALYS (Stacked) (total by age-group within the TARGET_PERIOD)"""
    return _df \
        .loc[_df.year.between(*[i.year for i in TARGET_PERIOD])] \
        .drop(columns=['date', 'sex', 'age_range', 'year']) \
        .sum()


def return_daly_summary(results_folder):
    dalys = extract_results(
        results_folder,
        module='tlo.methods.healthburden',
        key='dalys_stacked',
        custom_generate_series=num_dalys_by_cause,
        do_scaling=True
    )
    dalys.columns = dalys.columns.get_level_values(0)
    # combine two labels for non-AIDS TB (this now fixed in latest code)
    # dalys.loc['TB (non-AIDS)'] = dalys.loc['TB (non-AIDS)'] + dalys.loc['non_AIDS_TB']
    # dalys.drop(['non_AIDS_TB'], inplace=True)
    out = pd.DataFrame()
    out['median'] = dalys.median(axis=1).round(decimals=-3).astype(int)
    out['lower'] = dalys.quantile(q=0.025, axis=1).round(decimals=-3).astype(int)
    out['upper'] = dalys.quantile(q=0.975, axis=1).round(decimals=-3).astype(int)
    return out


dalys0 = return_daly_summary(results0)
dalys1 = return_daly_summary(results1)
dalys2 = return_daly_summary(results2)

dalys0.loc['Column_Total'] = dalys0.sum(numeric_only=True, axis=0)
dalys1.loc['Column_Total'] = dalys1.sum(numeric_only=True, axis=0)
dalys2.loc['Column_Total'] = dalys2.sum(numeric_only=True, axis=0)

# create full table for export
daly_table = pd.DataFrame()
daly_table['scenario0'] = dalys0['median'].astype(str) + \
                          " (" + dalys0['lower'].astype(str) + " - " + \
                          dalys0['upper'].astype(str) + ")"
daly_table['scenario1'] = dalys1['median'].astype(str) + \
                          " (" + dalys1['lower'].astype(str) + " - " + \
                          dalys1['upper'].astype(str) + ")"
daly_table['scenario2'] = dalys2['median'].astype(str) + \
                          " (" + dalys2['lower'].astype(str) + " - " + \
                          dalys2['upper'].astype(str) + ")"

daly_table.to_csv(outputspath / "daly_summary.csv")

#---------- RESULT --------------
print("DALYs caused by AIDS in scenario0:", daly_table.loc['AIDS', 'scenario0'])
print("DALYs caused by AIDS in scenario1:", daly_table.loc['AIDS', 'scenario1'])
print("DALYs caused by AIDS in scenario2:", daly_table.loc['AIDS', 'scenario2'])

million = 1000000
# Extracting DALYs (median, lower, upper) caused by AIDS for each scenario and scale to millions
aids_dalys_median = [dalys0.loc['AIDS', 'median']/million, dalys1.loc['AIDS', 'median']/million, dalys2.loc['AIDS', 'median']/million]
aids_dalys_lower = [dalys0.loc['AIDS', 'lower']/million, dalys1.loc['AIDS', 'lower']/million, dalys2.loc['AIDS', 'lower']/million]
aids_dalys_upper = [dalys0.loc['AIDS', 'upper']/million, dalys1.loc['AIDS', 'upper']/million, dalys2.loc['AIDS', 'upper']/million]

# Calculate the error below and above the median
error_below = [median - lower for median, lower in zip(aids_dalys_median, aids_dalys_lower)]
error_above = [upper - median for median, upper in zip(aids_dalys_median, aids_dalys_upper)]
error_bars = [error_below, error_above]

# Plotting
fig, ax = plt.subplots()
ax.bar(['Scenario 0', 'Scenario 1', 'Scenario 2'], aids_dalys_median, yerr=error_bars,
       color=[baseline_colour, sc1_colour, sc2_colour], capsize=10)

ax.set_title('DALYs caused by AIDS in Different Scenarios')
ax.set_ylabel('Number of DALYs')
plt.show()

# extract dalys averted by each scenario relative to scenario 0
# comparison should be run-by-run
full_dalys0 = extract_results(
    results0,
    module='tlo.methods.healthburden',
    key='dalys_stacked',
    custom_generate_series=num_dalys_by_cause,
    do_scaling=True
)

full_dalys1 = extract_results(
    results1,
    module='tlo.methods.healthburden',
    key='dalys_stacked',
    custom_generate_series=num_dalys_by_cause,
    do_scaling=True
)

full_dalys2 = extract_results(
    results2,
    module='tlo.methods.healthburden',
    key='dalys_stacked',
    custom_generate_series=num_dalys_by_cause,
    do_scaling=True
)

writer = pd.ExcelWriter(r"outputs/wz2016@ic.ac.uk/full_dalys.xlsx")
full_dalys0.to_excel(writer, sheet_name='sc0')
full_dalys1.to_excel(writer, sheet_name='sc1')
full_dalys2.to_excel(writer, sheet_name='sc2')
# writer.save()

# DALYs averted: baseline - scenario
# positive value will be DALYs averted due to interventions
# negative value will be higher DALYs reported, therefore increased health burden
sc1_sc0 = full_dalys0.subtract(full_dalys1, fill_value=0)
sc1_sc0_median = sc1_sc0.median(axis=1)
sc1_sc0_lower = sc1_sc0.quantile(q=0.025, axis=1)
sc1_sc0_upper = sc1_sc0.quantile(q=0.975, axis=1)

sc2_sc0 = full_dalys0.subtract(full_dalys2, fill_value=0)
sc2_sc0_median = sc2_sc0.median(axis=1)
sc2_sc0_lower = sc2_sc0.quantile(q=0.025, axis=1)
sc2_sc0_upper = sc2_sc0.quantile(q=0.975, axis=1)

# create full table for export
daly_averted_table = pd.DataFrame()
daly_averted_table['cause'] = sc1_sc0_median.index
daly_averted_table['scenario1_med'] = [int(round(x, -3)) for x in sc1_sc0_median]
daly_averted_table['scenario1_low'] = [int(round(x, -3)) for x in sc1_sc0_lower]
daly_averted_table['scenario1_upp'] = [int(round(x, -3)) for x in sc1_sc0_upper]
daly_averted_table['scenario2_med'] = [int(round(x, -3)) for x in sc2_sc0_median]
daly_averted_table['scenario2_low'] = [int(round(x, -3)) for x in sc2_sc0_lower]
daly_averted_table['scenario2_upp'] = [int(round(x, -3)) for x in sc2_sc0_upper]

daly_averted_table.to_csv(outputspath / "daly_averted_summary.csv")

# this is now unconstrained scenario first!!
aids_dalys_diff = [sc2_sc0_median['AIDS'],
                   sc1_sc0_median['AIDS']]

plt.style.use('ggplot')

aids_colour = "#8949ab"


# present DALYs in millions
million = 1000000
aids_dalys_diff = [x / million for x in aids_dalys_diff]

fig, ax1 = plt.subplots(nrows=1, ncols=1,
                                             figsize=(5, 4))
fig.suptitle('')

# DALYs
labels = ['Unconstrained scale-up', 'Constrained scale-up']
x = np.arange(len(labels))  # the label locations
width = 0.2  # the width of the bars

rects1 = ax1.bar(x - width, aids_dalys_diff, width, label='AIDS', color=aids_colour)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax1.set_ylabel('DALYs averted, millions')
ax1.set_title('')
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.legend(["AIDS", "TB", "Total"], frameon=False)

font = {'family': 'sans-serif',
        'color': 'black',
        'weight': 'bold',
        'size': 11,
        }

fig.tight_layout()
fig.savefig(outputspath / "DALYS.png")

plt.show()
