"""This file uses the results of the scenario runs to generate plots

*1 DALYs averted and HCW time required

"""

import os
import datetime
from pathlib import Path
import pickle
# import lacroix
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
# Set paths
outputpath = Path("./outputs/")

# Define paths for different scenarios
sc0_path = outputpath / "sc0/default_run.pickle"
sc1_path = outputpath / "sc1/default_run.pickle"
sc2_path = outputpath / "sc1/default_run.pickle"  # I assume you meant sc2 here

# Load pickle for scenario 0
if sc0_path.exists():
    with sc0_path.open("rb") as f:
        output0 = pickle.load(f)
else:
    print(f"File {sc0_path} does not exist!")

# Load pickle for scenario 1
if sc1_path.exists():
    with sc1_path.open("rb") as f1:
        output1 = pickle.load(f1)
else:
    print(f"File {sc1_path} does not exist!")

# Load pickle for scenario 2
if sc2_path.exists():
    with sc2_path.open("rb") as f2:
        output2 = pickle.load(f2)
else:
    print(f"File {sc2_path} does not exist!")


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
    dalys.loc['TB (non-AIDS)'] = dalys.loc['TB (non-AIDS)'] + dalys.loc['non_AIDS_TB']
    dalys.drop(['non_AIDS_TB'], inplace=True)
    out = pd.DataFrame()
    out['median'] = dalys.median(axis=1).round(decimals=-3).astype(int)
    out['lower'] = dalys.quantile(q=0.025, axis=1).round(decimals=-3).astype(int)
    out['upper'] = dalys.quantile(q=0.975, axis=1).round(decimals=-3).astype(int)

    return out


def return_daly_summary2(results_folder):
    dalys = extract_results(
        results_folder,
        module='tlo.methods.healthburden',
        key='dalys_stacked',
        custom_generate_series=num_dalys_by_cause,
        do_scaling=True
    )
    dalys.columns = dalys.columns.get_level_values(0)
    # combine two labels for non-AIDS TB (this now fixed in latest code)
    dalys.loc['TB (non-AIDS)'] = dalys.loc['TB (non-AIDS)'] + dalys.loc['non_AIDS_TB']
    # dalys.drop(['non_AIDS_TB'], inplace=True)
    out = pd.DataFrame()
    out['median'] = dalys.median(axis=1).round(decimals=-3).astype(int)
    out['lower'] = dalys.quantile(q=0.025, axis=1).round(decimals=-3).astype(int)
    out['upper'] = dalys.quantile(q=0.975, axis=1).round(decimals=-3).astype(int)

    return out


dalys0 = return_daly_summary("/Users/wenjiazhang/Documents/MSc_HDA/Summer/TLOmodel/outputs/sc0/")
dalys1 = return_daly_summary("/Users/wenjiazhang/Documents/MSc_HDA/Summer/TLOmodel/outputs/sc1/")
dalys2 = return_daly_summary2("/Users/wenjiazhang/Documents/MSc_HDA/Summer/TLOmodel/outputs/sc1/")

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

# extract dalys averted by each scenario relative to scenario 0
# comparison should be run-by-run
full_dalys0 = extract_results(
    results0,
    module='tlo.methods.healthburden',
    key='dalys_stacked',
    custom_generate_series=num_dalys_by_cause,
    do_scaling=True
)
full_dalys0.loc['TB (non-AIDS)'] = full_dalys0.loc['TB (non-AIDS)'] + full_dalys0.loc['non_AIDS_TB']
full_dalys0.drop(['non_AIDS_TB'], inplace=True)
full_dalys0.loc['Column_Total'] = full_dalys0.sum(numeric_only=True, axis=0)

full_dalys1 = extract_results(
    results1,
    module='tlo.methods.healthburden',
    key='dalys_stacked',
    custom_generate_series=num_dalys_by_cause,
    do_scaling=True
)
full_dalys1.loc['TB (non-AIDS)'] = full_dalys1.loc['TB (non-AIDS)'] + full_dalys1.loc['non_AIDS_TB']
full_dalys1.drop(['non_AIDS_TB'], inplace=True)
full_dalys1.loc['Column_Total'] = full_dalys1.sum(numeric_only=True, axis=0)

full_dalys2 = extract_results(
    results2,
    module='tlo.methods.healthburden',
    key='dalys_stacked',
    custom_generate_series=num_dalys_by_cause,
    do_scaling=True
)
full_dalys2.loc['Column_Total'] = full_dalys2.sum(numeric_only=True, axis=0)

writer = pd.ExcelWriter(r"outputs/t.mangal@imperial.ac.uk/full_dalys.xlsx")
full_dalys0.to_excel(writer, sheet_name='sc0')
full_dalys1.to_excel(writer, sheet_name='sc1')
full_dalys2.to_excel(writer, sheet_name='sc2')
writer.save()

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


aids_dalys_diff = [sc2_sc0_median['AIDS'],
                   sc1_sc0_median['AIDS']]
tb_dalys_diff = [sc2_sc0_median['TB (non-AIDS)'],
                 sc1_sc0_median['TB (non-AIDS)']]
total_dalys_diff = [sc2_sc0_median['Column_Total'],
                    sc1_sc0_median['Column_Total']]

plt.style.use('ggplot')

aids_colour = "#8949ab"
tb_colour = "#ed7e7a"
total_colour = "#eede77"

# present DALYs in millions
million = 1000000
aids_dalys_diff = [x / million for x in aids_dalys_diff]
tb_dalys_diff = [x / million for x in tb_dalys_diff]
total_dalys_diff = [x / million for x in total_dalys_diff]


fig, ax1 = plt.subplots(nrows=1, ncols=1,
                                             figsize=(5, 4))
fig.suptitle('')

# DALYs
labels = ['Unconstrained scale-up', 'Constrained scale-up']
x = np.arange(len(labels))  # the label locations
width = 0.2  # the width of the bars

rects1 = ax1.bar(x - width, aids_dalys_diff, width, label='AIDS', color=aids_colour)
rects2 = ax1.bar(x, tb_dalys_diff, width, label='TB', color=tb_colour)
rects3 = ax1.bar(x + width, total_dalys_diff, width, label='Total', color=total_colour)

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
