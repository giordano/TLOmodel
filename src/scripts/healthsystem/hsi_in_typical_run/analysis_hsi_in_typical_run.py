"""This file uses the run generated by `scenario_hsi_in_typical_run.py` to generate descriptions of the HSI that occur
in a typical run."""

# %% Declare the name of the file that specified the scenarios used in this run.
from pathlib import Path

import matplotlib.pyplot as plt
# import numpy as np
import pandas as pd
from matplotlib.cm import get_cmap

from tlo.analysis.utils import get_scenario_outputs, load_pickled_dataframes

scenario_filename = 'scenario_hsi_in_typical_run.py'

# %% Declare usual paths:
outputspath = Path('./outputs/bshe@ic.ac.uk')
rfp = Path('./resources')

# Find results folder (most recent run generated using that scenario_filename)
results_folder = get_scenario_outputs(scenario_filename, outputspath)[-1]
print(f"Results folder is: {results_folder}")

# Declare path for output graphs from this script
make_graph_file_name = lambda stub: results_folder / f"{stub}.png"  # noqa: E731

# %% Extract results
log = load_pickled_dataframes(results_folder)['tlo.methods.healthsystem']  # (There was only one draw and one run)

# %% Plot: Fraction of Total Healthcare Worker Time Used

cap = log['Capacity']
cap["date"] = pd.to_datetime(cap["date"])
cap = cap.set_index('date')

frac_time_used = cap['Frac_Time_Used_Overall']
frac_time_used_2014_2018 = frac_time_used.loc['2013-12-31':'2019-01-01']
frac_time_used_2016 = frac_time_used.loc['2015-12-31':'2017-01-01']

# Plot:
frac_time_used_2014_2018.plot()
plt.title("Fraction of Total Healthcare Worker Time Used (year 2014-2018)")
plt.xlabel("Date")
plt.tight_layout()
plt.savefig(make_graph_file_name('HSI_Frac_time_used_2014_2018'))
plt.show()

frac_time_used_2016.plot()
plt.title("Fraction of Total Healthcare Worker Time Used (year 2016)")
plt.xlabel("Date")
plt.tight_layout()
plt.savefig(make_graph_file_name('HSI_Frac_time_used_2016'))
plt.show()

# %% Number of HSI:

hsi = log['HSI_Event']
hsi["date"] = pd.to_datetime(hsi["date"])
hsi["month"] = hsi["date"].dt.month

# Number of HSI that are taking place by originating module, by month
year = 2016
hsi["Module"] = hsi["TREATMENT_ID"].str.split('_').apply(lambda x: x[0])
# todo: Rename Module HSI to Generic First Appointment? \
#  (Q: The treatment_id for module HSI seems not related to Generic First Appt; \
#  Besides, there are similar modules called GenericEmergency... and GenericFirstAppt...)
evs = hsi.loc[hsi.date.dt.year == year]\
    .groupby(by=['month', 'Module'])\
    .size().reset_index().rename(columns={0: 'count'})\
    .pivot_table(index='month', columns='Module', values='count', fill_value=0)

# Plot:
# Use colormap tab20 so that each module has a unique color
color_tab20 = get_cmap('tab20_r')
evs.plot.bar(stacked=True, color=color_tab20.colors)
plt.title(f"HSI by Module, per Month (year {year})")
plt.ylabel('Total per month')
plt.tight_layout()
plt.legend(ncol=3, loc='center', fontsize='xx-small')
plt.savefig(make_graph_file_name('HSI_per_module_per_month'))
plt.show()

# Plot the breakdown of all HSI, over all the years 2010-2018
evs = pd.DataFrame(hsi.groupby(by=['Module']).size())
# Calculate the fraction
evs[1] = 100*evs[0]/evs[0].sum()
patches, texts = plt.pie(evs[0], colors=color_tab20.colors)
labels = ['{0} - {1:1.2f} %'.format(i, j) for i, j in zip(evs.index, evs[1])]
# Sort legend
sort_legend = True
if sort_legend:
    patches, labels, dummy = zip(*sorted(zip(patches, labels, evs[0]),
                                         key=lambda x: x[2],
                                         reverse=True))
plt.legend(patches, labels, ncol=3, loc='lower center', fontsize='xx-small')
plt.title("HSI by Module (year 2010-2018)")
plt.tight_layout()
plt.savefig(make_graph_file_name('HSI_per_module'))
plt.show()

# %% Demand for appointments

num_hsi_by_treatment_id = hsi.groupby(hsi.TREATMENT_ID)['Number_By_Appt_Type_Code'].size()

# find the appt footprint for each treatment_id
appts_by_treatment_id = \
    hsi.set_index('TREATMENT_ID')['Number_By_Appt_Type_Code'].drop_duplicates().apply(pd.Series).fillna(0.0)

# Todo: Since the resulted appts_by_treatment_id deleted many hsi events \
#  (i.e., the hsi list is much shorter than in num_hsi_by_treatment_id), \
#  will try delete empty entries first then apply drop_duplicates().


# Plot...
# See the Sankey plot in analysis_sankey_appt_and_hsi.ipynb (in the same folder)
