"""
This script generates results on consumable scenarios to be presented at the TLM Think Tank meeting in Malawi - June 2023

Outputs:
*

Inputs:
* ResourceFile_Consumables_availability_small.csv` - This file contains the original consumable availability estimates
from OpenLMIS 2018 data

"""

import calendar
import datetime
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from plotnine import * # ggplot, aes, geom_point for ggplots from R
import seaborn as sns
import numpy as np
import pandas as pd

from tlo.methods.consumables import check_format_of_consumables_file

# Set local Dropbox source
path_to_dropbox = Path(  # <-- point to the TLO dropbox locally
    'C:/Users/sm2511/Dropbox/Thanzi la Onse'
)

# define a timestamp for script outputs
timestamp = datetime.datetime.now().strftime("_%Y_%m_%d_%H_%M")

# print the start time of the script
print('Script Start', datetime.datetime.now().strftime('%H:%M'))

# define a pathway to the data folder (note: currently outside the TLO model directory)
# remember to set working directory to TLOmodel/
outputfilepath = Path("./outputs")
resourcefilepath = Path("./resources")
path_for_new_resourcefiles = resourcefilepath / "healthsystem/consumables"
path_for_figures = outputfilepath / "think_tank_figures_june2023"

# 1. Import and clean data files
#**********************************
# 1.1 Import TLO model availability data
#------------------------------------------------------
tlo_availability_df = pd.read_csv(path_for_new_resourcefiles / "ResourceFile_Consumables_availability_small.csv")

# 1.1.1 Attach district, facility level, program to this dataset
#----------------------------------------------------------------
# Get TLO Facility_ID for each district and facility level
mfl = pd.read_csv(resourcefilepath / "healthsystem" / "organisation" / "ResourceFile_Master_Facilities_List.csv")
districts = set(pd.read_csv(resourcefilepath / 'demography' / 'ResourceFile_Population_2010.csv')['District'])
fac_levels = {'0', '1a', '1b', '2', '3', '4'}
tlo_availability_df = tlo_availability_df.merge(mfl[['District', 'Facility_Level', 'Facility_ID']],
                    on = ['Facility_ID'], how='left')

# 1.1.2 Attach programs
programs = pd.read_csv(path_for_new_resourcefiles / "ResourceFile_Consumables_availability_and_usage.csv")[['category', 'item_code', 'module_name']]
programs = programs.drop_duplicates('item_code')
tlo_availability_df = tlo_availability_df.merge(programs, on = ['item_code'], how = 'left')

# 1.1.3 Attach consumable names
cons_names = pd.read_csv(
    resourcefilepath / 'healthsystem' / 'consumables' / 'ResourceFile_Consumables_Items_and_Packages.csv'
)[['Item_Code', 'Items']].set_index('Item_Code').drop_duplicates()
cons_names['item_code'] = cons_names.index
tlo_availability_df = tlo_availability_df.merge(cons_names, on='item_code', how='left')

# 1.2 Get data ready for plotting
#------------------------------------------------------
tlo_availability_df = tlo_availability_df.rename({'available_prop': 'available_prop_actual'}, axis=1)

list_of_scenario_variables = ['available_prop_actual',
        'available_prop_scenario_fac_type', 'available_prop_scenario_fac_owner',
       'available_prop_scenario_functional_computer',
       'available_prop_scenario_incharge_drug_orders',
       'available_prop_scenario_functional_emergency_vehicle',
       'available_prop_scenario_service_diagnostic',
       'available_prop_scenario_dist_todh',
       'available_prop_scenario_dist_torms',
       'available_prop_scenario_drug_order_fulfilment_freq_last_3mts',
       'available_prop_scenario_all_features']

prefix = 'available_prop_'
list_of_scenarios = [s.replace(prefix, '') for s in list_of_scenario_variables]
#final_list_of_scenario_vars = ['available_prop_scenario_' + item for item in list_of_scenario_suffixes]

summary_availability_df = tlo_availability_df.groupby(['Facility_Level','category'])[list_of_scenario_variables].mean().reset_index()
summary_availability_df = summary_availability_df[summary_availability_df.Facility_Level.isin(['1a', '1b'])]

# 2. Generate plots for presentation
#************************************
# 2.1 Descriptives
#-------------------
# 2.1.1. Heatmap showing average availability by program and facility level
#---------------------------------------------------------------------------
fig = plt.figure(figsize = (10,10))

heatmap_df = tlo_availability_df.pivot_table(values='available_prop_actual', index='category', columns='Facility_Level', margins = True, aggfunc='mean')
heatmap_df = heatmap_df.sort_values('All')
# Move the "All" row to the bottom
heatmap_df = pd.concat([heatmap_df[heatmap_df.index != 'All'], heatmap_df[heatmap_df.index == 'All']])
# Generate heatmap
sns.heatmap(data=heatmap_df, square=False, annot=True , fmt = '.2f', annot_kws={'fontsize':15}, cmap="RdYlGn")
sns.set(font_scale=1.2)
# Set the x-axis and y-axis labels
plt.xlabel('Facility level')
plt.ylabel('Disease program/Module')
plt.xticks(rotation=45, fontsize = 15)
plt.yticks(fontsize = 15)
# Insert solid black lines to separate the aggregate values from the rest of the heatmap
ax = plt.gca()# Get the current axes
num_rows, num_cols = heatmap_df.shape # Get the number of rows and columns
# Draw a bold line separating the final column and final row
ax.hlines(num_rows - 1, -0.5, num_cols, linewidth=3, colors='black')
ax.vlines(num_cols - 1, -0.5, num_rows, linewidth=3, colors='black')
# Save figure
plt.savefig(path_for_figures / 'descriptive_heatmap.png', bbox_inches='tight', dpi=600)

# Heatmap showing detailed availability by consumable and facility ID
#---------------------------------------------------------------------------
fig = plt.figure(figsize = (15,15))

heatmap_df = tlo_availability_df.pivot_table(values='available_prop_actual', index='Items', columns='Facility_ID', margins = True, aggfunc='mean')
heatmap_df = heatmap_df.sort_values('All')
# Move the "All" row to the bottom
heatmap_df = pd.concat([heatmap_df[heatmap_df.index != 'All'], heatmap_df[heatmap_df.index == 'All']])
# Generate heatmap
sns.heatmap(data=heatmap_df, square=False, annot=False,  cmap="RdYlGn")
plt.xticks(rotation=45)
# Insert solid black lines to separate the aggregate values from the rest of the heatmap
ax = plt.gca()# Get the current axes
num_rows, num_cols = heatmap_df.shape # Get the number of rows and columns
# Draw a bold line separating the final column and final row
ax.hlines(num_rows - 1, -0.5, num_cols, linewidth=3, colors='black')
ax.vlines(num_cols - 1, -0.5, num_rows, linewidth=3, colors='black')
plt.savefig(path_for_figures / 'descriptive_heatmap_detailed.png', bbox_inches='tight', dpi=600)

# Generate figures demonstrating the change in consumable availability by program, facility level and district (% change)
# Option 1: Heatmap representing change from 2018 consumable availability estimates

# 2.2 Bar chart showing average consumable availability by program and facility level
#-----------------------------------------------------------------------------------------
# Group the data by 'Facility_level'
grouped_data = summary_availability_df.groupby('Facility_Level')

# Create a figure and axes outside the loop to hold the legend
fig_legend, ax_legend = plt.subplots()

# Iterate over the groups and create separate plots
for group_name, group_data in grouped_data:
    # Create a new plot for each group
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the data for the current group
    group_data.plot(x="category", y=list_of_scenario_variables, kind="bar", ax=ax)

    # Set the title and labels
    ax.set_title(f"Scenarios for Facility_level: {group_name}")
    ax.set_xlabel("Program/Module")
    ax.set_ylabel("Average probability of consumable availability")

    # Rotate x-axis labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=7)

    # Extract the legend separately
    handles, labels = ax.get_legend_handles_labels()
    ax.get_legend().remove() # remove legend

    # Show or save the plot
    fig_name = 'probability_by_scenario_fac_level' + group_name + '.png'
    plt.tight_layout()
    plt.savefig(path_for_figures /fig_name, dpi=600)

legend_mapping = {'available_prop_actual': 'Actual (2018)',
                  'available_prop_scenario_fac_type': "Scenario: Facility level 1b",
                  'available_prop_scenario_fac_owner': "Scenario: Facility owner CHAM",
                  'available_prop_scenario_functional_computer': "Scenario: Functional computer available",
                  'available_prop_scenario_incharge_drug_orders': "Scenario: Pharmacist in-charge of drug orders",
                  'available_prop_scenario_functional_emergency_vehicle': "Scenario: Emergency vehicle available",
                  'available_prop_scenario_service_diagnostic': "Scenario: Diagnostic services available",
                  'available_prop_scenario_dist_todh': "Scenario: Distance to DHO - 0-10 kms",
                  'available_prop_scenario_dist_torms': "Scenario: Distance to RMS - 0-10 kms" ,
                  'available_prop_scenario_drug_order_fulfilment_freq_last_3mts': "Scenario: Monthly drug order fulfillment",
                  'available_prop_scenario_all_features': "Scenario: All features"}

# Replace old labels with new labels based on the dictionary mapping
new_labels = [legend_mapping.get(label, label) for label in labels]

# Extract legend separately
figl, axl = plt.subplots()
axl.axis(False)
axl.legend(handles, new_labels, loc="center", bbox_to_anchor=(0.5, 0.5), prop={"size":15})
#axl.legend(*label_params, loc="center", bbox_to_anchor=(0.5, 0.5), prop={"size":10})
figl.savefig(path_for_figures / "probability_by_scenario_legend.png")
