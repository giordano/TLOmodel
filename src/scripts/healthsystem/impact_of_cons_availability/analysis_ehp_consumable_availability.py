
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

# Merge Treatment IDs

ResourceFile_Consumables_Items_and_Packages

# Merge Treatment ID prioritisation

contraception = tlo_availability_df[tlo_availability_df.category == 'contraception']
contraception_summary = contraception.groupby(['Items'])['available_prop'].mean().reset_index()

hivtb = tlo_availability_df[tlo_availability_df.category.isin(['hiv', 'tb'])]
cond = hivtb.Items.isin(['Treatment: second-line drugs', 'First-line ART regimen: adult', 'First line ART regimen: older child'])
art = hivtb[cond]
hivtb.to_csv(outputfilepath / 'hivtb_consumables_availability.csv')
art_summary = hivtb[cond].groupby(['Items', 'Facility_Level'])['available_prop'].mean()

tlo_availability_df.groupby(['month'])['available_prop'].mean()

all_drugs_data = tlo_availability_df[['Facility_ID', 'month', 'item_code', 'available_prop',
       'available_prop_scenario_fac_type', 'available_prop_scenario_fac_owner', 'available_prop_scenario_all_features', 'District', 'Facility_Level',
       'category', 'module_name', 'Items']]
all_drugs_data.to_csv(outputfilepath / 'all_drugs_data.csv')
