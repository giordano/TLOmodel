"""
This script generates the consumables availability dataset for regression analysis using the outputs of -
consumables_availability_estimation.py and clean_fac_locations.py

The scripts merges GIS data (facility distance from DHO) to the availability data.

Requirements:
- To run gmaps, an google API key is required.
"""

import calendar
import datetime
# Import Statements and initial declarations
from pathlib import Path

import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
#from matplotlib import pyplot # for figures
import seaborn as sns
import math

# Import modules for distance analysis
import googlemaps as gmaps # for the google maps distance matrix
from itertools import tee # for the google maps distance matrix
import requests, json # Perform request to use the Google Maps API web service
# API_key = # Placeholder to enter Google Maps API key
gmaps = gmaps.Client(key=API_key)

# Path to TLO directory
outputfilepath = Path("./outputs")
resourcefilepath = Path("./resources")
path_for_new_resourcefiles = resourcefilepath / "healthsystem/consumables"

# Set local Dropbox source
path_to_dropbox = Path(  # <-- point to the TLO dropbox locally
    'C:/Users/sm2511/Dropbox/Thanzi la Onse') # '/Users/tbh03/Dropbox (SPH Imperial College)/Thanzi la Onse Theme 1 SHARE'
path_to_files_in_the_tlo_dropbox = path_to_dropbox / "05 - Resources/Module-healthsystem/consumables raw files/"

# define a timestamp for script outputs
timestamp = datetime.datetime.now().strftime("_%Y_%m_%d_%H_%M")

# print the start time of the script
print('Script Start', datetime.datetime.now().strftime('%H:%M'))

# 1. DATA IMPORT AND CLEANING #
#########################################################################################
# --- 1.1 Import consumables availability data --- #
stkout_df = pd.read_csv(path_for_new_resourcefiles / "ResourceFile_Consumables_availability_and_usage.csv", low_memory=False)

# Drop rows which can't be used in regression analysis
regsubset_cond1 = stkout_df['data_source'] == 'original_lmis_data'
regsubset_cond2 = stkout_df['fac_type_tlo'] == 'Facility_level_0' # since only one facility from Mchinji reported in OpenLMIS
stkout_df_reg = stkout_df[regsubset_cond1 & ~regsubset_cond2]

# Clean some district names to match with master health facility registry
rename_districts = {
    'Nkhota Kota': 'Nkhotakota',
    'Nkhata bay': 'Nkhata Bay'
}
stkout_df['district'] = stkout_df['district'].replace(rename_districts)

# --- 1.2 Load and clean GIS data ---
fac_gis = pd.read_csv(path_for_new_resourcefiles / "ResourceFile_Facility_locations.csv")
# Keep rows with GIS locations
fac_gis = fac_gis[fac_gis['lat'].notna()]

# - 1.2.1 Assign relevant District Hospital to each facility - #
# Manual fixes before assigning DHO
# Master Health facility registry did not differentiate between Mzimba North and Mzimba South --> get this data
# and any other district discrepancies from LMIS
lmis_district = stkout_df[['fac_name', 'fac_type_tlo','district']]
lmis_district = lmis_district.drop_duplicates()
fac_gis = fac_gis.rename(columns = {'district':'district_mhfr'})
fac_gis = pd.merge(fac_gis, lmis_district, how = 'left', on = 'fac_name')

list_mhfr_district_is_correct = ['Chididi Health Centre', 'Chikowa Health Centre',
                        'Chileka Health Centre']
cond_mhfr_district_is_correct = fac_gis.fac_name.isin(list_mhfr_district_is_correct)
cond_lmis_district_missing = fac_gis.district.isna()
fac_gis.loc[cond_mhfr_district_is_correct|cond_lmis_district_missing, 'district'] = fac_gis.district_mhfr
fac_gis = fac_gis.drop(columns = ['Unnamed: 0', 'zone', 'district_mhfr', 'open_date', 'manual_entry'])

#Locate the corresponding DHO for each facility
cond1 = fac_gis['fac_name'].str.contains('DHO')
cond2 = fac_gis['fac_name'].str.contains('istrict')
# Create columns indicating the coordinates of the corresponding DHO for each facility
dho_df = fac_gis[cond1|cond2].reset_index()
# Rename columns
dho_df = dho_df.rename(columns = {'lat':'lat_dh', 'long':'long_dh'})

# Merge main GIS dataframe with corresponding DHO
fac_gis = pd.merge(fac_gis,dho_df[['district', 'lat_dh', 'long_dh']], how = 'left', on = 'district')

# - 1.2.2 Assign relevant CMST Regional Medical Store to each facility - #
# Create columns indicating the coordinates of the corresponding CMST warehouse (regional medical store) for each facility
fac_gis['lat_rms'] = np.nan
fac_gis['long_rms'] = np.nan
# RMS Center (-13.980394, 33.783521)
cond_center1 = fac_gis['district'].isin(['Kasungu', 'Ntchisi', 'Dowa', 'Mchinji', 'Lilongwe', 'Ntcheu',
                                        'Dedza', 'Nkhotakota', 'Salima'])
cond_center2 = fac_gis['fac_name'].str.contains('Kamuzu Central Hospital')
fac_gis.loc[cond_center1|cond_center2, 'lat_rms'] = -13.980394
fac_gis.loc[cond_center1|cond_center2, 'long_rms'] = 33.783521

# RMS North (-11.425590, 33.997467)
cond_north1 = fac_gis['district'].isin(['Nkhata Bay', 'Rumphi', 'Chitipa', 'Likoma', 'Karonga',
                                        'Mzimba North', 'Mzimba South'])
cond_north2 = fac_gis['fac_name'].str.contains('Mzuzu Central Hospital')
fac_gis.loc[cond_north1|cond_north2, 'lat_rms'] = -11.425590
fac_gis.loc[cond_north1|cond_north2, 'long_rms'] = 33.997467

# RMS South (-15.804544, 35.021192)
cond_south1 = fac_gis['district'].isin(['Blantyre', 'Balaka', 'Machinga', 'Zomba', 'Mangochi', 'Thyolo', 'Nsanje',
                                        'Chikwawa', 'Mwanza', 'Neno', 'Mulanje', 'Phalombe', 'Chiradzulu'])
cond_south2 = fac_gis['fac_name'].str.contains('Queen Elizabeth Central')
cond_south3 = fac_gis['fac_name'].str.contains('Zomba Central')
cond_south4 = fac_gis['fac_name'].str.contains('Zomba Mental')
fac_gis.loc[cond_south1|cond_south2|cond_south3|cond_south4, 'lat_rms'] = -15.804544
fac_gis.loc[cond_south1|cond_south2|cond_south3|cond_south4, 'long_rms'] = 35.021192
fac_gis['district'].unique()

# - 1.2.3 Create columns representing distance and travel time of each facility from the corresponding DHO -#
fac_gis['dist_todh'] = np.nan
fac_gis['drivetime_todh'] = np.nan
for i in range(len(fac_gis)):
    try:
        # print("Processing facility", i)
        latfac = fac_gis['lat'][i]
        longfac = fac_gis['long'][i]
        latdho = fac_gis['lat_dh'][i]
        longdho = fac_gis['long_dh'][i]
        origin = (latdho, longdho)
        dest = (latfac, longfac)

        fac_gis['dist_todh'][i] = \
        gmaps.distance_matrix(origin, dest, mode='driving')['rows'][0]['elements'][0]['distance']['value']
        fac_gis['drivetime_todh'][i] = \
        gmaps.distance_matrix(origin, dest, mode='driving')['rows'][0]['elements'][0]['duration']['value']
    except:
        pass

# - 1.2.4 Create columns representing distance and travel time of each facility from the corresponding RMS -#
fac_gis['dist_torms'] = np.nan
fac_gis['drivetime_torms'] = np.nan
for i in range(len(fac_gis)):
    try:
        # print("Processing facility", i)
        latfac = fac_gis['lat'][i]
        longfac = fac_gis['long'][i]
        latdho = fac_gis['lat_rms'][i]
        longdho = fac_gis['long_rms'][i]
        origin = (latdho, longdho)
        dest = (latfac, longfac)

        fac_gis['dist_torms'][i] = \
        gmaps.distance_matrix(origin, dest, mode='driving')['rows'][0]['elements'][0]['distance']['value']
        fac_gis['drivetime_torms'][i] = \
        gmaps.distance_matrix(origin, dest, mode='driving')['rows'][0]['elements'][0]['duration']['value']
    except:
        pass

# Update distance values from DH to 0 for levels 2 and above
cond1 = fac_gis['fac_type_tlo'] == 'Facility_level_2'
cond2 = fac_gis['fac_type_tlo'] == 'Facility_level_3'
cond3 = fac_gis['fac_type_tlo'] == 'Facility_level_4'
fac_gis.loc[cond1|cond2|cond3, 'dist_todh'] = 0
fac_gis.loc[cond1|cond2|cond3, 'drivetime_todh'] = 0

# - 1.2.5 Export distances file to dropbox - #
fac_gis.to_csv(path_to_files_in_the_tlo_dropbox / 'gis_data/facility_distances.csv')

# - 1.2.6 Descriptive graphs -#
groups = fac_gis.groupby('district')

# Scatterplot of distance and drive time to DHO
fig, ax = plt.subplots()
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
for name, group in groups:
    ax.plot(group.dist_todh/1000, group.drivetime_todh, marker='o', linestyle='', ms=5, label=name)
# Shrink current axis by 20% to fit legend
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.xlabel("Distance (kilometers)", fontsize=12)
plt.ylabel("Drive time (minutes)", fontsize=12)
plt.savefig('C:/Users/sm2511/OneDrive - University of York/Desktop/faclocation_wrtdh.png')

# Distance and drive time from DHO
fac_gis['lndist_todh'] = np.log(fac_gis['dist_todh'] + 1)
fac_gis['lndist_todh'].hist(alpha = 0.7) #(by=fullreg_df['fac_type_tlo'])
fac_gis['lndrivetime_todh'] = np.log(fac_gis['drivetime_todh']+1) # add 1 before taking the log of the values
fac_gis['lndrivetime_todh'].hist(alpha = 0.7) #(by=fullreg_df['fac_type_tlo'])
plt.savefig('C:/Users/sm2511/OneDrive - University of York/Desktop/faclocation_wrtdho.png')

# Distance and drive time from RMS
fac_gis['lndist_torms'] = np.log(fac_gis['dist_torms'] + 1)
fac_gis['lndrivetime_torms'] = np.log(fac_gis['drivetime_torms']+1) # add 1 before taking the log of the values
fac_gis['dist_torms'].hist(alpha = 0.7, color = 'green') #(by=fullreg_df['fac_type_tlo'])
fac_gis['drivetime_torms'].hist(alpha = 0.7, color = 'blue') #(by=fullreg_df['fac_type_tlo'])
plt.savefig('C:/Users/sm2511/OneDrive - University of York/Desktop/faclocation_wrtrms.png')

# --- 1.3 Merge cleaned LMIS data with GIS data --- #
consumables_df = pd.merge(stkout_df.drop(columns = ['district']), fac_gis[['district', 'lat', 'long', 'lat_dh', 'long_dh', 'dist_torms','drivetime_torms',
                                                   'dist_todh','drivetime_todh', 'fac_name', 'gis_source']], how = 'left', on = 'fac_name')
consumables_df.to_csv(path_to_files_in_the_tlo_dropbox / 'consumables_df.csv')
