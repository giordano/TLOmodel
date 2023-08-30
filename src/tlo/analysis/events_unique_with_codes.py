"""
Takes the CSV with each module-event-treatment-facility created by hsi_events.py with 'csv2' formatting. Saves new CSV
with unique treatments, along with the lists of hsi events and facility levels with which these treatments can be
performed.
"""

import pandas as pd
from pathlib import Path

# Get the path of the current script file
script_path = Path(__file__)

# Specify the file path to CSV file with all module-event-treatment-facility
file_path = script_path.parent.parent.parent.parent / 'docs'

# Load the CSV file into a DataFrame
df = pd.read_csv(Path(file_path) / 'module-event-treatment-facility.csv')
# NB. To create this CSV run:
# python src/tlo/analysis/hsi_events.py --output-file docs/module-event-treatment-facility.csv --output-format csv2

# OLD CODE ASSUMING HSI_EVENTS INCLUDE FACILITY LEVELS WHEN HSI EVENTS CAN OCCUR - NOT TRUE, INSTEAD WILL BE ADDED ['*']
# TREATMENT EXCLUDED FOR NOW AS NOT NEEDED - CAN BE RETURNED IF WE WANT TO
# ---------------------------------
# MEANING ALL LEVELS, AND COULD BE CHANGED TO A LIST OF LEVELS IF EQUIPMENT SET DIFFERS BETWEEN SOME OF THEM
# unique_hsi = \
#     df.copy().groupby('Event').agg({'Module': 'unique', 'Treatment': 'unique', 'Facility level': 'unique'}) \
#     .reset_index()
#
# assert unique_hsi['Module'].apply(lambda x: len(x) == 1).all()
# assert unique_hsi['Treatment'].apply(lambda x: len(x) == 1).all()
# assert unique_hsi['Facility level'].apply(lambda x: len(x) == 1).all()
#
# unique_hsi = \
#     df.copy().groupby('Event').agg({'Module': 'first', 'Treatment': 'first', 'Facility level': 'first'}) \
#     .reset_index()
#
# unique_hsi['Event_code'] = range(len(unique_hsi))
# unique_hsi = unique_hsi.loc[:, ['Module', 'Treatment', 'Facility level', 'Event', 'Event_code']]
# ---------------------------------

unique_hsi = \
    df.loc[:, ['Module', 'Event']].groupby('Event').agg({'Module': 'unique'}).reset_index()
assert unique_hsi['Module'].apply(lambda x: len(x) == 1).all()

unique_hsi = \
    df.loc[:, ['Module', 'Event']].groupby('Event').agg({'Module': 'first'}).reset_index()
unique_hsi['Facility levels'] = [['*']] * len(unique_hsi)
unique_hsi['Event_code'] = range(len(unique_hsi))

unique_hsi = unique_hsi.loc[:, ['Module', 'Facility levels', 'Event', 'Event_code']]

print("unique_hsi['Facility levels'][0]")
print(type(unique_hsi['Facility levels'][0]))
print(unique_hsi['Facility levels'][0])

# Save CSV with unique events
unique_hsi.to_csv(Path(file_path) / 'module-facilities-unique_event.csv', index=False)
