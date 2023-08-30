"""
(1) Can be used at the beginning when creating the lists of equipment used with hsi events.

Takes the RF 'file_to_update' prepared from copy of docs/module-facilities-unique_event.csv (created by
src/tlo/analysis/events_unique_with_codes.py for the first assignment) which is adjusted by hand:
- some equipment added (column 'Equipment')
- if multiple equipment assumed to be used for same Event, the row copied below and each Equipment included in a
separate row
- column 'Equipment_Code' created, but no codes need to be filled in yet
- columns 'Equipment_Cost', 'Equipment_ExpectedLifespan', 'Source_EquipCost', 'Source_ExpectedLifespan'
# TODO: might be better to have cost & lifespan in separate csv file to ensure they are unique for the same equipment

This script will assign unique code to each unique equipment name which has no code assigned yet. The codes are
assigned in order from the sequence 0, 1, 2, ....

(2) Can be used when new equipment names with missing codes are added later.

This script will fill in the existing codes for equipment with already assigned code and for equipment without existing
code will assign new code (continue in sequence, i.e. if the highest code is 5, it assigns new codes from the continuing
sequence 6, 7, 8, ...).

------
NB. Make sure the file_to_update is the file you want to update. The output will be named
'ResourceFile_Equipment_new.csv' to avoid unintentionally losing the previous version.
------
"""

import pandas as pd
from pathlib import Path

file_to_update = 'ResourceFile_Equipment_withoutEquipmentCodes.csv'

# Get the path of the current script file
script_path = Path(__file__)

# Specify the file path to RF csv file
file_path = script_path.parent.parent.parent.parent / 'resources/healthsystem/infrastructure_and_equipment'

# Load the CSV RF into a DataFrame ### CHANGE THIS IF YOU WANT TO USE DIFFERENT FILE AS INPUT ###
df = pd.read_csv(Path(file_path) / file_to_update)

# Find unique values in Equipment that have no code and are not None or empty
unique_values =\
    df.loc[df['Equipment_Code'].isna() & df['Equipment'].notna() & (df['Equipment'] != ''), 'Equipment'].unique()

# Create a mapping of unique values to codes
value_to_code = {}
# Initialize the starting code value
if not df['Equipment_Code'].isna().all():
    next_code = int(df['Equipment_Code'].max()) + 1
else:
    next_code = 0

# Iterate through unique values
for value in unique_values:
    # Check if there is at least one existing code for this value
    matching_rows = df.loc[df['Equipment'] == value, 'Equipment_Code'].dropna()
    if not matching_rows.empty:
        # Use the existing code for this value
        existing_code = int(matching_rows.iloc[0])
    else:
        # If no existing codes, start with the next available code
        existing_code = next_code
        next_code += 1
    value_to_code[value] = existing_code
    # Update the 'Equipment_Code' column for matching rows
    df.loc[df['Equipment'] == value, 'Equipment_Code'] = existing_code

# Convert 'Equipment_Code' column to integers
df['Equipment_Code'] = df['Equipment_Code'].astype('Int64')  # Convert to nullable integer typ

# Iterate through unique values
for value in unique_values:
    # Check if there is at least one existing code for this value
    matching_rows = df.loc[df['Equipment'] == value, 'Equipment_Code'].dropna()
    if not matching_rows.empty:
        # Use the existing code for this value
        existing_code = int(matching_rows.iloc[0])
    else:
        # If no existing codes, start with the next available code
        existing_code = next_code
        next_code += 1
    value_to_code[value] = existing_code
    # Update the 'code' column for matching rows and convert to integer
    df.loc[df['Equipment'] == value, 'Equipment_Code'] = existing_code

# Save CSV with equipment codes
df.to_csv(Path(file_path) / 'ResourceFile_Equipment_new.csv', index=False)
