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

# Verify that the same treatments are within the same Module
output_df_to_assert = \
    df.copy().groupby('Treatment').agg({'Module': 'unique', 'Event': 'unique', 'Facility level': 'unique'}) \
    .reset_index()

assert output_df_to_assert['Module'].apply(lambda x: len(x) == 1).all()

# Group by Treatment
output_df =\
    df.copy().groupby('Treatment').agg({'Module': 'first', 'Event': 'unique', 'Facility level': 'unique'}).reset_index()
# NB. 'Facility level' [nan] indicates the appropriate facility level is determined dynamically when running the model
# TODO: Could it be any level then, or which levels are included?

output_df['Treatment_code'] = range(len(output_df))
output_df = output_df.loc[:, ['Module', 'Event', 'Facility level', 'Treatment', 'Treatment_code']]

# Save CSV with unique treatments
output_df.to_csv(Path(file_path) / 'module-events-facilities-unique_treatment.csv', index=False)
