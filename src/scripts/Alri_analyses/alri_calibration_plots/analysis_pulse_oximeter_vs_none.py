"""
This will demonstrate the effect of different treatment.
"""

# %% Import Statements and initial declarations
import datetime
from pathlib import Path
from tlo.analysis.utils import parse_log_file

import pandas as pd
from matplotlib import pyplot as plt

from tlo import Date, Simulation, logging
from tlo.methods import (
    alri,
    demography,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    simplified_births,
    symptommanager,
)

# %%
resourcefilepath = Path("./resources")
outputpath = Path("./outputs")  # folder for convenience of storing outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# Scenarios Definitions:
# Define the 'service_availability' parameter for two sceanrios (without treatment, with treatment)
scenarios = dict()
scenarios['No_oximeter_and_oxygen'] = False
scenarios['With_oximeter_and_oxygen'] = True

# Create dict to capture the outputs
output_files = dict()

# %% Run the Simulation
start_date = Date(2010, 1, 1)
end_date = Date(2019, 12, 31)
popsize = 50000

for label, oximeter_avail in scenarios.items():

    log_config = {
        "filename": f"alri_{label}",
        "directory": "./outputs",
        "custom_levels": {
            "*": logging.WARNING,
            "tlo.methods.alri": logging.INFO,
            "tlo.methods.demography": logging.INFO,
        }
    }

    # add file handler for the purpose of logging
    sim = Simulation(start_date=start_date, log_config=log_config, show_progress_bar=True)

    sim.register(
        demography.Demography(resourcefilepath=resourcefilepath),
        enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
        simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
        symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
        healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
        healthburden.HealthBurden(resourcefilepath=resourcefilepath),
        healthsystem.HealthSystem(resourcefilepath=resourcefilepath, cons_availability='all'),
        alri.Alri(resourcefilepath=resourcefilepath),
        alri.AlriPropertiesOfOtherModules()
    )

    sim.make_initial_population(n=popsize)

    if oximeter_avail:
        sim.modules['HealthSystem'].override_availability_of_consumables({127: 1.0})
    else:
        sim.modules['HealthSystem'].override_availability_of_consumables({127: 0.0})

    # Assume perfect sensitivity in hw classification
    p = sim.modules['Alri'].parameters
    p['sensitivity_of_classification_of_fast_breathing_pneumonia_facility_level0'] = 1.0
    p['sensitivity_of_classification_of_danger_signs_pneumonia_facility_level0'] = 1.0
    p['sensitivity_of_classification_of_non_severe_pneumonia_facility_level1'] = 1.0
    p['sensitivity_of_classification_of_severe_pneumonia_facility_level1'] = 1.0
    p['sensitivity_of_classification_of_non_severe_pneumonia_facility_level2'] = 1.0
    p['sensitivity_of_classification_of_severe_pneumonia_facility_level2'] = 1.0

    sim.simulate(end_date=end_date)

    # Save the full set of results:
    output_files[label] = sim.log_filepath


# %% Extract the relevant outputs and make a graph:
def get_death_numbers_from_logfile(logfile):
    # parse the simulation logfile to get the output dataframes
    output = parse_log_file(logfile)

    # calculate death rate
    deaths_df = output['tlo.methods.demography']['death']
    deaths_df['year'] = pd.to_datetime(deaths_df['date']).dt.year
    deaths = deaths_df.loc[deaths_df['cause'].str.startswith('ALRI')].groupby('year').size()

    return deaths


deaths = dict()
for label, file in output_files.items():
    deaths[label] = \
        get_death_numbers_from_logfile(file)

# Plot death rates by year: across the scenarios
data = {}
for label in deaths.keys():
    data.update({label: deaths[label]})
pd.concat(data, axis=1).plot.bar()
plt.title('Number of Deaths Due to ALRI')
plt.savefig(outputpath / ("ALRI_deaths_by_scenario2" + datestamp + ".pdf"), format='pdf')
plt.show()
