"""Run simulations to demonstrate the impact of HIV interventions in combination.
This can be run remotely on Azure.
It creates the file:
"""

import pickle
from pathlib import Path
import pandas as pd
import os
from matplotlib import pyplot as plt

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    demography,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    simplified_births,
    symptommanager,
    alri
)

# Where will outputs go
outputpath = Path("./outputs")  # folder for convenience of storing outputs
results_filename = outputpath / 'combination_intervention_results'

log_filename1 = outputpath / 'coverage_PO_and_oxygen__2022-05-25T140506.log'
log_filename2 = outputpath / 'coverage_PO_and_oxygen__2022-05-25T140631.log'
log_filename3 = outputpath / 'coverage_PO_and_oxygen__2022-05-25T144019.log'
log_filename4 = outputpath / 'coverage_PO_and_oxygen__2022-05-25T150925.log'

logfile_list = [log_filename1, log_filename2, log_filename3, log_filename4]

# get the outputs
output1 = parse_log_file(log_filename1)
output2 = parse_log_file(log_filename2)
output3 = parse_log_file(log_filename3)
output4 = parse_log_file(log_filename4)

outputs_list = [output1, output2, output3, output4]

for log_filename in logfile_list:
    if not os.path.exists(log_filename):
    # If logfile does not exists, re-run the simulation:
    # Do not run this cell if you already have a logfile from a simulation:

        # %% Define the simulation run:
        def run_sim(scenario):

            # The resource files
            resourcefilepath = Path("./resources")

            start_date = Date(2010, 1, 1)
            end_date = Date(2020, 1, 1)
            popsize = 500

            # Establish the simulation object
            log_config = {
                'filename': 'Logfile_PO_and_oxygen',
                'directory': outputpath,
                'custom_levels': {
                    '*': logging.WARNING,
                    'tlo.methods.alri': logging.INFO,
                    "tlo.methods.demography": logging.INFO,
                }
            }

            # Register the appropriate modules
            sim = Simulation(start_date=start_date, seed=0, log_config=log_config)
            sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                         simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
                         enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                         healthsystem.HealthSystem(resourcefilepath=resourcefilepath),
                         symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                         healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                         healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                         alri.Alri(resourcefilepath=resourcefilepath),
                         alri.AlriPropertiesOfOtherModules()
                         )

            # Run the simulation and flush the logger
            sim.make_initial_population(n=popsize)

            # Update the parameters are given in the scenario dict
            sim.modules['HealthSystem'].override_availability_of_consumables(scenario)

            # start simulation
            sim.simulate(end_date=end_date)

            # Return the parsed_log-file
            return parse_log_file(sim.log_filepath)


        # %% Define the scenarios:
        ScenarioSet = {
            "no_coverage": {
                127: 0.0
            },

            "50%_coverage": {
                127: 0.5
            },

            "90%_coverage": {
                127: 0.9
            },

            "full_coverage": {
                127: 1.0
            },

        }

        # %% Run the scenarios:
        outputs = dict()
        for scenario in ScenarioSet:
            outputs[scenario] = run_sim(ScenarioSet[scenario])

for output in outputs_list:

    # Calculate the "incidence rate" and "mortality rate" from the output of event counts
    counts = output['tlo.methods.alri']['event_counts']
    counts['year'] = pd.to_datetime(counts['date']).dt.year
    counts.drop(columns='date', inplace=True)
    counts.set_index(
        'year',
        drop=True,
        inplace=True
    )

    # get person-years of < 5 year-olds
    py_ = output['tlo.methods.demography']['person_years']
    years = pd.to_datetime(py_['date']).dt.year
    py = pd.DataFrame(index=years, columns=['<5y'])
    for year in years:
        tot_py = (
            (py_.loc[pd.to_datetime(py_['date']).dt.year == year]['M']).apply(pd.Series) +
            (py_.loc[pd.to_datetime(py_['date']).dt.year == year]['F']).apply(pd.Series)
        ).transpose()
        tot_py.index = tot_py.index.astype(int)
        py.loc[year, '<5y'] = tot_py.loc[0:4].sum().values[0]

    # Mortality rate outputted from the ALRI model - using the tracker to get the number of deaths per year
    mort_rate = (counts.deaths.div(py['<5y'], axis=0).dropna()) * 100000

    start_date = 2010
    end_date = 2026

    # model output
    plt.plot(counts.index, mort_rate, color="mediumseagreen")  # model
    plt.title("ALRI Mortality per 100,000 children")
    plt.xlabel("Year")
    plt.xticks(rotation=90)
    plt.ylabel("Mortality (/100k)")
    plt.gca().set_xlim(start_date, end_date)
    plt.legend(["Model"])
    plt.tight_layout()
    # plt.savefig(outputpath / ("ALRI_Mortality_model_comparison" + datestamp + ".png"), format='png')

    plt.show()

# # %% Save the results
# with open(results_filename, 'increase_coverage') as f:
#     pickle.dump({
#         'ScenarioSet': ScenarioSet,
#         'outputs': outputs},
#         f, pickle.HIGHEST_PROTOCOL
#     )

# # %% Load the results
# with open(results_filename, 'rb') as f:
#     # The protocol version used is detected automatically, so we do not
#     # have to specify it.
#     X = pickle.load(f)
#
# ScenarioSet = X['ScenarioSet']
# outputs = X['outputs']
