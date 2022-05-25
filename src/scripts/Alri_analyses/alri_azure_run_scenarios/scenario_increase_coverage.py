"""Run simulations to demonstrate the impact of HIV interventions in combination.
This can be run remotely on Azure.
It creates the file:
"""

import pickle
from pathlib import Path

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
results_filename = outputpath / 'combination_intervention_results.pickle'


# %% Define the simulation run:
def run_sim(scenario):

    # The resource files
    resourcefilepath = Path("./resources")

    start_date = Date(2010, 1, 1)
    end_date = Date(2020, 1, 1)
    popsize = 500

    # Establish the simulation object
    log_config = {
        'filename': 'Logfile',
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
    sim.simulate(end_date=end_date)

    # Update the parameters are given in the scenario dict
    sim.modules['HealthSystem'].override_availability_of_consumables(scenario)

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

# %% Save the results
with open(results_filename, 'increase_coverage') as f:
    pickle.dump({
        'ScenarioSet': ScenarioSet,
        'outputs': outputs},
        f, pickle.HIGHEST_PROTOCOL
    )

# # %% Load the results
# with open(results_filename, 'rb') as f:
#     # The protocol version used is detected automatically, so we do not
#     # have to specify it.
#     X = pickle.load(f)
#
# ScenarioSet = X['ScenarioSet']
# outputs = X['outputs']
