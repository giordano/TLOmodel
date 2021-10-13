import ast
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    demography,
    dx_algorithm_adult,
    dx_algorithm_child,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    rti,
    simplified_births,
    symptommanager,
)

# =============================== Analysis description ========================================================
# This analysis file has essentially become the model fitting analysis, seeing what happens when we run the model
# and whether an ordinary model run will behave how we would expect it to, hitting the right demographics, producing
# the right injuries, measuring the percent of crashes involving alcohol

# ============================================== Model run ============================================================
log_config = {
    "filename": "rti_health_system_n_inj",  # The name of the output file (a timestamp will be appended).
    "directory": "./outputs/number_of_injuries",  # The default output path is `./outputs`. Change it here, if necessary
    "custom_levels": {  # Customise the output of specific loggers. They are applied in order:
        "*": logging.WARNING,  # Asterisk matches all loggers - we set the default level to WARNING
        "tlo.methods.rti": logging.INFO,
        "tlo.methods.healthsystem": logging.DEBUG,
        "tlo.methods.labour": logging.disable(logging.DEBUG)
    }
}
# The Resource files [NB. Working directory must be set to the root of TLO: TLOmodel]
resourcefilepath = Path('./resources')
# Establish the simulation object
yearsrun = 5
start_date = Date(year=2010, month=1, day=1)
end_date = Date(year=(2010 + yearsrun), month=1, day=1)
service_availability = ['*']
pop_size = 10000
nsim = 2
mean_n_inj = []

# Iterate over the number of simulations nsim
for i in range(0, nsim):
    # Create the simulation object
    sim = Simulation(start_date=start_date)
    # Register the modules
    sim.register(
        demography.Demography(resourcefilepath=resourcefilepath),
        enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
        healthsystem.HealthSystem(resourcefilepath=resourcefilepath, service_availability=['*']),
        symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
        dx_algorithm_adult.DxAlgorithmAdult(resourcefilepath=resourcefilepath),
        dx_algorithm_child.DxAlgorithmChild(resourcefilepath=resourcefilepath),
        healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
        healthburden.HealthBurden(resourcefilepath=resourcefilepath),
        rti.RTI(resourcefilepath=resourcefilepath),
        # simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath)
        )
    # Get the log file
    logfile = sim.configure_logging(filename="LogFile", directory='./outputs/number_of_injuries')
    # create and run the simulation
    sim.make_initial_population(n=pop_size)
    # Run the simulation
    sim.simulate(end_date=end_date)
    # Parse the logfile of this simulation
    log_df = parse_log_file(logfile)
    mean_n_inj.append(log_df['tlo.methods.rti']['number_of_injuries_in_hospital']['number_of_injuries'].mean())
print(np.mean(mean_n_inj))
