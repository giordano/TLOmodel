import ast
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    demography,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    rti,
    simplified_births,
    symptommanager,
)
resourcefilepath = Path('./resources')
yearsrun = 10

start_date = Date(year=2010, month=1, day=1)
end_date = Date(year=(2010 + yearsrun), month=1, day=1)
service_availability = ['*']
pop_size = 20000
# Create the simulation object
sim = Simulation(start_date=start_date, seed=1001268886)
# Register the modules
sim.register(
    demography.Demography(resourcefilepath=resourcefilepath),
    enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
    healthsystem.HealthSystem(resourcefilepath=resourcefilepath, service_availability=['*']),
    healthburden.HealthBurden(resourcefilepath=resourcefilepath),
    symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
    healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
    rti.RTI(resourcefilepath=resourcefilepath),
    simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
)
# Get the log file
logfile = sim.configure_logging(filename="Recreated_run")
# create and run the simulation
sim.make_initial_population(n=pop_size)
sim.modules['RTI'].parameters['prob_bleeding_leads_to_shock'] = 0.042222222222222223
sim.simulate(end_date=end_date)
