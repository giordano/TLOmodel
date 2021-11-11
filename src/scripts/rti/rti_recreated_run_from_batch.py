from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file, get_failed_batch_run_information, get_scenario_outputs
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

# outputspath = Path('./outputs/rmjlra2@ucl.ac.uk/')

resourcefilepath = Path('./resources')

# results_folder = get_scenario_outputs('rti_incidence_parameterisation.py', outputspath)[-1]

# seed, params, popsize, start_date = get_failed_batch_run_information(results_folder,
#                                                                      'rti_incidence_parameterisation.py', 2, 0)
start_date = Date(year=2010, month=1, day=1)
end_date = Date(year=(2010 + 5), month=1, day=1)
sim = Simulation(start_date=start_date)
# Register the modules
sim.register(
    demography.Demography(resourcefilepath=resourcefilepath),
    enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
    healthsystem.HealthSystem(resourcefilepath=resourcefilepath, disable=True, service_availability=['*']),
    healthburden.HealthBurden(resourcefilepath=resourcefilepath),
    symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
    healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
    rti.RTI(resourcefilepath=resourcefilepath),
    simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath)
)
# Get the log file
logfile = sim.configure_logging(filename="LogFile")
sim.simulate(end_date=end_date)

