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

# =============================== Analysis description ========================================================
# This analysis file will eventually become what I use to produce the introduction to RTI paper. Here I run the model
# initally only allowing one injury per person, capturing the incidence of RTI and incidence of RTI death, calibrating
# this to the GBD estimates. I then run the model with multiple injuries and compare the outputs, the question being
# asked here is what happens to road traffic injury deaths if we allow multiple injuries to occur

# ============================================== Model run ============================================================
# The Resource files [NB. Working directory must be set to the root of TLO: TLOmodel]
resourcefilepath = Path('./resources')
save_file_path = "C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/BlockedInterventions/"
# Establish the simulation object
yearsrun = 10

start_date = Date(year=2010, month=1, day=1)
end_date = Date(year=(2010 + yearsrun), month=1, day=1)
service_availability = ['*']
pop_size = 20000
nsim = 2
# Iterate over the number of simulations nsim
log_file_location = './outputs/blocked_interventions/'

# for i in range(0, nsim):
#     # Create the simulation object
#     sim = Simulation(start_date=start_date)
#     # Register the modules
#     sim.register(
#         demography.Demography(resourcefilepath=resourcefilepath),
#         enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
#         healthsystem.HealthSystem(resourcefilepath=resourcefilepath, service_availability=['*']),
#         symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
#         healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
#         healthburden.HealthBurden(resourcefilepath=resourcefilepath),
#         rti.RTI(resourcefilepath=resourcefilepath),
#         simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
#     )
#     # Get the log file
#     logfile = sim.configure_logging(filename="LogFile_all",
#                                     directory=log_file_location + "all")
#     # create and run the simulation
#     sim.make_initial_population(n=pop_size)
#     # Run the simulation
#     sim.simulate(end_date=end_date)
#
# for i in range(0, nsim):
#     # Create the simulation object
#     sim = Simulation(start_date=start_date)
#     # Register the modules
#     sim.register(
#         demography.Demography(resourcefilepath=resourcefilepath),
#         enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
#         healthsystem.HealthSystem(resourcefilepath=resourcefilepath, service_availability=['*']),
#         symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
#         healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
#         healthburden.HealthBurden(resourcefilepath=resourcefilepath),
#         rti.RTI(resourcefilepath=resourcefilepath),
#         simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
#     )
#     # Get the log file
#     logfile = sim.configure_logging(filename="LogFile_minor",
#                                     directory=log_file_location + "minor")
#     # create and run the simulation
#     sim.make_initial_population(n=pop_size)
#     sim.modules['RTI'].parameters['blocked_interventions'] = ['Minor Surgery']
#     # Run the simulation
#     sim.simulate(end_date=end_date)
#
# for i in range(0, nsim):
#     # Create the simulation object
#     sim = Simulation(start_date=start_date)
#     # Register the modules
#     sim.register(
#         demography.Demography(resourcefilepath=resourcefilepath),
#         enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
#         healthsystem.HealthSystem(resourcefilepath=resourcefilepath, service_availability=['*']),
#         symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
#         healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
#         healthburden.HealthBurden(resourcefilepath=resourcefilepath),
#         rti.RTI(resourcefilepath=resourcefilepath),
#         simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
#     )
#     # Get the log file
#     logfile = sim.configure_logging(filename="LogFile_major",
#                                     directory=log_file_location + "major")
#     # create and run the simulation
#     sim.make_initial_population(n=pop_size)
#     sim.modules['RTI'].parameters['blocked_interventions'] = ['Major Surgery']
#
#     sim.simulate(end_date=end_date)

# for i in range(0, nsim):
#     # Create the simulation object
#     sim = Simulation(start_date=start_date)
#     # Register the modules
#     sim.register(
#         demography.Demography(resourcefilepath=resourcefilepath),
#         enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
#         healthsystem.HealthSystem(resourcefilepath=resourcefilepath, service_availability=['*']),
#         symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
#         healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
#         healthburden.HealthBurden(resourcefilepath=resourcefilepath),
#         rti.RTI(resourcefilepath=resourcefilepath),
#         simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
#     )
#     # Get the log file
#     logfile = sim.configure_logging(filename="LogFile_casts",
#                                     directory=log_file_location + "casts")
#     # create and run the simulation
#     sim.make_initial_population(n=pop_size)
#     sim.modules['RTI'].parameters['blocked_interventions'] = ['Fracture Casts']
#
#     sim.simulate(end_date=end_date)

# for i in range(0, nsim):
#     # Create the simulation object
#     sim = Simulation(start_date=start_date)
#     # Register the modules
#     sim.register(
#         demography.Demography(resourcefilepath=resourcefilepath),
#         enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
#         healthsystem.HealthSystem(resourcefilepath=resourcefilepath, service_availability=[]),
#         symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
#         healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
#         healthburden.HealthBurden(resourcefilepath=resourcefilepath),
#         rti.RTI(resourcefilepath=resourcefilepath),
#         simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
#     )
#     # Get the log file
#     logfile = sim.configure_logging(filename="LogFile_no_hs",
#                                     directory=log_file_location + "no_hs")
#     # create and run the simulation
#     sim.make_initial_population(n=pop_size)
#
#     sim.simulate(end_date=end_date)

for i in range(0, nsim):
    # Create the simulation object
    sim = Simulation(start_date=start_date)
    # Register the modules
    sim.register(
        demography.Demography(resourcefilepath=resourcefilepath),
        enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
        healthsystem.HealthSystem(resourcefilepath=resourcefilepath, service_availability=['*']),
        symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
        healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
        healthburden.HealthBurden(resourcefilepath=resourcefilepath),
        rti.RTI(resourcefilepath=resourcefilepath),
        simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
    )
    # Get the log file
    logfile = sim.configure_logging(filename="LogFile_suture",
                                    directory=log_file_location + "suture")
    # create and run the simulation
    sim.make_initial_population(n=pop_size)
    sim.modules['RTI'].parameters['blocked_interventions'] = ['Suture']
    # Run the simulation
    sim.simulate(end_date=end_date)

for i in range(0, nsim):
    # Create the simulation object
    sim = Simulation(start_date=start_date)
    # Register the modules
    sim.register(
        demography.Demography(resourcefilepath=resourcefilepath),
        enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
        healthsystem.HealthSystem(resourcefilepath=resourcefilepath, service_availability=['*']),
        symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
        healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
        healthburden.HealthBurden(resourcefilepath=resourcefilepath),
        rti.RTI(resourcefilepath=resourcefilepath),
        simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
    )
    # Get the log file
    logfile = sim.configure_logging(filename="LogFile_burn",
                                    directory=log_file_location + "burn")
    # create and run the simulation
    sim.make_initial_population(n=pop_size)
    sim.modules['RTI'].parameters['blocked_interventions'] = ['Burn']
    # Run the simulation
    sim.simulate(end_date=end_date)

for i in range(0, nsim):
    # Create the simulation object
    sim = Simulation(start_date=start_date)
    # Register the modules
    sim.register(
        demography.Demography(resourcefilepath=resourcefilepath),
        enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
        healthsystem.HealthSystem(resourcefilepath=resourcefilepath, service_availability=['*']),
        symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
        healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
        healthburden.HealthBurden(resourcefilepath=resourcefilepath),
        rti.RTI(resourcefilepath=resourcefilepath),
        simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
    )
    # Get the log file
    logfile = sim.configure_logging(filename="LogFile_open",
                                    directory=log_file_location + "open_fracture")
    # create and run the simulation
    sim.make_initial_population(n=pop_size)
    sim.modules['RTI'].parameters['blocked_interventions'] = ['Open fracture']
    # Run the simulation
    sim.simulate(end_date=end_date)
