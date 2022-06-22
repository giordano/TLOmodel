import datetime
import pickle
import random
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    demography,
    enhanced_lifestyle,
    epi,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    hiv,
    simplified_births,
    symptommanager,
    tb,
)
# Where will outputs go
outputpath = Path("./outputs")  # folder for convenience of storing outputs

# date-stamp to label log files and any other outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# The resource files
resourcefilepath = Path("./resources")

# %% Run the simulation
start_date = Date(2010, 1, 1)
end_date = Date(2016, 1, 1)
popsize = 175000

# set up the log config
log_config = {
    "filename": "Logfile",
    "directory": outputpath,
    "custom_levels": {
        "*": logging.WARNING,
        "tlo.methods.hiv": logging.INFO,
        "tlo.methods.tb": logging.DEBUG,
        "tlo.methods.demography": logging.INFO,
        "tlo.methods.healthsystem.summary": logging.INFO,
        "tlo.methods.healthsystem": logging.INFO,
    },
}

# Register the appropriate modules
seed = random.randint(0, 50000)
# seed = 0  # set seed for reproducibility
sim = Simulation(start_date=start_date, seed=seed, log_config=log_config, show_progress_bar=True)
sim.register(
    demography.Demography(resourcefilepath=resourcefilepath),
    simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
    enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
    healthsystem.HealthSystem(
        resourcefilepath=resourcefilepath,
        service_availability=["*"],  # all treatment allowed
        mode_appt_constraints=0,  # mode of constraints to do with officer numbers and time
        cons_availability="all",  # mode for consumable constraints (if ignored, all consumables available)
        ignore_priority=True,  # do not use the priority information in HSI event to schedule
        capabilities_coefficient=1.0,  # multiplier for the capabilities of health officers
        disable=False,  # disables the healthsystem (no constraints and no logging) and every HSI runs
        disable_and_reject_all=False,  # disable healthsystem and no HSI runs
        store_hsi_events_that_have_run=False,  # convenience function for debugging
    ),
    symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
    healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
    healthburden.HealthBurden(resourcefilepath=resourcefilepath),
    epi.Epi(resourcefilepath=resourcefilepath),
    hiv.Hiv(resourcefilepath=resourcefilepath),
    tb.Tb(resourcefilepath=resourcefilepath),
)

# choose the scenario, 0=baseline, 4=shorter paediatric treatment
sim.modules["Tb"].parameters["scenario"] = 0
sim.modules["Tb"].parameters["scenario_start_date"] = Date(2010, 1, 1)
sim.modules["Demography"].parameters["max_age_initial"] = 16

# Run the simulation and flush the logger
sim.make_initial_population(n=popsize)
sim.simulate(end_date=end_date)

# parse the results
output = parse_log_file(sim.log_filepath)

# save the results, argument 'wb' means write using binary mode. use 'rb' for reading file
with open(outputpath / "default_run.pickle", "wb") as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(dict(output), f, pickle.HIGHEST_PROTOCOL)



# ---------------------------------- CONSUMABLES ANALYSIS ------------------------------------- #
# Get consumables summary
cons = output["tlo.methods.healthsystem.summary"]["Consumables"]
cons = cons.set_index('date')
cons = cons["Item_Available"].apply(pd.Series)

# Plot TB treatment use
fig, ax = plt.subplots()
ax.plot(cons.index, cons['178'], label='Child Tx', color='r')
ax.plot(cons.index, cons['179'], label='Child ReTx', color='g')
# ax.plot(cons.index, cons['2675'], label='Child Tx (Shorter)', color='y')
plt.xlabel('Year')
plt.ylabel('Quantity')
plt.title('TB Treatment Use')
plt.legend()
plt.show()

# Plot TB diagnostic test use
fig, ax = plt.subplots()
ax.plot(cons.index, cons['175'], label='X-Ray', color='r')
ax.plot(cons.index, cons['184'], label='Microscopy', color='g')
ax.plot(cons.index, cons['187'], label='Xpert', color='y')
plt.xlabel('Year')
plt.ylabel('Quantity')
plt.title('TB Diagnostic Test Use')
plt.legend()
plt.show()


# ---------------------------------- HSI EVENT ANALYSIS ------------------------------------- #
# Get HSI event summary
hsi = output["tlo.methods.healthsystem.summary"]["HSI_Event"]
hsi = hsi.set_index('date')
hsi = hsi["TREATMENT_ID"].apply(pd.Series)

# Plot HSI events
fig, ax = plt.subplots()
ax.plot(hsi.index, hsi['Tb_ScreeningAndRefer'], label='Screening and Refer', color='r')
ax.plot(hsi.index, hsi['Tb_Xray'], label='X-Ray', color='g')
ax.plot(hsi.index, hsi['Tb_Treatment_Initiation'], label='Treatment Initiation', color='y')
ax.plot(hsi.index, hsi['Tb_FollowUp'], label='Follow Up', color='m')
plt.xlabel('Year')
plt.ylabel('Quantity')
plt.title('HSI Event')
plt.legend()
plt.show()

# Get Appointment Type summary
appt_tpe = output["tlo.methods.healthsystem.summary"]["HSI_Event"]
appt_tpe = appt_tpe.set_index('date')
appt_tpe = appt_tpe["Number_By_Appt_Type_Code"].apply(pd.Series)

# Plot Appointment Types
fig, ax = plt.subplots()
ax.plot(appt_tpe.index, appt_tpe['TBNew'], label='TBNew', color='r')
ax.plot(appt_tpe.index, appt_tpe['DiagRadio'], label='DiagRadio', color='g')
ax.plot(appt_tpe.index, appt_tpe['TBFollowUp'], label='TBFollowUp', color='y')
ax.plot(appt_tpe.index, appt_tpe['LabTBMicro'], label='LabTBMicro', color='m')
ax.plot(appt_tpe.index, appt_tpe['LabMolec'], label='LabMolec', color='c')
ax.plot(appt_tpe.index, appt_tpe['Over5OPD'], label='Over5OPD', color='b')
ax.plot(appt_tpe.index, appt_tpe['Under5OPD'], label='Under5OPD', color='k')
plt.xlabel('Year')
plt.ylabel('Quantity')
plt.title('Appointment Types')
plt.legend()
plt.show()
