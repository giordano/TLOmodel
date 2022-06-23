
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
end_date = Date(2020, 1, 1)
popsize = 10000

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
sim.modules["Tb"].parameters["scenario"] = 4

# change scenario start date to speed up results
sim.modules["Tb"].parameters["scenario_start_date"] = Date(2014, 1, 1)

# create child (<16 years) population
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


# ----------------------------------------------------------------------------------------------- #
#                                         ANALYSIS PLOTS                                          #
# ----------------------------------------------------------------------------------------------- #
# ---------------------- TREATMENT REGIMEN VS CONSUMABLES REQUEST ANALYSIS ---------------------- #

# Get treatment regimen log
tb_tx_reg = output["tlo.methods.tb"]["tb_treatment_regimen"]
tb_tx_reg = tb_tx_reg.set_index('date')
tb_tx_reg.index = pd.DatetimeIndex(tb_tx_reg.index).year
tb_tx_reg = tb_tx_reg.groupby(tb_tx_reg.index).sum()

# Get consumables log
tb_cons = output["tlo.methods.healthsystem.summary"]["Consumables"]
tb_cons = tb_cons.set_index('date')
tb_cons.index = pd.DatetimeIndex(tb_cons.index).year
tb_cons = tb_cons['Item_Available'].apply(pd.Series).fillna(0)

# The following plots compare the number of patients initiated on treatment (tb module treatment regimen log)
# with the number of consumables requested (health system summary log). The values from the two logs should match.

# (1) Child Treatment
fig, ax = plt.subplots()
ax.plot(tb_tx_reg.index, tb_tx_reg['TBTxChild'], label='TB Treatment Regimen Log', color='r')
ax.plot(tb_cons.index, tb_cons['178'], label='Health System Consumables Log', color='b')
plt.xlabel('Year')
plt.ylabel('Quantity')
plt.title('Child Treatment')
plt.legend()
plt.show()

# (2) Child Treatment (Shorter)
fig, ax = plt.subplots()
ax.plot(tb_tx_reg.index, tb_tx_reg['TBTxChildShorter'], label='TB Treatment Regimen Log', color='r')
ax.plot(tb_cons.index, tb_cons['2675'], label='Health System Consumables Log', color='b')
plt.xlabel('Year')
plt.ylabel('Quantity')
plt.title('Child Treatment (Shorter)')
plt.legend()
plt.show()

# (3) Child Re-Treatment
fig, ax = plt.subplots()
ax.plot(tb_tx_reg.index, tb_tx_reg['TBRetxChild'], label='TB Treatment Regimen Log', color='r')
ax.plot(tb_cons.index, tb_cons['179'], label='Health System Consumables Log', color='b')
plt.xlabel('Year')
plt.ylabel('Quantity')
plt.title('Child Re-Treatment')
plt.legend()
plt.show()

# (4) Total Treatment
tb_cons_total_tx = tb_cons[["178", "179", "2675"]].sum(axis=1) # take out item code 2675 if running scenario 0

fig, ax = plt.subplots()
ax.plot(tb_tx_reg.index, tb_tx_reg['TBTx'], label='TB Treatment Regimen Log', color='r')
ax.plot(tb_cons_total_tx.index, tb_cons_total_tx, label='Health System Consumables Log', color='b')
plt.xlabel('Year')
plt.ylabel('Quantity')
plt.title('Total Treatment')
plt.legend()
plt.show()


