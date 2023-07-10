"""
Run the HIV/TB modules with intervention coverage specified at national level
save outputs for plotting (file: output_plots_tb.py)
 """

import datetime
import os
import pickle
import random
from pathlib import Path
from dateutil.relativedelta import relativedelta

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    # deviance_measure,
    demography,
    contraception,
    enhanced_lifestyle,
    epi,
    newborn_outcomes,
    pregnancy_supervisor,
    labour,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    hiv,
    postnatal_supervisor,
    care_of_women_during_pregnancy,
    symptommanager,
    tb
)

# Set the working directory
os.chdir('/Users/wenjiazhang/Documents/MSc_HDA/Summer/TLOmodel/')

# Where will outputs go
outputpath = Path("/Users/wenjiazhang/Documents/MSc_HDA/Summer/TLOmodel/outputs")  # folder for convenience of storing outputs

# date-stamp to label log files and any other outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# The resource files
resourcefilepath = Path("./resources")
#
# %% Run the simulation
start_date = Date(2010, 1, 1)
end_date = Date(2020, 1, 1)
popsize = 1000

# Set the simulation interval
interval = relativedelta(months=6)

scenario = 1

# set up the log config
log_config = {
    "filename": "test_runs",
    "directory": outputpath,
    "custom_levels": {
        "*": logging.WARNING,
        "tlo.methods.deviance_measure": logging.INFO,
        "tlo.methods.epi": logging.INFO,
        "tlo.methods.hiv": logging.INFO,
        "tlo.methods.tb": logging.INFO,
        "tlo.methods.newborn_outcomes": logging.INFO,
        "tlo.methods.care_of_women_during_pregnancy": logging.INFO,
        "tlo.methods.postnatal_supervisor": logging.INFO,
        "tlo.methods.labour": logging.INFO,
        "tlo.methods.demography": logging.INFO,
        "tlo.methods.demography.detail": logging.WARNING,
        "tlo.methods.healthsystem.summary": logging.INFO,
        "tlo.methods.healthsystem": logging.INFO,
        "tlo.methods.healthburden": logging.INFO,
    },
}

# Register the appropriate modules
# need to call epi before tb to get bcg vax
seed = 1  # set seed for reproducibility
sim = Simulation(start_date=start_date, seed=seed, log_config=log_config, show_progress_bar=True)
sim.register(
    #simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
    epi.Epi(resourcefilepath=resourcefilepath),
    demography.Demography(resourcefilepath=resourcefilepath),
    contraception.Contraception(resourcefilepath=resourcefilepath),
    enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
    healthburden.HealthBurden(resourcefilepath=resourcefilepath),
    symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
    healthsystem.HealthSystem(resourcefilepath=resourcefilepath, mode_appt_constraints=1, cons_availability='default'),
    newborn_outcomes.NewbornOutcomes(resourcefilepath=resourcefilepath),
    pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
    care_of_women_during_pregnancy.CareOfWomenDuringPregnancy(resourcefilepath=resourcefilepath),
    labour.Labour(resourcefilepath=resourcefilepath),
    postnatal_supervisor.PostnatalSupervisor(resourcefilepath=resourcefilepath),
    healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
    hiv.Hiv(resourcefilepath=resourcefilepath),
    tb.Tb(resourcefilepath=resourcefilepath)
    )


# set the scenario
# sim.modules["Hiv"].parameters["beta"] = 0.129671
# sim.modules["Tb"].parameters["scaling_factor_WHO"] = 1.5
# sim.modules["Tb"].parameters["scenario"] = scenario
#sim.modules["Tb"].parameters["scenario_start_date"] = Date(2010, 1, 1)
# sim.modules["Tb"].parameters["scenario_SI"] = "z"

# sim.modules["Tb"].parameters["rr_tb_hiv"] = 5  # default 13
# rr relapse if HIV+ 4.7
# sim.modules["Tb"].parameters["rr_tb_aids"] = 26  # default 26

# to cluster tests in positive people
# sim.modules["Hiv"].parameters["rr_test_hiv_positive"] = 1.1  # default 1.5

# to account for people starting-> defaulting, or not getting cons
# this not used now if perfect referral testing->treatment
# affects the prob of art start once diagnosed
# sim.modules["Hiv"].parameters["treatment_initiation_adjustment"] = 1  # default 1.5

# assume all defaulting is due to cons availability
# sim.modules["Hiv"].parameters["probability_of_being_retained_on_art_every_6_months"] = 1.0
# sim.modules["Hiv"].parameters["probability_of_seeking_further_art_appointment_if_drug_not_available"] = 1.0


# Run the simulation and flush the logger
sim.make_initial_population(n=popsize)
sim.simulate(end_date=end_date)

# Run the simulation and flush the logger
sim.make_initial_population(n=popsize)

# make the logging every 6 months
while sim.date < end_date:
    next_date = sim.date + relativedelta(months=6)
    sim.simulate(end_date=next_date)

# parse the results
output = parse_log_file(sim.log_filepath)

# save the results, argument 'wb' means write using binary mode. use 'rb' for reading file
with open(outputpath / "default_run.pickle", "wb") as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(dict(output), f, pickle.HIGHEST_PROTOCOL)

# load the results
with open(outputpath / "default_run.pickle", "rb") as f:
    output = pickle.load(f)
