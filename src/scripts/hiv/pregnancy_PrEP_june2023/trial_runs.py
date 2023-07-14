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
# os.chdir('/Users/wenjiazhang/Documents/MSc_HDA/Summer/TLOmodel/')

# Where will outputs go
# outputpath = Path("/Users/wenjiazhang/Documents/MSc_HDA/Summer/TLOmodel/outputs")  # folder for convenience of storing outputs
outputpath = Path("./outputs")  # folder for convenience of storing outputs

# date-stamp to label log files and any other outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# The resource files
resourcefilepath = Path("./resources")
#
# %% Run the simulation
start_date = Date(2010, 1, 1)
end_date = Date(2013, 1, 1)
popsize = 200

# Set the simulation interval
# interval = relativedelta(months=6)

# todo don't need this
# scenario = 1

# set up the log config
# todo remove unnecessary log files - they are huge and can crash simulations
log_config = {
    "filename": "test_runs",
    "directory": outputpath,
    "custom_levels": {
        "*": logging.WARNING,
        "tlo.methods.hiv": logging.INFO,
        "tlo.methods.demography": logging.INFO,
        "tlo.methods.healthsystem.summary": logging.INFO,
        "tlo.methods.healthburden": logging.INFO,
    },
}

# Register the appropriate modules
# need to call epi before tb to get bcg vax
seed = 1  # set seed for reproducibility
sim = Simulation(start_date=start_date, seed=seed, log_config=log_config, show_progress_bar=True)
# todo removed simplified births
sim.register(
    epi.Epi(resourcefilepath=resourcefilepath),
    demography.Demography(resourcefilepath=resourcefilepath),
    contraception.Contraception(resourcefilepath=resourcefilepath),
    enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
    healthburden.HealthBurden(resourcefilepath=resourcefilepath),
    symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
    healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                              service_availability=["*"],  # all treatment allowed
                              mode_appt_constraints=0,  # mode of constraints to do with officer numbers and time
                              cons_availability="default",
                              # mode for consumable constraints (if ignored, all consumables available)
                              ignore_priority=False,  # do not use the priority information in HSI event to schedule
                              capabilities_coefficient=1.0,  # multiplier for the capabilities of health officers
                              use_funded_or_actual_staffing="funded_plus",
                              # actual: use numbers/distribution of staff available currently
                              disable=False,
                              # disables the healthsystem (no constraints and no logging) and every HSI runs
                              disable_and_reject_all=False,  # disable healthsystem and no HSI runs
                              ),
    newborn_outcomes.NewbornOutcomes(resourcefilepath=resourcefilepath),
    pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
    care_of_women_during_pregnancy.CareOfWomenDuringPregnancy(resourcefilepath=resourcefilepath),
    labour.Labour(resourcefilepath=resourcefilepath),
    postnatal_supervisor.PostnatalSupervisor(resourcefilepath=resourcefilepath),
    healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
    hiv.Hiv(resourcefilepath=resourcefilepath),
    tb.Tb(resourcefilepath=resourcefilepath)
)

# todo you can comment/uncomment and change these and they will overwrite the parameters in the resourcefiles
# set the scenario
sim.modules["CareOfWomenDuringPregnancy"].parameters["prob_pregnant_woman_starts_prep"] = 1.0
# sim.modules["NewbornOutcomes"].parameters["prob_breastfeeding_woman_starts_prep"] = 0.2
# todo I changed this just for development so I can see women starting prep
# straightaway instead of waiting until 2023
sim.modules["CareOfWomenDuringPregnancy"].parameters["prep_for_pregnant_woman_start_year"] = 2010
sim.modules["Hiv"].parameters["probability_of_being_retained_on_prep_every_1_month"] = 1.0


# Run the simulation and flush the logger
sim.make_initial_population(n=popsize)
sim.simulate(end_date=end_date)

# Run the simulation and flush the logger
sim.make_initial_population(n=popsize)

# todo i've added a monthly logger for prep in the hiv module
# # make the logging every 6 months
# while sim.date < end_date:
#     next_date = sim.date + relativedelta(months=6)
#     sim.simulate(end_date=next_date)

# parse the results
output = parse_log_file(sim.log_filepath)

# save the results, argument 'wb' means write using binary mode. use 'rb' for reading file
with open(outputpath / "default_run.pickle", "wb") as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(dict(output), f, pickle.HIGHEST_PROTOCOL)

# load the results
with open(outputpath / "default_run.pickle", "rb") as f:
    output = pickle.load(f)
