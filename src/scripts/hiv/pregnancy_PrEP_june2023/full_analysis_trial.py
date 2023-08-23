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
    tb,
)

# Set the working directory
os.chdir('/Users/wenjiazhang/Documents/MSc_HDA/Summer/TLOmodel/')

# Where will outputs go
# outputpath = Path("/Users/wenjiazhang/Documents/MSc_HDA/Summer/TLOmodel/outputs")  # folder for convenience of storing outputs
outputpath = Path("./outputs")  # folder for convenience of storing outputs

# date-stamp to label log files and any other outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# The resource files
resourcefilepath = Path("./resources")

# %% Run the simulation
start_date = Date(2010, 1, 1)
end_date = Date(2012, 1, 1)
popsize = 5000

# set up the log config
log_config = {
    "filename": "test_runs",
    "directory": outputpath,
    "custom_levels": {
        "*": logging.WARNING,
        "tlo.methods.hiv": logging.INFO,
        "tlo.methods.demography": logging.INFO,
        "tlo.methods.care_of_women_during_pregnancy": logging.INFO,
        "tlo.methods.healthsystem.summary": logging.INFO,
        "tlo.methods.newborn_outcomes": logging.INFO,
        "tlo.methods.healthburden": logging.INFO,
    },
}


#seed = random.randint(0, 50000)
seed = 1  # set seed for reproducibility

sim = Simulation(start_date=start_date, seed=seed, log_config=log_config, show_progress_bar=True)
sim.register(
     epi.Epi(resourcefilepath=resourcefilepath),
     demography.Demography(resourcefilepath=resourcefilepath),
     contraception.Contraception(resourcefilepath=resourcefilepath),
     enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
     healthburden.HealthBurden(resourcefilepath=resourcefilepath),
     symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
     healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                               service_availability=["*"],  # all treatment allowed
                               mode_appt_constraints=0,
                               cons_availability="default",
                               ignore_priority=False,
                               capabilities_coefficient=1.0,
                               use_funded_or_actual_staffing="funded_plus",
                               disable=False,
                               disable_and_reject_all=False,  # disable healthsystem and no HSI runs
                               ),
     newborn_outcomes.NewbornOutcomes(resourcefilepath=resourcefilepath),
     pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
     care_of_women_during_pregnancy.CareOfWomenDuringPregnancy(resourcefilepath=resourcefilepath),
     labour.Labour(resourcefilepath=resourcefilepath),
     postnatal_supervisor.PostnatalSupervisor(resourcefilepath=resourcefilepath),
     healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
     hiv.Hiv(resourcefilepath=resourcefilepath),
     tb.Tb(resourcefilepath=resourcefilepath),
 )

# scenario 0 - adherence remains the same for all individuals
sim.modules["CareOfWomenDuringPregnancy"].parameters["prep_for_pregnant_woman_start_year"] = 2040


# scenario 1 - adherence remains the same for all individuals
#sim.modules["Hiv"].parameters["probability_of_being_retained_on_prep_every_1_month"] = 1.0
#sim.modules["Hiv"].parameters["probability_of_being_retained_on_prep_every_1_month_low"] = 1.0
#sim.modules["Hiv"].parameters["probability_of_being_retained_on_prep_every_1_month_high"] = 1.0
#sim.modules["Hiv"].parameters["probability_of_prep_consumables_being_available"] = 1

# scenario 2 - adjusted scenario

# scenario 3 - adherence remains the same for all individuals
sim.modules["Hiv"].parameters["probability_of_being_retained_on_prep_every_1_month"] = 1.0
sim.modules["Hiv"].parameters["probability_of_being_retained_on_prep_every_1_month_low"] = 1.0
sim.modules["Hiv"].parameters["probability_of_being_retained_on_prep_every_1_month_high"] = 1.0
sim.modules["Hiv"].parameters["probability_of_prep_consumables_being_available"] = 0.85

# Run the simulation and flush the logger
sim.make_initial_population(n=popsize)
sim.simulate(end_date=end_date)

# parse the results
output = parse_log_file(sim.log_filepath)

# save the results, argument 'wb' means write using binary mode. use 'rb' for reading file
with open(outputpath / "default_run.pickle", "wb") as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(dict(output), f, pickle.HIGHEST_PROTOCOL)

with open(outputpath / "default_run.pickle", "rb") as f:
    output = pickle.load(f)
