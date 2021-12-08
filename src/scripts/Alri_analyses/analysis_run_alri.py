from pathlib import Path
import datetime

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    chronicsyndrome,
    contraception,
    demography,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    symptommanager,
    simplified_births, alri
)


# The Resource files [NB. Working directory must be set to the root of TLO: TLOmodel]
outputpath = Path("./outputs")
resourcefilepath = Path("./resources")

# Create name for log-file
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# Create dict to capture the outputs
output_files = dict()


# Establish the simulation object
start_date = Date(2010, 1, 1)
end_date = Date(2012, 1, 1)
popsize = 10000

log_config = {
    "filename": "Logfile",
    "directory": outputpath,
    "custom_levels": {
        "*": logging.WARNING,
        "tlo.methods.alri": logging.INFO,
        "tlo.methods.demography": logging.INFO,
    }
}

sim = Simulation(start_date=start_date, seed=0, log_config=log_config, show_progress_bar=True)

# Register the appropriate 'core' modules and do not let healthsystem constraints operate
sim.register(
    # Core Modules
    demography.Demography(resourcefilepath=resourcefilepath),
    simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
    enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
    symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
    healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
    healthburden.HealthBurden(resourcefilepath=resourcefilepath),

    # The HealthSystem
    healthsystem.HealthSystem(resourcefilepath=resourcefilepath, disable=True),

    # Register disease modules of interest
    alri.Alri(resourcefilepath=resourcefilepath),
    alri.AlriPropertiesOfOtherModules()
)

# Run the simulation
sim.make_initial_population(n=popsize)
sim.simulate(end_date=end_date)

# Read the results
# output = parse_log_file(sim.log_filepath)

