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
popsize = 17500

# set up the log config
log_config = {
    "filename": "Logfile",
    "directory": outputpath,
    "custom_levels": {
        "*": logging.WARNING,
        "tlo.methods.hiv": logging.INFO,
        "tlo.methods.tb": logging.INFO,
        "tlo.methods.demography": logging.INFO,
        "tlo.methods.healthsystem": logging.INFO,
    },
}

# Register the appropriate modules
# seed = random.randint(0, 50000)
seed = 4  # set seed for reproducibility
sim = Simulation(start_date=start_date, seed=seed, log_config=log_config, show_progress_bar=True)
sim.register(
    demography.Demography(resourcefilepath=resourcefilepath, max_age_initial=16),
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

# Run the simulation and flush the logger
sim.make_initial_population(n=popsize)
sim.simulate(end_date=end_date)

# parse the results
output = parse_log_file(sim.log_filepath)

# save the results, argument 'wb' means write using binary mode. use 'rb' for reading file
with open(outputpath / "default_run.pickle", "wb") as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(dict(output), f, pickle.HIGHEST_PROTOCOL)


# -------------------------------------------------------------------------------------- #
#                                     ANALYSIS PLOTS                                     #
# -------------------------------------------------------------------------------------- #

# -------------------------------- SET UP ANALYSIS PLOTS ------------------------------- #

def make_plot(model=None, data_mid=None, data_low=None, data_high=None, title_str=None):
    assert model is not None
    assert title_str is not None

    fig, ax = plt.subplots()
    ax.plot(model.index, model.values, "-", color="r")

    if data_mid is not None:
        ax.plot(data_mid.index, data_mid.values, "-", color="b")
    if (data_low is not None) and (data_high is not None):
        ax.fill_between(data_low.index, data_low, data_high, color="b", alpha=0.2)
    plt.title(title_str)
    plt.xlabel("Years")
    plt.gca().set_ylim(bottom=0)
    plt.show()


# -------------------------------------- LOAD DATA ------------------------------------- #
data_who_tb_2020 = pd.read_excel(resourcefilepath / 'ResourceFile_TB.xlsx', sheet_name='WHO_activeTB2020')
data_who_tb_2020.index = pd.to_datetime(data_who_tb_2020['year'], format='%Y')
data_who_tb_2020 = data_who_tb_2020.drop(columns=['year'])


# ------------------------------------ ANALYSIS PLOTS ---------------------------------- #
# (1) Number of New Active TB Infections
sf = output['tlo.methods.demography']['scaling_factor']['scaling_factor'].values[0]

tb_incidence_children = output["tlo.methods.tb"]["tb_incidence"]
tb_incidence_children = tb_incidence_children.set_index('date')
tb_incidence_children.index = pd.to_datetime(tb_incidence_children.index)

make_plot(
    title_str="Number of New Active TB Infections (0-16 years)",
    model=tb_incidence_children['num_new_active_tb_child'],
    data_mid=data_who_tb_2020['estimated_inc_number_children'],
    data_low=data_who_tb_2020['estimated_inc_number_children_low'],
    data_high=data_who_tb_2020['estimated_inc_number_children_high'],
)

# (2) Active TB Incidence Rate
py_ = output['tlo.methods.demography']['person_years']
years = pd.to_datetime(py_['date']).dt.year
py = pd.Series(dtype='int64', index=years)
for year in years:
    tot_py = (
        (py_.loc[pd.to_datetime(py_['date']).dt.year == year]['M']).apply(pd.Series) +
        (py_.loc[pd.to_datetime(py_['date']).dt.year == year]['F']).apply(pd.Series)
    ).transpose()
    py[year] = tot_py.sum().values[0]
py.index = pd.to_datetime(years, format='%Y')

active_tb_incidence_rate_children = (tb_incidence_children['num_new_active_tb_child'] / py[0]) * 100000

make_plot(
    title_str="Active TB Incidence (per 100k person-years)",
    model=active_tb_incidence_rate_children,
)

# (3) Number of Diagnosed TB Cases
tb_treatment_children = output["tlo.methods.tb"]["tb_treatment"]
tb_treatment_children = tb_treatment_children.set_index('date')
tb_treatment_children.index = pd.to_datetime(tb_treatment_children.index)

make_plot(
    title_str="Number of New Diagnosed TB Cases (0-16 years)",
    model=tb_treatment_children['tbNewDiagnosis'],
)

# (4) Number of Treated TB Cases

make_plot(
    title_str="Number of Treated TB Cases (0-16 years)",
    model=tb_treatment_children['tbNewTreatment'],
)

