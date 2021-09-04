"""Run a simulation with no HSI constraints and plot the prevalence and incidence and program coverage trajectories"""
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

import pandas as pd
import datetime
import pickle
from pathlib import Path

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    demography,
    dx_algorithm_child,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    hiv,
    simplified_births,
    symptommanager,
)


resourcefilepath = Path("./resources")
outputpath = Path("./outputs")  # folder for convenience of storing outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")


# %% Run the simulation
start_date = Date(2010, 1, 1)
end_date = Date(2020, 1, 1)
popsize = 5000

# set up the logging file
log_config = {
    'filename': 'Logfile',
    'directory': outputpath,
    'custom_levels': {
        '*': logging.WARNING,
        'tlo.methods.hiv': logging.INFO,
        'tlo.methods.demography': logging.INFO
    }
}

# Register the appropriate modules
sim = Simulation(start_date=start_date, seed=100, log_config=log_config)
sim.register(demography.Demography(resourcefilepath=resourcefilepath),
             simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
             enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
             healthburden.HealthBurden(resourcefilepath=resourcefilepath),
             healthsystem.HealthSystem(
                 resourcefilepath=resourcefilepath,
                 service_availability=["*"],
                 mode_appt_constraints=0,
                 ignore_cons_constraints=True,
                 ignore_priority=True,
                 capabilities_coefficient=1.0,
                 disable=False,
             ),
             symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
             healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
             dx_algorithm_child.DxAlgorithmChild(resourcefilepath=resourcefilepath),
             hiv.Hiv(resourcefilepath=resourcefilepath)
             )

# # edit params to prevent all prep
# sim.modules['Hiv'].parameters["prob_prep_for_preg_after_hiv_test"] = 0


# Edit parameters to make a PrEP intervention that has 100% coverage, 100% adherence and 100% efficacy
sim.modules['Hiv'].parameters["prob_prep_for_preg_after_hiv_test"] = 1.0
sim.modules['Hiv'].parameters["prep_start_year"] = 2000
sim.modules['Hiv'].parameters["prep_start_year_preg"] = 2000
sim.modules['Hiv'].parameters["prob_prep_high_adherence"] = 1.0
sim.modules['Hiv'].parameters["prob_prep_mid_adherence"] = 0.0
sim.modules['Hiv'].parameters["prob_for_prep_selection"] = 1.0
sim.modules['Hiv'].parameters["proportion_reduction_in_risk_of_hiv_aq_if_on_prep"] = 1.0
sim.modules['Hiv'].parameters["rr_prep_high_adherence"] = 0.0
sim.modules['Hiv'].parameters["rr_prep_mid_adherence"] = 0.0
sim.modules['Hiv'].parameters["rr_prep_low_adherence"] = 0.0
sim.modules['Hiv'].parameters["probability_of_pregnant_woman_being_retained_on_prep_every_3_months"] = 1.0


# Run the simulation and flush the logger
sim.make_initial_population(n=popsize)
sim.simulate(end_date=end_date)

# parse the results
output = parse_log_file(sim.log_filepath)

# save the results
with open(outputpath / 'default_run.pickle', 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(output, f, pickle.HIGHEST_PROTOCOL)

# load the results
with open(outputpath / 'default_run.pickle', 'rb') as f:
    output = pickle.load(f)


# ---------------------------------------------------------------------- #
# %%: OUTPUTS
# ---------------------------------------------------------------------- #

# load the results
with open(outputpath / 'default_run.pickle', 'rb') as f:
    output = pickle.load(f)

# person-years all ages (irrespective of HIV status)
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

# treatment coverage
cov_over_time = output['tlo.methods.hiv']['hiv_program_coverage']
cov_over_time = cov_over_time.set_index('date')

# PrEP among Pregnant and Breastfeeding Women
prep_coverage=cov_over_time["prop_preg_and_bf_on_prep"]


# ----------------------------- HIV INCIDENCE ------------------------- #
prev_and_inc_over_time = output['tlo.methods.hiv'][
    'summary_inc_and_prev_for_adults_and_children_and_fsw']
prev_and_inc_over_time = prev_and_inc_over_time.set_index('date')

# Incidence Children:
inc_children = prev_and_inc_over_time['hiv_child_inc']*100

# Incidence Women of a Reproductive Age:
inc_women = prev_and_inc_over_time['hiv_women_reproductive_age_inc']*100

# Incidence Pregnant and Breastfeeding Women:
inc_preg = prev_and_inc_over_time['hiv_preg_and_bf_inc'] * 100

# Mother to Child Transmission Rate:
mtct = prev_and_inc_over_time['mtct'] * 100

list_of_series = [prep_coverage, inc_children, inc_women, inc_preg, mtct]
df = pd.DataFrame(list_of_series)
# df.to_excel(r'./outputs/NoPrep.xlsx')
df.to_excel(r'./outputs/PerfectPrep.xlsx')
