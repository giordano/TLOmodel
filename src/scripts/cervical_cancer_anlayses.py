"""
* Check key outputs for reporting in the calibration table of the write-up
* Produce representative plots for the default parameters

NB. To see larger effects
* Increase incidence of cancer (see tests)
* Increase symptom onset
* Increase progression rates (see tests)
"""

import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tlo import Date, Simulation
from tlo.analysis.utils import make_age_grp_types, parse_log_file
from tlo.methods import (
    cervical_cancer,
    demography,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    simplified_births,
    symptommanager
)

# Where will outputs go
outputpath = Path("./outputs")  # folder for convenience of storing outputs

# date-stamp to label log files and any other outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# The resource files
resourcefilepath = Path("./resources")

# Set parameters for the simulation
start_date = Date(2010, 1, 1)
end_date = Date(2010, 2, 1)
popsize = 1000


def run_sim(service_availability):
    # Establish the simulation object and set the seed
    sim = Simulation(start_date=start_date, seed=0)

    # Register the appropriate modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                           disable=False,
                                           cons_availability='all'),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 cervical_cancer.CervicalCancer(resourcefilepath=resourcefilepath)
                 )

    # Establish the logger
    logfile = sim.configure_logging(filename="LogFile")

    # Run the simulation
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)

    return logfile


def get_summary_stats(logfile):
    output = parse_log_file(logfile)

    # 1) TOTAL COUNTS BY STAGE OVER TIME
    counts_by_stage = output['tlo.methods.cervical_cancer']['summary_stats']
    counts_by_stage['date'] = pd.to_datetime(counts_by_stage['date'])
    counts_by_stage = counts_by_stage.set_index('date', drop=True)

    # 2) NUMBERS UNDIAGNOSED-DIAGNOSED-TREATED-PALLIATIVE CARE OVER TIME (SUMMED ACROSS TYPES OF CANCER)
    def get_cols_excl_none(allcols, stub):
        # helper function to some columns with a certain prefix stub - excluding the 'none' columns (ie. those
        #  that do not have cancer)
        cols = allcols[allcols.str.startswith(stub)]
        cols_not_none = [s for s in cols if ("none" not in s)]
        return cols_not_none

    summary = {
        'total': counts_by_stage[get_cols_excl_none(counts_by_stage.columns, 'total_')].sum(axis=1),
        'udx': counts_by_stage[get_cols_excl_none(counts_by_stage.columns, 'undiagnosed_')].sum(axis=1),
        'dx': counts_by_stage[get_cols_excl_none(counts_by_stage.columns, 'diagnosed_')].sum(axis=1),
        'tr': counts_by_stage[get_cols_excl_none(counts_by_stage.columns, 'treatment_')].sum(axis=1),
        'pc': counts_by_stage[get_cols_excl_none(counts_by_stage.columns, 'palliative_')].sum(axis=1)
    }
    counts_by_cascade = pd.DataFrame(summary)

    # 3) DALYS wrt age (total over whole simulation)
    dalys = output['tlo.methods.healthburden']['dalys']
    dalys = dalys.groupby(by=['age_range']).sum()
    dalys.index = dalys.index.astype(make_age_grp_types())
    dalys = dalys.sort_index()

    # 4) DEATHS wrt age (total over whole simulation)
    deaths = output['tlo.methods.demography']['death']
    deaths['age_group'] = deaths['age'].map(demography.Demography(resourcefilepath=resourcefilepath).AGE_RANGE_LOOKUP)

    x = deaths.loc[deaths.cause == 'CervicalCancer'].copy()
    x['age_group'] = x['age_group'].astype(make_age_grp_types())
    breast_cancer_deaths = x.groupby(by=['age_group']).size()

    # 5) Rates of diagnosis per year:
    counts_by_stage['year'] = counts_by_stage.index.year
    annual_count_of_dxtr = counts_by_stage.groupby(by='year')[['diagnosed_since_last_log',
                                                               'treated_since_last_log',
                                                               'palliative_since_last_log']].sum()

    return {
        'total_counts_by_stage_over_time': counts_by_stage,
        'counts_by_cascade': counts_by_cascade,
        'dalys': dalys,
        'deaths': deaths,
        'cervical_cancer_deaths': cervical_cancer_deaths,
        'annual_count_of_dxtr': annual_count_of_dxtr
    }


# %% Run the simulation with and without interventions being allowed

# With interventions:
logfile_with_healthsystem = run_sim(service_availability=['*'])

"""

results_with_healthsystem = get_summary_stats(logfile_with_healthsystem)

# Without interventions:
logfile_no_healthsystem = run_sim(service_availability=[])
results_no_healthsystem = get_summary_stats(logfile_no_healthsystem)

# %% Produce Summary Graphs:

# Examine Counts by Stage Over Time
counts = results_no_healthsystem['total_counts_by_stage_over_time']
counts.plot(y=['total_stage1', 'total_stage2a', 'total_stage2b', 'total_stage3', 'total_stage'])
plt.title('Count in Each Stage of Disease Over Time')
plt.xlabel('Time')
plt.ylabel('Count')
plt.show()

# Examine numbers in each stage of the cascade:
results_with_healthsystem['counts_by_cascade'].plot(y=['udx', 'dx', 'tr', 'pc'])
plt.title('With Health System')
plt.xlabel('Numbers of those With Cancer by Stage in Cascade')
plt.xlabel('Time')
plt.legend(['Undiagnosed', 'Diagnosed', 'On Treatment', 'On Palliative Care'])
plt.show()

results_no_healthsystem['counts_by_cascade'].plot(y=['udx', 'dx', 'tr', 'pc'])
plt.title('With No Health System')
plt.xlabel('Numbers of those With Cancer by Stage in Cascade')
plt.xlabel('Time')
plt.legend(['Undiagnosed', 'Diagnosed', 'On Treatment', 'On Palliative Care'])
plt.show()

# Examine DALYS (summed over whole simulation)
results_no_healthsystem['dalys'].plot.bar(
    y=['YLD_CervicalCancer_0', 'YLL_CervicalCancer_CervicalCancer'],
    stacked=True)
plt.xlabel('Age-group')
plt.ylabel('DALYS')
plt.legend()
plt.title("With No Health System")
plt.show()

# Examine Deaths (summed over whole simulation)
deaths = results_no_healthsystem['cervical_cancer_deaths']
deaths.index = deaths.index.astype(make_age_grp_types())
# # make a series with the right categories and zero so formats nicely in the grapsh:
agegrps = demography.Demography(resourcefilepath=resourcefilepath).AGE_RANGE_CATEGORIES
totdeaths = pd.Series(index=agegrps, data=np.nan)
totdeaths.index = totdeaths.index.astype(make_age_grp_types())
totdeaths = totdeaths.combine_first(deaths).fillna(0.0)
totdeaths.plot.bar()
plt.title('Deaths due to Cervical Cancer')
plt.xlabel('Age-group')
plt.ylabel('Total Deaths During Simulation')
# plt.gca().get_legend().remove()
plt.show()

# Compare Deaths - with and without the healthsystem functioning - sum over age and time
deaths = {
    'No_HealthSystem': sum(results_no_healthsystem['cervical_cancer_deaths']),
    'With_HealthSystem': sum(results_with_healthsystem['cervical_cancer_deaths'])
}

plt.bar(range(len(deaths)), list(deaths.values()), align='center')
plt.xticks(range(len(deaths)), list(deaths.keys()))
plt.title('Deaths due to Cervical Cancer')
plt.xlabel('Scenario')
plt.ylabel('Total Deaths During Simulation')
plt.show()


# %% Get Statistics for Table in write-up (from results_with_healthsystem);

# ** Current prevalence (end-2019) of people who have diagnosed with cervical
# cancer in 2020 (total; and current stage 1, 2, 3, 4), per 100,000 population aged 20+

counts = results_with_healthsystem['total_counts_by_stage_over_time'][[
    'total_stage1',
    'total_stage2a',
    'total_stage2b',
    'total_stage3',
    'total_stage4'
]].iloc[-1]

totpopsize = results_with_healthsystem['total_counts_by_stage_over_time'][[
    'total_none',
    'total_stage1',
    'total_stage2a',
    'total_stage2b',
    'total_stage3',
    'total_stage4'
]].iloc[-1].sum()

prev_per_100k = 1e5 * counts.sum() / totpopsize

# ** Number of deaths from cervical cancer per year per 100,000 population.
# average deaths per year = deaths over ten years divided by ten, * 100k/population size
(results_with_healthsystem['cervical_cancer_deaths'].sum()/10) * 1e5/popsize

# ** Incidence rate of diagnosis, treatment, palliative care for cervical cancer (all stages combined),
# per 100,000 population
(results_with_healthsystem['annual_count_of_dxtr']).mean() * 1e5/popsize


# ** 5-year survival following treatment
# See separate file

"""