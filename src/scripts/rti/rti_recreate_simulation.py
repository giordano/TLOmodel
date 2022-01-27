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
resourcefilepath = Path('./resources')
yearsrun = 5

start_date = Date(year=2010, month=1, day=1)
end_date = Date(year=(2010 + yearsrun), month=1, day=1)
service_availability = ['*']
pop_size = 20000
# Create the simulation object
# sim = Simulation(start_date=start_date, seed=1001268886)
sim = Simulation(start_date=start_date, seed=3806503318)
# Register the modules
sim.register(
    demography.Demography(resourcefilepath=resourcefilepath),
    enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
    healthsystem.HealthSystem(resourcefilepath=resourcefilepath, service_availability=['*']),
    healthburden.HealthBurden(resourcefilepath=resourcefilepath),
    symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
    healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
    rti.RTI(resourcefilepath=resourcefilepath),
    simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
)
# Get the log file
logfile = sim.configure_logging(filename="Recreated_run")
# create and run the simulation
sim.make_initial_population(n=pop_size)
sim.modules['RTI'].parameters['imm_death_proportion_rti'] = 0.007027777777777777
sim.simulate(end_date=end_date)
log = parse_log_file(logfile)
data = pd.read_csv("resources/demography/ResourceFile_Pop_Annual_WPP.csv")
sim_start_year = sim.start_date.year
sim_end_year = sim.date.year
sim_year_range = pd.Index(np.arange(sim_start_year, sim_end_year + 1))
Data_Pop = data.groupby(by="Year")["Count"].sum()
Data_Pop = Data_Pop.loc[sim_year_range]
model_pop_size = log['tlo.methods.demography']['population']['total'].tolist()
model_pop_size.append(len(sim.population.props.loc[sim.population.props.is_alive]))
scaling_df = pd.DataFrame({'total': model_pop_size})
scaling_df['pred_pop_size'] = Data_Pop.to_list()
scaling_df['scale_for_each_year'] = scaling_df['pred_pop_size'] / scaling_df['total']
scaling_df.index = sim_year_range
dalys_df = log['tlo.methods.healthburden']['dalys_stacked']
dalys_df = dalys_df.groupby('year').sum()
dalys_df['extrapolated_dalys'] = dalys_df['Transport Injuries'] * scaling_df['scale_for_each_year']
dalys_df = dalys_df.loc[~pd.isnull(dalys_df['extrapolated_dalys'])]
yll_stacked = log['tlo.methods.healthburden']['yll_by_causes_of_death_stacked']
yll_stacked['total_rti'] = [0.0] * len(yll_stacked)
for col in yll_stacked.columns:
    if 'RTI' in str(col):
        yll_stacked['total_rti'] += yll_stacked[col]
yll_stacked = yll_stacked.groupby('year').sum()
yll_stacked['extrapolated_yll'] = yll_stacked['total_rti'] * scaling_df['scale_for_each_year']
pred_dalys = dalys_df['extrapolated_dalys'].sum()
pred_yll = yll_stacked['extrapolated_yll'].sum()
