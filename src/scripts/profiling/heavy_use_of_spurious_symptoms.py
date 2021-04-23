"""This profiling script runs a simulation that uses spurious symptoms that incurs a heavy use of the SymptomManager and
 the GenericAppts"""

import os
from pathlib import Path

from pandas import DateOffset

from tlo import Date, Simulation
from tlo.methods import (
    chronicsyndrome,
    demography,
    dx_algorithm_child,
    enhanced_lifestyle,
    healthseekingbehaviour,
    healthsystem,
    mockitis,
    simplified_births,
    symptommanager,
)
from tlo.methods.symptommanager import (
    DuplicateSymptomWithNonIdenticalPropertiesError,
    Symptom,
    SymptomManager_AutoOnsetEvent,
    SymptomManager_AutoResolveEvent,
)

resourcefilepath = Path("./resources")

start_date = Date(2010, 1, 1)
end_date = start_date + DateOffset(years=1)
popsize = 1000

sim = Simulation(start_date=start_date)

# Register the core modules
sim.register(demography.Demography(resourcefilepath=resourcefilepath),
             enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
             healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                       disable=True),
             symptommanager.SymptomManager(resourcefilepath=resourcefilepath, spurious_symptoms=True),
             simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
             )

# Run the simulation
sim.make_initial_population(n=popsize)
sim.simulate(end_date=end_date)


