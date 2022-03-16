import datetime
from pathlib import Path

import pandas as pd

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    alri,
    bladder_cancer,
    breast_cancer,
    cardio_metabolic_disorders,
    care_of_women_during_pregnancy,
    contraception,
    demography,
    depression,
    diarrhoea,
    enhanced_lifestyle,
    epi,
    epilepsy,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    hiv,
    labour,
    malaria,
    measles,
    newborn_outcomes,
    oesophagealcancer,
    other_adult_cancers,
    postnatal_supervisor,
    pregnancy_supervisor,
    prostate_cancer,
    rti,
    stunting,
    symptommanager,
    wasting,
)

log_config = {
    "filename": "rti_analysis",   # The name of the output file (a timestamp will be appended).
    "directory": "./outputs",  # The default output path is `./outputs`. Change it here, if necessary
    "custom_levels": {  # Customise the output of specific loggers. They are applied in order:
        "*": logging.WARNING,  # Asterisk matches all loggers - we set the default level to WARNING
        "tlo.methods.rti": logging.INFO,
        "tlo.methods.healthsystem": logging.INFO,
    }
}

start_date = Date(2010, 1, 1)
end_date = Date(2020, 12, 31)
pop_size = 10000

# This creates the Simulation instance for this run. Because we've passed the `seed` and
# `log_config` arguments, these will override the default behaviour.
sim = Simulation(start_date=start_date, log_config=log_config)
resourcefilepath = Path('./resources')

# We register all modules in a single call to the register method, calling once with multiple
# objects. This is preferred to registering each module in multiple calls because we will be
# able to handle dependencies if modules are registered together
sim.register(
    # Core Modules
    demography.Demography(resourcefilepath=resourcefilepath),
    enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
    symptommanager.SymptomManager(resourcefilepath=resourcefilepath, spurious_symptoms=False),
    healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
    healthburden.HealthBurden(resourcefilepath=resourcefilepath),

    # Representations of the Healthcare System
    healthsystem.HealthSystem(resourcefilepath=resourcefilepath),
    epi.Epi(resourcefilepath=resourcefilepath),

    # - Contraception, Pregnancy and Labour
    contraception.Contraception(resourcefilepath=resourcefilepath, use_healthsystem=True),
    pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
    care_of_women_during_pregnancy.CareOfWomenDuringPregnancy(resourcefilepath=resourcefilepath),
    labour.Labour(resourcefilepath=resourcefilepath),
    newborn_outcomes.NewbornOutcomes(resourcefilepath=resourcefilepath),
    postnatal_supervisor.PostnatalSupervisor(resourcefilepath=resourcefilepath),

    # - Conditions of Early Childhood
    # diarrhoea.Diarrhoea(resourcefilepath=resourcefilepath),
    # alri.Alri(resourcefilepath=resourcefilepath),
    # stunting.Stunting(resourcefilepath=resourcefilepath),
    # wasting.Wasting(resourcefilepath=resourcefilepath),

    # - Communicable Diseases
    hiv.Hiv(resourcefilepath=resourcefilepath),
    malaria.Malaria(resourcefilepath=resourcefilepath),
    measles.Measles(resourcefilepath=resourcefilepath),
    # todo - add TB

    # - Non-Communicable Conditions
    # -- Cancers
    bladder_cancer.BladderCancer(resourcefilepath=resourcefilepath),
    breast_cancer.BreastCancer(resourcefilepath=resourcefilepath),
    oesophagealcancer.OesophagealCancer(resourcefilepath=resourcefilepath),
    other_adult_cancers.OtherAdultCancer(resourcefilepath=resourcefilepath),
    prostate_cancer.ProstateCancer(resourcefilepath=resourcefilepath),

    # -- Cardio-metabolic Disorders
    cardio_metabolic_disorders.CardioMetabolicDisorders(resourcefilepath=resourcefilepath),

    # -- Injuries
    rti.RTI(resourcefilepath=resourcefilepath),
    # todo - add RTI when it works reliably

    # -- Other Non-Communicable Conditions
    depression.Depression(resourcefilepath=resourcefilepath),
    epilepsy.Epilepsy(resourcefilepath=resourcefilepath),
)

# create and run the simulation
sim.make_initial_population(n=pop_size)
sim.simulate(end_date=end_date)

# parse the simulation logfile to get the output dataframes
log_df = parse_log_file(sim.log_filepath)


