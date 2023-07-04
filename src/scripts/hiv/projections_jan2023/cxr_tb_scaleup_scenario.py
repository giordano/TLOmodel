
"""
This file run scenarios for assesing unavailability of TB-related Development Assistamce for Health (DAH)

It can be submitted on Azure Batch by running:

   tlo batch-submit src/scripts/hiv/projections_jan2023/cxr_tb_scaleup_scenario.py
or locally using:
 tlo scenario-run src/scripts/hiv/projections_jan2023/cxr_tb_scaleup_scenario.py
  execute a single run:
 tlo scenario-run src/scripts/hiv/projections_jan2023/cxr_tb_scaleup_scenario.py --draw 1 0

 check the batch configuration gets generated without error:
tlo scenario-run --draw-only src/scripts/hiv/projections_jan2023/cxr_tb_scaleup_scenario.py

cxr_tb_scaleup_scenario-2023-06-23T213616Z

 """
import warnings
import random
import datetime
from tlo import Date, logging
from tlo.methods import (
    demography,
    simplified_births,
    enhanced_lifestyle,
    healthsystem,
    symptommanager,
    healthseekingbehaviour,
    healthburden,
    epi,
    hiv,
    tb
)
#scenario_start_date = datetime.date(2011, 1, 1)  # Set the scenario start date to January 1, 2011

# scenario_start_date = datetime.date(2011, 1, 1)
# datestamp = scenario_start_date.strftime("__%Y_%m_%d")

from tlo.scenario import BaseScenario

# Ignore warnings to avoid cluttering output from simulation - generally you do not
# need (and generally shouldn't) do this as warnings can contain useful information but
# we will do so here for the purposes of this example to keep things simple.
warnings.simplefilter("ignore", (UserWarning, RuntimeWarning))
class ImpactOfCxRScaleup(BaseScenario):
    def __init__(self):
        super().__init__(
            seed=random.randint(0, 50000),
            start_date=Date(2010, 1, 1),
            end_date=Date(2013, 12, 31),
            initial_population_size=1000,
            number_of_draws=1,
            runs_per_draw=2,
        )
    def log_configuration(self):
        return {
            'filename': 'cxrscaleup_scenario',
            'directory': './outputs/nic503@york.ac.uk',
            'custom_levels': {
                '*': logging.WARNING,
                'tlo.methods.demography': logging.INFO,
                'tlo.methods.population': logging.INFO,
                'tlo.methods.healthburden': logging.INFO,
                'tlo.methods.tb': logging.INFO,
                'tlo.methods.hiv': logging.INFO,
                'tlo.methods.healthsystem.summary': logging.INFO,
            },
        }
    def modules(self):
        return [
            demography.Demography(resourcefilepath=self.resources),
            simplified_births.SimplifiedBirths(resourcefilepath=self.resources),
            enhanced_lifestyle.Lifestyle(resourcefilepath=self.resources),
            healthsystem.HealthSystem(
                resourcefilepath=self.resources,
                service_availability=["*"],  # all treatment allowed
                mode_appt_constraints=0,  # mode of constraints to do with officer numbers and time
                cons_availability="default",  # mode for consumable constraints (if ignored, all consumables available)
                ignore_priority=False,  # do not use the priority information in HSI event to schedule
                capabilities_coefficient=1.0,  # multiplier for the capabilities of health officers
                use_funded_or_actual_staffing="funded_plus",
                # actual: use numbers/distribution of staff available currently
                disable=False,  # disables the healthsystem (no constraints and no logging) and every HSI runs
                disable_and_reject_all=False  # disable healthsystem and no HSI runs
            ),
            symptommanager.SymptomManager(resourcefilepath=self.resources),
            healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=self.resources),
            healthburden.HealthBurden(resourcefilepath=self.resources),
            epi.Epi(resourcefilepath=self.resources),
            hiv.Hiv(resourcefilepath=self.resources, run_with_checks=False),
            tb.Tb(resourcefilepath=self.resources),
        ]
    def draw_parameters(self, draw_number, rng):
        return {
            'Tb': {
                'scenario': 0,
                'probability_access_to_xray':  0.11
            },
            'HealthSystem': {
                'cons_availability': 'Item_Available'[draw_number],
            },
        }
if __name__ == '__main__':
    from tlo.cli import scenario_run
    scenario_run([__file__])

