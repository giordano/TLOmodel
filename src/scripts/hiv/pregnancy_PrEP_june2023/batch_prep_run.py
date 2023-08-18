"""
This file defines a batch run through which the hiv and tb modules are run across a set of parameter values

conda activate TLOmodel
find ~/Documents/MSc_HDA/Summer/TLOmodel -name 'batch_prep_runs.py'
tlo scenario-run --draw-only batch_prep_run.py

check the batch configuration gets generated without error:
tlo scenario-run --draw-only src/scripts/hiv/pregnancy_PrEP_june2023/batch_prep_run.py

Test the scenario starts running without problems:
tlo scenario-run src/scripts/hiv/pregnancy_PrEP_june2023/batch_prep_runs.py

or execute a single run:
tlo scenario-run src/scripts/hiv/trial_run.py --draw 1 0

Run on the batch system using:
tlo batch-submit ssrc/scripts/hiv/projections_jan2023/batch_test_runs.py

Display information about a job:
tlo batch-job tlo_q1_demo-123 --tasks

Download result files for a completed job:
tlo batch-download calibration_script-2022-04-12T190518Z

"""

import warnings

from tlo import Date, logging
from tlo.scenario import BaseScenario
# you need to import each module that you require
from tlo.methods import (
    demography,
    contraception,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    symptommanager,
    epi,
    hiv,
    tb,
    newborn_outcomes,
    pregnancy_supervisor,
    care_of_women_during_pregnancy,
    labour,
    postnatal_supervisor,
)

# Ignore warnings to avoid cluttering output from simulation
warnings.simplefilter("ignore", (UserWarning, RuntimeWarning))


class TestScenario(BaseScenario):
    # this imports the resource filepath automatically

    def __init__(self):
        super().__init__(
            seed=1,
            start_date=Date(2010, 1, 1),
            end_date=Date(2036, 1, 1),
            initial_population_size=50000,
            number_of_draws=4,
            runs_per_draw=5,
        )

    def log_configuration(self):
        return {
            'filename': 'test_runs',
            'directory': './outputs',
            'custom_levels': {
                '*': logging.WARNING,
                "tlo.methods.hiv": logging.INFO,
                "tlo.methods.demography": logging.INFO,
                "tlo.methods.care_of_women_during_pregnancy": logging.INFO,
                "tlo.methods.newborn_outcomes": logging.INFO,
                "tlo.methods.healthsystem.summary": logging.INFO,
                "tlo.methods.healthburden": logging.INFO,
            }
        }

    def modules(self):
        return [
            epi.Epi(resourcefilepath=self.resources),
            demography.Demography(resourcefilepath=self.resources),
            contraception.Contraception(resourcefilepath=self.resources),
            enhanced_lifestyle.Lifestyle(resourcefilepath=self.resources),
            healthburden.HealthBurden(resourcefilepath=self.resources),
            symptommanager.SymptomManager(resourcefilepath=self.resources),
            healthsystem.HealthSystem(resourcefilepath=self.resources,
                                      service_availability=["*"],  # all treatment allowed
                                      mode_appt_constraints=0,
                                      cons_availability="default",
                                      ignore_priority=False,
                                      capabilities_coefficient=1.0,
                                      use_funded_or_actual_staffing="funded_plus",
                                      disable=False,
                                      disable_and_reject_all=False,  # disable healthsystem and no HSI runs
                                      ),
            newborn_outcomes.NewbornOutcomes(resourcefilepath=self.resources),
            pregnancy_supervisor.PregnancySupervisor(resourcefilepath=self.resources),
            care_of_women_during_pregnancy.CareOfWomenDuringPregnancy(resourcefilepath=self.resources),
            labour.Labour(resourcefilepath=self.resources),
            postnatal_supervisor.PostnatalSupervisor(resourcefilepath=self.resources),
            healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=self.resources),
            hiv.Hiv(resourcefilepath=self.resources),
            tb.Tb(resourcefilepath=self.resources),
        ]

    # todo you can't use the term "default" in a list here, you need the actual value
    # for probability_of_prep_consumables_being_available, you can use 0.85 as a value which is close to default
    # are you sure you don't want prep to start until 2036?? that means no prep as your simulation ends in 2036
    def draw_parameters(self, draw_number, rng):
        return {
            'CareOfWomenDuringPregnancy': {
                'prep_for_pregnant_woman_start_year': [2036, "default", "default", "default"][draw_number]

            },
            'Hiv': {
                'probability_of_being_retained_on_prep_every_1_month': [1, 1, "default", 1][draw_number],
                'probability_of_being_retained_on_prep_every_1_month_high': [1, 1, "default", 1][draw_number],
                'probability_of_being_retained_on_prep_every_1_month_low': [1, 1, "default", 1][draw_number],
                'probability_of_prep_consumables_being_available': ["default", "default", "default", 0.8][draw_number]
            },
        }


if __name__ == '__main__':
    from tlo.cli import scenario_run

    scenario_run([__file__])
