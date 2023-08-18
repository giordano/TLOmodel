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
from tlo.methods import hiv_tb_calibration
from tlo.methods.fullmodel import fullmodel
from tlo.scenario import BaseScenario

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
        ]

    def draw_parameters(self, draw_number, rng):
        return {
            'CareOfWomenDuringPregnancy': {
                'prep_for_pregnant_woman_start_year': [2036,"default","default","default"][draw_number]

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
