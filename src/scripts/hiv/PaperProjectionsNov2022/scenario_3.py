"""
This file defines a batch run through which the hiv and tb modules are run across a grid of parameter values

check the batch configuration gets generated without error:
tlo scenario-run --draw-only src/scripts/hiv/hiv_tb_projections_may2022/scenario0.py

Run on the batch system using:
tlo batch-submit src/scripts/hiv/hiv_tb_projections_may2022/scenario0.py

Display information about a job:
tlo batch-job tlo_q1_demo-123 --tasks

Download result files for a completed job:
tlo batch-download scenario1-2022-04-20T112503Z

9th June
Job ID: scenario0-2022-06-09T170155Z



"""

from random import randint

from tlo import Date, logging
from tlo.methods.fullmodel import fullmodel
from tlo.scenario import BaseScenario


class TestScenario(BaseScenario):
    # this imports the resource filepath automatically

    def __init__(self):
        super().__init__()
        self.seed = 5
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2036, 1, 1)
        self.pop_size = 200000
        self.number_of_draws = 5
        self.runs_per_draw = 10

    def log_configuration(self):
        return {
            "filename": "scenario0",
            "directory": "./outputs",
            "custom_levels": {
                "*": logging.WARNING,
                "tlo.methods.hiv": logging.INFO,
                "tlo.methods.tb": logging.INFO,
                "tlo.methods.demography": logging.INFO,
                "tlo.methods.healthsystem.summary": logging.INFO,
                "tlo.methods.healthburden": logging.INFO,
            },
        }

    def modules(self):
        return fullmodel(
                resourcefilepath=self.resources,
                use_simplified_births=False,
                symptommanager_spurious_symptoms=True,
                healthsystem_disable=False,
                healthsystem_mode_appt_constraints=0,  # no constraints
                healthsystem_cons_availability="default",  # default cons availability
                healthsystem_beds_availability="all",  # all beds always available
                healthsystem_ignore_priority=True,  # ignore priority in HSI scheduling
                healthsystem_use_funded_or_actual_staffing="funded_plus",  # daily capabilities of staff
                healthsystem_capabilities_coefficient=1.0,  # if 'None' set to ratio of init 2010 pop
                healthsystem_record_hsi_event_details=False
            )

    def draw_parameters(self, draw_number, rng):
        return {
            'Tb': {
                'scenario': 3,
                'beta': [0.193668, 0.204125, 0.207443, 0.189727, 0.197228][draw_number]

            },
            'Hiv': {
                'beta': [0.127485, 0.135999, 0.1161, 0.122471, 0.128874][draw_number]
            },
        }


if __name__ == "__main__":
    from tlo.cli import scenario_run

    scenario_run([__file__])
