"""
This file defines a batch run through which the hiv modules are run across a grid of parameter values

Check the batch configuration gets generated without error:
tlo scenario-run --draw-only src/scripts/hiv/PrEP_analyses/hiv_perfect_prep.py

Test the scenario starts running without problems:
tlo scenario-run src/scripts/hiv/PrEP_analyses/hiv_perfect_prep.py

or execute a single run:
tlo scenario-run src/scripts/hiv/PrEP_analyses/hiv_perfect_prep --draw 1 0

Run on the batch system using:
tlo batch-submit  src/scripts/hiv/PrEP_analyses/hiv_perfect_prep.py

Display information about a job:
tlo batch-job tlo_q1_demo-123 --tasks

Download result files for a completed job:
tlo batch-download tlo_q1_demo-123

full run
hiv_prep_baseline_scenario-2021-08-27T173028Z


"""

import numpy as np

from tlo import Date
from tlo import logging

from tlo.methods import (
    demography,
    enhanced_lifestyle,
    simplified_births,
    dx_algorithm_child,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    symptommanager,
    hiv
)

from tlo.scenario import BaseScenario


class TestScenario(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 24
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2030, 12, 31)
        self.pop_size = 250000
        self.number_of_draws = 1
        self.runs_per_draw = 5

    def log_configuration(self):
        return {
            'filename': 'hiv_prep_baseline_scenario',
            'directory': './outputs',
            'custom_levels': {
                '*': logging.WARNING,
                'tlo.methods.hiv': logging.INFO,
                'tlo.methods.demography': logging.INFO,
                'tlo.scenario': logging.INFO
            }
        }

    def modules(self):
        return [
            demography.Demography(resourcefilepath=self.resources),
            simplified_births.SimplifiedBirths(resourcefilepath=self.resources),
            enhanced_lifestyle.Lifestyle(resourcefilepath=self.resources),
            healthsystem.HealthSystem(
                resourcefilepath=self.resources,
                disable=True,  # no event queueing, run all HSIs
                ignore_cons_constraints=True),
            symptommanager.SymptomManager(resourcefilepath=self.resources),
            healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=self.resources),
            healthburden.HealthBurden(resourcefilepath=self.resources),
            dx_algorithm_child.DxAlgorithmChild(resourcefilepath=self.resources),
            hiv.Hiv(resourcefilepath=self.resources),
        ]

    def draw_parameters(self, draw_number, rng):

        return {
            'Hiv': {
                'prob_prep_for_preg_after_hiv_test': 1.0,
                'prob_prep_high_adherence': 1.0,
                'prob_prep_mid_adherence': 0,
                "prob_for_prep_selection": 1.0,
                "proportion_reduction_in_risk_of_hiv_aq_if_on_prep": 1.0,
                "rr_prep_high_adherence": 0.0,
                "rr_prep_mid_adherence": 0.0,
                "rr_prep_low_adherence": 0.0,
                "probability_of_pregnant_woman_being_retained_on_prep_every_3_months": 1.0,
        }
        }

if __name__ == '__main__':
    from tlo.cli import scenario_run
    scenario_run([__file__])
