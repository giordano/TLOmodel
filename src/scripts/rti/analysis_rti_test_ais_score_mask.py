import numpy as np

from tlo import Date, logging
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
from tlo.scenario import BaseScenario


class TestScenario(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 12
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2015, 1, 1)
        self.pop_size = 20000
        self.smaller_pop_size = 20000
        self.number_of_draws = 6
        self.runs_per_draw = 2

    def log_configuration(self):
        return {
            'filename': 'analysis_rti_test_ais_score_mask.py',
            'directory': './outputs',
            'custom_levels': {
                '*': logging.INFO,
            }
        }

    def modules(self):
        return [
            demography.Demography(resourcefilepath=self.resources),
            enhanced_lifestyle.Lifestyle(resourcefilepath=self.resources),
            healthsystem.HealthSystem(resourcefilepath=self.resources, service_availability=[]),
            healthburden.HealthBurden(resourcefilepath=self.resources),
            symptommanager.SymptomManager(resourcefilepath=self.resources),
            healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=self.resources),
            rti.RTI(resourcefilepath=self.resources),
            simplified_births.SimplifiedBirths(resourcefilepath=self.resources)
        ]


    def draw_parameters(self, draw_number, rng):
        ais_min = 1
        ais_max = 6
        mask_range = np.linspace(ais_min, ais_max, self.number_of_draws)
        base_rate = 0.00433474669212151
        inc_scale = [0.8173112452023374, 0.8173488047629863, 0.8970126950452928, 0.8813224728274585, 0.9771925938143867,
                     0.846059563757938]
        return {
            'RTI': {'no_med_death_ais_mask': int(mask_range[draw_number]),
                    'base_rate_injrti': base_rate * inc_scale[draw_number]},
            }


if __name__ == '__main__':
    from tlo.cli import scenario_run
    scenario_run([__file__])
