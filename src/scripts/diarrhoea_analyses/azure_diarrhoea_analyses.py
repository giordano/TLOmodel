from tlo import Date
from tlo import logging
from tlo.scenario import BaseScenario
from tlo.methods import (demography, enhanced_lifestyle, dx_algorithm_child, healthseekingbehaviour,
                         healthsystem, healthburden, simplified_births, diarrhoea, symptommanager)


class MyTestScenario(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 12
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2020, 1, 1)
        self.pop_size = 1000
        self.number_of_draws = 2
        self.runs_per_draw = 2

    def log_configuration(self):
        return {
            'filename': 'diarrhoea_batch',
            'directory': './outputs',
            'custom_levels': {'*': logging.INFO}
        }

    def modules(self):
        return [
            demography.Demography(resourcefilepath=self.resources),
            enhanced_lifestyle.Lifestyle(resourcefilepath=self.resources),
            healthsystem.HealthSystem(resourcefilepath=self.resources, disable=True, service_availability=['*']),
            symptommanager.SymptomManager(resourcefilepath=self.resources),
            healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=self.resources),
            healthburden.HealthBurden(resourcefilepath=self.resources),
            simplified_births.SimplifiedBirths(resourcefilepath=self.resources),
            diarrhoea.Diarrhoea(resourcefilepath=self.resources),
            dx_algorithm_child.DxAlgorithmChild(resourcefilepath=self.resources)
        ]

    def draw_parameters(self, draw_number, rng):
        return {
            'Diarrhoea': {
                # 'init_p_urban': rng.randint(10, 20) / 100.0,
            }
        }
