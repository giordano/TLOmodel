import numpy as np

from tlo import Date, logging
from tlo.methods import (
    demography,
    dx_algorithm_adult,
    dx_algorithm_child,
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
        self.end_date = Date(2020, 1, 1)
        self.pop_size = 20000
        self.smaller_pop_size = 20000
        self.number_of_samples_in_parameter_range = 10
        self.number_of_draws = self.number_of_samples_in_parameter_range
        self.runs_per_draw = 3

    def log_configuration(self):
        return {
            'filename': 'rti_in_hospital_mortality_calibration',
            'directory': './outputs',
            'custom_levels': {
                '*': logging.INFO,
            }
        }

    def modules(self):
        return [
            demography.Demography(resourcefilepath=self.resources),
            dx_algorithm_adult.DxAlgorithmAdult(resourcefilepath=self.resources),
            dx_algorithm_child.DxAlgorithmChild(resourcefilepath=self.resources),
            enhanced_lifestyle.Lifestyle(resourcefilepath=self.resources),
            healthsystem.HealthSystem(resourcefilepath=self.resources, disable=True, service_availability=['*']),
            healthburden.HealthBurden(resourcefilepath=self.resources),
            symptommanager.SymptomManager(resourcefilepath=self.resources),
            healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=self.resources),
            rti.RTI(resourcefilepath=self.resources),
            simplified_births.SimplifiedBirths(resourcefilepath=self.resources)
        ]


    def draw_parameters(self, draw_number, rng):

        multiplication_factor_max = 1.325581395 + 0.25
        multiplication_factor_min = 1.325581395 - 0.25
        scale_factor = np.linspace(multiplication_factor_min, multiplication_factor_max,
                                   num=self.number_of_samples_in_parameter_range)
        return {
            'RTI': {'prob_death_iss_less_than_9': scale_factor[draw_number] * (102 / 11650),
                    'prob_death_iss_10_15': scale_factor[draw_number] * (7 / 528),
                    'prob_death_iss_16_24': scale_factor[draw_number] * (37 / 988),
                    'prob_death_iss_25_35': scale_factor[draw_number] * (52 / 325),
                    'prob_death_iss_35_plus': scale_factor[draw_number] * (37 / 136)},
            }


if __name__ == '__main__':
    from tlo.cli import scenario_run
    scenario_run([__file__])
