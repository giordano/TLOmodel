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
        self.end_date = Date(2020, 1, 1)
        self.pop_size = 20000
        self.smaller_pop_size = 20000
        self.number_of_samples_in_parameter_range = 6
        self.upper_iss_value = 6
        self.number_of_draws = self.number_of_samples_in_parameter_range * self.upper_iss_value
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
            enhanced_lifestyle.Lifestyle(resourcefilepath=self.resources),
            healthsystem.HealthSystem(resourcefilepath=self.resources, disable=True, service_availability=['*']),
            healthburden.HealthBurden(resourcefilepath=self.resources),
            symptommanager.SymptomManager(resourcefilepath=self.resources),
            healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=self.resources),
            rti.RTI(resourcefilepath=self.resources),
            simplified_births.SimplifiedBirths(resourcefilepath=self.resources)
        ]


    def draw_parameters(self, draw_number, rng):
        hsb_cutoff_max = self.upper_iss_value + 1
        hsb_cutoff_min = 1
        iss_cut_off_scores = range(hsb_cutoff_min, hsb_cutoff_max)
        multiplication_factor_max = 1.625581395 + 0.25
        multiplication_factor_min = 1.625581395 - 0.25
        scale_factor = np.linspace(multiplication_factor_min, multiplication_factor_max,
                                   num=self.number_of_samples_in_parameter_range)
        grid = self.make_grid(
            {'scale_factor': scale_factor,
             'rt_emergency_care_ISS_score_cut_off': iss_cut_off_scores}
        )
        return {
            'RTI': {'prob_death_iss_less_than_9': grid['scale_factor'][draw_number] * (102 / 11650),
                    'prob_death_iss_10_15': grid['scale_factor'][draw_number] * (7 / 528),
                    'prob_death_iss_16_24': grid['scale_factor'][draw_number] * (37 / 988),
                    'prob_death_iss_25_35': grid['scale_factor'][draw_number] * (52 / 325),
                    'prob_death_iss_35_plus': grid['scale_factor'][draw_number] * (37 / 136),
                    'rt_emergency_care_ISS_score_cut_off': int(grid['rt_emergency_care_ISS_score_cut_off'][draw_number])
                    },
            }


if __name__ == '__main__':
    from tlo.cli import scenario_run
    scenario_run([__file__])
