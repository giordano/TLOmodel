import numpy as np
import pandas as pd

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
        self.number_of_draws = self.number_of_samples_in_parameter_range
        self.runs_per_draw = 3

    def log_configuration(self):
        return {
            'filename': 'rti_analysis_calibrate_demographics.py',
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
        # 'rr_injrti_male'
        rr_male_lower = 1.1
        rr_male_upper = 2.1
        rr_range = np.linspace(rr_male_lower, rr_male_upper, num=self.number_of_draws)
        scale_to_age_dist = [0.2910658429014334, 0.7884210773764305, 1.0082309864810863, 1.1406954983602067,
                             0.9774304956506041, 0.8032970118799958, 0.7225489345554275, 0.7815129387736788,
                             0.9400591325082066, 0.9284209110664802, 1.0176103277218207, 1.3066486384947533,
                             1.6984139722946205, 2.316393313056162, 2.784439793835679, 4.226388037181285,
                             8.226983850544235, 25.21856019240709, 88.08030481663182, np.inf]
        age_ranges = ['1 to 4', '5 to 9', '10 to 14', '15 to 19', '20 to 24', '25 to 29', '30 to 34', '35 to 39',
                      '40 to 44', '45 to 49', '50 to 54', '55 to 59', '60 to 64', '65 to 69', '70 to 74', '75 to 79',
                      '80 to 84', '85 to 89', '90 to 94', '95 plus']
        age_scale_df = pd.DataFrame(data=scale_to_age_dist, index=age_ranges)

        return {
            'RTI': {'rr_injrti_age04': age_scale_df.loc['1 to 4'][0] * 0.5,
                    'rr_injrti_age59': age_scale_df.loc['5 to 9'][0] * 0.7,
                    'rr_injrti_age1017': ((age_scale_df.loc['10 to 14'][0] + age_scale_df.loc['15 to 19'][0]) / 2) * 0.9,
                    'rr_injrti_age1829': ((age_scale_df.loc['20 to 24'][0] + age_scale_df.loc['25 to 29'][0]) / 2) * 1.33,
                    'rr_injrti_age3039': ((age_scale_df.loc['30 to 34'][0] + age_scale_df.loc['35 to 39'][0]) / 2) * 1.4,
                    'rr_injrti_age4049': ((age_scale_df.loc['40 to 44'][0] + age_scale_df.loc['45 to 49'][0]) / 2) * 1.15,
                    'rr_injrti_age5059': ((age_scale_df.loc['50 to 54'][0] + age_scale_df.loc['55 to 59'][0]) / 2) * 1.15,
                    'rr_injrti_age6069': ((age_scale_df.loc['60 to 64'][0] + age_scale_df.loc['65 to 69'][0]) / 2) * 1.15,
                    'rr_injrti_age7079': ((age_scale_df.loc['70 to 74'][0] + age_scale_df.loc['75 to 79'][0]) / 2) * 1.15,
                    'rr_injrti_male': rr_range[draw_number]
                    },
            }


if __name__ == '__main__':
    from tlo.cli import scenario_run
    scenario_run([__file__])
