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
        self.upper_iss_value = 6
        self.number_of_draws = 6
        self.runs_per_draw = 4

    def log_configuration(self):
        return {
            'filename': 'rti_calibrated_single_injury_model_run.py',
            'directory': './outputs',
            'custom_levels': {
                "*": logging.WARNING,
                "tlo.methods.demography": logging.INFO,
                "tlo.methods.healthsystem": logging.WARNING,
                "tlo.methods.healthburden": logging.INFO,
                "tlo.methods.rti": logging.INFO,
            }
        }

    def modules(self):
        return [
            demography.Demography(resourcefilepath=self.resources),
            enhanced_lifestyle.Lifestyle(resourcefilepath=self.resources),
            healthsystem.HealthSystem(resourcefilepath=self.resources, service_availability=['*']),
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
        parameter_df = pd.DataFrame()
        parameter_df['rt_emergency_care_ISS_score_cut_off'] = iss_cut_off_scores
        ninj = [1, 2, 3, 4, 5, 6, 7, 8]
        injury_distributions = [[ninj, [1, 0, 0, 0, 0, 0, 0, 0]]] * len(parameter_df)
        parameter_df['number_of_injured_body_regions_distribution'] = injury_distributions
        parameter_df['scale_factor'] = [1.27210261, 0.97760263, 0.91028567, 0.96303679, 0.82095107, 0.71205461]
        scale_for_inc = [1.02456099, 1.02166648, 1.02467306, 1.00992407, 0.99241982, 0.99833061]
        current_inc = 0.00715091242587118
        parameter_df['base_rate_injrti'] = np.multiply(current_inc, scale_for_inc)
        return {
            'RTI': {'prob_death_iss_less_than_9': parameter_df['scale_factor'][draw_number] * (102 / 11650),
                    'prob_death_iss_10_15': parameter_df['scale_factor'][draw_number] * (7 / 528),
                    'prob_death_iss_16_24': parameter_df['scale_factor'][draw_number] * (37 / 988),
                    'prob_death_iss_25_35': parameter_df['scale_factor'][draw_number] * (52 / 325),
                    'prob_death_iss_35_plus': parameter_df['scale_factor'][draw_number] * (37 / 136),
                    'rt_emergency_care_ISS_score_cut_off': int(parameter_df['rt_emergency_care_ISS_score_cut_off'][
                                                                   draw_number]),
                    'number_of_injured_body_regions_distribution':
                        parameter_df['number_of_injured_body_regions_distribution'][draw_number],
                    'base_rate_injrti': parameter_df['base_rate_injrti'][draw_number]},
            }


if __name__ == '__main__':
    from tlo.cli import scenario_run
    scenario_run([__file__])
