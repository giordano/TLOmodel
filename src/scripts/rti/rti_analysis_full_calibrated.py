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
        self.upper_iss_value = 10
        self.number_of_draws = 10
        self.runs_per_draw = 4

    def log_configuration(self):
        return {
            'filename': 'rti_calibrated_model_run.py',
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
        injury_distributions = \
            [[ninj, [0.7093834795440467, 0.2061841889568684, 0.05992798112965075, 0.017418226588785873,
                     0.00506265373502088, 0.0014714737295484586, 0.00042768774047753785, 0.00012430857560121887]],
             [ninj, [0.7093834795440467, 0.2061841889568684, 0.05992798112965075, 0.017418226588785873,
                     0.00506265373502088, 0.0014714737295484586, 0.00042768774047753785, 0.00012430857560121887]],
             [ninj,  [0.7093834795440467, 0.2061841889568684, 0.05992798112965075, 0.017418226588785873,
                      0.00506265373502088, 0.0014714737295484586, 0.00042768774047753785, 0.00012430857560121887]],
             [ninj,  [0.7093834795440467, 0.2061841889568684, 0.05992798112965075, 0.017418226588785873,
                      0.00506265373502088, 0.0014714737295484586, 0.00042768774047753785, 0.00012430857560121887]],
             [ninj, [0.7362050184224718, 0.19421990546958515, 0.051237591074077896, 0.013517104401461814,
                     0.0035659777825202245, 0.0009407486372638106, 0.0002481810186400668, 6.547319397917849e-05]],
             [ninj, [0.7093834795440467, 0.2061841889568684, 0.05992798112965075, 0.017418226588785873,
                     0.00506265373502088, 0.0014714737295484586, 0.00042768774047753785, 0.00012430857560121887]],
             [ninj, [0.7093834795440467, 0.2061841889568684, 0.05992798112965075, 0.017418226588785873,
                     0.00506265373502088, 0.0014714737295484586, 0.00042768774047753785, 0.00012430857560121887]],
             [ninj, [0.7362050184224718, 0.19421990546958515, 0.051237591074077896, 0.013517104401461814,
                     0.0035659777825202245, 0.0009407486372638106, 0.0002481810186400668, 6.547319397917849e-05]],
             [ninj, [0.7093834795440467, 0.2061841889568684, 0.05992798112965075, 0.017418226588785873,
                     0.00506265373502088, 0.0014714737295484586, 0.00042768774047753785, 0.00012430857560121887]],
             [ninj, [0.7362050184224718, 0.19421990546958515, 0.051237591074077896, 0.013517104401461814,
                     0.0035659777825202245, 0.0009407486372638106, 0.0002481810186400668, 6.547319397917849e-05]],
             ]

        parameter_df['number_of_injured_body_regions_distribution'] = injury_distributions
        resulting_ihm = [0.018734551, 0.019035614, 0.026995837, 0.024213747, 0.016432719, 0.028644271, 0.022180777]
        target_ihm = 144 / 7416
        scale_to_ihm = np.divide(target_ihm, resulting_ihm)
        parameter_df['scale_factor'] = [1.27210261, 0.97760263, 0.91028567, 0.96303679 * 1.03645269,
                                        0.82095107 * 1.02006038, 0.71205461 * 0.71927667,
                                        0.71205461 * 0.80191949, 0.71205461 * 1.18163499, 0.71205461 * 0.6778834,
                                        0.71205461 * 0.8754191]
        # scale_for_inc = np.multiply([1.02456099, 1.02166648, 1.02467306, 1.00992407, 0.99241982, 0.99833061],
        #                             0.6129328641632665)
        new_inc = [0.004364235370981745, 0.004334746692121514, 0.004366273338474707, 0.004268219608865612,
                   0.0040991908466201646, 0.004070670102910139, 0.004070670102910139, 0.004070670102910139,
                   0.004070670102910139, 0.004070670102910139]
        new_inc = np.multiply(new_inc, [1.00357514, 1.00616128, 0.97546409, 1.01629594, 1.02107068, 1.02230112,
                                        1.02077057, 1.05356021, 1.02615006, 1.03587243])
        output = [930.90891369, 944.79547439, 953.95375283, 962.58209307, 945.64587678, 947.85714176, 943.57801207,
                  960.70361922, 945.81425988, 973.11198994]
        rescale = np.divide(952.2, output)
        new_inc = np.multiply(new_inc, rescale)

        # current_inc = 0.00715091242587118
        parameter_df['base_rate_injrti'] = new_inc
        parameter_df['imm_death_proportion_rti'] = [0.007] * len(new_inc)
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
                    'base_rate_injrti': parameter_df['base_rate_injrti'][draw_number],
                    'imm_death_proportion_rti': parameter_df['imm_death_proportion_rti'][draw_number]
                    }
            }


if __name__ == '__main__':
    from tlo.cli import scenario_run
    scenario_run([__file__])
