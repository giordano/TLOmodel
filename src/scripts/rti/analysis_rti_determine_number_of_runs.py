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
        self.number_of_draws = 1
        self.runs_per_draw = 20

    def log_configuration(self):
        return {
            'filename': 'analysis_rti_determine_number_of_runs.py',
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

        return {
            'RTI': {'prob_death_iss_less_than_9': 0.00733191462346883,
                    'prob_death_iss_10_15': 0.0111021545889758,
                    'prob_death_iss_16_24': 0.0313608577284893,
                    'prob_death_iss_25_35': 0.133987145668097,
                    'prob_death_iss_35_plus': 0.22782740761579,
                    'rt_emergency_care_ISS_score_cut_off': 5,
                    'number_of_injured_body_regions_distribution': [[1, 2, 3, 4, 5, 6, 7, 8],
                                                                    [0.7362050184224718, 0.19421990546958515,
                                                                     0.051237591074077896, 0.013517104401461814,
                                                                     0.0035659777825202245, 0.0009407486372638106,
                                                                     0.0002481810186400668, 6.547319397917849e-05]],
                    'base_rate_injrti': 0.00420322902234662,
                    }
            }


if __name__ == '__main__':
    from tlo.cli import scenario_run
    scenario_run([__file__])
