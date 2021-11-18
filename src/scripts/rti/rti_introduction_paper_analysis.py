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
        self.number_of_scenarios = 3
        self.number_of_samples_in_parameter_range = 11
        self.number_of_draws = 11
        self.runs_per_draw = 3

    def log_configuration(self):
        return {
            'filename': 'rti_introduction_paper_analysis',
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
        hsb_cutoff_max = 12
        hsb_cutoff_min = 1
        mult_distribution = [[1, 2, 3, 4, 5, 6, 7, 8],
                             [0.7362050184224718, 0.19421990546958515, 0.051237591074077896, 0.013517104401461814,
                              0.0035659777825202245, 0.0009407486372638106, 0.0002481810186400668,
                              6.547319397917849e-05]]
        sing_distribution = [[1, 2, 3, 4, 5, 6, 7, 8], [1, 0, 0, 0, 0, 0, 0, 0]]
        hsb_parameter_range = range(hsb_cutoff_min, hsb_cutoff_max)
        grid = self.make_grid(
            {'rt_emergency_care_ISS_score_cut_off': hsb_parameter_range,
             'scale_factor': [sing_distribution, mult_distribution]}
        )
        return {
            'RTI': {},
            }


if __name__ == '__main__':
    from tlo.cli import scenario_run
    scenario_run([__file__])
