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
        self.number_of_samples_in_parameter_range = 10
        self.number_of_draws = self.number_of_samples_in_parameter_range
        self.runs_per_draw = 3

    def log_configuration(self):
        return {
            'filename': 'rti_analysis_fit_incidence_of_on_scene.py',
            'directory': './outputs',
            'custom_levels': {
                '*': logging.INFO,
            }
        }

    def modules(self):
        return [
            demography.Demography(resourcefilepath=self.resources),
            enhanced_lifestyle.Lifestyle(resourcefilepath=self.resources),
            healthsystem.HealthSystem(resourcefilepath=self.resources),
            healthburden.HealthBurden(resourcefilepath=self.resources),
            symptommanager.SymptomManager(resourcefilepath=self.resources),
            healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=self.resources),
            rti.RTI(resourcefilepath=self.resources),
            simplified_births.SimplifiedBirths(resourcefilepath=self.resources)
        ]


    def draw_parameters(self, draw_number, rng):

        prob_imm_death_min = 0.068
        prob_imm_death_max = 0.089
        prob_imm_death_range = np.linspace(prob_imm_death_min, prob_imm_death_max,
                                           num=self.number_of_samples_in_parameter_range)
        return {
            'RTI': {'imm_death_proportion_rti': prob_imm_death_range[draw_number],}
        }


if __name__ == '__main__':
    from tlo.cli import scenario_run
    scenario_run([__file__])
