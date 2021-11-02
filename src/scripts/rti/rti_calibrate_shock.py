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
        self.number_of_samples_in_parameter_range = 10
        self.number_of_draws = self.number_of_samples_in_parameter_range
        self.runs_per_draw = 3

    def log_configuration(self):
        return {
            'filename': 'rti_calibrate_shock',
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

        prob_shock_min = 0.01
        prob_shock_max = 0.3
        prob_shock_range = np.linspace(prob_shock_min, prob_shock_max,
                                       num=self.number_of_samples_in_parameter_range)
        return {
            'RTI': {'prob_bleeding_leads_to_shock': prob_shock_range[draw_number],}
        }


if __name__ == '__main__':
    from tlo.cli import scenario_run
    scenario_run([__file__])
