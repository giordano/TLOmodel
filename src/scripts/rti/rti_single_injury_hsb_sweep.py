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
        self.number_of_samples_in_parameter_range = 11
        self.number_of_draws = self.number_of_samples_in_parameter_range
        self.runs_per_draw = 10

    def log_configuration(self):
        return {
            'filename': 'rti_single_injury_hsb_sweep',
            'directory': './outputs',
            'custom_levels': {
                '*': logging.INFO,
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

# Here I want to run the model with two sets of parameters multiple times. Once where only singular injuries
# are given out and once where we allow multiple injuries

    def draw_parameters(self, draw_number, rng):
        hsb_cutoff_max = 12
        hsb_cutoff_min = 1
        parameter_range = range(hsb_cutoff_min, hsb_cutoff_max)
        return {
            'RTI': {'rt_emergency_care_ISS_score_cut_off': int(parameter_range[draw_number]),
                    'number_of_injured_body_regions_distribution': [[1, 2, 3, 4, 5, 6, 7, 8], [1, 0, 0, 0, 0, 0, 0, 0]]}
            }


if __name__ == '__main__':
    from tlo.cli import scenario_run
    scenario_run([__file__])
