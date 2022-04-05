from random import randint

from tlo import Date, logging

from tlo.methods import (
    demography,
    enhanced_lifestyle,
    epi,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    hiv,
    simplified_births,
    symptommanager,
    tb,
)

from tlo.scenario import BaseScenario


class TestTbShineBaselineScenario(BaseScenario):

    def __init__(self):
        super().__init__()
        self.seed = randint(0, 5000)
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2035, 1, 1)
        self.pop_size = 250_000
        self.number_of_draws = 1
        self.runs_per_draw = 5

    def log_configuration(self):
        return {
            'filename': 'tb_shine_baseline_scenario',
            'directory': './outputs',
            'custom_levels': {
                '*': logging.WARNING,
                'tlo.methods.hiv': logging.INFO,
                'tlo.methods.tb': logging.INFO,
                'tlo.methods.demography': logging.INFO,
            }
        }

    def modules(self):
        return [
            demography.Demography(resourcefilepath=self.resources, max_age_initial=16),
            simplified_births.SimplifiedBirths(resourcefilepath=self.resources),
            enhanced_lifestyle.Lifestyle(resourcefilepath=self.resources),
            healthsystem.HealthSystem(
                resourcefilepath=self.resources,
                service_availability=["*"],
                mode_appt_constraints=0,
                cons_availability="all",
                ignore_priority=True,
                capabilities_coefficient=1.0,
                disable=False,
                disable_and_reject_all=False,
                store_hsi_events_that_have_run=False
            ),
            symptommanager.SymptomManager(resourcefilepath=self.resources),
            healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=self.resources),
            healthburden.HealthBurden(resourcefilepath=self.resources),
            epi.Epi(resourcefilepath=self.resources),
            hiv.Hiv(resourcefilepath=self.resources),
            tb.Tb(resourcefilepath=self.resources),
        ]

    def draw_parameters(self, draw_number, rng):
        return {
            'Tb': {'scenario': 0}
        }


if __name__ == '__main__':
    from tlo.cli import scenario_run

    scenario_run([__file__])
