import numpy as np
from tlo import Date, logging
from tlo.methods import (
    care_of_women_during_pregnancy,
    contraception,
    demography,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    hiv,
    labour,
    newborn_outcomes,
    postnatal_supervisor,
    pregnancy_supervisor,
    symptommanager,
)
from tlo.scenario import BaseScenario


class MaxPregnancyRun(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 123
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2011, 1, 1)
        self.pop_size = 1000
        self.number_of_draws = 3
        self.runs_per_draw = 5

    def log_configuration(self):
        return {
            'filename': 'max_pregnancy_run_250k', 'directory': './outputs',
            "custom_levels": {
                "*": logging.WARNING,
                "tlo.methods.demography": logging.INFO,
                "tlo.methods.demography.detail": logging.INFO,
                "tlo.methods.depression": logging.INFO,
                "tlo.methods.contraception": logging.INFO,
                "tlo.methods.healthsystem.summary": logging.INFO,
                "tlo.methods.healthburden": logging.INFO,
                "tlo.methods.hiv": logging.INFO,
                "tlo.methods.labour": logging.INFO,
                "tlo.methods.labour.detail": logging.INFO,
                "tlo.methods.malaria": logging.INFO,
                "tlo.methods.newborn_outcomes": logging.INFO,
                "tlo.methods.care_of_women_during_pregnancy": logging.INFO,
                "tlo.methods.pregnancy_supervisor": logging.INFO,
                "tlo.methods.postnatal_supervisor": logging.INFO,
                "tlo.methods.tb": logging.INFO,
            }
        }

    def modules(self):
        return [demography.Demography(resourcefilepath=self.resources),
                contraception.Contraception(resourcefilepath=self.resources),
                enhanced_lifestyle.Lifestyle(resourcefilepath=self.resources),
                healthburden.HealthBurden(resourcefilepath=self.resources),
                symptommanager.SymptomManager(resourcefilepath=self.resources),
                healthsystem.HealthSystem(resourcefilepath=self.resources,
                                          mode_appt_constraints=1,
                                          cons_availability='default'),
                newborn_outcomes.NewbornOutcomes(resourcefilepath=self.resources),
                pregnancy_supervisor.PregnancySupervisor(resourcefilepath=self.resources),
                care_of_women_during_pregnancy.CareOfWomenDuringPregnancy(resourcefilepath=self.resources),
                labour.Labour(resourcefilepath=self.resources),
                postnatal_supervisor.PostnatalSupervisor(resourcefilepath=self.resources),
                healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=self.resources),
                hiv.DummyHivModule(),

                ]

    def draw_parameters(self, draw_number, rng):

        grid = self.make_grid(
            {'prob_delay_one_two_fd': [[0.0, 0.0], [1.0, 1.0], [0.38, 0.38]]})

        return {'PregnancySupervisor': {'analysis_year': 2010,
                                        'set_all_pregnant': True},

                'Labour': {'prob_delay_one_two_fd': grid['prob_delay_one_two_fd'][draw_number]}

        }


if __name__ == '__main__':
    from tlo.cli import scenario_run
    scenario_run([__file__])
