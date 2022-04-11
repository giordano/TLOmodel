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


class TestScenario(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 333
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2011, 1, 1)
        self.pop_size = 1000
        self.number_of_draws = 1
        self.runs_per_draw = 1

    def log_configuration(self):
        return {
            'filename': 'baseline_scenario_20k', 'directory': './outputs',
            "custom_levels": {  # Customise the output of specific loggers. They are applied in order:
                "*": logging.WARNING,
                "tlo.methods.demography": logging.INFO,
                "tlo.methods.contraception": logging.INFO,
                "tlo.methods.healthsystem": logging.INFO,
                "tlo.methods.healthburden": logging.INFO,
                "tlo.methods.labour": logging.INFO,
                "tlo.methods.newborn_outcomes": logging.INFO,
                "tlo.methods.care_of_women_during_pregnancy": logging.INFO,
                "tlo.methods.pregnancy_supervisor": logging.INFO,
                "tlo.methods.postnatal_supervisor": logging.INFO,
            }
        }

    def modules(self):
        return [
            demography.Demography(resourcefilepath=self.resources),
            contraception.Contraception(resourcefilepath=self.resources),
            enhanced_lifestyle.Lifestyle(resourcefilepath=self.resources),
            healthburden.HealthBurden(resourcefilepath=self.resources),
            healthsystem.HealthSystem(resourcefilepath=self.resources,
                                      mode_appt_constraints=1,
                                      cons_availability='default'),
            symptommanager.SymptomManager(resourcefilepath=self.resources),
            healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=self.resources),
            hiv.DummyHivModule(),
            pregnancy_supervisor.PregnancySupervisor(resourcefilepath=self.resources),
            care_of_women_during_pregnancy.CareOfWomenDuringPregnancy(resourcefilepath=self.resources),
            labour.Labour(resourcefilepath=self.resources),
            postnatal_supervisor.PostnatalSupervisor(resourcefilepath=self.resources),
            newborn_outcomes.NewbornOutcomes(resourcefilepath=self.resources),
        ]

    def draw_parameters(self, draw_number, rng):
        return {
            'Labour': {'activate_perfect_bemonc': True,
                       'activate_perfect_cemonc': True}
        }



if __name__ == '__main__':
    from tlo.cli import scenario_run
    scenario_run([__file__])
