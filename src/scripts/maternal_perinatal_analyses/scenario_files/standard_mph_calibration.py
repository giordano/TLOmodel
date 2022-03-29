from tlo import Date, logging
from tlo.methods import (
    alri,
    cardio_metabolic_disorders,
    care_of_women_during_pregnancy,
    contraception,
    demography,
    depression,
    diarrhoea,
    enhanced_lifestyle,
    epi,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    hiv,
    labour,
    malaria,
    newborn_outcomes,
    postnatal_supervisor,
    pregnancy_supervisor,
    stunting,
    symptommanager,
    wasting
)
from tlo.scenario import BaseScenario


class TestScenario(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 5596
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2021, 1, 2)
        self.pop_size = 30000
        self.number_of_draws = 1
        self.runs_per_draw = 5

    def log_configuration(self):
        return {
            'filename': 'all_modules_15k',
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
            symptommanager.SymptomManager(resourcefilepath=self.resources),
            healthsystem.HealthSystem(resourcefilepath=self.resources,
                                      mode_appt_constraints=1,
                                      cons_availability='default'),  # went set disable=true, cant check HSI queue
            newborn_outcomes.NewbornOutcomes(resourcefilepath=self.resources),
            pregnancy_supervisor.PregnancySupervisor(resourcefilepath=self.resources),
            care_of_women_during_pregnancy.CareOfWomenDuringPregnancy(resourcefilepath=self.resources),
            labour.Labour(resourcefilepath=self.resources),
            postnatal_supervisor.PostnatalSupervisor(resourcefilepath=self.resources),
            healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=self.resources),

            # Register all the modules that are reference in the maternal perinatal health suite (including their
            # dependencies)
            alri.Alri(resourcefilepath=self.resources),
            hiv.Hiv(resourcefilepath=self.resources),
            malaria.Malaria(resourcefilepath=self.resources),
            cardio_metabolic_disorders.CardioMetabolicDisorders(resourcefilepath=self.resources),
            depression.Depression(resourcefilepath=self.resources),
            stunting.Stunting(resourcefilepath=self.resources),
            wasting.Wasting(resourcefilepath=self.resources),
            diarrhoea.Diarrhoea(resourcefilepath=self.resources),
            epi.Epi(resourcefilepath=self.resources)
        ]

    def draw_parameters(self, draw_number, rng):
        return {
        }


if __name__ == '__main__':
    from tlo.cli import scenario_run
    scenario_run([__file__])
