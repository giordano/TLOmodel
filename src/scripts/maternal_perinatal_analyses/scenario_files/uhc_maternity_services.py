from tlo import Date, logging
from tlo.methods.fullmodel import fullmodel
from tlo.scenario import BaseScenario


class UHCMaternityServices(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 537184
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2031, 1, 1)
        self.pop_size = 250_000
        self.number_of_draws = 1
        self.runs_per_draw = 20

    def log_configuration(self):
        return {
            'filename': 'uhc_250k', 'directory': './outputs',
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
        return fullmodel(
            resourcefilepath=self.resources,
            module_kwargs={"HealthSystem": {"mode_appt_constraints": 1,
                                            'cons_availability': 'default'}}
        )

    def draw_parameters(self, draw_number, rng):
        return {
            'Labour': {'alternative_bemonc_availability': True,
                       'alternative_cemonc_availability': True,
                       'bemonc_availability': 0.9,
                       'cemonc_availability': 0.9,
                       'bemonc_cons_availability': 1.0,
                       'cemonc_cons_availability': 1.0,
                       'alternative_pnc_coverage': True,
                       'alternative_pnc_quality': True,
                       'pnc_availability_odds': 15.0,
                       'pnc_availability_probability': 1.0,
                       'analysis_year': 2023},

            'PregnancySupervisor': {'alternative_anc_coverage': True,
                                    'alternative_anc_quality': True,
                                    'alternative_ip_anc_quality': True,
                                    'anc_availability_odds': 9.0,
                                    'anc_availability_probability': 1.0,
                                    'ip_anc_availability_probability': 1.0,
                                    'analysis_year': 2023}

        }


if __name__ == '__main__':
    from tlo.cli import scenario_run
    scenario_run([__file__])
