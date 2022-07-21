import pandas as pd
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


class UnivariateSensitivityAnalysis(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 123
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2011, 1, 1)
        self.pop_size = 100_000
        self.params_of_interest = {'PregnancySupervisor': ['treatment_effect_modifier_all_delays',
                                                           'treatment_effect_modifier_one_delay'],

                                   'Labour': ['prob_haemostatis_uterotonics',
                                              'prob_successful_manual_removal_placenta',
                                              'pph_treatment_effect_mrp_md',
                                              'success_rate_pph_surgery',
                                              'pph_treatment_effect_surg_md',
                                              'pph_treatment_effect_hyst_md',
                                              'pph_bt_treatment_effect_md',

                                              'sepsis_treatment_effect_md',

                                              'eclampsia_treatment_effect_severe_pe',
                                              'eclampsia_treatment_effect_md',
                                              'anti_htns_treatment_effect_md',

                                              'prob_hcw_avail_uterotonic',
                                              'prob_hcw_avail_man_r_placenta',
                                              'prob_hcw_avail_blood_tran',

                                              'prob_hcw_avail_iv_abx',

                                              'prob_hcw_avail_anticonvulsant',

                                              'treatment_effect_modifier_one_delay',
                                              'treatment_effect_modifier_all_delays',
                                              'mean_hcw_competence_hc',
                                              'mean_hcw_competence_hp']}

        # Each parameter in turn will be set each of these values sequentially for a given draw. Any number of values
        # can be set here
        self.values_for_params = [0.0, 0.5, 1.0]

        # Three draws per parameter for a low, medium, high value plus and additional draw with no parameter changes for
        # comparison
        self.number_of_draws = \
            (sum(len(l) for l in self.params_of_interest.values()) * (len(self.values_for_params))) + 1

        self.runs_per_draw = 5
        self.param_df = self._get_param_df()

    def log_configuration(self):
        return {
            'filename': 'max_pregnancy_run_100k', 'directory': './outputs',
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

    def _get_param_df(self):

        # create DF with the draw number as the index and the row containing the module, parameter and value to be
        # changed for that run
        df = pd.DataFrame(columns=['module', 'parameter', 'value'], index=list(range(self.number_of_draws - 1)))
        df['value'] = self.values_for_params * int(((self.number_of_draws - 1) / (len(self.values_for_params))))

        pvals = list(self.params_of_interest.values())
        new_list = list()
        for l in range(len(pvals)):
            new_list += pvals[l]

        df['parameter'] = [p for p in new_list for k in range((len(self.values_for_params)))]

        mod_list = list()
        for k in self.params_of_interest:
            for l in range(len(self.params_of_interest[k] * (len(self.values_for_params)))):
                mod_list.append(k)

        df['module'] = mod_list

        return df

    def _update_parameter_dictionary(self, param_dict, draw_number):
        # Update the dictionary of parameters to be returned in draw_parameters. Use the draw number to create
        # a new row from the parameter dictionary

        df = self.param_df
        new_row = {df.at[draw_number, 'parameter']: [df.at[draw_number, 'value'],
                                                     df.at[draw_number, 'value']]}

        if df.at[draw_number, 'module'] in param_dict.keys():
            param_dict[df.at[draw_number, 'module']].update(new_row)
        else:
            param_dict.update({df.at[draw_number, 'module']: new_row})

    def draw_parameters(self, draw_number, rng):
        # For all draws set these parameters so that all women are set to being pregnant from the first day of the sim
        param_dict = {'PregnancySupervisor': {'analysis_year': 2010,
                                              'set_all_pregnant': True}}

        # then for all draws, apart from the last, update the dictionary with the parameter to be changed and its value
        if not draw_number == list(range(self.number_of_draws))[-1]:
            self._update_parameter_dictionary(param_dict, draw_number)

        return param_dict


if __name__ == '__main__':
    from tlo.cli import scenario_run
    scenario_run([__file__])
