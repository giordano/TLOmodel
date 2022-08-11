from pathlib import Path

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
        self.seed = 666
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2011, 1, 1)
        self.pop_size = 100_000

        self.params_of_interest = {'Labour': {'prob_hcw_avail_man_r_placenta': 0.82,
                                              'prob_hcw_avail_blood_tran': 0.86,
                                              'prob_hcw_avail_anticonvulsant': 0.93,
                                              'treatment_effect_modifier_one_delay': 0.75,
                                              'treatment_effect_modifier_all_delays': 0.5,
                                              'mean_hcw_competence_hp': 0.662}}
        self.number_of_params = 6

        # Each parameter in turn will be set each of these values sequentially for a given draw. Any number of values
        # can be set here
        self.values_for_params = [0.0, 0.25, 0.5, 0.75, 1.0]

        # Three draws per parameter for a low, medium, high value plus and additional draw with no parameter changes for
        # comparison
        self.number_of_draws = self.number_of_params * (len(self.values_for_params) + 1)

        self.runs_per_draw = 5
        self.param_df = self._get_param_df()

    def log_configuration(self):
        return {
            'filename': 'max_pregnancy_run_100k', 'directory': './outputs',
            "custom_levels": {
                "*": logging.WARNING,
                "tlo.methods.demography": logging.INFO,
                "tlo.methods.demography.detail": logging.INFO,
                "tlo.methods.contraception": logging.INFO,
                "tlo.methods.healthsystem.summary": logging.INFO,
                "tlo.methods.healthburden": logging.INFO,
                "tlo.methods.labour": logging.INFO,
                "tlo.methods.labour.detail": logging.INFO,
                "tlo.methods.newborn_outcomes": logging.INFO,
                "tlo.methods.care_of_women_during_pregnancy": logging.INFO,
                "tlo.methods.pregnancy_supervisor": logging.INFO,
                "tlo.methods.postnatal_supervisor": logging.INFO,
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
        df = pd.DataFrame(columns=['module', 'parameter', 'value'], index=list(range(self.number_of_draws)))

        pvals = list(self.params_of_interest.values())
        new_list = list()
        for l in range(len(pvals)):
            new_list += pvals[l]

        df['parameter'] = [p for p in new_list for k in range((len(self.values_for_params) + 1))]

        mod_list = list()
        for k in self.params_of_interest:
            for l in range(len(self.params_of_interest[k].keys()) * (len(self.values_for_params) + 1)):
                mod_list.append(k)

        df['module'] = mod_list

        val_column = list()
        for p in new_list:
            nl = [i for i in self.values_for_params]
            nl.append(self.params_of_interest['Labour'][p])
            nl.sort()
            for i in nl:
                val_column.append(i)

        df['value'] = val_column

        return df

    def _update_parameter_dictionary(self, param_dict, draw_number):
        # Update the dictionary of parameters to be returned in draw_parameters. Use the draw number to create
        # a new row from the parameter dictionary

        df = self.param_df

        # todo: fix
        if 'mean_hcw_competence' in df.at[draw_number, 'parameter']:
            p_value = [[df.at[draw_number, 'value'], df.at[draw_number, 'value']],
                       [df.at[draw_number, 'value'], df.at[draw_number, 'value']]]
        else:
            p_value = [df.at[draw_number, 'value'], df.at[draw_number, 'value']]

        new_row = {df.at[draw_number, 'parameter']: p_value}

        if df.at[draw_number, 'module'] in param_dict.keys():
            param_dict[df.at[draw_number, 'module']].update(new_row)
        else:
            param_dict.update({df.at[draw_number, 'module']: new_row})

    def draw_parameters(self, draw_number, rng):
        # For all draws set these parameters so that all women are set to being pregnant from the first day of the sim
        param_dict = {'PregnancySupervisor': {'analysis_year': 2010,
                                              'set_all_pregnant': True}}

        # then for all draws, apart from the last, update the dictionary with the parameter to be changed and its value
        self._update_parameter_dictionary(param_dict, draw_number)

        return param_dict


if __name__ == '__main__':
    from tlo.cli import scenario_run
    scenario_run([__file__])
