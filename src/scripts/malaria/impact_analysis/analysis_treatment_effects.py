"""
This scenario file sets up the scenarios for simulating the effects of removing sets of services

The scenarios are:
*1 remove HIV-related services
*2 remove TB-related services
*3 remove malaria-related services
*4 remove all three sets of services and impose constraints on appt times (mode 2)

For scenarios 1-3, keep all default health system settings

check the batch configuration gets generated without error:
tlo scenario-run --draw-only src/scripts/malaria/impact_analysis/analysis_scenarios.py

Run on the batch system using:
tlo batch-submit src/scripts/malaria/impact_analysis/analysis_scenarios.py

or locally using:
tlo scenario-run src/scripts/malaria/impact_analysis/analysis_scenarios.py

or execute a single run:
tlo scenario-run src/scripts/malaria/impact_analysis/analysis_scenarios.py --draw 1 0

"""

from pathlib import Path
from typing import Dict

import os
import pandas as pd

from tlo import Date, logging
from tlo.analysis.utils import get_filtered_treatment_ids, get_parameters_for_status_quo, mix_scenarios
from tlo.methods.fullmodel import fullmodel
from tlo.scenario import BaseScenario


class EffectOfProgrammes(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 0
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2020, 1, 1)
        self.pop_size = 100_000
        self._scenarios = self._get_scenarios()
        self.number_of_draws = len(self._scenarios)
        self.runs_per_draw = 5

        self.treatment_effects = pd.read_excel(
            os.path.join(self.resources, "../../../../resources/ResourceFile_malaria.xlsx"),
            sheet_name="treatment_effects",
        )

    def log_configuration(self):
        return {
            'filename': 'effect_of_treatment_packages',
            'directory': Path('./outputs'),  # <- (specified only for local running)
            'custom_levels': {
                '*': logging.WARNING,
                'tlo.methods.hiv': logging.INFO,
                'tlo.methods.tb': logging.INFO,
                'tlo.methods.malaria': logging.INFO,
                'tlo.methods.demography': logging.INFO,
                'tlo.methods.healthsystem.summary': logging.INFO,
                'tlo.methods.healthburden': logging.INFO
            }
        }

    def modules(self):
        return fullmodel(resourcefilepath=self.resources)

    def draw_parameters(self, draw_number, rng):
        return list(self._scenarios.values())[draw_number]

    def _get_scenarios(self) -> Dict[str, Dict]:
        """ create a dict which modifies the health system settings for each scenario."""

        # Generate list of TREATMENT_IDs
        treatments = get_filtered_treatment_ids(depth=1)

        # get service availability with all select services removed
        services_to_remove = ['Hiv_*', 'Tb_*', 'Malaria_*']
        service_availability = dict({"Everything": ["*"]})

        # create service packages with one set of interventions removed
        for service in services_to_remove:
            service_availability.update(
                {f"No {service}": [v for v in treatments if v != service]}
            )

        # create service package with all three sets of interventions removed
        # service_availability.update(
        #     {f"No_Hiv_TB_Malaria": [v for v in treatments if v not in services_to_remove]}
        # )

        # Assuming you have already loaded your Excel data into a DataFrame 'self.treatment_effects'

        # Filter the DataFrame based on the conditions
        f_df = self.treatment_effects[
            (self.treatment_effects['module'] == 'hiv') & (self.treatment_effects['parameter'] == 'rr_ipt')]

        # Check if any rows meet the conditions
        if not filtered_df.empty:
            # Access the value in the 'value' column (or whatever the name of the column with the value is)
            value = filtered_df.iloc[0]['value']
            print("The value for module=hiv and parameter=rr_ipt is:", value)
        else:
            print("No matching rows found.")
        Make sure to replace'value'
        with the actual name of the column that contains the value you want to retrieve from your DataFrame.
        This code filters the DataFrame based on your conditions and retrieves the value from the first matching row, if any.



        return {
            "Default":
                mix_scenarios(
                    get_parameters_for_status_quo(),
                    {'HealthSystem': {'use_funded_or_actual_staffing': 'funded',
                                      'mode_appt_constraints': 1,
                                      }
                     }
                ),

            "Remove_HIV_effects":
                mix_scenarios(
                    get_parameters_for_status_quo(),
                    {'Hiv': {'prob_viral_suppression': self.treatment_effects.loc[
            (self.treatment_effects['module'] == 'hiv') & (self.treatment_effects['parameter'] == 'rr_ipt')],
                                      'use_funded_or_actual_staffing': 'funded',
                                      }
                     }
                ),

            "Remove_TB_effects":
                mix_scenarios(
                    get_parameters_for_status_quo(),
                    {'Tb': {'Service_Availability': service_availability['No Tb_*'],
                                      'use_funded_or_actual_staffing': 'funded',
                                      }
                     }
                ),

            "Remove_malaria_services":
                mix_scenarios(
                    get_parameters_for_status_quo(),
                    {'HealthSystem': {'Service_Availability': service_availability['No Malaria_*'],
                                      'use_funded_or_actual_staffing': 'funded',
                                      }
                     }
                ),

            # "Remove_HIV_TB_malaria_under_constraints":
            #     mix_scenarios(
            #         get_parameters_for_status_quo(),
            #         {
            #             'HealthSystem': {
            #                 'Service_Availability': service_availability['No_Hiv_TB_Malaria'],
            #                 'use_funded_or_actual_staffing': 'funded',
            #                 'mode_appt_constraints': 2,
            #                 'policy_name': 'VerticalProgrammes',
            #             }
            #         }
            #     ),
        }


if __name__ == '__main__':
    from tlo.cli import scenario_run

    scenario_run([__file__])
