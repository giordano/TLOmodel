"""
This scenario file sets up the scenarios for simulating the effects of removing sets of services

The scenarios are:
*1 remove HIV-related services
*2 remove TB-related services
*3 remove malaria-related services
*4 remove all three sets of services and impose constraints on appt times (mode 2)

For scenarios 1-3, keep all default health system settings

check the batch configuration gets generated without error:
tlo scenario-run --draw-only src/scripts/malaria/impact_analysis/analysis_treatment_effects.py

Run on the batch system using:
tlo batch-submit src/scripts/malaria/impact_analysis/analysis_treatment_effects.py

or locally using:
tlo scenario-run src/scripts/malaria/impact_analysis/analysis_treatment_effects.py

or execute a single run:
tlo scenario-run src/scripts/malaria/impact_analysis/analysis_treatment_effects.py --draw 1 0

"""

from pathlib import Path
from typing import Dict

import os
import pandas as pd

from tlo import Date, logging
from tlo.analysis.utils import get_parameters_for_status_quo, mix_scenarios
from tlo.methods.fullmodel import fullmodel
from tlo.scenario import BaseScenario


class EffectOfProgrammes(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 0
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2020, 1, 1)
        self.pop_size = 100_000
        self.number_of_draws = 6
        self.runs_per_draw = 5

        self.treatment_effects = pd.read_excel(
            os.path.join(self.resources, "ResourceFile_HIV.xlsx"),
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
        return {
            'HealthSystem': {
                'use_funded_or_actual_staffing': 'funded',
                'mode_appt_constraints': [1, 1, 1, 1, 2, 2][draw_number],
                'policy_name': ['naive', 'naive', 'naive', 'naive', 'VerticalProgrammes', 'VerticalProgrammes'][
                    draw_number],
            },
            'Hiv': {
                'scenario': [0, 1, 2, 3, 0, 5][draw_number],
            },
        }


if __name__ == '__main__':
    from tlo.cli import scenario_run

    scenario_run([__file__])
