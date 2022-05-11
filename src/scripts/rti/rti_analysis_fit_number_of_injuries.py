import numpy as np

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
from scipy.optimize import curve_fit
import pandas as pd
class TestScenario(BaseScenario):
    def __init__(self):
        super().__init__()
        self.seed = 12
        self.start_date = Date(2010, 1, 1)
        self.end_date = Date(2015, 1, 1)
        self.pop_size = 20000
        self.smaller_pop_size = 20000
        self.number_of_samples_in_parameter_range = 6
        self.upper_iss_value = 10
        self.number_of_draws = 7 * 6
        self.runs_per_draw = 2

    def log_configuration(self):
        return {
            'filename': 'rti_analysis_fit_number_of_injuries',
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


    def draw_parameters(self, draw_number, rng):
        # use if fitting to data
        # percent_multiple = [34.4, 21.9, 24.85, 18.91, 20.8, 20.6]
        # percent_multiple_as_decimal = np.divide(percent_multiple, 100)
        # sources = ['Madubueze et al.', 'Sanyang et al.', 'Qi et al. 2006', 'Ganveer & Tiwani', 'Thani & Kehinde',
        #            'Akinpea et al.']
        # use if creating own distrubtion
        percent_multiple_max = 0.29901999999999995 + 0.03
        percent_multiple_min = 0.29901999999999995 - 0.03
        percent_multiple_as_decimal = np.linspace(percent_multiple_min, percent_multiple_max,
                                                  self.number_of_samples_in_parameter_range)
        def exponentialdecay(x, a, k):
            y = a * np.exp(k * x)
            return y

        probability_distributions = []
        for percentage in percent_multiple_as_decimal:

            data_dict = {'Ninj': [1, 2, 9],
                         'dist': [(1 - percentage), percentage / 2 + 0.04, 0]}
            data = pd.DataFrame(data_dict)

            xdata = data['Ninj']
            ydata = data['dist']
            popt, pcov = curve_fit(exponentialdecay, xdata, ydata, p0=[1, -1])
            exponential_prediction = []
            allnumb = range(1, 10, 1)
            for i in allnumb:
                exponential_prediction.append(exponentialdecay(i, *popt))
            # Normalize the 70-30 distribution
            exponential_prediction = exponential_prediction[:-1]
            exponential_prediction = list(np.divide(exponential_prediction, sum(exponential_prediction)))
            probability_distributions.append(exponential_prediction)
        hsb_cutoff_max = self.upper_iss_value + 1
        hsb_cutoff_min = 4
        iss_cut_off_scores = range(hsb_cutoff_min, hsb_cutoff_max)
        ninj_df = pd.DataFrame()
        ninj_df['ninj'] = [[[1, 2, 3, 4, 5, 6, 7, 8]]] * len(probability_distributions)
        ninj_df['dist'] = [[dist] for dist in probability_distributions]
        ninj_df['number_of_injured_body_regions_distribution'] = ninj_df['ninj'] + ninj_df['dist']
        grid = self.make_grid(
            {'number_of_injured_body_regions_distribution': ninj_df['number_of_injured_body_regions_distribution'],
             'rt_emergency_care_ISS_score_cut_off': iss_cut_off_scores}
        )
        return {
            'RTI': {'number_of_injured_body_regions_distribution':
                        grid['number_of_injured_body_regions_distribution'][draw_number],
                    'rt_emergency_care_ISS_score_cut_off': grid['rt_emergency_care_ISS_score_cut_off'][draw_number]
                    },
            }


if __name__ == '__main__':
    from tlo.cli import scenario_run
    scenario_run([__file__])
