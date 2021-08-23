"""
Childhood stunting module
Documentation: '04 - Methods Repository/Undernutrition module - Description.docx'

Overview
=======
This module applies the prevalence of stunting at the population-level, and schedules new incidences of stunting

"""
import copy
from pathlib import Path
from scipy.stats import norm

import numpy as np
import pandas as pd

from tlo import DateOffset, Module, Parameter, Property, Types, logging
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.lm import LinearModel, LinearModelType, Predictor
from tlo.methods import Metadata
from tlo.methods.healthsystem import HSI_Event

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------------------------------------
#   MODULE DEFINITIONS
# ---------------------------------------------------------------------------------------------------------


class Stunting(Module):
    """
    This module applies the prevalence of stunting at the population-level,
    based on the Malawi DHS Survey 2015-2016.
    The definitions:
    - moderate stunting: height-for-age Z-score (HAZ) <-2 SD from the reference mean
    - severe stunting: height-for-age Z-score (HAZ) <-3 SD from the reference mean

    """

    # Declare Metadata
    METADATA = {
        Metadata.DISEASE_MODULE,
        Metadata.USES_SYMPTOMMANAGER,
        Metadata.USES_HEALTHSYSTEM,
        Metadata.USES_HEALTHBURDEN
    }

    stunting_states = ['HAZ<-3', '-3<=HAZ<-2', 'HAZ>=-2']

    PARAMETERS = {
        # prevalence of stunting by age group
        'prev_HAZ_distribution_age_0_5mo': Parameter(
            Types.LIST, 'distribution of HAZ among less than 6 months of age in 2015'),
        'prev_HAZ_distribution_age_6_11mo': Parameter(
            Types.LIST, 'distribution of HAZ among 6 months and 1 year of age in 2015'),
        'prev_HAZ_distribution_age_12_23mo': Parameter(
            Types.LIST, 'distribution of HAZ among 1 year olds in 2015'),
        'prev_HAZ_distribution_age_24_35mo': Parameter(
            Types.LIST, 'distribution of HAZ among 2 year olds in 2015'),
        'prev_HAZ_distribution_age_36_47mo': Parameter(
            Types.LIST, 'distribution of HAZ among 3 year olds in 2015'),
        'prev_HAZ_distribution_age_48_59mo': Parameter(
            Types.LIST, 'distribution of HAZ among 4 year olds  in 2015'),
        # effect of risk factors on stunting prevalence
        'or_stunting_male': Parameter(
            Types.REAL, 'odds ratio of stunting if male gender'),
        'or_stunting_preterm_and_AGA': Parameter(
            Types.REAL, 'odds ratio of stunting if born preterm and adequate for gestational age'),
        'or_stunting_SGA_and_term': Parameter(
            Types.REAL, 'odds ratio of stunting if born term and small for geatational age'),
        'or_stunting_SGA_and_preterm': Parameter(
            Types.REAL, 'odds ratio of stunting if born preterm and small for gestational age'),
        'or_stunting_hhwealth_Q5': Parameter(
            Types.REAL, 'odds ratio of stunting if household wealth is poorest Q5, ref group Q1'),
        'or_stunting_hhwealth_Q4': Parameter(
            Types.REAL, 'odds ratio of stunting if household wealth is poorer Q4, ref group Q1'),
        'or_stunting_hhwealth_Q3': Parameter(
            Types.REAL, 'odds ratio of stunting if household wealth is middle Q3, ref group Q1'),
        'or_stunting_hhwealth_Q2': Parameter(
            Types.REAL, 'odds ratio of stunting if household wealth is richer Q2, ref group Q1'),
        # incidence parameters
        'base_inc_rate_stunting_by_agegp': Parameter(
            Types.LIST, 'List with baseline incidence of stunting by age group'),
        'rr_stunting_preterm_and_AGA': Parameter(
            Types.REAL, 'relative risk of stunting if born preterm and adequate for gestational age'),
        'rr_stunting_SGA_and_term': Parameter(
            Types.REAL, 'relative risk of stunting if born term and small for gestational age'),
        'rr_stunting_SGA_and_preterm': Parameter(
            Types.REAL, 'relative risk of stunting if born preterm and small for gestational age'),
        'rr_stunting_prior_wasting': Parameter(
            Types.REAL, 'relative risk of stunting if prior wasting in the last 3 months'),
        'rr_stunting_untreated_HIV': Parameter(
            Types.REAL, 'relative risk of stunting for untreated HIV+'),
        'rr_stunting_wealth_level': Parameter(
            Types.REAL, 'relative risk of stunting by increase in wealth level'),
        'rr_stunting_no_exclusive_breastfeeding': Parameter(
            Types.REAL, 'relative risk of stunting for not exclusively breastfed babies < 6 months'),
        'rr_stunting_no_continued_breastfeeding': Parameter(
            Types.REAL, 'relative risk of stunting for not continued breasfed infants 6-24 months'),
        'rr_stunting_per_diarrhoeal_episode': Parameter(
            Types.REAL, 'relative risk of stunting for recent diarrhoea episode'),
        # progression parameters
        'r_progression_severe_stunting_by_agegp': Parameter(
            Types.LIST, 'list with rates of progression to severe stunting by age group'),
        'rr_progress_severe_stunting_untreated_HIV': Parameter(
            Types.REAL, 'relative risk of severe stunting for untreated HIV+'),
        'rr_progress_severe_stunting_previous_wasting': Parameter(
            Types.REAL, 'relative risk of severe stunting if previously wasted'),
        'prob_remained_stunted_in_the_next_3months': Parameter(
            Types.REAL, 'probability of stunted remained stunted in the next 3 month period'),
        # intervention parameters
        'un_effectiveness_complementary_feeding_promo_education_only_in_stunting_reduction': Parameter(
            Types.REAL, 'effectiveness of complementary feeding promotion (education only) in reducing stunting'),
        'un_effectiveness_complementary_feeding_promo_with_food_supplementation_in_stunting_reduction': Parameter(
            Types.REAL,
            'effectiveness of complementary feeding promotion with food supplementation in reducing stunting'),
        'un_effectiveness_zinc_supplementation_in_stunting_reduction': Parameter(
            Types.REAL, 'effectiveness of zinc supplementation in reducing stunting'),
    }

    PROPERTIES = {
        'un_ever_stunted': Property(Types.BOOL, 'had stunting before (HAZ <-2)'),
        'un_HAZ_category': Property(Types.CATEGORICAL, 'height-for-age z-score group',
                                    categories=['HAZ<-3', '-3<=HAZ<-2', 'HAZ>=-2']),
        'un_last_stunting_date_of_onset': Property(Types.DATE, 'date of onset of lastest stunting episode'),
        'un_stunting_recovery_date': Property(Types.DATE, 'recovery date, when HAZ>=-2'),
        'un_cm_treatment_type': Property(Types.CATEGORICAL, 'treatment types for of chronic malnutrition',
                                         categories=['education_on_complementary_feeding',
                                                     'complementary_feeding_with_food_supplementation'] +
                                                    ['none'] + ['not_applicable']),
        'un_stunting_tx_start_date': Property(Types.DATE, 'start date of treatment for stunting')
    }

    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath

        # dict to hold counters for the number of episodes by stunting-type and age-group
        blank_counter = dict(zip(self.stunting_states, [list() for _ in self.stunting_states]))
        self.stunting_incident_case_tracker_blank = {
            '0y': copy.deepcopy(blank_counter),
            '1y': copy.deepcopy(blank_counter),
            '2y': copy.deepcopy(blank_counter),
            '3y': copy.deepcopy(blank_counter),
            '4y': copy.deepcopy(blank_counter),
            '5+y': copy.deepcopy(blank_counter)
        }
        self.stunting_incident_case_tracker = copy.deepcopy(self.stunting_incident_case_tracker_blank)

        zeros_counter = dict(zip(self.stunting_states, [0] * len(self.stunting_states)))
        self.stunting_incident_case_tracker_zeros = {
            '0y': copy.deepcopy(zeros_counter),
            '1y': copy.deepcopy(zeros_counter),
            '2y': copy.deepcopy(zeros_counter),
            '3y': copy.deepcopy(zeros_counter),
            '4y': copy.deepcopy(zeros_counter),
            '5+y': copy.deepcopy(zeros_counter)
        }

        # dict to hold the DALY weights
        self.daly_wts = dict()  # no dalys directly from stunting, but will have cognitive deficiencies in the future

        # --------------------- linear models of the natural history --------------------- #
        # set the linear model equations for prevalence and incidence
        self.prevalence_equations_by_age = dict()
        self.stunting_incidence_equation = dict()

        # set the linear model equation for progression to severe stunting state
        self.severe_stunting_progression_equation = dict()

        # --------------------- linear models following HSI interventions --------------------- #

        # set the linear models for stunting improvement by intervention
        self.stunting_improvement_based_on_interventions = dict()

    def read_parameters(self, data_folder):
        """
        :param data_folder: path of a folder supplied to the Simulation containing data files.
              Typically modules would read a particular file within here.
        :return:
        """
        # Update parameters from the resource dataframe
        dfd = pd.read_excel(Path(self.resourcefilepath) / 'ResourceFile_Undernutrition.xlsx',
                            sheet_name='Parameter_values_CM')
        self.load_parameters_from_dataframe(dfd)

        p = self.parameters

        # Check that every value has been read-in successfully
        for param_name, param_type in self.PARAMETERS.items():
            assert param_name in p, f'Parameter "{param_name}" is not read in correctly from the resourcefile.'
            assert param_name is not None, f'Parameter "{param_name}" is not read in correctly from the resourcefile.'
            assert isinstance(p[param_name],
                              param_type.python_type), f'Parameter "{param_name}" is not read in correctly from the ' \
                                                       f'resourcefile.'

        # Stunting does not have specific symptoms, but is a consequence of poor nutrition and repeated infections

        # no DALYs for stunting in the TLO daly weights

    def initialise_population(self, population):
        """
        Set our property values for the initial population.

        This method is called by the simulation when creating the initial population, and is
        responsible for assigning initial values, for every individual, of those properties
        'owned' by this module, i.e. those declared in the PROPERTIES dictionary above.

        :param population:
        :return:
        """
        df = population.props
        p = self.parameters

        # Set initial properties
        df.loc[df.is_alive, 'un_ever_stunted'] = False
        df.loc[df.is_alive, 'un_HAZ_category'] = 'HAZ>=-2'  # not undernourished
        df.loc[df.is_alive, 'un_last_stunting_date_of_onset'] = pd.NaT
        df.loc[df.is_alive, 'un_stunting_recovery_date'] = pd.NaT
        df.loc[df.is_alive, 'un_cm_treatment_type'] = 'not_applicable'
        df.loc[df.is_alive, 'un_stunting_tx_start_date'] = pd.NaT

        # -----------------------------------------------------------------------------------------------------
        # # # # # allocate initial prevalence of stunting at the start of the simulation # # # # #

        def make_scaled_linear_model_stunting(agegp):
            """Makes the unscaled linear model with intercept of baseline odds of stunting (HAZ <-2).
            Calculates the mean odds of stunting by age group and then creates a new linear model
            with adjusted intercept so odds in 1-year-olds matches the specified value in the model
            when averaged across the population
            """
            def get_odds_stunting(agegp):
                """
                This function will calculate the HAZ scores by categories and return the odds of stunting
                :param agegp: age grouped in months
                :return:
                """
                # generate random numbers from N(meean, sd)
                baseline_HAZ_prevalence_by_agegp = f'prev_HAZ_distribution_age_{agegp}'
                HAZ_normal_distribution = norm(loc=p[baseline_HAZ_prevalence_by_agegp][0],
                                               scale=p[baseline_HAZ_prevalence_by_agegp][1])

                # get all stunting: HAZ <-2
                probability_over_or_equal_minus2sd = HAZ_normal_distribution.sf(-2)
                probability_less_than_minus2sd = 1 - probability_over_or_equal_minus2sd

                # convert probability to odds
                base_odds_of_stunting = probability_less_than_minus2sd / (1-probability_less_than_minus2sd)

                return base_odds_of_stunting

            def make_linear_model_stunting(agegp, intercept=get_odds_stunting(agegp=agegp)):
                return LinearModel(
                    LinearModelType.LOGISTIC,
                    get_odds_stunting(agegp=agegp),  # base odds
                    Predictor('sex').when('M', p['or_stunting_male']),
                    Predictor('li_wealth')  .when(2, p['or_stunting_hhwealth_Q2'])
                                            .when(3, p['or_stunting_hhwealth_Q3'])
                                            .when(4, p['or_stunting_hhwealth_Q4'])
                                            .when(5, p['or_stunting_hhwealth_Q5']),
                    Predictor().when('(nb_size_for_gestational_age == "small_for_gestational_age") '
                                     '& (nb_late_preterm == False) & (nb_early_preterm == False)',
                                     p['or_stunting_SGA_and_term']),
                    Predictor().when('(nb_size_for_gestational_age == "small_for_gestational_age") '
                                     '& ((nb_late_preterm == True) | (nb_early_preterm == True))',
                                     p['or_stunting_SGA_and_preterm']),
                    Predictor().when('(nb_size_for_gestational_age == "average_for_gestational_age") '
                                     '& ((nb_late_preterm == True) | (nb_early_preterm == True))',
                                     p['or_stunting_preterm_and_AGA']),
                )

            unscaled_lm = make_linear_model_stunting(agegp, intercept=get_odds_stunting(agegp=agegp))
            target_mean = get_odds_stunting(agegp='12_23mo')
            actual_mean = unscaled_lm.predict(df.loc[df.is_alive & (df.age_years == 1)]).mean()
            scaled_intercept = get_odds_stunting(agegp) * (target_mean / actual_mean) if \
                (target_mean != 0 and actual_mean != 0 and ~np.isnan(actual_mean)) else get_odds_stunting(agegp)
            scaled_lm = make_linear_model_stunting(agegp, intercept=scaled_intercept)
            return scaled_lm

        # the linear model returns the probability that is implied by the model prob = odds / (1 + odds)
        for agegp in ['0_5mo', '6_11mo', '12_23mo', '24_35mo', '36_47mo', '48_59mo']:
            self.prevalence_equations_by_age[agegp] = make_scaled_linear_model_stunting(agegp)

        # get the initial prevalence values for each age group using the lm equation (scaled)
        prevalence_of_stunting = pd.DataFrame(index=df.loc[df.is_alive & (df.age_exact_years < 5)].index)

        prevalence_of_stunting['0_5mo'] = self.prevalence_equations_by_age['0_5mo']\
            .predict(df.loc[df.is_alive & (df.age_exact_years < 0.5)])
        prevalence_of_stunting['6_11mo'] = self.prevalence_equations_by_age['6_11mo']\
            .predict(df.loc[df.is_alive & ((df.age_exact_years >= 0.5) & (df.age_exact_years < 1))])
        prevalence_of_stunting['12_23mo'] = self.prevalence_equations_by_age['12_23mo'] \
            .predict(df.loc[df.is_alive & ((df.age_exact_years >= 1) & (df.age_exact_years < 2))])
        prevalence_of_stunting['24_35mo'] = self.prevalence_equations_by_age['24_35mo'] \
            .predict(df.loc[df.is_alive & ((df.age_exact_years >= 2) & (df.age_exact_years < 3))])
        prevalence_of_stunting['36_47mo'] = self.prevalence_equations_by_age['36_47mo'] \
            .predict(df.loc[df.is_alive & ((df.age_exact_years >= 3) & (df.age_exact_years < 4))])
        prevalence_of_stunting['48_59mo'] = self.prevalence_equations_by_age['48_59mo'] \
            .predict(df.loc[df.is_alive & ((df.age_exact_years >= 4) & (df.age_exact_years < 5))])

        def get_prob_severe_in_overall_stunting(agegp):
            """
            This function will calculate the HAZ scores by categories and return probability of severe stunting
            among those with stunting
            :param agegp: age grouped in months
            :return:
            """
            # generate random numbers from N(meean, sd)
            baseline_HAZ_prevalence_by_agegp = f'prev_HAZ_distribution_age_{agegp}'
            HAZ_normal_distribution = norm(loc=p[baseline_HAZ_prevalence_by_agegp][0],
                                           scale=p[baseline_HAZ_prevalence_by_agegp][1])

            # get all stunting: HAZ <-2
            probability_over_or_equal_minus2sd = HAZ_normal_distribution.sf(-2)
            probability_less_than_minus2sd = 1 - probability_over_or_equal_minus2sd
            # get severe stunting zcores: HAZ <-3
            probability_over_or_equal_minus3sd = HAZ_normal_distribution.sf(-3)
            probability_less_than_minus3sd = 1 - probability_over_or_equal_minus3sd

            # make HAZ <-2 as the 100% and get the adjusted probability of severe stunting
            proportion_severe_in_overall_stunting = probability_less_than_minus3sd * probability_less_than_minus2sd

            # get a list with probability of severe stunting, and moderate stunting
            return proportion_severe_in_overall_stunting

        # further differentiate between severe stunting and moderate stunting, and normal HAZ
        for agegp in ['0_5mo', '6_11mo', '12_23mo', '24_35mo', '36_47mo', '48_59mo']:
            stunted = self.rng.random_sample(len(prevalence_of_stunting[agegp])) < prevalence_of_stunting[agegp]
            for id in stunted[stunted].index:
                probability_of_severe = get_prob_severe_in_overall_stunting(agegp)
                stunted_category = self.rng.choice(['HAZ<-3', '-3<=HAZ<-2'],
                                                   p=[probability_of_severe, 1 - probability_of_severe])
                df.at[id, 'un_HAZ_category'] = stunted_category
                df.at[id, 'un_last_stunting_date_of_onset'] = self.sim.date
                df.at[id, 'un_ever_stunted'] = True
                df.at[id, 'un_cm_treatment_type'] = 'none'  # start without treatment

            df.loc[stunted[stunted == False].index, 'un_HAZ_category'] = 'HAZ>=-2'

        # -----------------------------------------------------------------------------------------------------

    def count_all_previous_diarrhoea_episodes(self, today, index):
        """
        Get all diarrhoea episodes since birth prior to today's date for non-stunted children;
        for already moderately stunted children, get all diarrhoea episodes since the onset of stunting
        :param today:
        :param index:
        :return:
        """
        df = self.sim.population.props
        list_dates = []

        for person in index:
            if df.at[person, 'un_HAZ_category'] == 'HAZ>=-2':
                delta_dates = df.at[person, 'date_of_birth'] - today
                for i in range(delta_dates.days):
                    day = today - DateOffset(days=i)
                    while df.gi_last_diarrhoea_date_of_onset[person] == day:
                        list_dates.append(day)

            if df.at[person, 'un_HAZ_category'] == '-3<=HAZ<-2':
                delta_dates = df.at[person, 'un_last_stunting_date_of_onset'] - today
                for i in range(delta_dates.days):
                    day = today - DateOffset(days=i)
                    while df.gi_last_diarrhoea_date_of_onset[person] == day:
                        list_dates.append(day)

        total_diarrhoea_count_to_date = len(list_dates)

        return total_diarrhoea_count_to_date

    def initialise_simulation(self, sim):
        """Prepares for simulation:
        * Schedules the main polling event
        * Schedules the main logging event
        * Establishes the linear models and other data structures using the parameters that have been read-in
        * Store the consumables that are required in each of the HSI
        """
        df = self.sim.population.props
        p = self.parameters

        # Schedule the main polling event
        sim.schedule_event(StuntingPollingEvent(self), sim.date + DateOffset(months=3))

        # Schedule recoveries by interventions
        sim.schedule_event(StuntingRecoveryPollingEvent(self), sim.date + DateOffset(months=3))

        # Schedule the main logging event (to first occur in one year)
        sim.schedule_event(StuntingLoggingEvent(self), sim.date + DateOffset(years=1))

        # Get DALY weights
        # no DALYs for stunting directly, but cognitive impairment should be added later

        # --------------------------------------------------------------------------------------------
        # Make a linear model equation that govern the probability that a person becomes stunted HAZ<-2
        def make_scaled_lm_stunting_incidence():
            """
            Makes the unscaled linear model with default intercept of 1. Calculates the mean incidents rate for
            1-year-olds and then creates a new linear model with adjusted intercept so incidents in 1-year-olds
            matches the specified value in the model when averaged across the population
            """
            def make_lm_stunting_incidence(intercept=1.0):
                return LinearModel(
                    LinearModelType.MULTIPLICATIVE,
                    intercept,
                    Predictor('age_exact_years').when('<0.5', p['base_inc_rate_stunting_by_agegp'][0])
                                                .when('.between(0.5,0.9999)', p['base_inc_rate_stunting_by_agegp'][1])
                                                .when('.between(1,1.9999)', p['base_inc_rate_stunting_by_agegp'][2])
                                                .when('.between(2,2.9999)', p['base_inc_rate_stunting_by_agegp'][3])
                                                .when('.between(3,3.9999)', p['base_inc_rate_stunting_by_agegp'][4])
                                                .when('.between(4,4.9999)', p['base_inc_rate_stunting_by_agegp'][5])
                                                .otherwise(0.0),
                    Predictor().when('(nb_size_for_gestational_age == "small_for_gestational_age") '
                                     '& (nb_late_preterm == False) & (nb_early_preterm == False)',
                                     p['rr_stunting_SGA_and_term']),
                    Predictor().when('(nb_size_for_gestational_age == "small_for_gestational_age") '
                                     '& ((nb_late_preterm == True) | (nb_early_preterm == True))',
                                     p['rr_stunting_SGA_and_preterm']),
                    Predictor().when('(nb_size_for_gestational_age == "average_for_gestational_age") '
                                     '& ((nb_late_preterm == True) | (nb_early_preterm == True))',
                                     p['rr_stunting_preterm_and_AGA']),
                    Predictor().when('(hv_inf == True) & (hv_art == "not")', p['rr_stunting_untreated_HIV']),
                    Predictor('li_wealth').apply(lambda x: 1 if x == 1 else (x - 1) ** (p['rr_stunting_wealth_level'])),
                    Predictor('un_ever_wasted').when(True, p['rr_stunting_prior_wasting']),
                    Predictor('nb_breastfeeding_status').when('non_exclusive | none',
                                                              p['rr_stunting_no_exclusive_breastfeeding']),
                    Predictor().when('((nb_breastfeeding_status == "non_exclusive") | '
                                     '(nb_breastfeeding_status == "none")) & (age_exact_years < 0.5)',
                                     p['rr_stunting_no_exclusive_breastfeeding']),
                    Predictor().when('(nb_breastfeeding_status == "none") & (age_exact_years.between(0.5,2))',
                                     p['rr_stunting_no_continued_breastfeeding']),
                    Predictor('previous_diarrhoea_episodes', external=True).apply(
                        lambda x: x ** (p['rr_stunting_per_diarrhoeal_episode'])),
                )

            unscaled_lm = make_lm_stunting_incidence()
            target_mean = p[f'base_inc_rate_stunting_by_agegp'][2]
            actual_mean = unscaled_lm.predict(
                df.loc[df.is_alive & (df.age_years == 1) & (df.un_HAZ_category == 'HAZ>=-2')],
                previous_diarrhoea_episodes=self.count_all_previous_diarrhoea_episodes(
                    today=sim.date, index=df.loc[df.is_alive & (df.age_years == 1) &
                                                 (df.un_HAZ_category == 'HAZ>=-2')].index)).mean()
            scaled_intercept = 1.0 * (target_mean / actual_mean) \
                if (target_mean != 0 and actual_mean != 0 and ~np.isnan(actual_mean)) else 1.0
            scaled_lm = make_lm_stunting_incidence(intercept=scaled_intercept)
            return scaled_lm

        self.stunting_incidence_equation = make_scaled_lm_stunting_incidence()

        # --------------------------------------------------------------------------------------------
        # Make a linear model equation that govern the probability that a person becomes severely stunted HAZ<-3
        # (natural history only, no interventions)
        def make_scaled_lm_severe_stunting():
            """
            Makes the unscaled linear model with default intercept of 1. Calculates the mean progression rate for
            1-year-olds and then creates a new linear model with adjusted intercept so progression in 1-year-olds
            matches the specified value in the model when averaged across the population
            """
            def make_lm_severe_stunting(intercept=1.0):
                return LinearModel(
                    LinearModelType.MULTIPLICATIVE,
                    intercept,
                    Predictor('age_exact_years').when('<0.5', p['r_progression_severe_stunting_by_agegp'][0])
                                                .when('.between(0.5,0.9999)',
                                                      p['r_progression_severe_stunting_by_agegp'][1])
                                                .when('.between(1,1.9999)',
                                                      p['r_progression_severe_stunting_by_agegp'][2])
                                                .when('.between(2,2.9999)',
                                                      p['r_progression_severe_stunting_by_agegp'][3])
                                                .when('.between(3,3.9999)',
                                                      p['r_progression_severe_stunting_by_agegp'][4])
                                                .when('.between(4,4.9999)',
                                                      p['r_progression_severe_stunting_by_agegp'][5])
                                                .otherwise(0.0),
                    Predictor('un_ever_wasted').when(True, p['rr_progress_severe_stunting_previous_wasting']),
                    Predictor().when('(hv_inf == True) & (hv_art == "not")', p['rr_stunting_untreated_HIV']),
                    Predictor('previous_diarrhoea_episodes', external=True).apply(
                        lambda x: x ** (p['rr_stunting_per_diarrhoeal_episode'])),
                )

            unscaled_lm = make_lm_severe_stunting()
            target_mean = p[f'base_inc_rate_stunting_by_agegp'][2]
            actual_mean = unscaled_lm.predict(
                df.loc[df.is_alive & (df.age_years == 1) & (df.un_HAZ_category == '-3<=HAZ<-2')],
                previous_diarrhoea_episodes=self.count_all_previous_diarrhoea_episodes(
                    today=sim.date, index=df.loc[df.is_alive & (df.age_years == 1) &
                                                 (df.un_HAZ_category == '-3<=HAZ<-2')].index)).mean()
            scaled_intercept = 1.0 * (target_mean / actual_mean) \
                if (target_mean != 0 and actual_mean != 0 and ~np.isnan(actual_mean)) else 1.0
            scaled_lm = make_lm_severe_stunting(intercept=scaled_intercept)
            return scaled_lm

        self.severe_stunting_progression_equation = make_scaled_lm_severe_stunting()

        # --------------------------------------------------------------------------------------------
        # Make a linear model equation that govern the probability that a person improves in stunting state
        self.stunting_improvement_based_on_interventions = \
            LinearModel(LinearModelType.MULTIPLICATIVE,
                        1.0,
                        Predictor('un_cm_treatment_type')
                        .when('complementary_feeding_with_food_supplementation',
                              p['un_effectiveness_complementary_feeding_promo_'
                                'with_food_supplementation_in_stunting_reduction'])
                        .when('education_on_complementary_feeding',
                              p['un_effectiveness_complementary_feeding_promo_'
                                'education_only_in_stunting_reduction'])
                        .otherwise(0.0)
                        )

    def on_birth(self, mother_id, child_id):
        pass

    def report_daly_values(self):
        df = self.sim.population.props

        total_daly_values = pd.Series(data=0.0, index=df.index[df.is_alive])

        return total_daly_values

    def do_when_chronic_malnutrition_assessment(self, person_id):
        """
        This is called by the a generic HSI event when chronic malnutrition/ stunting is checked.
        :param person_id:
        :param hsi_event: The HSI event that has called this event
        :return:
        """
        # Interventions for stunting

        # Check for coverage of complementary feeding, by assuming
        # these interventions are given in supplementary feeding programmes (in wasting)
        if self.sim.modules['Wasting'].parameters['coverage_supplementary_feeding_program'] > self.rng.rand():
            # schedule HSI for complementary feeding program
            self.sim.modules['HealthSystem'].schedule_hsi_event(
                hsi_event=HSI_complementary_feeding_with_supplementary_foods
                (module=self,
                 person_id=person_id),
                priority=0,
                topen=self.sim.date
            )
        else:
            # if not in supplementary feeding program, education only will be provided in outpatient health centres
            # schedule HSI for complementary feeding program
            self.sim.modules['HealthSystem'].schedule_hsi_event(
                hsi_event=HSI_complementary_feeding_education_only
                (module=self,
                 person_id=person_id),
                priority=0,
                topen=self.sim.date
            )
            # ASSUMPTION : ALL PATIENTS WILL GET ADVICE/COUNSELLING AT OUTPATIENT VISITS

    def do_when_cm_treatment(self, person_id):
        """
        This function will apply the linear model of recovery based on intervention given
        :param person_id:
        :param intervention:
        :return:
        """
        df = self.sim.population.props

        stunting_improvement = self.stunting_improvement_based_on_interventions.predict(
            df.loc[[person_id]]).values[0]
        if self.rng.rand() < stunting_improvement:
            # schedule recovery date
            self.sim.schedule_event(
                event=StuntingRecoveryEvent(module=self, person_id=person_id),
                date=df.at[person_id, 'un_stunting_tx_start_date'] + DateOffset(weeks=4))
            # cancel progression to severe stunting date (in ProgressionEvent)
        else:
            # remained stunted or severe stunted
            return


class StuntingPollingEvent(RegularEvent, PopulationScopeEventMixin):
    """
    Regular event that updates all stunting properties for the population
    It determines who will be stunted and schedules individual incident cases to represent onset.
    """

    AGE_GROUPS = {0: '0y', 1: '1y', 2: '2y', 3: '3y', 4: '4y'}

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=1))
        assert isinstance(module, Stunting)

    def apply(self, population):
        df = population.props
        rng = self.module.rng

        days_until_next_polling_event = (self.sim.date + self.frequency - self.sim.date) / np.timedelta64(1, 'D')

        # Determine who will be onset with stunting among those who are not currently stunted
        incidence_of_stunting = self.module.stunting_incidence_equation.predict(
            df.loc[df.is_alive & (df.age_exact_years < 5) & (df.un_HAZ_category == 'HAZ>=-2')],
            previous_diarrhoea_episodes=self.module.count_all_previous_diarrhoea_episodes(
                today=self.sim.date, index=df.loc[df.is_alive & (df.age_exact_years < 5) &
                                                  (df.un_HAZ_category == 'HAZ>=-2')].index))
        stunted = rng.random_sample(len(incidence_of_stunting)) < incidence_of_stunting

        # determine the time of onset and other disease characteristics for each individual
        for person_id in stunted[stunted].index:
            # Allocate a date of onset for stunting episode
            date_onset = self.sim.date + DateOffset(days=rng.randint(0, days_until_next_polling_event))

            # Create the event for the onset of stunting (start with moderate stunting)
            self.sim.schedule_event(
                event=StuntingOnsetEvent(module=self.module,
                                         person_id=person_id), date=date_onset)


class StuntingRecoveryPollingEvent(RegularEvent, PopulationScopeEventMixin):
    """
    Regular event that determines those that will improve their stunting state
     and schedules individual recoveries, these are based on interventions
    """
    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=3))
        assert isinstance(module, Stunting)

    def apply(self, population):
        df = population.props
        rng = self.module.rng

        days_until_next_polling_event = (self.sim.date + self.frequency - self.sim.date) / np.timedelta64(1, 'D')

        # determine those individuals that will improve stunting state
        improvement_of_stunting_state = self.module.stunting_improvement_based_on_interventions.predict(
            df.loc[df.is_alive & (df.age_exact_years < 5) & ((df.un_HAZ_category == '-3<=HAZ<-2') |
                                                             (df.un_HAZ_category == 'HAZ<-3'))])
        improved_stunting_state = rng.random_sample(len(improvement_of_stunting_state)) < improvement_of_stunting_state

        # determine the onset date of severe stunting and schedule event
        for person_id in improved_stunting_state[improved_stunting_state].index:
            # Allocate a date of onset for stunting episode
            date_recovery_stunting = self.sim.date + DateOffset(days=rng.randint(0, days_until_next_polling_event))
            # Create the event for the onset of stunting recovery by 1sd in HAZ
            self.sim.schedule_event(
                event=StuntingRecoveryEvent(module=self.module,
                                            person_id=person_id), date=date_recovery_stunting)


class StuntingOnsetEvent(Event, IndividualScopeEventMixin):
    """
    This Event is for the onset of stunting (stunting with HAZ <-2).
     * Refreshes all the properties so that they pertain to this current episode of stunting
     * Imposes the symptoms
     * Schedules relevant natural history event {(ProgressionSevereStuntingEvent) and
       (either StuntingRecoveryEvent or StuntingDeathEvent)}
    """

    AGE_GROUPS = {0: '0y', 1: '1y', 2: '2y', 3: '3y', 4: '4y'}

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        df = self.sim.population.props  # shortcut to the dataframe
        m = self.module
        p = m.parameters
        rng = m.rng

        df.at[person_id, 'un_ever_stunted'] = True
        df.at[person_id, 'un_HAZ_category'] = '-3<=HAZ<-2'  # start as moderate stunting
        df.at[person_id, 'un_last_stunting_date_of_onset'] = self.sim.date

        # -------------------------------------------------------------------------------------------
        # Add this incident case to the tracker
        stunting_state = df.at[person_id, 'un_HAZ_category']
        age_group = StuntingOnsetEvent.AGE_GROUPS.get(df.loc[person_id].age_years, '5+y')
        m.stunting_incident_case_tracker[age_group][stunting_state].append(self.sim.date)
        # -------------------------------------------------------------------------------------------

        # determine if this person will progress to severe stunting # # # # # # # # # # #
        progression_to_severe_stunting = self.module.severe_stunting_progression_equation.predict(
            df.loc[[person_id]],
            previous_diarrhoea_episodes=self.module.count_all_previous_diarrhoea_episodes(
                today=self.sim.date, index=df.loc[[person_id]].index))

        if rng.rand() < progression_to_severe_stunting:
            # Allocate a date of onset for stunting episode
            date_onset_severe_stunting = self.sim.date + DateOffset(months=3)

            # Create the event for the onset of severe stunting
            self.sim.schedule_event(
                event=ProgressionSevereStuntingEvent(module=self.module,
                                                     person_id=person_id), date=date_onset_severe_stunting
            )

        # determine if this person will improve stunting state without interventions # # # # # # # # # # #
        improved_stunting_state = 1 - p['prob_remained_stunted_in_the_next_3months']
        if rng.rand() < improved_stunting_state:
            # Allocate a date of onset for improvement of stunting episode
            date_recovery_stunting = self.sim.date + DateOffset(months=3)

            # Create the event for the onset of stunting recovery by 1sd in HAZ
            self.sim.schedule_event(
                event=StuntingRecoveryEvent(module=self.module,
                                            person_id=person_id), date=date_recovery_stunting)


class ProgressionSevereStuntingEvent(Event, IndividualScopeEventMixin):
    """
    This Event is for the onset of severe stunting (with HAZ <-3).
     * Refreshes all the properties so that they pertain to this current episode of stunting
     * Imposes the symptoms
     * Schedules relevant natural history event {(either WastingRecoveryEvent or WastingDeathEvent)}
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        df = self.sim.population.props  # shortcut to the dataframe
        m = self.module
        p = m.parameters
        rng = m.rng

        if not df.at[person_id, 'is_alive']:
            return

        # before progression to severe stunting, check those who started complementary feeding interventions
        if df.at[person_id,
                 'un_last_stunting_date_of_onset'] < df.at[person_id,
                                                           'un_stunting_tx_start_date'] < self.sim.date:
            return

        # update properties
        df.at[person_id, 'un_HAZ_category'] = 'HAZ<-3'

        # -------------------------------------------------------------------------------------------
        # Add this incident case to the tracker
        stunting_state = df.at[person_id, 'un_HAZ_category']
        age_group = StuntingOnsetEvent.AGE_GROUPS.get(df.loc[person_id].age_years, '5+y')
        m.stunting_incident_case_tracker[age_group][stunting_state].append(self.sim.date)
        # -------------------------------------------------------------------------------------------

        # determine if this person will improve stunting state # # # # # # # # # # #
        improved_stunting_state = 1 - p['prob_remained_stunted_in_the_next_3months']
        if rng.rand() < improved_stunting_state:
            # Allocate a date of onset for improvement of stunting episode
            date_recovery_stunting = self.sim.date + DateOffset(months=3)

            # Create the event for the onset of stunting recovery by 1sd in HAZ
            self.sim.schedule_event(
                event=StuntingRecoveryEvent(module=self.module,
                                            person_id=person_id), date=date_recovery_stunting)


class StuntingRecoveryEvent(Event, IndividualScopeEventMixin):
    """
    This event sets the properties back to normal state
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        df = self.sim.population.props  # shortcut to the dataframe

        if not df.at[person_id, 'is_alive']:
            return

        if df.at[person_id, 'un_HAZ_category'] == '-3<=HAZ<-2':
            df.at[person_id, 'un_HAZ_category'] = 'HAZ>=-2'
            df.at[person_id, 'un_stunting_recovery_date'] = self.sim.date
        if df.at[person_id, 'un_HAZ_category'] == 'HAZ<-3':
            df.at[person_id, 'un_HAZ_category'] = '-3<=HAZ<-2'


class HSI_complementary_feeding_education_only(HSI_Event, IndividualScopeEventMixin):
    """
    This HSI is for education (only) of complementary feeding / without provision of supplementary foods
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Stunting)

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules["HealthSystem"].get_blank_appt_footprint()
        the_appt_footprint['GrowthMon'] = 1  # This requires one out patient

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'complementary_feeding_education_only'
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):

        df = self.sim.population.props

        # Stop the person from dying of acute malnutrition (if they were going to die)
        if not df.at[person_id, 'is_alive']:
            return

        # Do here whatever happens to an individual during this health system interaction event
        # ~~~~~~~~~~~~~~~~~~~~~~
        # Make request for some consumables
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']

        # individual items
        item_code_complementary_feeding_education = pd.unique(
            consumables.loc[consumables['Items'] ==
                            'Complementary feeding--education only drugs/supplies to service a client', 'Item_Code'])[0]

        # check availability of consumables
        if self.get_all_consumables(item_codes=item_code_complementary_feeding_education):
            logger.debug(key='debug', data='item_code_complementary_feeding_education is available, so use it.')
            # Update properties
            df.at[person_id, 'un_stunting_tx_start_date'] = self.sim.date
            df.at[person_id, 'un_cm_treatment_type'] = 'education_on_complementary_feeding'
            self.module.do_when_cm_treatment(person_id)
        else:
            logger.debug(key='debug', data="item_code_complementary_feeding_education is not available, "
                                           "so can't use it.")

        # --------------------------------------------------------------------------------------------------
        # Return the actual appt footprints
        actual_appt_footprint = self.EXPECTED_APPT_FOOTPRINT  # The actual time take is double what is expected
        actual_appt_footprint['GrowthMon'] = actual_appt_footprint['GrowthMon'] * 2
        return actual_appt_footprint

    def did_not_run(self):
        logger.debug("HSI_complementary_feeding_education_only: did not run")
        pass


class HSI_complementary_feeding_with_supplementary_foods(HSI_Event, IndividualScopeEventMixin):
    """
    This HSI is for complementary feeding with provision of supplementary foods
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Stunting)

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.sim.modules["HealthSystem"].get_blank_appt_footprint()
        the_appt_footprint['GrowthMon'] = 1  # This requires one out patient

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'complementary_feeding_with_supplementary_foods'
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):

        df = self.sim.population.props

        # Stop the person from dying of acute malnutrition (if they were going to die)
        if not df.at[person_id, 'is_alive']:
            return

        # Do here whatever happens to an individual during this health system interaction event
        # ~~~~~~~~~~~~~~~~~~~~~~
        # Make request for some consumables
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']

        # individual items
        item_code_complementary_feeding_with_supplements = pd.unique(
            consumables.loc[consumables['Items'] ==
                            'Supplementary spread, sachet 92g/CAR-150', 'Item_Code'])[0]

        # check availability of consumables
        if self.get_all_consumables(item_codes=item_code_complementary_feeding_with_supplements):
            logger.debug(key='debug', data='item_code_complementary_feeding_with_supplements is available, so use it.')
            # Update properties
            df.at[person_id, 'un_stunting_tx_start_date'] = self.sim.date
            df.at[person_id, 'un_cm_treatment_type'] = 'complementary_feeding_with_food_supplementation'
            self.module.do_when_cm_treatment(person_id)
        else:
            logger.debug(key='debug', data="item_code_complementary_feeding_with_supplements is not available, "
                                           "so can't use it.")

        # --------------------------------------------------------------------------------------------------
        # Return the actual appt footprints
        actual_appt_footprint = self.EXPECTED_APPT_FOOTPRINT  # The actual time take is double what is expected
        actual_appt_footprint['GrowthMon'] = actual_appt_footprint['GrowthMon'] * 2
        return actual_appt_footprint

    def did_not_run(self):
        logger.debug("HSI_complementary_feeding_with_supplementary_foods: did not run")
        pass


class StuntingLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    """
        This Event logs the number of incident cases that have occurred since the previous logging event.
        Analysis scripts expect that the frequency of this logging event is once per year.
        """

    def __init__(self, module):
        # This event to occur every year
        self.repeat = 12
        super().__init__(module, frequency=DateOffset(months=self.repeat))
        self.date_last_run = self.sim.date

    def apply(self, population):
        # Convert the list of timestamps into a number of timestamps
        # and check that all the dates have occurred since self.date_last_run
        counts_cm = copy.deepcopy(self.module.stunting_incident_case_tracker_zeros)

        for age_grp in self.module.stunting_incident_case_tracker.keys():
            for state in self.module.stunting_states:
                list_of_times = self.module.stunting_incident_case_tracker[age_grp][state]
                counts_cm[age_grp][state] = len(list_of_times)
                for t in list_of_times:
                    assert self.date_last_run <= t <= self.sim.date

        logger.info(key='stunting_incidence_count', data=counts_cm)

        # Reset the counters and the date_last_run
        self.module.stunting_incident_case_tracker = copy.deepcopy(self.module.stunting_incident_case_tracker_blank)
        self.date_last_run = self.sim.date


class PropertiesOfOtherModules(Module):
    """For the purpose of the testing, this module generates the properties upon which the Wasting module relies"""

    PROPERTIES = {
        'hv_inf': Property(Types.BOOL, 'temporary property'),
        'hv_art': Property(Types.CATEGORICAL, 'temporary property',
                           categories=['not', 'on_VL_suppressed', 'on_not_VL_suppressed']),
        'nb_low_birth_weight_status': Property(Types.CATEGORICAL, 'temporary property',
                                               categories=['extremely_low_birth_weight', 'very_low_birth_weight',
                                                           'low_birth_weight', 'normal_birth_weight']),
        'nb_size_for_gestational_age': Property(Types.CATEGORICAL, 'temporary property',
                                                categories=['small_for_gestational_age',
                                                            'average_for_gestational_age']),
        'nb_late_preterm': Property(Types.BOOL, 'temporary property'),
        'nb_early_preterm': Property(Types.BOOL, 'temporary property'),

        'nb_breastfeeding_status': Property(Types.CATEGORICAL, 'temporary property',
                                            categories=['none', 'non_exclusive', 'exclusive']),

    }

    def __init__(self, name=None):
        super().__init__(name)

    def read_parameters(self, data_folder):
        pass

    def initialise_population(self, population):
        df = population.props
        df.loc[df.is_alive, 'hv_inf'] = False
        df.loc[df.is_alive, 'hv_art'] = 'not'
        df.loc[df.is_alive, 'nb_low_birth_weight_status'] = 'normal_birth_weight'
        df.loc[df.is_alive, 'nb_breastfeeding_status'] = 'exclusive'
        df.loc[df.is_alive, 'nb_size_for_gestational_age'] = 'average_for_gestational_age'
        df.loc[df.is_alive, 'nb_late_preterm'] = False
        df.loc[df.is_alive, 'nb_early_preterm'] = False

    def initialise_simulation(self, sim):
        pass

    def on_birth(self, mother, child):
        df = self.sim.population.props
        df.at[child, 'hv_inf'] = False
        df.at[child, 'hv_art'] = 'not'
        df.at[child, 'nb_low_birth_weight_status'] = 'normal_birth_weight'
        df.at[child, 'nb_breastfeeding_status'] = 'exclusive'
        df.at[child, 'nb_size_for_gestational_age'] = 'average_for_gestational_age'
        df.at[child, 'nb_late_preterm'] = False
        df.at[child, 'nb_early_preterm'] = False
