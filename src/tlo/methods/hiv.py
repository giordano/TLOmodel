"""
The HIV Module
Overview:
HIV infection ---> AIDS onset Event (defined by the presence of AIDS symptoms) --> AIDS Death Event
Testing is spontaneously taken-up and can lead to accessing intervention services (ART, VMMC, PrEP).
AIDS symptoms can also lead to care-seeking and there is routine testing for HIV at all non-emergency Generic HSI
 events.
Persons can be on ART -
    - with viral suppression: when the person with not develop AIDS, or if they have already, it is relieved and they
        will not die of AIDS; and the person is not infectious
    - without viral suppression: when there is no benefit in avoiding AIDS and infectiousness is unchanged.
Maintenance on ART and PrEP is re-assessed at periodic 'Decision Events', at which is it is determined if the person
  will attend the "next" HSI for continuation of the service; and if not, they are removed from that service and "stop
  treatment". If a stock-out or non-availability of health system resources prevent treatment continuation, the person
  "stops treatment". Stopping treatment leads to a new AIDS Event being scheduled. Persons can restart treatment. If a
  person has developed AIDS, starts treatment and then defaults from treatment, their 'original' AIDS Death Event will
  still occur.
If PrEP is not available due to limitations in the HealthSystem, the person defaults to not being on PrEP.
# Things to note:
    * Need to incorporate testing for HIV at first ANC appointment (as it does in generic HSI)
    * Need to incorporate testing for infants born to HIV-positive mothers (currently done in on_birth here).
    * Need to incorporate cotrim for infants born to HIV-positive mothers (not done here)
    * Cotrimoxazole is not included - either in effect of consumption of the drug (because the effect is not known).
    * Calibration has not been done: most things look OK - except HIV-AIDS deaths
"""

import os

import numpy as np
import pandas as pd

from tlo import DAYS_IN_YEAR, DateOffset, Module, Parameter, Property, Types, logging
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.lm import LinearModel, LinearModelType, Predictor
from tlo.methods import Metadata, demography
from tlo.methods.dxmanager import DxTest
from tlo.methods.healthsystem import HSI_Event
from tlo.methods.symptommanager import Symptom
from tlo.util import create_age_range_lookup

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Hiv(Module):
    """
    The HIV Disease Module
    """

    def __init__(self, name=None, resourcefilepath=None, run_with_checks=False):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath

        assert isinstance(run_with_checks, bool)
        self.run_with_checks = run_with_checks

        self.stored_test_numbers = []  # create empty list for storing hiv test numbers

        self.daly_wts = dict()
        self.lm = dict()
        self.footprints_for_consumables_required = dict()

    METADATA = {
        Metadata.DISEASE_MODULE,
        Metadata.USES_SYMPTOMMANAGER,
        Metadata.USES_HEALTHSYSTEM,
        Metadata.USES_HEALTHBURDEN
    }

    PROPERTIES = {
        # --- Core Properties
        "hv_inf": Property(
            Types.BOOL,
            "Is person currently infected with HIV (NB. AIDS status is determined by prescence of the AIDS Symptom."),
        "hv_art": Property(
            Types.CATEGORICAL,
            "ART status of person, whether on ART or not; and whether viral load is suppressed or not if on ART.",
            categories=["not", "on_VL_suppressed", "on_not_VL_suppressed"]),
        "hv_is_on_prep": Property(
            Types.BOOL,
            "Whether the person is currently taking and receiving a protective effect from Pre-Exposure Prophylaxis."),
        "hv_behaviour_change": Property(
            Types.BOOL,
            "Has this person been exposed to HIV prevention counselling following a negative HIV test result"),
        "hv_diagnosed": Property(Types.BOOL, "Knows that they are HIV+: i.e. is HIV+ and tested as HIV+"),
        "hv_number_tests": Property(Types.INT, "Number of HIV tests ever taken"),
        "hv_last_test_date": Property(Types.DATE, "Date of last HIV test"),

        # --- Dates on which things have happened:
        "hv_date_inf": Property(Types.DATE, "Date infected with HIV"),

    }

    PARAMETERS = {
        # Baseline characteristics
        "time_inf": Parameter(Types.DATA_FRAME, "prob of time since infection for baseline adult pop"),
        "art_coverage": Parameter(Types.DATA_FRAME, "coverage of ART at baseline"),

        "fraction_of_those_infected_that_have_aids_at_initiation": Parameter(
            Types.REAL, "Fraction of persons living with HIV at baseline that have developed AIDS"),
        "testing_coverage_male": Parameter(Types.REAL, "proportion of adult male population tested"),
        "testing_coverage_female": Parameter(Types.REAL, "proportion of adult female population tested"),

        # Natural history - transmission - overall rates
        "beta": Parameter(Types.REAL, "Transmission rate"),
        "prob_mtct_untreated": Parameter(Types.REAL, "Probability of mother to child transmission"),
        "prob_mtct_treated": Parameter(Types.REAL, "Probability of mother to child transmission, mother on ART"),
        "prob_mtct_incident_preg": Parameter(
            Types.REAL, "Probability of mother to child transmission, mother infected during pregnancy"),
        "monthly_prob_mtct_bf_untreated": Parameter(
            Types.REAL, "Probability of mother to child transmission during breastfeeding"),
        "monthly_prob_mtct_bf_treated": Parameter(
            Types.REAL, "Probability of mother to child transmission, mother infected during breastfeeding"),

        # Natural history - transmission - relative risk of HIV acquisition (non-intervention)
        "rr_fsw": Parameter(Types.REAL, "Relative risk of HIV with female sex work"),
        "rr_circumcision": Parameter(Types.REAL, "Relative risk of HIV with circumcision"),
        "rr_rural": Parameter(Types.REAL, "Relative risk of HIV in rural location"),
        "rr_windex_poorer": Parameter(Types.REAL, "Relative risk of HIV with wealth level poorer"),
        "rr_windex_middle": Parameter(Types.REAL, "Relative risk of HIV with wealth level middle"),
        "rr_windex_richer": Parameter(Types.REAL, "Relative risk of HIV with wealth level richer"),
        "rr_windex_richest": Parameter(Types.REAL, "Relative risk of HIV with wealth level richest"),
        "rr_sex_f": Parameter(Types.REAL, "Relative risk of HIV if female"),
        "rr_age_gp20": Parameter(Types.REAL, "Relative risk of HIV if age 20-24 compared with 15-19"),
        "rr_age_gp25": Parameter(Types.REAL, "Relative risk of HIV if age 25-29"),
        "rr_age_gp30": Parameter(Types.REAL, "Relative risk of HIV if age 30-34"),
        "rr_age_gp35": Parameter(Types.REAL, "Relative risk of HIV if age 35-39"),
        "rr_age_gp40": Parameter(Types.REAL, "Relative risk of HIV if age 40-44"),
        "rr_age_gp45": Parameter(Types.REAL, "Relative risk of HIV if age 45-49"),
        "rr_age_gp50": Parameter(Types.REAL, "Relative risk of HIV if age 50+"),
        "rr_edlevel_primary": Parameter(Types.REAL, "Relative risk of HIV with primary education"),
        "rr_edlevel_secondary": Parameter(Types.REAL, "Relative risk of HIV with secondary education"),
        "rr_edlevel_higher": Parameter(Types.REAL, "Relative risk of HIV with higher education"),

        # Natural history - transmission - relative risk of HIV acquisition (interventions)
        "rr_behaviour_change": Parameter(Types.REAL, "Relative risk of HIV with behaviour modification"),
        "proportion_reduction_in_risk_of_hiv_aq_if_on_prep": Parameter(
            Types.REAL,
            "Proportion reduction in risk of HIV acquisition if on PrEP. 0 for no efficacy; 1.0 for perfect efficacy."),

        # Natural history - survival (adults)
        "mean_months_between_aids_and_death": Parameter(
            Types.REAL, "Mean number of months (distributed exponentially) for the time between AIDS and AIDS Death"),
        "mean_months_between_aids_and_death_infant": Parameter(
            Types.REAL, "Mean number of months for the time between AIDS and AIDS Death for infants"),
        "infection_to_death_weibull_shape_1519": Parameter(
            Types.REAL,
            "Shape parameter for Weibull describing time between infection and death for 15-19 yo (units: years)"),
        "infection_to_death_weibull_shape_2024": Parameter(
            Types.REAL,
            "Shape parameter for Weibull describing time between infection and death for 20-24 yo (units: years)"),
        "infection_to_death_weibull_shape_2529": Parameter(
            Types.REAL,
            "Shape parameter for Weibull describing time between infection and death for 25-29 yo (units: years)"),
        "infection_to_death_weibull_shape_3034": Parameter(
            Types.REAL,
            "Shape parameter for Weibull describing time between infection and death for 30-34 yo (units: years)"),
        "infection_to_death_weibull_shape_3539": Parameter(
            Types.REAL,
            "Shape parameter for Weibull describing time between infection and death for 35-39 yo (units: years)"),
        "infection_to_death_weibull_shape_4044": Parameter(
            Types.REAL,
            "Shape parameter for Weibull describing time between infection and death for 40-44 yo (units: years)"),
        "infection_to_death_weibull_shape_4549": Parameter(
            Types.REAL,
            "Shape parameter for Weibull describing time between infection and death for 45-49 yo (units: years)"),
        "infection_to_death_weibull_scale_1519": Parameter(
            Types.REAL,
            "Scale parameter for Weibull describing time between infection and death for 15-19 yo (units: years)"),
        "infection_to_death_weibull_scale_2024": Parameter(
            Types.REAL,
            "Scale parameter for Weibull describing time between infection and death for 20-24 yo (units: years)"),
        "infection_to_death_weibull_scale_2529": Parameter(
            Types.REAL,
            "Scale parameter for Weibull describing time between infection and death for 25-29 yo (units: years)"),
        "infection_to_death_weibull_scale_3034": Parameter(
            Types.REAL,
            "Scale parameter for Weibull describing time between infection and death for 30-34 yo (units: years)"),
        "infection_to_death_weibull_scale_3539": Parameter(
            Types.REAL,
            "Scale parameter for Weibull describing time between infection and death for 35-39 yo (units: years)"),
        "infection_to_death_weibull_scale_4044": Parameter(
            Types.REAL,
            "Scale parameter for Weibull describing time between infection and death for 40-44 yo (units: years)"),
        "infection_to_death_weibull_scale_4549": Parameter(
            Types.REAL,
            "Scale parameter for Weibull describing time between infection and death for 45-49 yo (units: years)"),
        "art_default_to_aids_mean_years": Parameter(
            Types.REAL,
            "Mean years between when a person (any change) stops being on treatment to when AIDS is onset (if the "
            "absence of resuming treatment)."),

        # Natural history - survival (children)
        "mean_survival_for_infants_infected_prior_to_birth": Parameter(
            Types.REAL, "Exponential rate parameter for mortality in infants who are infected before birth"),
        "infection_to_death_infant_infection_after_birth_weibull_scale": Parameter(
            Types.REAL, "Weibull scale parameter for mortality in infants who are infected after birth"),
        "infection_to_death_infant_infection_after_birth_weibull_shape": Parameter(
            Types.REAL, "Weibull shape parameter for mortality in infants who are infected after birth"),

        # Uptake of Interventions
        "prob_spontaneous_test_12m": Parameter(
            Types.REAL, "Probability that a person will seek HIV testing per 12 month period."),
        "prob_start_art_after_hiv_test": Parameter(
            Types.REAL, "Probability that a person will start treatment, if HIV-positive, following testing"),
        "rr_start_art_if_aids_symptoms": Parameter(
            Types.REAL, "Relative probability of a person starting treatment if they have aids_symptoms compared to if"
                        "they do not."
        ),
        "prob_behav_chg_after_hiv_test": Parameter(
            Types.REAL, "Probability that a person will change risk behaviours, if HIV-negative, following testing"),
        "prob_prep_for_fsw_after_hiv_test": Parameter(
            Types.REAL, "Probability that a FSW will start PrEP, if HIV-negative, following testing"),
        "prob_circ_after_hiv_test": Parameter(
            Types.REAL, "Probability that a male will be circumcised, if HIV-negative, following testing"),
        "probability_of_being_retained_on_prep_every_3_months": Parameter(
            Types.REAL, "Probability that someone who has initiated on prep will attend an appointment and be on prep "
                        "for the next 3 months, until the next appointment."),
        "probability_of_being_retained_on_art_every_6_months": Parameter(
            Types.REAL, "Probability that someone who has initiated on treatment will attend an appointment and be on "
                        "treatment for next 6 months, until the next appointment."),
        "probability_of_seeking_further_art_appointment_if_drug_not_available": Parameter(
            Types.REAL, "Probability that a person who 'should' be on art will seek another appointment (the following "
                        "day and try for each of the next 7 days) if drugs were not available."),
        "probability_of_seeking_further_art_appointment_if_appointment_not_available": Parameter(
            Types.REAL, "Probability that a person who 'should' be on art will seek another appointment if the health-"
                        "system has not been able to provide them with an appointment"),
        "vls_m": Parameter(Types.REAL, "Rates of viral load suppression males"),
        "vls_f": Parameter(Types.REAL, "Rates of viral load suppression males"),
        "vls_child": Parameter(Types.REAL, "Rates of viral load suppression in children 0-14 years"),
        "prep_start_year": Parameter(Types.REAL, "Year from which PrEP is available")
    }

    def read_parameters(self, data_folder):
        """
        * 1) Reads the ResourceFiles
        * 2) Declare the Symptoms
        """

        # 1) Read the ResourceFiles

        # Short cut to parameters dict
        p = self.parameters

        workbook = pd.read_excel(
            os.path.join(self.resourcefilepath, "ResourceFile_HIV.xlsx"),
            sheet_name=None,
        )
        self.load_parameters_from_dataframe(workbook["parameters"])

        # Load data on HIV prevalence
        p["hiv_prev"] = workbook["hiv_prevalence"]

        # Load assumed time since infected at baseline (year 2010)
        p["time_inf"] = workbook["time_since_infection_at_baselin"]

        # Load assumed ART coverage at baseline (year 2010)
        p["art_coverage"] = workbook["art_coverage"]

        # DALY weights
        # get the DALY weight that this module will use from the weight database (these codes are just random!)
        if "HealthBurden" in self.sim.modules.keys():
            # Chronic infection but not AIDS (including if on ART)
            # (taken to be equal to "Symptomatic HIV without anaemia")
            self.daly_wts['hiv_infection_but_not_aids'] = self.sim.modules["HealthBurden"].get_daly_weight(17)

            #  AIDS without anti-retroviral treatment without anemia
            self.daly_wts['aids'] = self.sim.modules["HealthBurden"].get_daly_weight(19)

        # 2)  Declare the Symptoms.
        self.sim.modules['SymptomManager'].register_symptom(
            Symptom(name="aids_symptoms",
                    odds_ratio_health_seeking_in_adults=3.0,  # High chance of seeking care when aids_symptoms onset
                    odds_ratio_health_seeking_in_children=3.0)  # High chance of seeking care when aids_symptoms onset
        )

    def pre_initialise_population(self):
        """
        * Establish the Linear Models
        *
        """
        p = self.parameters

        # ---- LINEAR MODELS -----
        # LinearModel for the relative risk of becoming infected during the simulation
        self.lm['rr_of_infection'] = LinearModel.multiplicative(
            Predictor('age_years')  .when('<15', 0.0)
                                    .when('<20', 1.0)
                                    .when('<25', p["rr_age_gp20"])
                                    .when('<30', p["rr_age_gp25"])
                                    .when('<35', p["rr_age_gp30"])
                                    .when('<40', p["rr_age_gp35"])
                                    .when('<45', p["rr_age_gp40"])
                                    .when('<50', p["rr_age_gp45"])
                                    .when('<80', p["rr_age_gp50"])
                                    .otherwise(0.0),
            Predictor('sex').when('F', p["rr_sex_f"]),
            Predictor('li_is_sexworker').when(True, p["rr_fsw"]),
            Predictor('li_is_circ').when(True, p["rr_circumcision"]),
            Predictor('hv_is_on_prep').when(True, 1.0 - p['proportion_reduction_in_risk_of_hiv_aq_if_on_prep']),
            Predictor('li_urban').when(False, p["rr_rural"]),
            Predictor('li_wealth')  .when(2, p["rr_windex_poorer"])
                                    .when(3, p["rr_windex_middle"])
                                    .when(4, p["rr_windex_richer"])
                                    .when(5, p["rr_windex_richest"]),
            Predictor('li_ed_lev')  .when(2, p["rr_edlevel_primary"])
                                    .when(3, p["rr_edlevel_secondary"]),
            Predictor('hv_behaviour_change').when(True, p["rr_behaviour_change"])
        )

        # LinearModels to give the shape and scale for the Weibull distribution describing time from infection to death
        self.lm['scale_parameter_for_infection_to_death'] = LinearModel.multiplicative(
            Predictor('age_years')  .when('<20', p["infection_to_death_weibull_scale_1519"])
                                    .when('<25', p["infection_to_death_weibull_scale_2024"])
                                    .when('<30', p["infection_to_death_weibull_scale_2529"])
                                    .when('<35', p["infection_to_death_weibull_scale_3034"])
                                    .when('<40', p["infection_to_death_weibull_scale_3539"])
                                    .when('<45', p["infection_to_death_weibull_scale_4044"])
                                    .when('<50', p["infection_to_death_weibull_scale_4549"])
                                    .otherwise(p["infection_to_death_weibull_scale_4549"])
        )

        self.lm['shape_parameter_for_infection_to_death'] = LinearModel.multiplicative(
            Predictor('age_years')  .when('<20', p["infection_to_death_weibull_shape_1519"])
                                    .when('<25', p["infection_to_death_weibull_shape_2024"])
                                    .when('<30', p["infection_to_death_weibull_shape_2529"])
                                    .when('<35', p["infection_to_death_weibull_shape_3034"])
                                    .when('<40', p["infection_to_death_weibull_shape_3539"])
                                    .when('<45', p["infection_to_death_weibull_shape_4044"])
                                    .when('<50', p["infection_to_death_weibull_shape_4549"])
                                    .otherwise(p["infection_to_death_weibull_shape_4549"])
        )

        # -- Linear Models for the Uptake of Services
        # Linear model that give the probability of seeking a 'Spontaneous' Test for HIV
        # (= sum of probabilities for accessing any HIV service when not ill)

        self.lm['lm_spontaneous_test_12m'] = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            p["prob_spontaneous_test_12m"],
            Predictor('hv_diagnosed').when(True, 0.0).otherwise(1.0)
        )

        # Linear model if the person will start ART, following when the person has been diagnosed:
        self.lm['lm_art'] = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            p["prob_start_art_after_hiv_test"],
            Predictor('hv_inf').when(True, 1.0).otherwise(0.0),
            Predictor('has_aids_symptoms', external=True).when(True, p["rr_start_art_if_aids_symptoms"])
        )

        # Linear model for changing behaviour following an HIV-negative test
        self.lm['lm_behavchg'] = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            p["prob_behav_chg_after_hiv_test"],
            Predictor('hv_inf').when(False, 1.0).otherwise(0.0)
        )

        # Linear model for starting PrEP (if F/sex-workers), following when the person has tested HIV -ve:
        self.lm['lm_prep'] = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            p["prob_prep_for_fsw_after_hiv_test"],
            Predictor('hv_inf').when(False, 1.0).otherwise(0.0),
            Predictor('sex').when('F', 1.0).otherwise(0.0),
            Predictor('li_is_sexworker').when(True, 1.0).otherwise(0.0)
        )

        # Linear model for circumcision (if M) following when the person has been diagnosed:
        self.lm['lm_circ'] = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            p["prob_circ_after_hiv_test"],
            Predictor('hv_inf').when(False, 1.0).otherwise(0.0),
            Predictor('sex').when('M', 1.0).otherwise(0.0),
        )

    def initialise_population(self, population):
        """Set our property values for the initial population.
        """

        df = population.props

        # --- Current status
        df.loc[df.is_alive, "hv_inf"] = False
        df.loc[df.is_alive, "hv_art"] = "not"
        df.loc[df.is_alive, "hv_is_on_prep"] = False
        df.loc[df.is_alive, "hv_behaviour_change"] = False
        df.loc[df.is_alive, "hv_diagnosed"] = False
        df.loc[df.is_alive, "hv_number_tests"] = 0

        # --- Dates on which things have happened
        df.loc[df.is_alive, "hv_date_inf"] = pd.NaT
        df.loc[df.is_alive, "hv_last_test_date"] = pd.NaT

        # Launch sub-routines for allocating the right number of people into each category
        self.initialise_baseline_prevalence(population)  # allocate baseline prevalence
        self.initialise_baseline_art(population)  # allocate baseline art coverage
        self.initialise_baseline_tested(population)  # allocate baseline art coverage

    def initialise_baseline_prevalence(self, population):
        """
        Assign baseline HIV prevalence, according to age, sex and key other variables (established in analysis of DHS
        data).
        """

        params = self.parameters
        df = population.props

        # prob of infection based on age and sex in baseline year (2010:
        prevalence_db = params["hiv_prev"]
        prev_2010 = prevalence_db.loc[prevalence_db.year == 2010, ['age_from', 'sex', 'prev_prop']]
        prev_2010 = prev_2010.rename(columns={'age_from': 'age_years'})
        prob_of_infec = df.loc[df.is_alive, ['age_years', 'sex']].merge(prev_2010, on=['age_years', 'sex'], how='left')[
            'prev_prop']

        # probability based on risk factors
        rel_prob_by_risk_factor = LinearModel.multiplicative(
            Predictor("li_is_sexworker").when(True, params["rr_fsw"]),
            Predictor("li_is_circ").when(True, params["rr_circumcision"]),
            Predictor("li_urban").when(False, params["rr_rural"]),
            Predictor("li_wealth")  .when(2, params["rr_windex_poorer"])
                                    .when(3, params["rr_windex_middle"])
                                    .when(4, params["rr_windex_richer"])
                                    .when(5, params["rr_windex_richest"]),
            Predictor("li_ed_lev")  .when(2, params["rr_edlevel_primary"])
                                    .when(3, params["rr_edlevel_secondary"])
        ).predict(df.loc[df.is_alive])

        # Rescale relative probability of infection so that its average is 1.0 within each age/sex group
        p = pd.DataFrame({
            'age_years': df['age_years'],
            'sex': df['sex'],
            'prob_of_infec': prob_of_infec,
            'rel_prob_by_risk_factor': rel_prob_by_risk_factor
        })

        p['mean_of_rel_prob_within_age_sex_group'] = p.groupby(['age_years', 'sex'])[
            'rel_prob_by_risk_factor'].transform('mean')
        p['scaled_rel_prob_by_risk_factor'] = p['rel_prob_by_risk_factor'] / p['mean_of_rel_prob_within_age_sex_group']
        p['overall_prob_of_infec'] = p['scaled_rel_prob_by_risk_factor'] * p['prob_of_infec']
        infec = self.rng.random_sample(len(p['overall_prob_of_infec'])) < p['overall_prob_of_infec']

        # Assign the designated person as infected in the population.props dataframe:
        df.loc[infec, 'hv_inf'] = True

        # Assign date that persons were infected by drawing from assumed distribution (for adults)
        # Clipped to prevent dates of infection before before the person was born.
        years_ago_inf = self.rng.choice(
            self.time_inf["year"],
            size=len(infec),
            replace=True,
            p=self.time_inf["scaled_prob"],
        )

        hv_date_inf = pd.Series(self.sim.date - pd.to_timedelta(years_ago_inf * DAYS_IN_YEAR, unit="d"))
        df.loc[infec, "hv_date_inf"] = hv_date_inf.clip(lower=df.date_of_birth)

    def initialise_baseline_art(self, population):
        """ assign initial art coverage levels
        also assign hiv test properties if allocated ART
        """
        df = population.props

        # 1) Determine who is currently on ART
        worksheet = self.parameters["art_coverage"]
        art_data = worksheet.loc[
            worksheet.year == 2010, ["year", "single_age", "sex", "prop_coverage"]
        ]

        # merge all susceptible individuals with their coverage probability based on sex and age
        prob_art = df.loc[df.is_alive, ['age_years', 'sex']].merge(
            art_data,
            left_on=["age_years", "sex"],
            right_on=["single_age", "sex"],
            how="left",
        )['prop_coverage']

        prob_art = prob_art.fillna(0)

        art_idx = prob_art.index[
            (self.rng.random_sample(len(prob_art)) < prob_art)
            & df.is_alive
            & df.hv_inf
            ]

        # 2) Determine adherence levels for those currently on ART, for each of adult men, adult women and children
        adult_f_art_idx = df.loc[(df.index.isin(art_idx) & (df.sex == 'F') & (df.age_years >= 15))].index
        adult_m_art_idx = df.loc[(df.index.isin(art_idx) & (df.sex == 'M') & (df.age_years >= 15))].index
        child_art_idx = df.loc[(df.index.isin(art_idx) & (df.age_years < 15))].index

        suppr = list()  # list of all indices for persons on ART and suppressed
        notsuppr = list()  # list of all indices for persons on ART and not suppressed

        def split_into_vl_and_notvl(all_idx, prob):
            vl_suppr = self.rng.random_sample(len(all_idx)) < prob
            suppr.extend(all_idx[vl_suppr])
            notsuppr.extend(all_idx[~vl_suppr])

        split_into_vl_and_notvl(adult_f_art_idx, self.parameters['vls_f'])
        split_into_vl_and_notvl(adult_m_art_idx, self.parameters['vls_m'])
        split_into_vl_and_notvl(child_art_idx, self.parameters['vls_child'])

        # Set ART status:
        df.loc[suppr, "hv_art"] = "on_VL_suppressed"
        df.loc[notsuppr, "hv_art"] = "on_not_VL_suppressed"

        # check that everyone on ART is labelled as such
        assert not (df.loc[art_idx, "hv_art"] == "not").any()

        # for logical consistency, ensure that all persons on ART have been tested and diagnosed
        df.loc[art_idx, "hv_number_tests"] = 1
        df.loc[art_idx, "hv_last_test_date"] = self.sim.date
        df.loc[art_idx, "hv_diagnosed"] = True

        # all those on ART need to have event scheduled for continuation/cessation of treatment
        for person in art_idx:
            days = self.rng.randint(low=100, high=200, size=1, dtype=np.int64)[0]
            self.sim.schedule_event(
                Hiv_DecisionToContinueTreatment(person_id=person, module=self),
                self.sim.date + pd.to_timedelta(days)
            )

    def initialise_baseline_tested(self, population):
        """ assign initial hiv testing levels, only for adults
        all who have been allocated ART will already have property hv_number_tests=1
        use the hiv testing coverage levels to assign any remaining hiv tests
        """
        df = population.props
        p = self.parameters

        random_draw = self.rng.random_sample(size=len(df))
        testing_dict = {'F': p['testing_coverage_female'], 'M': p['testing_coverage_male']}

        # test_index = {}
        for sex in ['F', 'M']:
            hiv_test = len(df[df.is_alive & (df.hv_number_tests > 0) & (df.sex == sex) & (df.age_years >= 15)])
            pop = len(df[df.is_alive & (df.sex == sex) & (df.age_years >= 15)])
            hiv_test_coverage = hiv_test / pop
            hiv_test_deficit = testing_dict[sex] - hiv_test_coverage

            if hiv_test_deficit > 0:
                # assign more tests to fill testing coverage deficit
                test_index = df.index[
                    (random_draw < hiv_test_deficit)
                    & df.is_alive
                    & (df.sex == sex)
                    & (df.age_years >= 15)
                    ]

                # assign hiv tests to males and females
                df.loc[test_index, 'hv_number_tests'] = 1
                # dummy date for date last hiv test (before sim start), otherwise see big spike in testing 01-01-2010
                df.loc[test_index, 'hv_last_test_date'] = self.sim.date - pd.DateOffset(years=3)

        # person assumed to be diagnosed if they have had a test and are currently HIV positive:
        df.loc[((df.hv_number_tests > 0) & df.is_alive & df.hv_inf), 'hv_diagnosed'] = True

    def initialise_simulation(self, sim):
        """
        * 1) Schedule the Main HIV Regular Polling Event
        * 2) Schedule the Logging Event
        * 3) Determine who has AIDS and impose the Symptoms 'aids_symptoms'
        * 4) Schedule the AIDS onset events and AIDS death event for those infected already
        * 5) (Optionally) Schedule the event to check the configuration of all properties
        * 6) Define the DxTests
        * 7) Look-up and save the codes for consumables
        """
        df = sim.population.props

        # 1) Schedule the Main HIV Regular Polling Event
        sim.schedule_event(HivRegularPollingEvent(self), sim.date)

        # 2) Schedule the Logging Event
        sim.schedule_event(HivLoggingEvent(self), sim.date)

        # 3) Determine who has AIDS and impose the Symptoms 'aids_symptoms'

        # Those on ART currently (will not get any further events scheduled):
        on_art_idx = df.loc[
            df.is_alive &
            df.hv_inf &
            (df.hv_art != "not")
            ].index

        # Those that lived more than ten years and not currently on ART are assumed to currently have AIDS
        #  (will have AIDS Death event scheduled)
        has_aids_idx = df.loc[
            df.is_alive &
            df.hv_inf &
            ((self.sim.date - df.hv_date_inf).dt.days > 10 * 365) &
            (df.hv_art == "not")
            ].index

        # Those that are in neither category are "before AIDS" (will have AIDS Onset Event scheduled)
        before_aids_idx = set(df.loc[df.is_alive & df.hv_inf].index) - set(has_aids_idx) - set(on_art_idx)

        # Impose the symptom to those that have AIDS (the symptom is the definition of having AIDS)
        self.sim.modules['SymptomManager'].change_symptom(
            person_id=has_aids_idx.tolist(),
            symptom_string='aids_symptoms',
            add_or_remove='+',
            disease_module=self
        )

        # 4) Schedule the AIDS onset events and AIDS death event for those infected already
        # AIDS Onset Event for those who are infected but not yet AIDS and have not ever started ART
        # NB. This means that those on ART at the start of the simulation may not have an AIDS event --
        # like it happened at some point in the past

        for person_id in before_aids_idx:
            # get days until develops aids, repeating sampling until a positive number is obtained.
            days_until_aids = 0
            while days_until_aids <= 0:
                days_since_infection = (self.sim.date - df.at[person_id, 'hv_date_inf']).days
                days_infection_to_aids = np.round((self.get_time_from_infection_to_aids(person_id)).months * 30.5)
                days_until_aids = days_infection_to_aids - days_since_infection

            date_onset_aids = self.sim.date + pd.DateOffset(days=days_until_aids)
            sim.schedule_event(
                HivAidsOnsetEvent(person_id=person_id, module=self),
                date=date_onset_aids
            )

        # Schedule the AIDS death events for those who have got AIDS already
        for person_id in has_aids_idx:
            date_aids_death = self.sim.date + self.get_time_from_aids_to_death()  # (assumes AIDS onset on this day)
            sim.schedule_event(
                HivAidsDeathEvent(person_id=person_id, module=self),
                date=date_aids_death
            )

        # 5) (Optionally) Schedule the event to check the configuration of all properties
        if self.run_with_checks:
            sim.schedule_event(HivCheckPropertiesEvent(self), sim.date + pd.DateOffset(months=1))

        # 6) Define the DxTests
        # HIV Rapid Diagnostic Test:

        consumables = self.sim.modules["HealthSystem"].parameters["Consumables"]
        pkg_code_hiv_rapid_test = consumables.loc[
            consumables["Intervention_Pkg"] == "HIV Testing Services",
            "Intervention_Pkg_Code"].values[0]

        # NB. The rapid test is assumed to be 100% specific and sensitive. This is used to guarantee that all persons
        #  that start ART are truly HIV-pos.
        self.sim.modules['HealthSystem'].dx_manager.register_dx_test(
            hiv_rapid_test=DxTest(
                property='hv_inf',
                cons_req_as_footprint={'Intervention_Package_Code': {pkg_code_hiv_rapid_test: 1}, 'Item_Code': {}}
            )
        )

        # Test for Early Infect Diagnosis
        #  - Consumables required:
        item1 = pd.unique(
            consumables.loc[
                consumables["Items"] == "Blood collecting tube, 5 ml", "Item_Code"
            ]
        )[0]
        item2 = pd.unique(
            consumables.loc[
                consumables["Items"] == "Gloves, exam, latex, disposable, pair",
                "Item_Code",
            ]
        )[0]
        item3 = pd.unique(
            consumables.loc[consumables["Items"] == "HIV EIA Elisa test", "Item_Code"]
        )[0]

        self.sim.modules['HealthSystem'].dx_manager.register_dx_test(
            hiv_early_infant_test=DxTest(
                property='hv_inf',
                sensitivity=1.0,
                specificity=1.0,
                cons_req_as_footprint={'Intervention_Package_Code': {}, 'Item_Code': {item1: 1, item2: 1, item3: 1}}
            )
        )

        # 7) Look-up and store the codes for the consumables used in the interventions.
        consumables = self.sim.modules["HealthSystem"].parameters["Consumables"]

        # Circumcison:
        pkg_codes_for_circ = pd.unique(
            consumables.loc[
                consumables["Intervention_Pkg"] == "Male circumcision ",
                "Intervention_Pkg_Code",
            ]
        )[0]
        self.footprints_for_consumables_required['circ'] = {
            "Intervention_Package_Code": {pkg_codes_for_circ: 1},
            "Item_Code": {}
        }

        # PrEP:
        item_code_for_prep = pd.unique(
            consumables.loc[
                consumables["Items"]
                == "Tenofovir (TDF)/Emtricitabine (FTC), tablet, 300/200 mg",
                "Item_Code",
            ]
        )[0]
        self.footprints_for_consumables_required['prep'] = {
            "Intervention_Package_Code": {},
            "Item_Code": {item_code_for_prep: 1}
        }

        # ART for adults
        item_code_for_art = pd.unique(
            consumables.loc[
                consumables["Items"] == "Adult First line 1A d4T-based", "Item_Code"
            ]
        )[0]
        item_code_for_art2 = pd.unique(
            consumables.loc[
                consumables["Items"] == "Cotrimoxizole, 960mg pppy", "Item_Code"
            ]
        )[0]  # NB spelling error in consumables file "Cotrimoxizole"
        self.footprints_for_consumables_required['art_adult'] = {
            "Intervention_Package_Code": {},
            "Item_Code": {item_code_for_art: 1, item_code_for_art2: 1}
        }

        # ART for children:
        self.footprints_for_consumables_required['art_child'] = {
            "Intervention_Package_Code": {
                pd.unique(consumables.loc[
                              consumables["Intervention_Pkg"] == "Cotrimoxazole for children",
                              "Intervention_Pkg_Code"])[0]: 1},
            "Item_Code": {
                pd.unique(consumables.loc[
                              consumables[
                                  "Items"] == "Lamiduvine/Zidovudine/Nevirapine (3TC + AZT + NVP), tablet, 150 + 300"
                                              " + 200 mg", "Item_Code"])[
                    0]: 1}
        }

        # Viral Load monitoring
        item_code_for_viral_load = pd.unique(
            consumables.loc[
                consumables["Intervention_Pkg"] == "Viral Load", "Intervention_Pkg_Code"
            ]
        )[0]
        self.footprints_for_consumables_required['vl_measurement'] = {
            "Intervention_Package_Code": {},
            "Item_Code": {item_code_for_viral_load: 1}
        }

    def on_birth(self, mother_id, child_id):
        """
        * Initialise our properties for a newborn individual;
        * schedule testing;
        * schedule infection during breastfeeding
        """
        params = self.parameters
        df = self.sim.population.props

        # Default Settings:
        # --- Current status
        df.at[child_id, "hv_inf"] = False
        df.at[child_id, "hv_art"] = "not"
        df.at[child_id, "hv_is_on_prep"] = False
        df.at[child_id, "hv_behaviour_change"] = False
        df.at[child_id, "hv_diagnosed"] = False
        df.at[child_id, "hv_number_tests"] = 0

        # --- Dates on which things have happened
        df.at[child_id, "hv_date_inf"] = pd.NaT
        df.at[child_id, "hv_last_test_date"] = pd.NaT

        # ----------------------------------- MTCT - AT OR PRIOR TO BIRTH --------------------------
        #  DETERMINE IF THE CHILD IS INFECTED WITH HIV FROM THEIR MOTHER DURING PREGNANCY / DELIVERY
        mother = df.loc[mother_id]

        mother_infected_prior_to_pregnancy = mother.hv_inf & (mother.hv_date_inf <= mother.date_of_last_pregnancy)
        mother_infected_during_pregnancy = mother.hv_inf & (mother.hv_date_inf > mother.date_of_last_pregnancy)

        if mother_infected_prior_to_pregnancy:
            if mother.hv_art == "on_VL_suppressed":
                #  mother has existing infection, mother ON ART and VL suppressed at time of delivery
                child_infected = self.rng.random_sample() < params["prob_mtct_treated"]
            else:
                # mother was infected prior to prgenancy but is not on VL suppressed at time of delivery
                child_infected = self.rng.random_sample() < params["prob_mtct_untreated"]

        elif mother_infected_during_pregnancy:
            #  mother has incident infection during pregnancy, NO ART
            child_infected = self.rng.random_sample() < params["prob_mtct_incident_preg"]

        else:
            # mother is not infected
            child_infected = False

        if child_infected:
            self.do_new_infection(child_id)

        # ----------------------------------- MTCT - DURING BREASTFEEDING --------------------------
        # If child is not infected and is being breastfed, then expose them to risk of MTCT through breastfeeding
        # TODO: note for AT/TH - neonatal breastfeeding property replaced HIV temp property as discussed 19/02/21.
        #  We need to make sure newborn outcomes on_birth is always called before HIV so breastfeeding status is set
        #  prior to this function being called
        if (not child_infected) and (df.at[child_id, "nb_breastfeeding_status"] != 'none') and mother.hv_inf:
            self.mtct_during_breastfeeding(mother_id, child_id)

    def on_hsi_alert(self, person_id, treatment_id):
        raise NotImplementedError

    def report_daly_values(self):
        """Report DALYS for HIV, based on current symptomatic state of persons."""
        df = self.sim.population.props

        dalys = pd.Series(data=0, index=df.loc[df.is_alive].index)

        # All those infected get the 'infected but not AIDS' daly_wt:
        dalys.loc[df.hv_inf] = self.daly_wts['hiv_infection_but_not_aids']

        # Overwrite the value for those that currently have symptoms of AIDS with the 'AIDS' daly_wt:
        dalys.loc[self.sim.modules['SymptomManager'].who_has('aids_symptoms')] = self.daly_wts['aids']

        dalys.name = 'hiv'
        return dalys

    def mtct_during_breastfeeding(self, mother_id, child_id):
        """
        Compute risk of mother-to-child transmission and schedule HivInfectionDuringBreastFeedingEvent.
        If the child is breastfeeding currently, consider the time-until-infection assuming a constantly monthly risk of
         transmission. If the breastfeeding has ceased by the time of the scheduled infection, then it will not run.
        (This means that this event can be run at birth or at the time of the mother's infection without the need for
        further polling etc.)
        """

        df = self.sim.population.props
        params = self.parameters

        if df.at[mother_id, "hv_art"] == "on_VL_suppressed":
            monthly_prob_mtct_bf = params["monthly_prob_mtct_bf_treated"]
        else:
            monthly_prob_mtct_bf = params["monthly_prob_mtct_bf_untreated"]

        if monthly_prob_mtct_bf > 0.0:
            months_to_infection = int(self.rng.exponential(1 / monthly_prob_mtct_bf))
            date_of_infection = self.sim.date + pd.DateOffset(months=months_to_infection)
            self.sim.schedule_event(
                HivInfectionDuringBreastFeedingEvent(person_id=child_id, module=self),
                date_of_infection
            )

    def do_new_infection(self, person_id):
        """
        Enact that this person is infected with HIV
        * Update their hv_inf status and hv_date_inf
        * Schedule the AIDS onset event for this person
        """
        df = self.sim.population.props

        # Update HIV infection status for this person
        df.at[person_id, "hv_inf"] = True
        df.at[person_id, "hv_date_inf"] = self.sim.date

        # Schedule AIDS onset events for this person
        date_onset_aids = self.sim.date + self.get_time_from_infection_to_aids(person_id=person_id)
        self.sim.schedule_event(event=HivAidsOnsetEvent(self, person_id), date=date_onset_aids)

    def get_time_from_infection_to_aids(self, person_id):
        """Gives time between onset of infection and AIDS, returning a pd.DateOffset.
        For those infected prior to, or at, birth: (this is a draw from an exponential distribution)
        For those infected after birth but before reaching age 5.0 (this is drawn from a weibull distribution)
        For adults: (this is a drawn from a weibull distribution (with scale depending on age);
        * NB. It is further assumed that the time from aids to death is 18 months.
        """

        df = self.sim.population.props
        age = df.at[person_id, 'age_exact_years']
        p = self.parameters

        if age == 0.0:
            # The person is infected prior to, or at, birth:
            months_to_death = int(
                max(0.0, self.rng.exponential(scale=p["mean_survival_for_infants_infected_prior_to_birth"]) * 12))
            months_to_aids = int(
                max(0.0, np.round(months_to_death - self.parameters['mean_months_between_aids_and_death_infant'])))
        elif age < 5.0:
            # The person is infected after birth but before age 5.0:
            months_to_death = int(
                max(0.0, self.rng.weibull(
                    p["infection_to_death_infant_infection_after_birth_weibull_shape"]) *
                    p["infection_to_death_infant_infection_after_birth_weibull_scale"] * 12)
            )
            months_to_aids = int(
                max(0.0, np.round(months_to_death - self.parameters['mean_months_between_aids_and_death_infant'])))
        else:
            # The person is infected after age 5.0
            # - get the shape parameters (unit: years)
            scale = self.lm['scale_parameter_for_infection_to_death'].predict(
                self.sim.population.props.loc[[person_id]]).values[0]
            # - get the scale parameter (unit: years)
            shape = self.lm['shape_parameter_for_infection_to_death'].predict(
                self.sim.population.props.loc[[person_id]]).values[0]
            # - draw from Weibull and convert to months
            months_to_death = self.rng.weibull(shape) * scale * 12
            # - compute months to aids, which is somewhat shorter than the months to death
            months_to_aids = int(
                max(0.0, np.round(months_to_death - self.parameters['mean_months_between_aids_and_death'])))

        return pd.DateOffset(months=months_to_aids)

    def get_time_from_aids_to_death(self):
        """Gives time between onset of AIDS and death, returning a pd.DateOffset.
        Assumes that the time between onset of AIDS symptoms and deaths is exponentially distributed.
        """
        mean = self.parameters['mean_months_between_aids_and_death']
        draw_number_of_months = int(np.round(self.rng.exponential(mean)))
        return pd.DateOffset(months=draw_number_of_months)

    def do_when_hiv_diagnosed(self, person_id):
        """Things to do when a person has been tested and found (newly) be be HIV-positive:.
        * Consier if ART should be initiated, and schedule HSI if so.
        The person should not yet be on ART.
        """
        df = self.sim.population.props

        if not (df.loc[person_id, 'hv_art'] == 'not'):
            logger.warning("This event should not be running. do_when_diagnosed is for persons being newly dianogsed.")

        # Consider if the person will be referred to start ART
        has_aids_symptoms = 'aids_symptoms' in self.sim.modules['SymptomManager'].has_what(person_id)

        starts_art = self.lm['lm_art'].predict(
            df=df.loc[[person_id]],
            rng=self.rng,
            has_aids_symptoms=has_aids_symptoms
        )

        if starts_art:
            self.sim.modules['HealthSystem'].schedule_hsi_event(
                HSI_Hiv_StartOrContinueTreatment(person_id=person_id, module=self),
                topen=self.sim.date,
                tclose=None,
                priority=0
            )

    def stops_treatment(self, person_id):
        """Helper function that is called when someone stops being on ART.
        Sets the flag for ART status. If the person was already on ART, it schedules a new AIDSEvent"""

        df = self.sim.population.props

        # Schedule a new AIDS onset event if the person was on ART up until now
        if df.at[person_id, "hv_art"] != "not":
            months_to_aids = int(np.floor(
                self.rng.exponential(scale=self.parameters["art_default_to_aids_mean_years"]) * 12.0))
            self.sim.schedule_event(event=HivAidsOnsetEvent(person_id=person_id, module=self),
                                    date=self.sim.date + pd.DateOffset(months=months_to_aids)
                                    )

        # Set that the person is no longer on ART
        df.at[person_id, "hv_art"] = "not"

    def per_capita_testing_rate(self):
        """ this calculates the numbers of hiv tests performed in each time period
        it looks at the cumulative number of tests ever performed and subtracts the
        number calculated at the last time point
        values are converted to per capita testing rates
        this function is called by the logger and can be called at any frequency
        """

        df = self.sim.population.props

        # get number of tests performed in last time period
        if self.sim.date.year == 2010:
            number_tests_new = df.hv_number_tests.sum()
            previous_test_numbers = 0

        else:
            previous_test_numbers = self.stored_test_numbers[-1]

            # calculate number of tests now performed - cumulative, include those who have died
            number_tests_new = df.hv_number_tests.sum()

        self.stored_test_numbers.append(number_tests_new)

        # number of tests performed in last time period
        number_tests_in_last_period = number_tests_new - previous_test_numbers

        # per-capita testing rate
        per_capita_testing = number_tests_in_last_period / len(df[df.is_alive])

        # return updated value for time-period
        return per_capita_testing

    def check_config_of_properties(self):
        """check that the properties are currently configured correctly"""
        df = self.sim.population.props
        df_alive = df.loc[df.is_alive]

        # basic check types of columns and dtypes
        orig = self.sim.population.new_row
        assert (df.dtypes == orig.dtypes).all()

        def is_subset(col_for_set, col_for_subset):
            # Confirms that the series of col_for_subset is true only for a subset of the series for col_for_set
            return set(col_for_subset.loc[col_for_subset].index).issubset(col_for_set.loc[col_for_set].index)

        # Check that core properties of current status are never None/NaN/NaT
        assert not df_alive.hv_inf.isna().any()
        assert not df_alive.hv_art.isna().any()
        assert not df_alive.hv_behaviour_change.isna().any()
        assert not df_alive.hv_diagnosed.isna().any()
        assert not df_alive.hv_number_tests.isna().any()

        # Check that the core HIV properties are 'nested' in the way expected.
        assert is_subset(col_for_set=df_alive.hv_inf, col_for_subset=df_alive.hv_diagnosed)
        assert is_subset(col_for_set=df_alive.hv_diagnosed, col_for_subset=(df_alive.hv_art != "not"))

        # Check that if person is not infected, the dates of HIV events are None/NaN/NaT
        assert df_alive.loc[~df_alive.hv_inf, "hv_date_inf"].isna().all()

        # Check that dates consistent for those infected with HIV
        assert not df_alive.loc[df_alive.hv_inf].hv_date_inf.isna().any()
        assert (df_alive.loc[df_alive.hv_inf].hv_date_inf >= df_alive.loc[df_alive.hv_inf].date_of_birth).all()

        # Check alignment between AIDS Symptoms and status and infection and ART status
        has_aids_symptoms = set(self.sim.modules['SymptomManager'].who_has('aids_symptoms'))
        assert has_aids_symptoms.issubset(df_alive.loc[df_alive.is_alive & df_alive.hv_inf].index)
        assert 0 == len(has_aids_symptoms.intersection(df_alive.loc[df_alive.is_alive & (
            df_alive.hv_art == "on_VL_suppressed")].index))

# ---------------------------------------------------------------------------
#   Main Polling Event
# ---------------------------------------------------------------------------


class HivRegularPollingEvent(RegularEvent, PopulationScopeEventMixin):
    """ The HIV Regular Polling Events
    * Schedules persons becoming newly infected through horizontal transmission
    * Schedules who will present for voluntary ("spontaneous") testing
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=12))  # repeats every 12 months, but this can be changed

    def apply(self, population):

        df = population.props
        params = self.module.parameters

        fraction_of_year_between_polls = self.frequency.months / 12
        beta = params["beta"] * fraction_of_year_between_polls

        # ----------------------------------- HORIZONTAL TRANSMISSION -----------------------------------
        def horizontal_transmission(to_sex, from_sex):
            # Count current number of alive 15-80 year-olds at risk of transmission
            # (= those infected and not VL suppressed):
            n_infectious = len(df.loc[
                                   df.is_alive &
                                   df.age_years.between(15, 80) &
                                   df.hv_inf &
                                   (df.hv_art != "on_VL_suppressed") &
                                   (df.sex == from_sex)
                                   ])

            # Get Susceptible (non-infected alive 15-80 year-old) persons:
            susc_idx = df.loc[df.is_alive & ~df.hv_inf & df.age_years.between(15, 80) & (df.sex == to_sex)].index
            n_susceptible = len(susc_idx)

            # Compute chance that each susceptible person becomes infected:
            #  - relative chance of infection (acts like a scaling-factor on 'beta')
            rr_of_infection = self.module.lm['rr_of_infection'].predict(df.loc[susc_idx])

            #  - probability of infection = beta * I/N
            p_infection = rr_of_infection * beta * (n_infectious / (n_infectious + n_susceptible))

            # New infections:
            will_be_infected = self.module.rng.random_sample(len(p_infection)) < p_infection
            idx_new_infection = will_be_infected[will_be_infected].index

            # Schedule the date of infection for each new infection:
            for idx in idx_new_infection:
                date_of_infection = self.sim.date + \
                                    pd.DateOffset(days=self.module.rng.randint(0, 365 * fraction_of_year_between_polls))
                self.sim.schedule_event(HivInfectionEvent(self.module, idx), date_of_infection)

        # Horizontal transmission: Male --> Female
        horizontal_transmission(from_sex='M', to_sex='F')

        # Horizontal transmission: Female --> Male
        horizontal_transmission(from_sex='F', to_sex='M')

        # ----------------------------------- SPONTANEOUS TESTING -----------------------------------
        prob_spontaneous_test = self.module.lm['lm_spontaneous_test_12m'].predict(
            df.loc[df.is_alive]) * fraction_of_year_between_polls
        will_test = self.module.rng.random_sample(len(prob_spontaneous_test)) < prob_spontaneous_test
        idx_will_test = will_test[will_test].index

        for person_id in idx_will_test:
            date_test = self.sim.date + \
                        pd.DateOffset(days=self.module.rng.randint(0, 365 * fraction_of_year_between_polls))
            self.sim.modules['HealthSystem'].schedule_hsi_event(
                hsi_event=HSI_Hiv_TestAndRefer(person_id=person_id, module=self.module),
                priority=1,
                topen=date_test,
                tclose=self.sim.date + pd.DateOffset(months=self.frequency.months)  # (to occur before next polling)
            )


# ---------------------------------------------------------------------------
#   Natural History Events
# ---------------------------------------------------------------------------

class HivInfectionEvent(Event, IndividualScopeEventMixin):
    """ This person will become infected.
    * Do the infection process
    * Check for onward transmission through MTCT if the infection is to a mother who is currently breastfeeding.
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        df = self.sim.population.props

        # Check person is_alive
        if not df.at[person_id, 'is_alive']:
            return

        # Onset the infection for this person (which will schedule progression etc)
        self.module.do_new_infection(person_id)

        # Consider mother-to-child-transmission (MTCT) from this person to their children:
        children_of_this_person_being_breastfed = df.loc[(df.mother_id == person_id) &
                                                         (df.nb_breastfeeding_status != 'none')].index
        # - Do the MTCT routine for each child:
        for child_id in children_of_this_person_being_breastfed:
            self.module.mtct_during_breastfeeding(person_id, child_id)


class HivInfectionDuringBreastFeedingEvent(Event, IndividualScopeEventMixin):
    """ This person will become infected during breastfeeding
    * Do the infection process
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        df = self.sim.population.props

        # Check person is_alive
        if not df.at[person_id, 'is_alive']:
            return

        # Check person is breastfed currently
        if df.at[person_id, "nb_breastfeeding_status"] == 'none':
            return

        # Onset the infection for this person (which will schedule progression etc)
        self.module.do_new_infection(person_id)


class HivAidsOnsetEvent(Event, IndividualScopeEventMixin):
    """ This person has developed AIDS.
    * Update their symptomatic status
    * Record the date at which AIDS onset
    * Schedule the AIDS death
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):

        df = self.sim.population.props

        # Check person is_alive
        if not df.at[person_id, 'is_alive']:
            return

        # Do nothing if person is now on ART and VL suppressed (non-VL suppressed has no effect)
        if df.at[person_id, "hv_art"] == "on_VL_suppressed":
            return

        # Update Symptoms
        self.sim.modules["SymptomManager"].change_symptom(
            person_id=person_id,
            symptom_string="aids_symptoms",
            add_or_remove="+",
            disease_module=self.module,
        )

        # Schedule AidsDeath
        date_of_aids_death = self.sim.date + self.module.get_time_from_aids_to_death()
        self.sim.schedule_event(event=HivAidsDeathEvent(person_id=person_id, module=self.module),
                                date=date_of_aids_death)


class HivAidsDeathEvent(Event, IndividualScopeEventMixin):
    """
    Causes someone to die of AIDS, if they are not VL suppressed on ART.
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        df = self.sim.population.props

        # Check person is_alive
        if not df.at[person_id, 'is_alive']:
            return

        # Do nothing if person is now on ART and VL suppressed (non VL suppressed has no effect)
        if df.at[person_id, "hv_art"] == "on_VL_suppressed":
            return

        # Cause the death to happen immediately
        demography.InstantaneousDeath(self.module, individual_id=person_id, cause="AIDS").apply(person_id)


class Hiv_DecisionToContinueOnPrEP(Event, IndividualScopeEventMixin):
    """Helper event that is used to 'decide' if someone on PrEP should continue on PrEP.
    This event is scheduled by 'HSI_Hiv_StartOrContinueOnPrep' 3 months after it is run.
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        df = self.sim.population.props
        person = df.loc[person_id]
        m = self.module

        # If the person is no longer alive, and sex worker and not diagnosed, they will not continue on PrEP
        if (not person["is_alive"]) or (not person["li_is_sexworker"]) or (person["hv_diagnosed"]):
            return

        # Check that there are on PrEP currently:
        if not person["hv_is_on_prep"]:
            logger.warning('This event should not be running')

        # Determine if this appointment is actually attended by the person who has already started on PrEP
        if m.rng.random_sample() < m.parameters['probability_of_being_retained_on_prep_every_3_months']:
            # Continue on PrEP - and schedule an HSI for a refill appointment today
            self.sim.modules['HealthSystem'].schedule_hsi_event(
                HSI_Hiv_StartOrContinueOnPrep(person_id=person_id, module=m),
                topen=self.sim.date,
                tclose=self.sim.date + pd.DateOffset(days=7),
                priority=0
            )

        else:
            # Defaults to being off PrEP - reset flag and take no further action
            df.at[person_id, "hv_is_on_prep"] = False


class Hiv_DecisionToContinueTreatment(Event, IndividualScopeEventMixin):
    """Helper event that is used to 'decide' if someone on Treatment should continue on Treatment.
    This event is scheduled by 'HSI_Hiv_StartOrContinueTreatment' 6 months after it is run.
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        df = self.sim.population.props
        person = df.loc[person_id]
        m = self.module

        if not person["is_alive"]:
            return

        # Check that there are on Treatment currently:
        if not (person["hv_art"] in ["on_VL_suppressed", "on_not_VL_suppressed"]):
            logger.warning('This event should not be running')

        # Determine if this appointment is actually attended by the person who has already started on PrEP
        if m.rng.random_sample() < m.parameters['probability_of_being_retained_on_art_every_6_months']:
            # Continue on Treatment - and schedule an HSI for a continuation appointment today
            self.sim.modules['HealthSystem'].schedule_hsi_event(
                HSI_Hiv_StartOrContinueTreatment(person_id=person_id, module=m),
                topen=self.sim.date,
                tclose=self.sim.date + pd.DateOffset(days=14),
                priority=0
            )

        else:
            # Defaults to being off Treatment
            m.stops_treatment(person_id)


# ---------------------------------------------------------------------------
#   Health System Interactions (HSI)
# ---------------------------------------------------------------------------

class HSI_Hiv_TestAndRefer(HSI_Event, IndividualScopeEventMixin):
    """
    The is the Test-and-Refer HSI. Individuals may seek an HIV test at any time. From this, they can be referred on to
    other services.
    This event is scheduled by:
        * the main event poll,
        * when someone presents for any care through a Generic HSI.
        * when an infant is born to an HIV-positive mother
    Following the test, they may or may not go on to present for uptake an HIV service: ART (if HIV-positive), VMMC (if
    HIV-negative and male) or PrEP (if HIV-negative and a female sex worker).
    If this event is called within another HSI, it may be desirable to limit the functionality of the HSI: do this
    using the arguments:
        * do_not_refer_if_neg=False : if the person is HIV-neg they will not be referred to VMMC or PrEP
        * suppress_footprint=True : the HSI will not have any footprint
    """

    def __init__(self, module, person_id, do_not_refer_if_neg=False, suppress_footprint=False):
        super().__init__(module, person_id=person_id)
        assert isinstance(
            module, Hiv
        )

        assert isinstance(do_not_refer_if_neg, bool)
        self.do_not_refer_if_neg = do_not_refer_if_neg

        assert isinstance(suppress_footprint, bool)
        self.suppress_footprint = suppress_footprint

        # Define the necessary information for an HSI
        self.TREATMENT_ID = "Hiv_TestAndRefer"
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'VCTNegative': 1})
        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        """Do the testing and referring to other services"""

        df = self.sim.population.props
        person = df.loc[person_id]

        if not person['is_alive']:
            return

        # If the person has previously been diagnosed do nothing do not occupy any resources
        if person['hv_diagnosed']:
            return self.sim.modules['HealthSystem'].get_blank_appt_footprint()

        # Run test
        if person['age_years'] < 1.0:
            test_result = self.sim.modules['HealthSystem'].dx_manager.run_dx_test(
                dx_tests_to_run='hiv_early_infant_test',
                hsi_event=self
            )
        else:
            test_result = self.sim.modules['HealthSystem'].dx_manager.run_dx_test(
                dx_tests_to_run='hiv_rapid_test',
                hsi_event=self
            )

        # Update number of tests:
        df.at[person_id, 'hv_number_tests'] += 1
        df.at[person_id, 'hv_last_test_date'] = self.sim.date

        if test_result is not None:
            # Offer services as needed:
            if test_result:
                # The test_result is HIV positive
                ACTUAL_APPT_FOOTPRINT = self.make_appt_footprint({'VCTPositive': 1})

                # Update diagnosis if the person is indeed HIV positive;
                if person['hv_inf']:
                    df.at[person_id, 'hv_diagnosed'] = True
                    self.module.do_when_hiv_diagnosed(person_id=person_id)

            else:
                # The test_result is HIV negative
                ACTUAL_APPT_FOOTPRINT = self.make_appt_footprint({'VCTNegative': 1})

                if not self.do_not_refer_if_neg:
                    # The test was negative: make referrals to other services:

                    # Consider if the person's risk will be reduced by behaviour change counselling
                    if self.module.lm['lm_behavchg'].predict(df.loc[[person_id]], self.module.rng):
                        df.at[person_id, 'hv_behaviour_change'] = True

                    # If person is a man, and not circumcised, then consider referring to VMMC
                    if (person['sex'] == 'M') & (~person['li_is_circ']):
                        x = self.module.lm['lm_circ'].predict(df.loc[[person_id]], self.module.rng)
                        if x:
                            self.sim.modules['HealthSystem'].schedule_hsi_event(
                                HSI_Hiv_Circ(person_id=person_id, module=self.module),
                                topen=self.sim.date,
                                tclose=None,
                                priority=0
                            )

                    # If person is a woman and FSW, and not currently on PrEP then consider referring to PrEP
                    # available 2018 onwards
                    if (
                        (person['sex'] == 'F') &
                        person['li_is_sexworker'] &
                        ~person['hv_is_on_prep'] &
                        (self.sim.date.year >= self.module.parameters['prep_start_year'])
                    ):
                        if self.module.lm['lm_prep'].predict(df.loc[[person_id]], self.module.rng):
                            self.sim.modules['HealthSystem'].schedule_hsi_event(
                                HSI_Hiv_StartOrContinueOnPrep(person_id=person_id, module=self.module),
                                topen=self.sim.date,
                                tclose=None,
                                priority=0
                            )
        else:
            # Test was not possible, so do nothing:
            ACTUAL_APPT_FOOTPRINT = self.make_appt_footprint({'VCTNegative': 1})

        # Return the footprint. If it should be suppressed, return a blank footprint.
        if self.suppress_footprint:
            return self.make_appt_footprint({})
        else:
            return ACTUAL_APPT_FOOTPRINT


class HSI_Hiv_Circ(HSI_Event, IndividualScopeEventMixin):
    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        self.TREATMENT_ID = "Hiv_Circumcision"
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({"MinorSurg": 1})
        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        """Do the circumcision for this man"""

        df = self.sim.population.props  # shortcut to the dataframe

        person = df.loc[person_id]

        # Do not run if the person is not alive or is already circumcised
        if not (person["is_alive"] & ~person["li_is_circ"]):
            return

        # Check/log use of consumables, and do circumcision if materials available
        # NB. If materials not available, it is assumed that the procedure is not carried out for this person following
        # this particular referral.
        if self.get_all_consumables(footprint=self.module.footprints_for_consumables_required['circ']):
            # Update circumcision state
            df.at[person_id, "li_is_circ"] = True


class HSI_Hiv_StartOrContinueOnPrep(HSI_Event, IndividualScopeEventMixin):
    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Hiv)

        self.TREATMENT_ID = "Hiv_StartOrContinueOnPrep"
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({"Over5OPD": 1})
        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        """Start PrEP for this person; or continue them on PrEP for 3 more months"""

        df = self.sim.population.props
        person = df.loc[person_id]

        # Do not run if the person is not alive, or is not currently a sex worker, or is diagnosed
        if (not person["is_alive"]) or (not person["li_is_sexworker"]) or (person["hv_diagnosed"]):
            return

        # Run an HIV test
        test_result = self.sim.modules['HealthSystem'].dx_manager.run_dx_test(
            dx_tests_to_run='hiv_rapid_test',
            hsi_event=self
        )
        df.at[person_id, 'hv_number_tests'] += 1
        df.at[person_id, 'hv_last_test_date'] = self.sim.date

        # If test is positive, flag as diagnosed and refer to ART
        if test_result is True:
            # label as diagnosed
            df.at[person_id, 'hv_diagnosed'] = True

            # Do actions for when a person has been diagnosed with HIV
            self.module.do_when_hiv_diagnosed(person_id=person_id)

            return self.make_appt_footprint({"Over5OPD": 1, "VCTPositive": 1})

        # Check that PrEP is available and if it is, initiate or continue  PrEP:
        if self.get_all_consumables(footprint=self.module.footprints_for_consumables_required['prep']):
            df.at[person_id, "hv_is_on_prep"] = True

            # Schedule 'decision about whether to continue on PrEP' for 3 months time
            self.sim.schedule_event(
                Hiv_DecisionToContinueOnPrEP(person_id=person_id, module=self.module),
                self.sim.date + pd.DateOffset(months=3)
            )

        else:
            # If PrEP is not available, the person will default and not be on PrEP
            df.at[person_id, "hv_is_on_prep"] = False

    def never_ran(self, *args, **kwargs):
        """This is called if this HSI was never run.
        Default the person to being off PrEP"""
        self.sim.population.props.at[self.target, "hv_is_on_prep"] = False


class HSI_Hiv_StartOrContinueTreatment(HSI_Event, IndividualScopeEventMixin):
    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Hiv)

        self.TREATMENT_ID = "Hiv_Treatment_InitiationOrContinuation"
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({"Over5OPD": 1, "NewAdult": 1})
        self.ACCEPTED_FACILITY_LEVEL = 1
        self.ALERT_OTHER_DISEASES = []

        self.counter_for_did_not_run = 0

    def apply(self, person_id, squeeze_factor):
        """This is a Health System Interaction Event - start or continue HIV treatment for 6 more months"""
        df = self.sim.population.props
        person = df.loc[person_id]
        art_status_at_beginning_of_hsi = person["hv_art"]

        if not person["is_alive"]:
            return

        # Confirm that the person is diagnosed (this should not run if they are not)
        assert person["hv_diagnosed"]

        if art_status_at_beginning_of_hsi == "not":
            # Do a confirmatory test and do not run the rest of the event if negative.
            # NB. It is assumed that the sensitivity and specificiy of the raoid test is perfect.
            test_result = self.sim.modules['HealthSystem'].dx_manager.run_dx_test(
                dx_tests_to_run='hiv_rapid_test',
                hsi_event=self
            )
            df.at[person_id, 'hv_number_tests'] += 1
            df.at[person_id, 'hv_last_test_date'] = self.sim.date

            if not test_result:
                return self.make_appt_footprint({"Over5OPD": 1})

            assert person["hv_inf"]  # after the test results, it can be guaranteed that the person is HIV-pos.

            # Try to initiate the person onto ART:
            drugs_were_available = self.do_at_initiation(person_id)
        else:
            # Try to continue the person on ART:
            drugs_were_available = self.do_at_continuation(person_id)

        if drugs_were_available:
            # If person has been placed/continued on ART, schedule 'decision about whether to continue on Treatment for
            # 6 months later
            self.sim.schedule_event(
                Hiv_DecisionToContinueTreatment(person_id=person_id, module=self.module),
                self.sim.date + pd.DateOffset(months=6)
            )
        else:
            # As drugs were not available, the person will default to being off ART (...if they were on ART at the
            # beginning of the HSI.)
            # NB. If the person was not on ART at the beginning of the HSI, then there is no need to stop them (which
            #  causes a new AIDSOnsetEvent to be scheduled.)
            if art_status_at_beginning_of_hsi != "not":
                self.module.stops_treatment(person_id)

            # If person 'decides to' seek another appointment, schedule a new HSI appointment for tomorrow.
            # NB. With a probability of 1.0, this will keep occurring, and the person will never give-up coming back to
            # pick-up medication.
            if (
                self.module.rng.random_sample() <
                self.module.parameters["probability_of_seeking_further_art_appointment_if_drug_not_available"]
            ):
                self.sim.modules['HealthSystem'].schedule_hsi_event(
                    hsi_event=HSI_Hiv_StartOrContinueTreatment(person_id=person_id, module=self.module),
                    topen=self.sim.date + pd.DateOffset(days=1),
                    tclose=self.sim.date + pd.DateOffset(days=15),
                    priority=1
                )

    def do_at_initiation(self, person_id):
        """Things to do when this the first appointment ART"""
        df = self.sim.population.props
        person = df.loc[person_id]

        # Check if drugs are available, and provide drugs:
        drugs_available = self.get_drugs(age_of_person=person['age_years'])

        if drugs_available:
            # Assign person to be have suppressed or un-suppressed viral load
            # (If person is VL suppressed This will prevent the Onset of AIDS, or an AIDS death if AIDS has already
            # onset,)
            vl_status = self.determine_vl_status(sex_of_person=person["sex"], age_of_person=person["age_years"])
            df.at[person_id, "hv_art"] = vl_status

            # If VL suppressed, remove any symptoms caused by this module
            if vl_status == "on_VL_suppressed":
                self.sim.modules['SymptomManager'].clear_symptoms(
                    person_id=person_id,
                    disease_module=self.module
                )

        # Consider if TB treatment should start
        self.consider_tb(person_id)

        return drugs_available

    def do_at_continuation(self, person_id):
        """Things to do when the person is already on ART"""

        df = self.sim.population.props
        person = df.loc[person_id]

        # Viral Load Monitoring
        # NB. This does not have a direct effect on outcomes for the person.
        _ = self.get_all_consumables(footprint=self.module.footprints_for_consumables_required['vl_measurement'])

        # Check if drugs are available, and provide drugs:
        drugs_available = self.get_drugs(age_of_person=person['age_years'])

        return drugs_available

    def determine_vl_status(self, sex_of_person, age_of_person):
        """Helper function to determine the VL status that the person will have.
        Return what will be the status of "hv_art"
        """
        p = self.module.parameters

        if age_of_person < 15:
            prob_vs = p["vls_child"]
        else:
            if sex_of_person == "M":
                prob_vs = p["vls_m"]
            else:
                prob_vs = p["vls_f"]

        return "on_VL_suppressed" if (self.module.rng.random_sample() < prob_vs) else "on_not_VL_suppressed"

    def get_drugs(self, age_of_person):
        """Helper function to get the ART according to the age of the person being treated. Returns bool to indicate
        whether drugs were available"""
        if age_of_person < 5.0:
            # Formulation for children
            drugs_available = self.get_all_consumables(
                footprint=self.module.footprints_for_consumables_required['art_child'])
        else:
            # Formulation for adults
            drugs_available = self.get_all_consumables(
                footprint=self.module.footprints_for_consumables_required['art_adult'])

        return drugs_available

    def consider_tb(self, person_id):
        # todo - TB treatment when person starts ART - complete when TB module is completed.
        pass
        """
        Consider whether IPT is needed at this time. This is run only when treatment is initiated.
        # if 'Tb' in self.sim.modules:
            district = df.at[person_id, "district_of_residence"]
            eligible = df.at[person_id, "tb_inf"].startswith("active")
            if (
                (district in params["tb_high_risk_distr"].values)
                & (self.sim.date.year > 2012)
                & eligible
                & (self.module.rng.rand() < params["???"])
            ):
                # Schedule the TB treatment event:
                self.sim.modules["HealthSystem"].schedule_hsi_event(
                    tb.HSI_Tb_IptHiv(self.module['Tb'], person_id=person_id),
                    priority=1,
                    topen=self.sim.date,
                    tclose=None
                )
            """

    def never_ran(self):
        """This is called if this HSI was never run.
        * Default the person to being off ART.
        * Determine if they will re-seek care themselves in the future:
        """
        # stop treatment for this person
        person_id = self.target
        self.module.stops_treatment(person_id)

        # determine if will seek another HSI:
        if (
            self.module.rng.random_sample() <
            self.module.parameters["probability_of_seeking_further_art_appointment_if_appointment_not_available"]
        ):
            self.sim.modules['HealthSystem'].schedule_hsi_event(
                HSI_Hiv_StartOrContinueTreatment(person_id=person_id, module=self.module),
                topen=self.sim.date + pd.DateOffset(days=14),
                tclose=self.sim.date + pd.DateOffset(days=21),
                priority=1
            )

# ---------------------------------------------------------------------------
#   Logging
# ---------------------------------------------------------------------------


class HivLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        """ Log Current status of the population, every year
        """

        self.repeat = 12
        super().__init__(module, frequency=DateOffset(months=self.repeat))

    def apply(self, population):
        # get some summary statistics
        df = population.props
        now = self.sim.date

        # ------------------------------------ SUMMARIES ------------------------------------
        # adult prevalence
        adult_prev_15plus = (
            len(df[df.hv_inf & df.is_alive & (df.age_years >= 15)])
            / len(df[df.is_alive & (df.age_years >= 15)])
        )

        adult_prev_1549 = (
            len(df[df.hv_inf & df.is_alive & df.age_years.between(15, 49)])
            / len(df[df.is_alive & (df.age_years >= 15)])
        )

        # child prevalence
        child_prev = (
            len(df[df.hv_inf & df.is_alive & (df.age_years < 15)])
            / len(df[df.is_alive & (df.age_years < 15)])
        )

        # incidence in the period since the last log for 15+ and 15-49 year-olds (denominator is approximate)
        n_new_infections_adult_15plus = len(
            df.loc[
                (df.age_years >= 15)
                & df.is_alive
                & (df.hv_date_inf > (now - DateOffset(months=self.repeat)))
                ]
        )
        denom_adults_15plus = len(df[df.is_alive & ~df.hv_inf & (df.age_years >= 15)])
        adult_inc_15plus = n_new_infections_adult_15plus / denom_adults_15plus

        n_new_infections_adult_1549 = len(
            df.loc[
                df.age_years.between(15, 49)
                & df.is_alive
                & (df.hv_date_inf > (now - DateOffset(months=self.repeat)))
                ]
        )
        denom_adults_1549 = len(df[df.is_alive & ~df.hv_inf & df.age_years.between(15, 49)])
        adult_inc_1549 = n_new_infections_adult_1549 / denom_adults_1549

        # incidence in the period since the last log for 0-14 year-olds (denominator is approximate)
        n_new_infections_children = len(
            df.loc[
                (df.age_years < 15)
                & df.is_alive
                & (df.hv_date_inf > (now - DateOffset(months=self.repeat)))
                ]
        )
        denom_children = len(df[df.is_alive & ~df.hv_inf & (df.age_years < 15)])
        child_inc = (n_new_infections_children / denom_children)

        # hiv prev among female sex workers (aged 15-49)
        n_fsw = len(df.loc[
                        df.is_alive &
                        df.li_is_sexworker &
                        (df.sex == "F") &
                        df.age_years.between(15, 49)
                        ])
        prev_hiv_fsw = 0 if n_fsw == 0 else \
            len(df.loc[
                    df.is_alive &
                    df.hv_inf &
                    df.li_is_sexworker &
                    (df.sex == "F") &
                    df.age_years.between(15, 49)
                    ]) / n_fsw

        logger.info(key='summary_inc_and_prev_for_adults_and_children_and_fsw',
                    description='Summary of HIV among adult (15+ and 15-49) and children (0-14s) and female sex workers'
                                ' (15-49)',
                    data={
                        "hiv_prev_adult_15plus": adult_prev_15plus,
                        "hiv_prev_adult_1549": adult_prev_1549,
                        "hiv_prev_child": child_prev,
                        "hiv_adult_inc_15plus": adult_inc_15plus,
                        "hiv_adult_inc_1549": adult_inc_1549,
                        "hiv_child_inc": child_inc,
                        "hiv_prev_fsw": prev_hiv_fsw
                    }
                    )

        # ------------------------------------ PREVALENCE BY AGE and SEX  ------------------------------------

        # Prevalence by Age/Sex (to make every category be output, do separately by 'sex')
        prev_by_age_and_sex = {}
        for sex in ['F', 'M']:
            n_hiv = df.loc[df.sex == sex].groupby(by=['age_range'])['hv_inf'].sum()
            n_pop = df.loc[df.sex == sex].groupby(by=['age_range'])['hv_inf'].count()
            prev_by_age_and_sex[sex] = (n_hiv / n_pop).to_dict()

        logger.info(key='prev_by_age_and_sex',
                    data=prev_by_age_and_sex,
                    description='Prevalence of HIV split by age and sex')

        # ------------------------------------ TESTING ------------------------------------

        # proportion of adult population tested in past year
        n_tested = len(df.loc[df.is_alive & (df.hv_number_tests > 0) & (df.age_years >= 15) &
                              (df.hv_last_test_date > (now - DateOffset(months=self.repeat)))])
        n_pop = len(df.loc[df.is_alive & (df.age_years >= 15)])
        tested = (n_tested / n_pop)

        # proportion of adult population tested in past year by sex
        testing_by_sex = {}
        for sex in ['F', 'M']:
            n_tested = len(df.loc[(df.sex == sex) & (df.hv_number_tests > 0) & (df.age_years >= 15) &
                                  (df.hv_last_test_date > (now - DateOffset(months=self.repeat)))])
            n_pop = len(df.loc[(df.sex == sex) & (df.age_years >= 15)])
            testing_by_sex[sex] = (n_tested / n_pop)

        # per_capita_testing_rate: number of tests administered divided by population
        current_testing_rate = self.module.per_capita_testing_rate()

        # ------------------------------------ TREATMENT ------------------------------------
        def treatment_counts(subset):
            # total number of subset (subset is a true/false series)
            count = sum(subset)
            # proportion of subset living with HIV that are diagnosed:
            proportion_diagnosed = sum(subset & df.hv_diagnosed) / count if count > 0 else 0
            # proportions of subset living with HIV on treatment:
            art = sum(subset & (df.hv_art != "not"))
            art_cov = art / count if count > 0 else 0
            # proportion of subset on treatment that have good VL suppression
            art_vs = sum(subset & (df.hv_art == "on_VL_suppressed"))
            art_cov_vs = art_vs / art if art > 0 else 0
            return proportion_diagnosed, art_cov, art_cov_vs

        alive_infected = df.is_alive & df.hv_inf
        dx_adult, art_cov_adult, art_cov_vs_adult = treatment_counts(alive_infected & (df.age_years >= 15))
        dx_children, art_cov_children, art_cov_vs_children = treatment_counts(alive_infected & (df.age_years < 15))

        # ------------------------------------ BEHAVIOUR CHANGE ------------------------------------

        # proportion of adults (15+) exposed to behaviour change intervention
        prop_adults_exposed_to_behav_intv = len(
            df[df.is_alive & df.hv_behaviour_change & (df.age_years >= 15)]
        ) / len(df[df.is_alive & (df.age_years >= 15)])

        # ------------------------------------ PREP AMONG FSW ------------------------------------
        prop_fsw_on_prep = 0 if n_fsw == 0 else len(
            df[df.is_alive & df.li_is_sexworker & (df.age_years >= 15) & df.hv_is_on_prep]
        ) / len(df[df.is_alive & df.li_is_sexworker & (df.age_years >= 15)])

        # ------------------------------------ MALE CIRCUMCISION ------------------------------------
        # NB. Among adult men
        prop_men_circ = len(
            df[df.is_alive & (df.sex == 'M') & (df.age_years >= 15) & df.li_is_circ]
        ) / len(df[df.is_alive & (df.sex == 'M') & (df.age_years >= 15)])

        logger.info(key='hiv_program_coverage',
                    description='Coverage of interventions for HIV among adult (15+) and children (0-14s)',
                    data={
                        "prop_tested_adult": tested,
                        "prop_tested_adult_male": testing_by_sex['M'],
                        "prop_tested_adult_female": testing_by_sex['F'],
                        "per_capita_testing_rate": current_testing_rate,
                        "dx_adult": dx_adult,
                        "dx_childen": dx_children,
                        "art_coverage_adult": art_cov_adult,
                        "art_coverage_adult_VL_suppression": art_cov_vs_adult,
                        "art_coverage_child": art_cov_children,
                        "art_coverage_child_VL_suppression": art_cov_vs_children,
                        "prop_adults_exposed_to_behav_intv": prop_adults_exposed_to_behav_intv,
                        "prop_fsw_on_prep": prop_fsw_on_prep,
                        "prop_men_circ": prop_men_circ
                    }
                    )

# ---------------------------------------------------------------------------
#   Debugging / Checking Events
# ---------------------------------------------------------------------------


class HivCheckPropertiesEvent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=1))  # runs every month

    def apply(self, population):
        self.module.check_config_of_properties()

# ---------------------------------------------------------------------------
#   Helper functions for analysing outputs
# ---------------------------------------------------------------------------


def set_age_group(ser):
    AGE_RANGE_CATEGORIES, AGE_RANGE_LOOKUP = create_age_range_lookup(
        min_age=demography.MIN_AGE_FOR_RANGE,
        max_age=demography.MAX_AGE_FOR_RANGE,
        range_size=demography.AGE_RANGE_SIZE
    )
    ser = ser.astype("category")
    AGE_RANGE_CATEGORIES_filtered = [a for a in AGE_RANGE_CATEGORIES if a in ser.values]
    return ser.cat.reorder_categories(AGE_RANGE_CATEGORIES_filtered)


def map_to_age_group(ser):
    AGE_RANGE_CATEGORIES, AGE_RANGE_LOOKUP = create_age_range_lookup(
        min_age=demography.MIN_AGE_FOR_RANGE,
        max_age=demography.MAX_AGE_FOR_RANGE,
        range_size=demography.AGE_RANGE_SIZE
    )
    ser = ser.map(AGE_RANGE_LOOKUP)
    ser = set_age_group(ser)
    return ser


def unpack_raw_output_dict(raw_dict):
    x = pd.DataFrame.from_dict(data=raw_dict, orient='index')
    x = x.reset_index()
    x.rename(columns={'index': 'age_group', 0: 'value'}, inplace=True)
    x['age_group'] = set_age_group(x['age_group'])
    return x
