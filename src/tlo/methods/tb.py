"""
    This module schedules TB infection and natural history
    It schedules TB treatment and follow-up appointments along with preventive therapy
    for eligible people (HIV+ and paediatric contacts of active TB cases
"""

import os

import numpy as np
import pandas as pd

from tlo import DateOffset, Module, Parameter, Property, Types, logging
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.lm import LinearModel, LinearModelType, Predictor
from tlo.methods import Metadata, hiv
from tlo.methods.causes import Cause
from tlo.methods.dxmanager import DxTest
from tlo.methods.healthsystem import HSI_Event
from tlo.methods.symptommanager import Symptom

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Tb(Module):
    """Set up the baseline population with TB prevalence"""

    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)

        self.resourcefilepath = resourcefilepath
        self.daly_wts = dict()
        self.lm = dict()
        self.footprints_for_consumables_required = dict()
        self.symptom_list = {"fever", "respiratory_symptoms", "fatigue", "night_sweats"}
        self.district_list = list()
        self.item_codes_for_consumables_required = dict()

        # tb outputs needed for calibration/
        keys = ["date",
                "num_new_active_tb",
                "tbPrevLatent"
                ]
        # initialise empty dict with set keys
        self.tb_outputs = {k: [] for k in keys}

    INIT_DEPENDENCIES = {"Demography", "HealthSystem", "Lifestyle", "SymptomManager", "Epi"}

    OPTIONAL_INIT_DEPENDENCIES = {"HealthBurden", "Hiv"}

    METADATA = {
        Metadata.DISEASE_MODULE,
        Metadata.USES_SYMPTOMMANAGER,
        Metadata.USES_HEALTHSYSTEM,
        Metadata.USES_HEALTHBURDEN,
    }

    # Declare Causes of Death
    CAUSES_OF_DEATH = {
        "TB": Cause(gbd_causes="Tuberculosis", label="non_AIDS_TB"),
        "AIDS_TB": Cause(gbd_causes="HIV/AIDS", label="AIDS"),
    }

    CAUSES_OF_DISABILITY = {
        "TB": Cause(gbd_causes="Tuberculosis", label="non_AIDS_TB"),
    }

    # Declaration of the specific symptoms that this module will use
    SYMPTOMS = {"fatigue", "night_sweats"}

    PROPERTIES = {
        # ------------------ natural history ------------------ #
        "tb_inf": Property(
            Types.CATEGORICAL,
            categories=[
                "uninfected",
                "latent",
                "active",
            ],
            description="tb status",
        ),
        "tb_strain": Property(
            Types.CATEGORICAL,
            categories=[
                "none",
                "ds",
                "mdr",
            ],
            description="tb strain: drug-susceptible (ds) or multi-drug resistant (mdr)",
        ),
        "tb_date_latent": Property(
            Types.DATE, "Date acquired tb infection (latent stage)"
        ),
        "tb_scheduled_date_active": Property(
            Types.DATE, "Date active tb is scheduled to start"
        ),
        "tb_date_active": Property(Types.DATE, "Date active tb started"),
        "tb_smear": Property(
            Types.BOOL,
            "smear positivity with active infection: False=negative, True=positive",
        ),
        # ------------------ testing status ------------------ #
        "tb_ever_tested": Property(Types.BOOL, "ever had a tb test"),
        "tb_diagnosed": Property(
            Types.BOOL, "person has current diagnosis of active tb"
        ),
        "tb_date_diagnosed": Property(Types.DATE, "date most recent tb diagnosis"),
        "tb_diagnosed_mdr": Property(
            Types.BOOL, "person has current diagnosis of active mdr-tb"
        ),
        # ------------------ treatment status ------------------ #
        "tb_on_treatment": Property(Types.BOOL, "on tb treatment regimen"),
        "tb_date_treated": Property(
            Types.DATE, "date most recent tb treatment started"
        ),
        "tb_treatment_regimen": Property(
            Types.CATEGORICAL,
            categories=[
                "none",
                "tb_tx_adult",
                "tb_tx_child",
                "tb_tx_child_shorter",
                "tb_retx_adult",
                "tb_retx_child",
                "tb_mdrtx"
            ],
            description="current tb treatment regimen",
        ),
        "tb_ever_treated": Property(Types.BOOL, "if ever treated for active tb"),
        "tb_treatment_failure": Property(Types.BOOL, "failed first line tb treatment"),
        "tb_treated_mdr": Property(Types.BOOL, "on tb treatment MDR regimen"),
        "tb_date_treated_mdr": Property(Types.DATE, "date tb MDR treatment started"),
        "tb_on_ipt": Property(Types.BOOL, "if currently on ipt"),
        "tb_date_ipt": Property(Types.DATE, "date ipt started"),
    }

    PARAMETERS = {
        # ------------------ workbooks ------------------ #
        "prop_active_2010": Parameter(
            Types.REAL, "Proportion of population with active tb in 2010"
        ),
        "pulm_tb": Parameter(Types.DATA_FRAME, "probability of pulmonary tb"),
        "followup_times": Parameter(
            Types.DATA_FRAME,
            "times(weeks) tb treatment monitoring required after tx start",
        ),
        "tb_high_risk_distr": Parameter(Types.LIST, "list of ten high-risk districts"),
        "ipt_coverage": Parameter(
            Types.DATA_FRAME,
            "national estimates of coverage of IPT in PLHIV and paediatric contacts",
        ),
        "who_incidence": Parameter(
            Types.DATA_FRAME,
            "WHO estimates of incidence, prevalence and mortality",
        ),

        # ------------------ baseline population ------------------ #
        "prop_mdr2010": Parameter(
            Types.REAL,
            "Proportion of active tb cases with multidrug resistance in 2010",
        ),
        # ------------------ natural history ------------------ #
        "prob_latent_tb_0_14": Parameter(
            Types.REAL, "probability of latent infection in ages 0-14 years"
        ),
        "prob_latent_tb_15plus": Parameter(
            Types.REAL, "probability of latent infection in ages 15+"
        ),
        "incidence_active_tb_2010": Parameter(
            Types.REAL, "incidence of active tb in 2010 in all ages"
        ),
        "transmission_rate": Parameter(Types.REAL, "TB transmission rate, calibrated"),
        "mixing_parameter": Parameter(
            Types.REAL,
            "mixing parameter adjusts transmission rate for force of infection "
            "between districts, value 1=completely random mixing across districts, "
            "value=0 no between-district transmission",
        ),
        "rel_inf_smear_ng": Parameter(
            Types.REAL, "relative infectiousness of tb in hiv+ compared with hiv-"
        ),
        "rr_bcg_inf": Parameter(
            Types.REAL, "relative risk of tb infection with bcg vaccination"
        ),
        "monthly_prob_relapse_tx_complete": Parameter(
            Types.REAL, "monthly probability of relapse once treatment complete"
        ),
        "monthly_prob_relapse_tx_incomplete": Parameter(
            Types.REAL, "monthly probability of relapse if treatment incomplete"
        ),
        "monthly_prob_relapse_2yrs": Parameter(
            Types.REAL,
            "monthly probability of relapse 2 years after treatment complete",
        ),
        "rr_relapse_hiv": Parameter(
            Types.REAL, "relative risk of relapse for HIV-positive people"
        ),
        # ------------------ progression ------------------ #
        "prop_fast_progressor": Parameter(
            Types.REAL,
            "Proportion of infections that progress directly to active stage",
        ),
        "prop_fast_progressor_hiv": Parameter(
            Types.REAL,
            "proportion of HIV+ people not on ART progressing directly to active TB disease after infection",
        ),
        "prog_active": Parameter(
            Types.REAL, "risk of progressing to active tb within two years"
        ),
        "prog_1yr": Parameter(
            Types.REAL, "proportion children aged <1 year progressing to active disease"
        ),
        "prog_1_2yr": Parameter(
            Types.REAL,
            "proportion children aged 1-2 year2 progressing to active disease",
        ),
        "prog_2_5yr": Parameter(
            Types.REAL,
            "proportion children aged 2-5 years progressing to active disease",
        ),
        "prog_5_10yr": Parameter(
            Types.REAL,
            "proportion children aged 5-10 years progressing to active disease",
        ),
        "prog_10yr": Parameter(
            Types.REAL,
            "proportion children aged 10-15 years progressing to active disease",
        ),
        "duration_active_disease_years": Parameter(
            Types.REAL, "duration of active disease from onset to cure or death"
        ),
        # ------------------ clinical features ------------------ #
        "prop_smear_positive": Parameter(
            Types.REAL, "proportion of new active cases that will be smear-positive"
        ),
        "prop_smear_positive_hiv": Parameter(
            Types.REAL, "proportion of hiv+ active tb cases that will be smear-positive"
        ),
        # ------------------ mortality ------------------ #
        # untreated
        "death_rate_smear_pos_untreated": Parameter(
            Types.REAL,
            "probability of death in smear-positive tb cases with untreated tb",
        ),
        "death_rate_smear_neg_untreated": Parameter(
            Types.REAL,
            "probability of death in smear-negative tb cases with untreated tb",
        ),
        # treated
        "death_rate_child0_4_treated": Parameter(
            Types.REAL, "probability of death in child aged 0-4 years with treated tb"
        ),
        "death_rate_child5_14_treated": Parameter(
            Types.REAL, "probability of death in child aged 5-14 years with treated tb"
        ),
        "death_rate_adult_treated": Parameter(
            Types.REAL, "probability of death in adult aged >=15 years with treated tb"
        ),
        # ------------------ progression to active disease ------------------ #
        "rr_tb_bcg": Parameter(
            Types.REAL,
            "relative risk of progression to active disease for children with BCG vaccine",
        ),
        "rr_tb_hiv": Parameter(
            Types.REAL, "relative risk of progression to active disease for PLHIV"
        ),
        "rr_tb_aids": Parameter(
            Types.REAL,
            "relative risk of progression to active disease for PLHIV with AIDS",
        ),
        "rr_tb_art_adult": Parameter(
            Types.REAL,
            "relative risk of progression to active disease for adults with HIV on ART",
        ),
        "rr_tb_art_child": Parameter(
            Types.REAL,
            "relative risk of progression to active disease for adults with HIV on ART",
        ),
        "rr_tb_obese": Parameter(
            Types.REAL, "relative risk of progression to active disease if obese"
        ),
        "rr_tb_diabetes1": Parameter(
            Types.REAL,
            "relative risk of progression to active disease with type 1 diabetes",
        ),
        "rr_tb_alcohol": Parameter(
            Types.REAL,
            "relative risk of progression to active disease with heavy alcohol use",
        ),
        "rr_tb_smoking": Parameter(
            Types.REAL, "relative risk of progression to active disease with smoking"
        ),
        "rr_ipt_adult": Parameter(
            Types.REAL, "relative risk of active TB with IPT in adults"
        ),
        "rr_ipt_child": Parameter(
            Types.REAL, "relative risk of active TB with IPT in children"
        ),
        "rr_ipt_adult_hiv": Parameter(
            Types.REAL, "relative risk of active TB with IPT in adults with hiv"
        ),
        "rr_ipt_child_hiv": Parameter(
            Types.REAL, "relative risk of active TB with IPT in children with hiv"
        ),
        "rr_ipt_art_adult": Parameter(
            Types.REAL, "relative risk of active TB with IPT and ART in adults"
        ),
        "rr_ipt_art_child": Parameter(
            Types.REAL, "relative risk of active TB with IPT and ART in children"
        ),
        # ------------------ diagnostic tests ------------------ #
        "sens_xpert_smear_negative": Parameter(
            Types.REAL, "sensitivity of Xpert test in smear negative TB cases"),
        "sens_xpert_smear_positive": Parameter(
            Types.REAL, "sensitivity of Xpert test in smear positive TB cases"),
        "spec_xpert_smear_negative": Parameter(
            Types.REAL, "specificity of Xpert test in smear negative TB cases"),
        "spec_xpert_smear_positive": Parameter(
            Types.REAL, "specificity of Xpert test in smear positive TB cases"),
        "sens_sputum_smear_positive": Parameter(
            Types.REAL,
            "sensitivity of sputum smear microscopy in sputum positive cases",
        ),
        "spec_sputum_smear_positive": Parameter(
            Types.REAL,
            "specificity of sputum smear microscopy in sputum positive cases",
        ),
        "sens_clinical": Parameter(
            Types.REAL, "sensitivity of clinical diagnosis in detecting active TB"
        ),
        "spec_clinical": Parameter(
            Types.REAL, "specificity of clinical diagnosis in detecting TB"
        ),
        "sens_xray_smear_negative": Parameter(
            Types.REAL, "sensitivity of x-ray diagnosis in smear negative TB cases"
        ),
        "sens_xray_smear_positive": Parameter(
            Types.REAL, "sensitivity of x-ray diagnosis in smear positive TB cases"
        ),
        "spec_xray_smear_negative": Parameter(
            Types.REAL, "specificity of x-ray diagnosis in smear negative TB cases"
        ),
        "spec_xray_smear_positive": Parameter(
            Types.REAL, "specificity of x-ray diagnosis in smear positive TB cases"
        ),
        # ------------------ treatment success rates ------------------ #
        "prob_tx_success_ds": Parameter(
            Types.REAL, "Probability of treatment success for new and relapse TB cases"
        ),
        "prob_tx_success_mdr": Parameter(
            Types.REAL, "Probability of treatment success for MDR-TB cases"
        ),
        "prob_tx_success_0_4": Parameter(
            Types.REAL, "Probability of treatment success for children aged 0-4 years"
        ),
        "prob_tx_success_5_14": Parameter(
            Types.REAL, "Probability of treatment success for children aged 5-14 years"
        ),
        "prob_tx_success_shorter": Parameter(
            Types.REAL, "Probability of treatment success for children aged <16 years on shorter regimen"
        ),
        # ------------------ testing rates ------------------ #
        "rate_testing_general_pop": Parameter(
            Types.REAL,
            "rate of screening / testing per month in general population",
        ),
        "rate_testing_active_tb": Parameter(
            Types.REAL,
            "rate of screening / testing per month in population with active tb",
        ),
        "rate_treatment_baseline_active": Parameter(
            Types.REAL,
            "probability of screening for baseline population with active tb",
        ),
        # ------------------ treatment regimens ------------------ #
        "ds_treatment_length": Parameter(
            Types.REAL,
            "length of treatment for drug-susceptible tb (first case) in months",
        ),
        "ds_retreatment_length": Parameter(
            Types.REAL,
            "length of treatment for drug-susceptible tb (secondary case) in months",
        ),
        "mdr_treatment_length": Parameter(
            Types.REAL, "length of treatment for mdr-tb in months"
        ),
        "child_shorter_treatment_length": Parameter(
            Types.REAL, "length of treatment for shorter paediatric regimen in months"
        ),
        "prob_retained_ipt_6_months": Parameter(
            Types.REAL,
            "probability of being retained on IPT every 6 months if still eligible",
        ),
        "age_eligibility_for_ipt": Parameter(
            Types.REAL,
            "eligibility criteria (years of age) for IPT given to contacts of TB cases",
        ),
        "ipt_start_date": Parameter(
            Types.INT,
            "year from which IPT is available for paediatric contacts of diagnosed active TB cases",
        ),
        "scenario": Parameter(
            Types.INT,
            "integer value labelling the scenario to be run: default is 0"
        ),
        "scenario_start_date": Parameter(
            Types.DATE,
            "date from which different scenarios are run"
        ),
        "first_line_test": Parameter(
            Types.STRING,
            "name of first test to be used for TB diagnosis"
        ),
        "second_line_test": Parameter(
            Types.STRING,
            "name of second test to be used for TB diagnosis"
        ),
        "probability_access_to_xray": Parameter(
            Types.REAL,
            "probability a person will have access to chest x-ray"
        ),
        "adjusted_active_testing_rate": Parameter(
            Types.REAL,
            "used to adjust active TB screening rate"
        )
    }

    def read_parameters(self, data_folder):
        """
        * 1) Reads the ResourceFiles
        * 2) Declares the DALY weights
        * 3) Declares the Symptoms
        """

        # 1) Read the ResourceFiles
        workbook = pd.read_excel(
            os.path.join(self.resourcefilepath, "ResourceFile_TB.xlsx"), sheet_name=None
        )
        self.load_parameters_from_dataframe(workbook["parameters"])

        p = self.parameters

        # assume cases distributed equally across districts
        # todo this is not used for national-level model
        p["prop_active_2010"] = workbook["cases2010district"]

        p["rate_testing_active_tb"] = workbook["testing_rates"]
        p["pulm_tb"] = workbook["pulm_tb"]
        p["followup_times"] = workbook["followup"]
        p["who_incidence"] = workbook["WHO_activeTB2020"]

        # if using national-level model, include all districts in IPT coverage
        # p['tb_high_risk_distr'] = workbook['IPTdistricts']
        p["tb_high_risk_distr"] = workbook["all_districts"]

        p["ipt_coverage"] = workbook["ipt_coverage"]

        self.district_list = (
            self.sim.modules["Demography"]
                .parameters["pop_2010"]["District"]
                .unique()
                .tolist()
        )

        # 2) Get the DALY weights
        if "HealthBurden" in self.sim.modules.keys():
            # HIV-negative
            # Drug-susceptible tuberculosis, not HIV infected
            self.daly_wts["daly_tb"] = self.sim.modules["HealthBurden"].get_daly_weight(
                0
            )
            # multi-drug resistant tuberculosis, not HIV infected
            self.daly_wts["daly_mdr_tb"] = self.sim.modules[
                "HealthBurden"
            ].get_daly_weight(1)

            # HIV-positive
            # Drug-susceptible Tuberculosis, HIV infected and anemia, moderate
            self.daly_wts["daly_tb_hiv_anaemia"] = self.sim.modules[
                "HealthBurden"
            ].get_daly_weight(5)
            # Multi-drug resistant Tuberculosis, HIV infected and anemia, moderate
            self.daly_wts["daly_mdr_tb_hiv_anaemia"] = self.sim.modules[
                "HealthBurden"
            ].get_daly_weight(10)

        # 3) Declare the Symptoms
        # additional healthcare-seeking behaviour with these symptoms
        self.sim.modules["SymptomManager"].register_symptom(
            Symptom(
                name="fatigue",
                odds_ratio_health_seeking_in_adults=5.0,
                odds_ratio_health_seeking_in_children=5.0,
            )
        )

        self.sim.modules["SymptomManager"].register_symptom(
            Symptom(
                name="night_sweats",
                odds_ratio_health_seeking_in_adults=5.0,
                odds_ratio_health_seeking_in_children=5.0,
            )
        )

    def pre_initialise_population(self):
        """
        * Establish the Linear Models
        """
        p = self.parameters

        # probability of death
        self.lm["death_rate"] = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            1,
            Predictor().when(
                "(tb_on_treatment == True) & "
                "(age_years <=4)",
                p["death_rate_child0_4_treated"],
            ),
            Predictor().when(
                "(tb_on_treatment == True) & "
                "(age_years <=14)",
                p["death_rate_child5_14_treated"],
            ),
            Predictor().when(
                "(tb_on_treatment == True) & "
                "(age_years >=15)",
                p["death_rate_adult_treated"],
            ),
            Predictor().when(
                "(tb_on_treatment == False) & "
                "(tb_smear == True)",
                p["death_rate_smear_pos_untreated"],
            ),
            Predictor().when(
                "(tb_on_treatment == False) & "
                "(tb_smear == False)",
                p["death_rate_smear_neg_untreated"],
            ),
        )

    def send_for_screening(self, population):

        df = population.props
        p = self.parameters
        rng = self.rng

        active_testing_rates = p["rate_testing_active_tb"]
        current_active_testing_rate = active_testing_rates.loc[
                                          (
                                              active_testing_rates.year == self.sim.date.year), "testing_rate_active_cases"].values[
                                          0] / 100
        current_active_testing_rate = current_active_testing_rate * p["adjusted_active_testing_rate"]
        current_active_testing_rate = current_active_testing_rate / 12  # adjusted for monthly poll
        random_draw = rng.random_sample(size=len(df))

        # randomly select some individuals for screening and testing
        screen_idx = df.index[
            df.is_alive
            & ~df.tb_diagnosed
            & ~df.tb_on_treatment
            & (random_draw < p["rate_testing_general_pop"])
            ]

        # randomly select some symptomatic individuals for screening and testing
        # this rate increases by year
        screen_active_idx = df.index[
            df.is_alive
            & ~df.tb_diagnosed
            & ~df.tb_on_treatment
            & (df.tb_inf == "active")
            & (random_draw < current_active_testing_rate)
            ]

        all_screened = screen_idx.union(screen_active_idx).drop_duplicates()

        for person in all_screened:
            self.sim.modules["HealthSystem"].schedule_hsi_event(
                HSI_Tb_ScreeningAndRefer(person_id=person, module=self),
                topen=self.sim.date,
                tclose=None,
                priority=0,
            )

    def select_tb_test(self, person_id):

        df = self.sim.population.props
        p = self.parameters
        person = df.loc[person_id]

        # xpert tests limited to 60% coverage
        # if selected test is xpert, check for availability
        # give sputum smear as back-up
        # assume sputum smear always available
        test = None

        # primary cases, no HIV diagnosed and >= 5 years
        if not person["tb_ever_treated"] and not person["hv_diagnosed"]:
            test = p["first_line_test"]

        # previously diagnosed/treated or hiv+ -> xpert
        # assume ~60% have access to Xpert, some data in 2019 NTP report but not exact proportions
        if person["tb_ever_treated"] or person["hv_diagnosed"]:
            test = "xpert"

        return (
            "xpert"
            if (test == "xpert")
            else "sputum"
        )

    def initialise_population(self, population):

        df = population.props

        # if HIV is not registered, create a dummy property
        if "Hiv" not in self.sim.modules:
            population.make_test_property("hv_inf", Types.BOOL)
            population.make_test_property("sy_aids_symptoms", Types.INT)
            population.make_test_property("hv_art", Types.STRING)

            df["hv_inf"] = False
            df["sy_aids_symptoms"] = 0
            df["hv_art"] = "not"

        # Set our property values for the initial population
        df["tb_inf"].values[:] = "uninfected"
        df["tb_strain"].values[:] = "none"

        df["tb_date_latent"] = pd.NaT
        df["tb_scheduled_date_active"] = pd.NaT
        df["tb_date_active"] = pd.NaT
        df["tb_smear"] = False

        # ------------------ testing status ------------------ #
        df["tb_ever_tested"] = False
        df["tb_diagnosed"] = False
        df["tb_date_diagnosed"] = pd.NaT
        df["tb_diagnosed_mdr"] = False

        # ------------------ treatment status ------------------ #
        df["tb_on_treatment"] = False
        df["tb_date_treated"] = pd.NaT
        df["tb_treatment_regimen"].values[:] = "none"
        df["tb_ever_treated"] = False
        df["tb_treatment_failure"] = False

        df["tb_on_ipt"] = False
        df["tb_date_ipt"] = pd.NaT

    def initialise_simulation(self, sim):
        """
        * 1) Schedule the regular TB events
        * 2) Schedule the Logging Event
        * 3) Define the DxTests
        * 4) Define the treatment options
        """

        # 1) Regular events
        sim.schedule_event(TbChildrensPoll(self), sim.date + DateOffset(days=0))
        sim.schedule_event(TbActiveEvent(self), sim.date + DateOffset(days=0))

        sim.schedule_event(TbEndTreatmentEvent(self), sim.date + DateOffset(months=1))
        sim.schedule_event(TbSelfCureEvent(self), sim.date + DateOffset(months=1))

        sim.schedule_event(ScenarioSetupEvent(self), self.parameters["scenario_start_date"])

        # 2) Logging
        sim.schedule_event(TbLoggingEvent(self), sim.date + DateOffset(days=364))
        sim.schedule_event(TbTreatmentLoggingEvent(self), sim.date)

        # 3) -------- Define the DxTests and get the consumables required --------

        p = self.parameters
        # consumables = self.sim.modules["HealthSystem"].parameters["Consumables"]
        hs = self.sim.modules["HealthSystem"]

        # TB Sputum smear test
        # assume that if smear-positive, sputum smear test is 100% specific and sensitive
        self.item_codes_for_consumables_required['sputum_test'] = \
            hs.get_item_codes_from_package_name("Microscopy Test")

        self.sim.modules['HealthSystem'].dx_manager.register_dx_test(
            tb_sputum_test_smear_positive=DxTest(
                property='tb_inf',
                target_categories=["active"],
                sensitivity=p["sens_sputum_smear_positive"],
                specificity=p["spec_sputum_smear_positive"],
                item_codes=self.item_codes_for_consumables_required['sputum_test']
            )
        )
        self.sim.modules['HealthSystem'].dx_manager.register_dx_test(
            tb_sputum_test_smear_negative=DxTest(
                property='tb_inf',
                target_categories=["active"],
                sensitivity=0.0,
                specificity=0.0,
                item_codes=self.item_codes_for_consumables_required['sputum_test']
            )
        )

        # TB GeneXpert
        self.item_codes_for_consumables_required['xpert_test'] = \
            hs.get_item_codes_from_package_name("Xpert test")

        # sensitivity/specificity set for smear status of cases
        self.sim.modules["HealthSystem"].dx_manager.register_dx_test(
            tb_xpert_test_smear_positive=DxTest(
                property="tb_inf",
                target_categories=["active"],
                sensitivity=p["sens_xpert_smear_positive"],
                specificity=p["spec_xpert_smear_positive"],
                item_codes=self.item_codes_for_consumables_required['xpert_test']
            )
        )
        self.sim.modules["HealthSystem"].dx_manager.register_dx_test(
            tb_xpert_test_smear_negative=DxTest(
                property="tb_inf",
                target_categories=["active"],
                sensitivity=p["sens_xpert_smear_negative"],
                specificity=p["spec_xpert_smear_negative"],
                item_codes=self.item_codes_for_consumables_required['xpert_test']
            )
        )

        # TB Chest x-ray
        self.item_codes_for_consumables_required['chest_xray'] = {
            hs.get_item_code_from_item_name("X-ray"): 1}

        # sensitivity/specificity set for smear status of cases
        self.sim.modules["HealthSystem"].dx_manager.register_dx_test(
            tb_xray_smear_positive=DxTest(
                property="tb_inf",
                target_categories=["active"],
                sensitivity=p["sens_xray_smear_positive"],
                specificity=p["spec_xray_smear_positive"],
                item_codes=self.item_codes_for_consumables_required['chest_xray']
            )
        )
        self.sim.modules["HealthSystem"].dx_manager.register_dx_test(
            tb_xray_smear_negative=DxTest(
                property="tb_inf",
                target_categories=["active"],
                sensitivity=p["sens_xray_smear_negative"],
                specificity=p["spec_xray_smear_negative"],
                item_codes=self.item_codes_for_consumables_required['chest_xray']
            )
        )

        # TB clinical diagnosis
        self.sim.modules["HealthSystem"].dx_manager.register_dx_test(
            tb_clinical=DxTest(
                property="tb_inf",
                target_categories=["active"],
                sensitivity=p["sens_clinical"],
                specificity=p["spec_clinical"],
                item_codes=[]
            )
        )

        # 4) -------- Define the treatment options --------
        # adult treatment - primary
        self.item_codes_for_consumables_required['tb_tx_adult'] = \
            hs.get_item_code_from_item_name("Cat. I & III Patient Kit A")

        # child treatment - primary
        self.item_codes_for_consumables_required['tb_tx_child'] = \
            hs.get_item_code_from_item_name("Cat. I & III Patient Kit B")

        # child treatment - primary, shorter regimen
        self.item_codes_for_consumables_required['tb_tx_child_shorter'] = \
            hs.get_item_code_from_item_name("Cat. I & III Patient Kit B shorter")

        # adult treatment - secondary
        self.item_codes_for_consumables_required['tb_retx_adult'] = \
            hs.get_item_code_from_item_name("Cat. II Patient Kit A1")

        # child treatment - secondary
        self.item_codes_for_consumables_required['tb_retx_child'] = \
            hs.get_item_code_from_item_name("Cat. II Patient Kit A2")

        # mdr treatment
        self.item_codes_for_consumables_required['tb_mdrtx'] = {
            hs.get_item_code_from_item_name("Category IV"): 1}

        # ipt
        self.item_codes_for_consumables_required['tb_ipt'] = {
            hs.get_item_code_from_item_name("Isoniazid/Pyridoxine, tablet 300 mg"): 1}

        # chest x-rays
        self.item_codes_for_consumables_required['chest_xray_2'] = {
            hs.get_item_code_from_item_name("X-ray"): 2}

    def on_birth(self, mother_id, child_id):
        """Initialise properties for a newborn individual
        allocate IPT for child if mother diagnosed with TB
        """

        df = self.sim.population.props
        now = self.sim.date

        df.at[child_id, "tb_inf"] = "uninfected"
        df.at[child_id, "tb_strain"] = "none"

        df.at[child_id, "tb_date_latent"] = pd.NaT
        df.at[child_id, "tb_scheduled_date_active"] = pd.NaT
        df.at[child_id, "tb_date_active"] = pd.NaT
        df.at[child_id, "tb_smear"] = False

        # ------------------ testing status ------------------ #
        df.at[child_id, "tb_ever_tested"] = False

        df.at[child_id, "tb_diagnosed"] = False
        df.at[child_id, "tb_date_diagnosed"] = pd.NaT
        df.at[child_id, "tb_diagnosed_mdr"] = False

        # ------------------ treatment status ------------------ #
        df.at[child_id, "tb_on_treatment"] = False
        df.at[child_id, "tb_date_treated"] = pd.NaT
        df.at[child_id, "tb_treatment_regimen"] = "none"
        df.at[child_id, "tb_treatment_failure"] = False
        df.at[child_id, "tb_ever_treated"] = False

        df.at[child_id, "tb_on_ipt"] = False
        df.at[child_id, "tb_date_ipt"] = pd.NaT

        if "Hiv" not in self.sim.modules:
            df.at[child_id, "hv_inf"] = False
            df.at[child_id, "sy_aids_symptoms"] = 0
            df.at[child_id, "hv_art"] = "not"

    def report_daly_values(self):
        """
        This must send back a pd.Series or pd.DataFrame that reports on the average daly-weights that have been
        experienced by persons in the previous month. Only rows for alive-persons must be returned.
        The names of the series of columns is taken to be the label of the cause of this disability.
        It will be recorded by the healthburden module as <ModuleName>_<Cause>.
        """
        df = self.sim.population.props  # shortcut to population properties dataframe

        # health_values = pd.Series(0, index=df.index)

        # to avoid errors when hiv module not running
        df_tmp = df.loc[df.is_alive]
        health_values = pd.Series(0, index=df_tmp.index)

        # hiv-negative
        health_values.loc[
            df_tmp.is_alive
            & (df_tmp.tb_inf == "active")
            & (df_tmp.tb_strain == "ds")
            & ~df_tmp.hv_inf
            ] = self.daly_wts["daly_tb"]
        health_values.loc[
            df_tmp.is_alive
            & (df_tmp.tb_inf == "active")
            & (df_tmp.tb_strain == "mdr")
            & ~df_tmp.hv_inf
            ] = self.daly_wts["daly_tb"]

        # hiv-positive
        health_values.loc[
            df_tmp.is_alive
            & (df_tmp.tb_inf == "active")
            & (df_tmp.tb_strain == "ds")
            & df_tmp.hv_inf
            ] = self.daly_wts["daly_tb_hiv_anaemia"]
        health_values.loc[
            df_tmp.is_alive
            & (df_tmp.tb_inf == "active")
            & (df_tmp.tb_strain == "mdr")
            & df_tmp.hv_inf
            ] = self.daly_wts["daly_mdr_tb_hiv_anaemia"]

        health_values.name = "TB"  # label the cause of this disability

        return health_values.loc[df.is_alive]

    def consider_ipt_for_those_initiating_art(self, person_id):
        """
        this is called by HIV when person is initiating ART
        checks whether person is eligible for IPT
        """
        df = self.sim.population.props

        if df.loc[person_id, "tb_diagnosed"] or df.loc[person_id, "tb_diagnosed_mdr"]:
            pass

        high_risk_districts = self.parameters["tb_high_risk_distr"]
        district = df.at[person_id, "district_of_residence"]
        eligible = df.at[person_id, "tb_inf"] != "active"

        # select coverage rate by year:
        ipt = self.parameters["ipt_coverage"]
        ipt_year = ipt.loc[ipt.year == self.sim.date.year]
        ipt_coverage_plhiv = ipt_year.coverage_plhiv

        if (
            (district in high_risk_districts.district_name.values)
            & eligible
            & (self.rng.rand() < ipt_coverage_plhiv.values)
        ):
            # Schedule the TB treatment event:
            self.sim.modules["HealthSystem"].schedule_hsi_event(
                HSI_Tb_Start_or_Continue_Ipt(self, person_id=person_id),
                priority=1,
                topen=self.sim.date,
                tclose=None,
            )


# # ---------------------------------------------------------------------------
# #   TB infection event
# # ---------------------------------------------------------------------------
class ScenarioSetupEvent(RegularEvent, PopulationScopeEventMixin):
    """ This event exists to change parameters or functions
    depending on the scenario for projections which has been set
    * scenario 0 is the default which uses baseline parameters
    * scenario 1 optimistic, achieving all program targets
    * scenario 2 realistic, program constraints, tx/dx test stockouts, high dropout
    * scenario 3 additional measure to reduce incidence
    * scenario 4 SHINE trial

    It only occurs once at param: scenario_start_date,
    called by initialise_simulation
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(years=100))

    def apply(self, population):

        p = self.module.parameters
        scenario = p["scenario"]

        logger.debug(
            key="message", data=f"ScenarioSetupEvent: scenario {scenario}"
        )

        # baseline scenario 0: no change to parameters/functions
        if scenario == 0:
            return

        if (scenario == 1) | (scenario == 3):
            # increase testing/diagnosis rates, default 2020 0.03/0.25 -> 93% dx
            self.sim.modules["Hiv"].parameters["hiv_testing_rates"]["annual_testing_rate_children"] = 0.1
            self.sim.modules["Hiv"].parameters["hiv_testing_rates"]["annual_testing_rate_adults"] = 0.3

            # ANC testing - value for mothers and infants testing
            self.sim.modules["Hiv"].parameters["prob_anc_test_at_delivery"] = 0.95

            # prob ART start if dx, this is already 95% at 2020
            # self.sim.modules["Hiv"].parameters["prob_start_art_after_hiv_test"] = 0.95

            # viral suppression rates
            # adults already at 95% by 2020
            # change all column values
            self.sim.modules["Hiv"].parameters["prob_viral_suppression"]["virally_suppressed_on_art"] = 95

            # change first-line testing for TB to xpert
            p["first_line_test"] = "xpert"
            p["second_line_test"] = "sputum"

        # health system constraints
        if scenario == 2:

            # set consumables availability to 0.6 for all required cons in hiv/tb modules
            hiv_item_codes = set()
            for f in self.sim.modules['Hiv'].item_codes_for_consumables_required.values():
                hiv_item_codes = hiv_item_codes.union(f.keys())
            self.sim.modules["HealthSystem"].prob_item_codes_available.loc[hiv_item_codes] = 0.75

            tb_item_codes = set()
            for f in self.sim.modules['Tb'].item_codes_for_consumables_required.values():
                tb_item_codes = tb_item_codes.union(f.keys())
            self.sim.modules["HealthSystem"].prob_item_codes_available.loc[tb_item_codes] = 0.6

            # drop viral suppression for all PLHIV
            self.sim.modules["Hiv"].parameters["prob_viral_suppression"]["virally_suppressed_on_art"] = 80

            # lower tb treatment success rates
            self.sim.modules["Tb"].parameters["prob_tx_success_ds"] = 0.6
            self.sim.modules["Tb"].parameters["prob_tx_success_mdr"] = 0.6
            self.sim.modules["Tb"].parameters["prob_tx_success_0_4"] = 0.6
            self.sim.modules["Tb"].parameters["prob_tx_success_5_14"] = 0.6
            self.sim.modules["Tb"].parameters["prob_tx_success_shorter"] = 0.6

        # improve preventive measures
        if scenario == 3:
            # reduce risk of HIV - applies to whole adult population
            self.sim.modules["Hiv"].parameters["beta"] = self.sim.modules["Hiv"].parameters["beta"] * 0.9

            # increase PrEP coverage for FSW after HIV test
            self.sim.modules["Hiv"].parameters["prob_prep_for_fsw_after_hiv_test"] = 0.5

            # prep poll for AGYW - target to highest risk
            # increase retention to 75% for FSW and AGYW
            self.sim.modules["Hiv"].parameters["prob_prep_for_agyw"] = 0.1
            self.sim.modules["Hiv"].parameters["probability_of_being_retained_on_prep_every_3_months"] = 0.75

            # increase probability of VMMC after hiv test
            self.sim.modules["Hiv"].parameters["prob_circ_after_hiv_test"] = 0.25

            # change IPT eligibility for TB contacts to all years
            p["age_eligibility_for_ipt"] = 100

            # increase coverage of IPT
            p["ipt_coverage"]["coverage_plhiv"] = 0.6
            p["ipt_coverage"]["coverage_paediatric"] = 0.8  # this will apply to contacts of all ages


class TbChildrensPoll(RegularEvent, PopulationScopeEventMixin):
    """The Tb Regular Poll Event for assigning active infections to children
    * selects children and schedules onset of active tb
    * schedules tb screening / testing
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=1))

    def apply(self, population):

        # ds-tb cases
        # the outcome of this will be an updated df with new tb cases
        self.assign_active_tb(strain="ds")

        # # schedule some background rates of tb testing (non-symptom driven)
        self.module.send_for_screening(population)

    def assign_active_tb(self, strain):
        """
        select children and assign scheduled date of active tb onset
        update properties as needed
        assumes all infections are ds-tb, no mdr-tb
        symptoms and smear status are assigned in the TbActiveEvent
        """

        df = self.sim.population.props
        p = self.module.parameters
        rng = self.module.rng
        now = self.sim.date
        year = now.year if now.year <= 2020 else 2020

        # WHO estimates of active TB in children
        inc_estimates = p["who_incidence"]
        number_active_tb = int((inc_estimates.loc[
            (inc_estimates.year == year), "estimated_inc_number_children"
        ].values[0]) / 12)

        # identify eligible children, under 16 and not currently with active tb infection
        eligible = df.loc[
            df.is_alive
            & (df.age_years <= 16)
            & (df.tb_inf != "active")
            ].index

        # need to scale the sampling if small population size
        if len(eligible) < number_active_tb:
            number_active_tb = len(eligible)

        # probability based on risk factors, 25x higher if HIV+
        risk_of_tb = LinearModel.multiplicative(
            Predictor("age_years").when(">16", 0.0).otherwise(1.0),
            Predictor("tb_inf").when("active", 0.0).otherwise(1.0),
            Predictor("hv_inf").when(True, p["rr_tb_hiv"]),
        )
        risk_of_progression = risk_of_tb.predict(
            df.loc[eligible]
        )
        # scale risk
        risk_of_progression = risk_of_progression / sum(risk_of_progression)  # must sum to 1
        new_active = rng.choice(df.loc[eligible].index, size=number_active_tb, replace=False, p=risk_of_progression)

        df.loc[new_active, "tb_strain"] = strain

        # schedule onset of active tb
        # schedule for time now up to 1 month
        for person_id in new_active:
            date_progression = now + pd.DateOffset(
                days=rng.randint(0, 30)
            )

            # set date of active tb - properties will be updated at TbActiveEvent every month
            df.at[person_id, "tb_scheduled_date_active"] = date_progression


class TbActiveEvent(RegularEvent, PopulationScopeEventMixin):
    """
    * check for those with dates of active tb onset within last time-period
    *1 change individual properties for active disease
    *2 assign symptoms
    *3 if HIV+, assign smear status and schedule AIDS onset
    *4 if HIV-, assign smear status and schedule death
    *5 schedule screening for general population and symptomatic active cases
    """

    def __init__(self, module):

        self.repeat = 1
        super().__init__(module, frequency=DateOffset(months=self.repeat))

    def apply(self, population):
        df = population.props
        now = self.sim.date
        p = self.module.parameters
        rng = self.module.rng

        # find people eligible for progression to active disease
        # date of active disease scheduled to occur within the last month
        # some will be scheduled for future dates
        # if on IPT or treatment - do nothing
        active_idx = df.loc[
            df.is_alive
            & (df.tb_scheduled_date_active >= (now - DateOffset(months=self.repeat)))
            & (df.tb_scheduled_date_active <= now)
            & ~df.tb_on_ipt
            & ~df.tb_on_treatment
            ].index

        # -------- 1) change individual properties for active disease --------
        df.loc[active_idx, "tb_inf"] = "active"
        df.loc[active_idx, "tb_date_active"] = now
        df.loc[active_idx, "tb_smear"] = False  # default property

        # -------- 2) assign symptoms --------
        for person_id in active_idx:
            for symptom in self.module.symptom_list:
                self.sim.modules["SymptomManager"].change_symptom(
                    person_id=person_id,
                    symptom_string=symptom,
                    add_or_remove="+",
                    disease_module=self.module,
                    duration_in_days=None,
                )

        # -------- 3) if HIV+ assign smear status and schedule AIDS onset --------
        active_and_hiv = df.loc[
            df.is_alive
            & (df.tb_scheduled_date_active >= (now - DateOffset(months=self.repeat)))
            & (df.tb_scheduled_date_active <= now)
            & ~df.tb_on_ipt
            & ~df.tb_on_treatment
            & df.hv_inf
            ].index

        # higher probability of being smear positive than HIV-
        smear_pos = (
            rng.random_sample(len(active_and_hiv)) < p["prop_smear_positive_hiv"]
        )
        active_and_hiv_smear_pos = active_and_hiv[smear_pos]
        df.loc[active_and_hiv_smear_pos, "tb_smear"] = True

        if "Hiv" in self.sim.modules:
            for person_id in active_and_hiv:
                self.sim.schedule_event(
                    hiv.HivAidsOnsetEvent(
                        self.sim.modules["Hiv"], person_id, cause="AIDS_TB"
                    ),
                    now,
                )

        # -------- 4) if HIV- assign smear status and schedule death --------
        active_no_hiv = active_idx[~active_idx.isin(active_and_hiv)]
        smear_pos = rng.random_sample(len(active_no_hiv)) < p["prop_smear_positive"]
        active_no_hiv_smear_pos = active_no_hiv[smear_pos]
        df.loc[active_no_hiv_smear_pos, "tb_smear"] = True

        for person_id in active_no_hiv:
            date_of_tb_death = self.sim.date + pd.DateOffset(
                months=int(rng.uniform(low=1, high=6))
            )
            self.sim.schedule_event(
                event=TbDeathEvent(person_id=person_id, module=self.module, cause="TB"),
                date=date_of_tb_death,
            )

        # -------- 5) schedule screening for asymptomatic and symptomatic people --------

        # schedule some background rates of tb testing (non-symptom + symptom-driven)
        self.module.send_for_screening(population)


class TbEndTreatmentEvent(RegularEvent, PopulationScopeEventMixin):
    """
    * check for those eligible to finish treatment
    * sample for treatment failure and refer for follow-up screening/testing
    * if treatment has finished, change individual properties
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=1))

    def apply(self, population):
        df = population.props
        now = self.sim.date
        p = self.module.parameters
        rng = self.module.rng

        # check across population on tb treatment and end treatment if required
        # if current date is after (treatment start date + treatment length) -> end tx

        # ---------------------- treatment end: first case ds-tb (6 months) ---------------------- #
        # end treatment for new tb (ds) cases
        end_ds_tx_idx = df.loc[
            df.is_alive
            & df.tb_on_treatment
            & ((df.tb_treatment_regimen == "tb_tx_adult") | (df.tb_treatment_regimen == "tb_tx_child"))
            & (
                now
                > (df.tb_date_treated + pd.DateOffset(months=p["ds_treatment_length"]))
            )
            ].index

        # ---------------------- treatment end: retreatment ds-tb (7 months) ---------------------- #
        # end treatment for retreatment cases
        end_ds_retx_idx = df.loc[
            df.is_alive
            & df.tb_on_treatment
            & ((df.tb_treatment_regimen == "tb_retx_adult") | (df.tb_treatment_regimen == "tb_retx_child"))
            & (
                now
                > (
                    df.tb_date_treated
                    + pd.DateOffset(months=p["ds_retreatment_length"])
                )
            )
            ].index

        # ---------------------- treatment end: mdr-tb (24 months) ---------------------- #
        # end treatment for mdr-tb cases
        end_mdr_tx_idx = df.loc[
            df.is_alive
            & df.tb_on_treatment
            & (df.tb_treatment_regimen == "tb_mdrtx")
            & (
                now
                > (df.tb_date_treated + pd.DateOffset(months=p["mdr_treatment_length"]))
            )
            ].index

        # ---------------------- treatment end: shorter paediatric regimen ---------------------- #
        # end treatment for paediatric cases on 4 month regimen
        end_tx_shorter_idx = df.loc[
            df.is_alive
            & df.tb_on_treatment
            & (df.tb_treatment_regimen == "tb_tx_child_shorter")
            & (
                now
                > (df.tb_date_treated + pd.DateOffset(months=p["child_shorter_treatment_length"]))
            )
            ].index

        # join indices
        end_tx_idx = end_ds_tx_idx.union(end_ds_retx_idx)
        end_tx_idx = end_tx_idx.union(end_mdr_tx_idx)
        end_tx_idx = end_tx_idx.union(end_tx_shorter_idx)

        # ---------------------- treatment failure ---------------------- #
        # sample some to have treatment failure
        # assume all retreatment cases will cure
        random_var = rng.random_sample(size=len(df))

        # children aged 0-4 ds-tb
        ds_tx_failure0_4_idx = df.loc[
            (df.index.isin(end_ds_tx_idx))
            & (df.age_years < 5)
            & (random_var < (1 - p["prob_tx_success_0_4"]))
            ].index

        # children aged 5-14 ds-tb
        ds_tx_failure5_14_idx = df.loc[
            (df.index.isin(end_ds_tx_idx))
            & (df.age_years.between(5, 14))
            & (random_var < (1 - p["prob_tx_success_5_14"]))
            ].index

        # children aged <16 and on shorter regimen
        ds_tx_failure_shorter_idx = df.loc[
            (df.index.isin(end_tx_shorter_idx))
            & (df.age_years < 16)
            & (random_var < (1 - p["prob_tx_success_shorter"]))
            ].index

        # adults ds-tb
        ds_tx_failure_adult_idx = df.loc[
            (df.index.isin(end_ds_tx_idx))
            & (df.age_years >= 15)
            & (random_var < (1 - p["prob_tx_success_ds"]))
            ].index

        # all mdr cases on ds tx will fail
        failure_in_mdr_with_ds_tx_idx = df.loc[
            (df.index.isin(end_ds_tx_idx))
            & (df.tb_strain == "mdr")
            ].index

        # some mdr cases on mdr treatment will fail
        failure_due_to_mdr_idx = df.loc[
            (df.index.isin(end_mdr_tx_idx))
            & (df.tb_strain == "mdr")
            & (random_var < (1 - p["prob_tx_success_mdr"]))

            ].index

        # join indices of failing cases together
        tx_failure = (
            list(ds_tx_failure0_4_idx)
            + list(ds_tx_failure5_14_idx)
            + list(ds_tx_failure_shorter_idx)
            + list(ds_tx_failure_adult_idx)
            + list(failure_in_mdr_with_ds_tx_idx)
            + list(failure_due_to_mdr_idx)
        )

        if tx_failure:
            df.loc[tx_failure, "tb_treatment_failure"] = True
            df.loc[
                tx_failure, "tb_ever_treated"
            ] = True  # ensure classed as retreatment case

            for person in tx_failure:
                self.sim.modules["HealthSystem"].schedule_hsi_event(
                    HSI_Tb_ScreeningAndRefer(person_id=person, module=self.module),
                    topen=self.sim.date,
                    tclose=None,
                    priority=0,
                )

        # remove any treatment failure indices from the treatment end indices
        cure_idx = list(set(end_tx_idx) - set(tx_failure))

        # change individual properties for all to off treatment
        df.loc[end_tx_idx, "tb_diagnosed"] = False
        df.loc[end_tx_idx, "tb_on_treatment"] = False
        df.loc[end_tx_idx, "tb_treated_mdr"] = False
        df.loc[end_tx_idx, "tb_treatment_regimen"] = "none"
        # this will indicate that this person has had one complete course of tb treatment
        # subsequent infections will be classified as retreatment
        df.loc[end_tx_idx, "tb_ever_treated"] = True

        # if cured, move infection status back to latent
        # leave tb_strain property set in case of relapse
        df.loc[cure_idx, "tb_inf"] = "latent"
        df.loc[cure_idx, "tb_smear"] = False


class TbSelfCureEvent(RegularEvent, PopulationScopeEventMixin):
    """annual event which allows some individuals to self-cure
    approximate time from infection to self-cure is 3 years
    HIV+ and not virally suppressed cannot self-cure
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=12))

    def apply(self, population):
        p = self.module.parameters
        now = self.sim.date
        rng = self.module.rng

        df = population.props

        prob_self_cure = 1 / p["duration_active_disease_years"]

        # self-cure - move from active to latent, excludes cases that just became active
        random_draw = rng.random_sample(size=len(df))

        # hiv-negative
        self_cure = df.loc[
            (df.tb_inf == "active")
            & df.is_alive
            & ~df.hv_inf
            & (df.tb_date_active < now)
            & (random_draw < prob_self_cure)
            ].index

        # hiv-positive, on art and virally suppressed
        self_cure_art = df.loc[
            (df.tb_inf == "active")
            & df.is_alive
            & df.hv_inf
            & (df.hv_art == "on_VL_suppressed")
            & (df.tb_date_active < now)
            & (random_draw < prob_self_cure)
            ].index

        # resolve symptoms and change properties
        all_self_cure = [*self_cure, *self_cure_art]

        # leave tb strain set in case of relapse
        df.loc[all_self_cure, "tb_inf"] = "latent"
        df.loc[all_self_cure, "tb_diagnosed"] = False
        df.loc[all_self_cure, "tb_smear"] = False

        for person_id in all_self_cure:
            # this will clear all tb symptoms
            self.sim.modules["SymptomManager"].clear_symptoms(
                person_id=person_id, disease_module=self.module
            )


# ---------------------------------------------------------------------------
#   Health System Interactions (HSI)
# ---------------------------------------------------------------------------


class HSI_Tb_ScreeningAndRefer(HSI_Event, IndividualScopeEventMixin):
    """
    The is the Screening-and-Refer HSI.
    A positive outcome from symptom-based screening will prompt referral to tb tests (sputum/xpert/xray)
    no consumables are required for screening (4 clinical questions)

    This event is scheduled by:
        * the main event poll,
        * when someone presents for care through a Generic HSI with tb-like symptoms
        * active screening / contact tracing programmes

    If this event is called within another HSI, it may be desirable to limit the functionality of the HSI: do this
    using the arguments:
        * suppress_footprint=True : the HSI will not have any footprint

    This event will:
    * screen individuals for TB symptoms
    * administer appropriate TB test
    * schedule treatment if needed
    * give IPT for paediatric contacts of diagnosed case
    """

    def __init__(self, module, person_id, suppress_footprint=False):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Tb)

        assert isinstance(suppress_footprint, bool)
        self.suppress_footprint = suppress_footprint

        # Define the necessary information for an HSI
        self.TREATMENT_ID = "Tb_ScreeningAndRefer"
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({"Over5OPD": 1})
        self.ACCEPTED_FACILITY_LEVEL = '1a'
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        """Do the screening and referring to next tests"""

        df = self.sim.population.props
        now = self.sim.date
        p = self.module.parameters
        rng = self.module.rng
        person = df.loc[person_id]

        # If the person is dead, do nothing do not occupy any resources
        if not person["is_alive"]:
            return self.sim.modules["HealthSystem"].get_blank_appt_footprint()

        # If the person is already diagnosed, do nothing do not occupy any resources
        if person["tb_diagnosed"]:
            return self.sim.modules["HealthSystem"].get_blank_appt_footprint()

        # If the person is >16, do nothing do not occupy any resources
        if person["age_years"] > 16:
            return self.sim.modules["HealthSystem"].get_blank_appt_footprint()

        logger.debug(
            key="message", data=f"HSI_Tb_ScreeningAndRefer: person {person_id}"
        )

        smear_status = person["tb_smear"]

        # If the person is already on treatment and not failing, do nothing do not occupy any resources
        if person["tb_on_treatment"] and not person["tb_treatment_failure"]:
            return self.sim.modules["HealthSystem"].get_blank_appt_footprint()

        # ------------------------- screening ------------------------- #

        # check if patient has: cough, fever, night sweat, weight loss
        # if none of the above conditions are present, no further action
        persons_symptoms = self.sim.modules["SymptomManager"].has_what(person_id)
        if not any(x in self.module.symptom_list for x in persons_symptoms):
            return

        # ------------------------- testing ------------------------- #
        # if screening indicates presumptive tb
        test = None
        test_result = None
        ACTUAL_APPT_FOOTPRINT = self.EXPECTED_APPT_FOOTPRINT

        # refer for HIV testing: all ages
        self.sim.modules["HealthSystem"].schedule_hsi_event(
            hsi_event=hiv.HSI_Hiv_TestAndRefer(
                person_id=person_id, module=self.sim.modules["Hiv"], referred_from='Tb'
            ),
            priority=1,
            topen=self.sim.date,
            tclose=None,
        )

        # child under 5 -> chest x-ray, but access is limited
        # if xray not available, HSI_Tb_Xray_level1b will refer
        if person["age_years"] < 5:
            ACTUAL_APPT_FOOTPRINT = self.make_appt_footprint(
                {"Under5OPD": 1}
            )

            logger.debug(
                key="message", data=f"HSI_Tb_ScreeningAndRefer: person {person_id} scheduling xray at level 1b"
            )
            # this HSI will choose relevant sensitivity/specificity depending on person's smear status
            self.sim.modules["HealthSystem"].schedule_hsi_event(
                HSI_Tb_Xray_level1b(person_id=person_id, module=self.module),
                topen=now,
                tclose=None,
                priority=0,
            )
            test_result = False  # to avoid calling a clinical diagnosis

        # for all presumptive cases over 5 years of age
        else:
            # this selects a test for the person
            # if selection is xpert, will check for availability and return sputum if xpert not available
            test = self.module.select_tb_test(person_id)
            assert test in ["sputum", "xpert"]

            if test == "sputum":
                ACTUAL_APPT_FOOTPRINT = self.make_appt_footprint(
                    {"Over5OPD": 1, "LabTBMicro": 1}
                )
                # relevant test depends on smear status (changes parameters on sensitivity/specificity
                if smear_status:
                    test_result = self.sim.modules["HealthSystem"].dx_manager.run_dx_test(
                        dx_tests_to_run="tb_sputum_test_smear_positive", hsi_event=self
                    )
                else:
                    test_result = self.sim.modules["HealthSystem"].dx_manager.run_dx_test(
                        dx_tests_to_run="tb_sputum_test_smear_negative", hsi_event=self
                    )

            elif test == "xpert":
                ACTUAL_APPT_FOOTPRINT = self.make_appt_footprint(
                    {"Over5OPD": 1}
                )
                # relevant test depends on smear status (changes parameters on sensitivity/specificity
                if smear_status:
                    test_result = self.sim.modules["HealthSystem"].dx_manager.run_dx_test(
                        dx_tests_to_run="tb_xpert_test_smear_positive", hsi_event=self
                    )
                # for smear-negative people
                else:
                    test_result = self.sim.modules["HealthSystem"].dx_manager.run_dx_test(
                        dx_tests_to_run="tb_xpert_test_smear_negative", hsi_event=self
                    )

        # ------------------------- testing referrals ------------------------- #

        # if none of the tests are available, try again for sputum
        # requires another appointment - added in ACTUAL_APPT_FOOTPRINT
        if test_result is None:

            logger.debug(
                key="message", data=f"HSI_Tb_ScreeningAndRefer: person {person_id} test not available"
            )

            if smear_status:
                test_result = self.sim.modules["HealthSystem"].dx_manager.run_dx_test(
                    dx_tests_to_run="tb_sputum_test_smear_positive", hsi_event=self
                )
            else:
                test_result = self.sim.modules["HealthSystem"].dx_manager.run_dx_test(
                    dx_tests_to_run="tb_sputum_test_smear_negative", hsi_event=self
                )

            ACTUAL_APPT_FOOTPRINT = self.make_appt_footprint(
                {"Over5OPD": 2, "LabTBMicro": 1}
            )

        # if still no result available, rely on clinical diagnosis
        if test_result is None:
            test_result = self.sim.modules["HealthSystem"].dx_manager.run_dx_test(
                dx_tests_to_run="tb_clinical", hsi_event=self
            )

        # if sputum test result is negative but patient still displays symptoms that indicate active TB,
        # refer to clinical diagnosis (this is to avoid smear negative patients being missed via sputum test)
        if (test == "sputum") and not test_result and not smear_status:
            logger.debug(
                key="message", data=f"HSI_Tb_ScreeningAndRefer: person {person_id} referring for clinical dx"
            )

            test_result = self.sim.modules["HealthSystem"].dx_manager.run_dx_test(
                dx_tests_to_run="tb_clinical", hsi_event=self
            )

        # if a test has been performed, update person's properties
        if test_result is not None:
            df.at[person_id, "tb_ever_tested"] = True

        # if any test returns positive result, refer for appropriate treatment
        if test_result:
            df.at[person_id, "tb_diagnosed"] = True
            df.at[person_id, "tb_date_diagnosed"] = now

            logger.debug(
                key="message",
                data=f"schedule HSI_Tb_StartTreatment for person {person_id}",
            )

            self.sim.modules["HealthSystem"].schedule_hsi_event(
                HSI_Tb_StartTreatment(person_id=person_id, module=self.module),
                topen=now,
                tclose=None,
                priority=0,
            )

        # Return the footprint. If it should be suppressed, return a blank footprint.
        if self.suppress_footprint:
            return self.make_appt_footprint({})
        else:
            return ACTUAL_APPT_FOOTPRINT


class HSI_Tb_Xray_level1b(HSI_Event, IndividualScopeEventMixin):
    """
    The is the x-ray HSI
    usually used for testing children unable to produce sputum
    positive result will prompt referral to start treatment

    """

    def __init__(self, module, person_id, suppress_footprint=False):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Tb)

        assert isinstance(suppress_footprint, bool)
        self.suppress_footprint = suppress_footprint

        # Define the necessary information for an HSI
        self.TREATMENT_ID = "Tb_Xray"
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({"DiagRadio": 1})
        self.ACCEPTED_FACILITY_LEVEL = '1b'
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):

        df = self.sim.population.props

        # if person not alive or already diagnosed, do nothing
        if not df.at[person_id, "is_alive"] or df.at[person_id, "tb_diagnosed"]:
            return self.sim.modules["HealthSystem"].get_blank_appt_footprint()

        smear_status = df.at[person_id, "tb_smear"]

        ACTUAL_APPT_FOOTPRINT = self.EXPECTED_APPT_FOOTPRINT

        # select sensitivity/specificity of test based on smear status
        if smear_status:
            test_result = self.sim.modules["HealthSystem"].dx_manager.run_dx_test(
                dx_tests_to_run="tb_xray_smear_positive", hsi_event=self
            )
        else:
            test_result = self.sim.modules["HealthSystem"].dx_manager.run_dx_test(
                dx_tests_to_run="tb_xray_smear_negative", hsi_event=self
            )

        # if consumables not available, either refer to level 2 or use clinical diagnosis
        if test_result is None:

            # if smear-positive, assume symptoms strongly predictive of TB
            if smear_status:
                test_result = self.sim.modules["HealthSystem"].dx_manager.run_dx_test(
                    dx_tests_to_run="tb_clinical", hsi_event=self
                )
                # add another clinic appointment
                ACTUAL_APPT_FOOTPRINT = self.make_appt_footprint(
                    {"Under5OPD": 1, "DiagRadio": 1}
                )

            # if smear-negative, assume still some uncertainty around dx, refer for another x-ray
            else:
                self.sim.modules["HealthSystem"].schedule_hsi_event(
                    HSI_Tb_Xray_level2(person_id=person_id, module=self.module),
                    topen=self.sim.date + pd.DateOffset(weeks=1),
                    tclose=None,
                    priority=0,
                )

        # if test returns positive result, refer for appropriate treatment
        if test_result:
            df.at[person_id, "tb_diagnosed"] = True
            df.at[person_id, "tb_date_diagnosed"] = self.sim.date

            self.sim.modules["HealthSystem"].schedule_hsi_event(
                HSI_Tb_StartTreatment(person_id=person_id, module=self.module),
                topen=self.sim.date,
                tclose=None,
                priority=0,
            )

        # Return the footprint. If it should be suppressed, return a blank footprint.
        if self.suppress_footprint:
            return self.make_appt_footprint({})
        else:
            return ACTUAL_APPT_FOOTPRINT


class HSI_Tb_Xray_level2(HSI_Event, IndividualScopeEventMixin):
    """
    The is the x-ray HSI performed at level 2
    usually used for testing children unable to produce sputum
    positive result will prompt referral to start treatment
    """

    def __init__(self, module, person_id, suppress_footprint=False):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Tb)

        assert isinstance(suppress_footprint, bool)
        self.suppress_footprint = suppress_footprint

        # Define the necessary information for an HSI
        self.TREATMENT_ID = "Tb_Xray"
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({"DiagRadio": 1})
        self.ACCEPTED_FACILITY_LEVEL = '2'
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):

        df = self.sim.population.props

        # if person not alive or already diagnosed, do nothing
        if not df.at[person_id, "is_alive"] or df.at[person_id, "tb_diagnosed"]:
            return self.sim.modules["HealthSystem"].get_blank_appt_footprint()

        smear_status = df.at[person_id, "tb_smear"]

        ACTUAL_APPT_FOOTPRINT = self.EXPECTED_APPT_FOOTPRINT

        # select sensitivity/specificity of test based on smear status
        if smear_status:
            test_result = self.sim.modules["HealthSystem"].dx_manager.run_dx_test(
                dx_tests_to_run="tb_xray_smear_positive", hsi_event=self
            )
        else:
            test_result = self.sim.modules["HealthSystem"].dx_manager.run_dx_test(
                dx_tests_to_run="tb_xray_smear_negative", hsi_event=self
            )

        # if consumables not available, rely on clinical diagnosis
        if test_result is None:
            test_result = self.sim.modules["HealthSystem"].dx_manager.run_dx_test(
                dx_tests_to_run="tb_clinical", hsi_event=self
            )
            # add another clinic appointment
            ACTUAL_APPT_FOOTPRINT = self.make_appt_footprint(
                {"Under5OPD": 1, "DiagRadio": 1}
            )

        # if test returns positive result, refer for appropriate treatment
        if test_result:
            df.at[person_id, "tb_diagnosed"] = True
            df.at[person_id, "tb_date_diagnosed"] = self.sim.date

            self.sim.modules["HealthSystem"].schedule_hsi_event(
                HSI_Tb_StartTreatment(person_id=person_id, module=self.module),
                topen=self.sim.date,
                tclose=None,
                priority=0,
            )

        # Return the footprint. If it should be suppressed, return a blank footprint.
        if self.suppress_footprint:
            return self.make_appt_footprint({})
        else:
            return ACTUAL_APPT_FOOTPRINT


# # ---------------------------------------------------------------------------
# #   Treatment
# # ---------------------------------------------------------------------------
# # the consumables at treatment initiation include the cost for the full course of treatment
# # so the follow-up appts don't need to account for consumables, just appt time


class HSI_Tb_StartTreatment(HSI_Event, IndividualScopeEventMixin):
    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Tb)

        self.TREATMENT_ID = "Tb_Treatment_Initiation"
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({"TBNew": 1})
        self.ACCEPTED_FACILITY_LEVEL = '1a'
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        """This is a Health System Interaction Event - start TB treatment
        select appropriate treatment and request
        if available, change person's properties
        """
        df = self.sim.population.props
        now = self.sim.date
        person = df.loc[person_id]

        if not person["is_alive"]:
            return self.sim.modules["HealthSystem"].get_blank_appt_footprint()

        if person["tb_on_treatment"]:
            return self.sim.modules["HealthSystem"].get_blank_appt_footprint()

        if person["age_years"] > 16:
            return self.sim.modules["HealthSystem"].get_blank_appt_footprint()

        if not person["tb_diagnosed"]:
            return self.sim.modules["HealthSystem"].get_blank_appt_footprint()

        treatment_regimen = self.select_treatment(person_id)
        # todo remove
        # logger.info(
        #     key="tb_debug",
        #     description="debug",
        #     data={
        #         "person": person_id,
        #         "treatment": treatment_regimen,
        #     },
        # )

        treatment_available = self.get_consumables(
            item_codes=self.module.item_codes_for_consumables_required[treatment_regimen],
            optional_item_codes=self.module.item_codes_for_consumables_required['chest_xray_2']
        )

        logger.debug(
            key="message", data=f"Starting treatment {treatment_regimen}: person {person_id}"
        )

        if treatment_available:
            # start person on tb treatment - update properties
            df.at[person_id, "tb_on_treatment"] = True
            df.at[person_id, "tb_date_treated"] = now
            df.at[person_id, "tb_treatment_regimen"] = treatment_regimen

            if person["tb_diagnosed_mdr"]:
                df.at[person_id, "tb_treated_mdr"] = True
                df.at[person_id, "tb_date_treated_mdr"] = now

            # schedule first follow-up appointment
            follow_up_date = self.sim.date + DateOffset(months=1)
            logger.debug(
                key="message",
                data=f"HSI_Tb_StartTreatment: scheduling first follow-up "
                     f"for person {person_id} on {follow_up_date}",
            )

            self.sim.modules["HealthSystem"].schedule_hsi_event(
                HSI_Tb_FollowUp(person_id=person_id, module=self.module),
                topen=follow_up_date,
                tclose=None,
                priority=0,
            )

        # if treatment not available, return for treatment start in 1 week
        else:
            print("restart treatment")
            self.sim.modules["HealthSystem"].schedule_hsi_event(
                HSI_Tb_StartTreatment(person_id=person_id, module=self.module),
                topen=self.sim.date + DateOffset(weeks=1),
                tclose=None,
                priority=0,
            )

    def select_treatment(self, person_id):
        """
        helper function to select appropriate treatment and check whether
        consumables are available to start drug course
        treatment will always be for ds-tb unless mdr has been identified
        :return: treatment_regimen[STR]
        """
        df = self.sim.population.props
        person = df.loc[person_id]

        treatment_regimen = None  # default return value

        # -------- MDR-TB -------- #

        if person["tb_diagnosed_mdr"]:

            treatment_regimen = "tb_mdrtx"

        # -------- First TB infection -------- #
        # could be undiagnosed mdr or ds-tb: treat as ds-tb

        elif not person["tb_ever_treated"]:

            if person["age_years"] > 16:
                # treatment for ds-tb: adult
                treatment_regimen = "tb_tx_adult"
            else:
                # treatment for ds-tb: child
                treatment_regimen = "tb_tx_child"

        # -------- Secondary TB infection -------- #
        # person has been treated before
        # possible treatment failure or subsequent reinfection
        else:

            if person["age_years"] > 16:
                # treatment for reinfection ds-tb: adult
                treatment_regimen = "tb_retx_adult"

            else:
                # treatment for reinfection ds-tb: child
                treatment_regimen = "tb_retx_child"

        # -------- SHINE Trial shorter paediatric regimen -------- #
        if (self.module.parameters["scenario"] == 4) \
            & (self.sim.date >= self.module.parameters["scenario_start_date"]) \
            & (person["age_years"] <= 16) \
            & (not person["tb_smear"]) \
            & (not person["tb_ever_treated"]) \
            & (not person["tb_diagnosed_mdr"]) \
            & (not person["is_pregnant"]):
            # shorter treatment for child with minimal tb
            treatment_regimen = "tb_tx_child_shorter"

        return treatment_regimen


# # ---------------------------------------------------------------------------
# #   Follow-up appts
# # ---------------------------------------------------------------------------
class HSI_Tb_FollowUp(HSI_Event, IndividualScopeEventMixin):
    """
    This is a Health System Interaction Event
    clinical monitoring for tb patients on treatment
    will schedule sputum smear test if needed
    if positive sputum smear, schedule xpert test for drug sensitivity
    then schedule the next follow-up appt if needed
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Tb)

        # Get a blank footprint and then edit to define call on resources of this treatment event
        the_appt_footprint = self.make_appt_footprint({"TBFollowUp": 1})

        # Define the necessary information for an HSI
        self.TREATMENT_ID = "Tb_FollowUp"
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = '1a'
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        p = self.module.parameters
        df = self.sim.population.props
        person = df.loc[person_id]

        # Do not run if the person is not alive, or is not currently on treatment
        if (not person["is_alive"]) or (not person["tb_on_treatment"]):
            return

        ACTUAL_APPT_FOOTPRINT = self.EXPECTED_APPT_FOOTPRINT

        # months since treatment start - to compare with monitoring schedule
        # make sure it's an integer value
        months_since_tx = int(
            (self.sim.date - df.at[person_id, "tb_date_treated"]).days / 30.5
        )

        logger.debug(
            key="message",
            data=f"HSI_Tb_FollowUp: person {person_id} on month {months_since_tx} of treatment",
        )

        # default clinical monitoring schedule for first infection ds-tb
        xperttest_result = None
        follow_up_times = p["followup_times"]
        clinical_fup = follow_up_times["ds_clinical_monitor"].dropna()
        sputum_fup = follow_up_times["ds_sputum"].dropna()
        treatment_length = p["ds_treatment_length"]

        # assign clinical monitoring schedule to each treatment regimen:
        if ((person["tb_treatment_regimen"] == "tb_tx_adult") or
            (person["tb_treatment_regimen"] == "tb_tx_child")):

            # if hiv+:
            if person["hv_inf"]:
                clinical_fup = follow_up_times["ds_clinical_monitor_hiv_positive"].dropna()

            # if hiv-:
            else:
                clinical_fup = follow_up_times["ds_clinical_monitor_hiv_negative"].dropna()

        # if previously treated:
        elif ((person["tb_treatment_regimen"] == "tb_retx_adult") or
              (person["tb_treatment_regimen"] == "tb_retx_child")):

            clinical_fup = follow_up_times["ds_retreatment_clinical"].dropna()

        # if person diagnosed with mdr - this treatment schedule takes precedence
        elif person["tb_treatment_regimen"] == "tb_mdrtx":

            clinical_fup = follow_up_times["mdr_clinical_monitor"].dropna()

        # if person on shorter paediatric regimen
        elif person["tb_treatment_regimen"] == "tb_tx_child_shorter":

            # if hiv+:
            if person["hv_inf"]:
                clinical_fup = follow_up_times["shine_clinical_monitor_hiv_positive"].dropna()

            # if hiv-:
            else:
                clinical_fup = follow_up_times["shine_clinical_monitor_hiv_negative"].dropna()

        # return a blank footprint if no clinical monitoring is scheduled for this month
        if months_since_tx not in clinical_fup:
            ACTUAL_APPT_FOOTPRINT = self.sim.modules["HealthSystem"].get_blank_appt_footprint()

        # assign sputum monitoring follow up schedule to each treatment regimen:
        # if previously treated:
        if ((person["tb_treatment_regimen"] == "tb_retx_adult") or
            (person["tb_treatment_regimen"] == "tb_retx_child")):

            # if strain is ds and person previously treated:
            sputum_fup = follow_up_times["ds_retreatment_sputum"].dropna()
            treatment_length = p["ds_retreatment_length"]

        # if person diagnosed with mdr - this treatment schedule takes precedence
        elif person["tb_treatment_regimen"] == "tb_mdrtx":

            sputum_fup = follow_up_times["mdr_sputum"].dropna()
            treatment_length = p["mdr_treatment_length"]

        # if person on shorter paediatric regimen
        elif person["tb_treatment_regimen"] == "tb_tx_child_shorter":
            sputum_fup = follow_up_times["shine_sputum"].dropna()
            treatment_length = p["child_shorter_treatment_length"]

        # check schedule for sputum test and perform if necessary
        # note: sputum test monitoring is conducted in smear-positive patients only
        if ((person["tb_smear"]) and (months_since_tx in sputum_fup)):
            ACTUAL_APPT_FOOTPRINT = self.make_appt_footprint(
                {"TBFollowUp": 1, "LabTBMicro": 1}
            )

            # choose test parameters based on smear status
            if person["tb_smear"]:
                test_result = self.sim.modules["HealthSystem"].dx_manager.run_dx_test(
                    dx_tests_to_run="tb_sputum_test_smear_positive", hsi_event=self
                )
            else:
                test_result = self.sim.modules["HealthSystem"].dx_manager.run_dx_test(
                    dx_tests_to_run="tb_sputum_test_smear_negative", hsi_event=self
                )

            # if sputum test was available and returned positive and not diagnosed with mdr, schedule xpert test
            if test_result and not person["tb_diagnosed_mdr"]:
                ACTUAL_APPT_FOOTPRINT = self.make_appt_footprint(
                    {"TBFollowUp": 1, "LabTBMicro": 1, "LabMolec": 1}
                )
                if person["tb_smear"]:
                    xperttest_result = self.sim.modules["HealthSystem"].dx_manager.run_dx_test(
                        dx_tests_to_run="tb_xpert_test_smear_positive", hsi_event=self
                    )
                else:
                    xperttest_result = self.sim.modules["HealthSystem"].dx_manager.run_dx_test(
                        dx_tests_to_run="tb_xpert_test_smear_negative", hsi_event=self
                    )

        # if xpert test returns new mdr-tb diagnosis
        if xperttest_result and (df.at[person_id, "tb_strain"] == "mdr"):
            df.at[person_id, "tb_diagnosed_mdr"] = True
            # already diagnosed with active tb so don't update tb_date_diagnosed
            df.at[person_id, "tb_treatment_failure"] = True

            # restart treatment (new regimen) if newly diagnosed with mdr-tb
            self.sim.modules["HealthSystem"].schedule_hsi_event(
                HSI_Tb_StartTreatment(person_id=person_id, module=self.module),
                topen=self.sim.date,
                tclose=None,
                priority=0,
            )

        # for all ds cases and known mdr cases:
        # schedule next clinical follow-up appt if still within treatment length
        elif months_since_tx < treatment_length:
            follow_up_date = self.sim.date + DateOffset(months=1)
            logger.debug(
                key="message",
                data=f"HSI_Tb_FollowUp: scheduling next follow-up "
                     f"for person {person_id} on {follow_up_date}",
            )

            self.sim.modules["HealthSystem"].schedule_hsi_event(
                HSI_Tb_FollowUp(person_id=person_id, module=self.module),
                topen=follow_up_date,
                tclose=None,
                priority=0,
            )

        return ACTUAL_APPT_FOOTPRINT

# ---------------------------------------------------------------------------
#   IPT
# ---------------------------------------------------------------------------
class HSI_Tb_Start_or_Continue_Ipt(HSI_Event, IndividualScopeEventMixin):
    """
    This is a Health System Interaction Event - give ipt to reduce risk of active TB
    It can be scheduled by:
    * HIV.HSI_Hiv_StartOrContinueTreatment for PLHIV, diagnosed and on ART
    * Tb.HSI_Tb_StartTreatment for up to 5 contacts of diagnosed active TB case

    if person referred by ART initiation (HIV+), IPT given for 36 months
    paediatric IPT is 6-9 months
     """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        self.TREATMENT_ID = "Tb_Ipt"
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({"Over5OPD": 1})
        self.ACCEPTED_FACILITY_LEVEL = '1a'
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):

        logger.debug(key="message", data=f"Starting IPT for person {person_id}")

        df = self.sim.population.props  # shortcut to the dataframe

        person = df.loc[person_id]

        # Do not run if the person is not alive or already on IPT or diagnosed active infection
        if (
            (not person["is_alive"])
            or person["tb_on_ipt"]
            or person["tb_diagnosed"]
        ):
            return

        # if currently have symptoms of TB, refer for screening/testing
        persons_symptoms = self.sim.modules["SymptomManager"].has_what(person_id)
        if any(x in self.module.symptom_list for x in persons_symptoms):

            self.sim.modules["HealthSystem"].schedule_hsi_event(
                HSI_Tb_ScreeningAndRefer(person_id=person_id, module=self.module),
                topen=self.sim.date,
                tclose=self.sim.date + pd.DateOffset(days=14),
                priority=0,
            )

        else:
            # Check/log use of consumables, and give IPT if available
            # if not available, reschedule IPT start
            if self.get_consumables(
                item_codes=self.module.item_codes_for_consumables_required["tb_ipt"]
            ):
                # Update properties
                df.at[person_id, "tb_on_ipt"] = True
                df.at[person_id, "tb_date_ipt"] = self.sim.date

                # schedule decision to continue or end IPT after 6 months
                self.sim.schedule_event(
                    Tb_DecisionToContinueIPT(self.module, person_id),
                    self.sim.date + DateOffset(months=6),
                )
            else:
                self.sim.modules["HealthSystem"].schedule_hsi_event(
                    HSI_Tb_Start_or_Continue_Ipt(person_id=person_id, module=self.module),
                    topen=self.sim.date,
                    tclose=self.sim.date + pd.DateOffset(days=14),
                    priority=0,
                )


class Tb_DecisionToContinueIPT(Event, IndividualScopeEventMixin):
    """Helper event that is used to 'decide' if someone on IPT should continue or end
    This event is scheduled by 'HSI_Tb_Start_or_Continue_Ipt' after 6 months

    * end IPT for all
    * schedule further IPT for HIV+ if still eligible (no active TB diagnosed, <36 months IPT)
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        df = self.sim.population.props
        person = df.loc[person_id]
        m = self.module

        if not (person["is_alive"]):
            return

        # default update properties for all
        df.at[person_id, "tb_on_ipt"] = False

        # decide whether PLHIV will continue
        if (
            (not person["tb_diagnosed"])
            and (
            person["tb_date_ipt"] < (self.sim.date - pd.DateOffset(days=36 * 30.5))
        )
            and (m.rng.random_sample() < m.parameters["prob_retained_ipt_6_months"])
        ):
            self.sim.modules["HealthSystem"].schedule_hsi_event(
                HSI_Tb_Start_or_Continue_Ipt(person_id=person_id, module=m),
                topen=self.sim.date,
                tclose=self.sim.date + pd.DateOffset(days=14),
                priority=0,
            )


# ---------------------------------------------------------------------------
#   Deaths
# ---------------------------------------------------------------------------


class TbDeathEvent(Event, IndividualScopeEventMixin):
    """
    The scheduled death for a tb case
    check whether this death should occur using a linear model
    will depend on treatment status, smear status and age
    """

    def __init__(self, module, person_id, cause):
        super().__init__(module, person_id=person_id)
        self.cause = cause

    def apply(self, person_id):
        df = self.sim.population.props

        if not df.at[person_id, "is_alive"]:
            return

        logger.debug(
            key="message",
            data=f"TbDeathEvent: checking whether death should occur for person {person_id}",
        )

        # use linear model to determine whether this person will die:
        rng = self.module.rng
        result = self.module.lm["death_rate"].predict(df.loc[[person_id]], rng=rng)

        if result:
            logger.debug(
                key="message",
                data=f"TbDeathEvent: cause this death for person {person_id}",
            )

            self.sim.modules["Demography"].do_death(
                individual_id=person_id,
                cause=self.cause,
                originating_module=self.module,
            )


# ---------------------------------------------------------------------------
#   Logging
# ---------------------------------------------------------------------------


class TbLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        """produce some outputs to check"""
        # run this event every 12 months
        self.repeat = 12
        super().__init__(module, frequency=DateOffset(months=self.repeat))

    def apply(self, population):
        # get some summary statistics
        df = population.props
        now = self.sim.date

        # ------------------------------------ INCIDENCE ------------------------------------
        # total number of new active cases in last year - ds + mdr
        # may have died in the last year but still counted as active case for the year

        # number of new active cases
        new_tb_cases = len(
            df[(df.tb_date_active > (now - DateOffset(months=self.repeat)))]
        )

        # number of new active cases (0 - 16 years)
        new_tb_cases_child = len(
            df[(df.tb_date_active > (now - DateOffset(months=self.repeat)))
               & (df.age_years <= 16)]
        )

        # number of new active cases in HIV+
        inc_active_hiv = len(
            df[
                (df.tb_date_active > (now - DateOffset(months=self.repeat)))
                & df.hv_inf
                ]
        )

        # number of new active cases in HIV+ children
        inc_active_hiv_child = len(
            df[
                (df.tb_date_active > (now - DateOffset(months=self.repeat)))
                & df.hv_inf
                & (df.age_years <= 16)
                ]
        )

        # proportion of active TB cases in the last year who are HIV-positive
        prop_hiv = inc_active_hiv / new_tb_cases if new_tb_cases else 0

        # proportion of active TB cases in the last year who are HIV-positive and aged < 16 years
        prop_hiv_child = inc_active_hiv_child / new_tb_cases_child if new_tb_cases_child else 0

        logger.info(
            key="tb_incidence",
            description="Number new active and latent TB cases, total and in PLHIV",
            data={
                "num_new_active_tb": new_tb_cases,
                "num_new_active_tb_child": new_tb_cases_child,
                "num_new_active_tb_in_hiv": inc_active_hiv,
                "num_new_active_tb_in_hiv_child": inc_active_hiv_child,
                "prop_active_tb_in_plhiv": prop_hiv,
                "prop_active_tb_in_plhiv_child": prop_hiv_child,
            },
        )

        # save outputs to dict for calibration
        self.module.tb_outputs["date"] += [self.sim.date.year]
        self.module.tb_outputs["num_new_active_tb"] += [new_tb_cases]

        # ------------------------------------ PREVALENCE ------------------------------------
        # number of current active cases divided by population alive

        # ACTIVE
        num_active_tb_cases = len(df[(df.tb_inf == "active") & df.is_alive])
        prev_active = num_active_tb_cases / len(df[df.is_alive])

        assert prev_active <= 1

        # prevalence of active TB in adults
        num_active_adult = len(
            df[(df.tb_inf == "active") & (df.age_years >= 16) & df.is_alive]
        )
        prev_active_adult = num_active_adult / len(
            df[(df.age_years >= 16) & df.is_alive]
        )
        assert prev_active_adult <= 1

        # prevalence of active TB in children
        num_active_child = len(
            df[(df.tb_inf == "active") & (df.age_years < 16) & df.is_alive]
        )
        prev_active_child = num_active_child / len(
            df[(df.age_years <= 16) & df.is_alive]
        )
        assert prev_active_child <= 1

        logger.info(
            key="tb_prevalence",
            description="Prevalence of active and latent TB cases, total and in PLHIV",
            data={
                "tbPrevActive": prev_active,
                "tbPrevActiveAdult": prev_active_adult,
                "tbPrevActiveChild": prev_active_child,
            },
        )

        # ------------------------------------ MDR ------------------------------------
        # number new mdr tb cases
        new_mdr_cases = len(
            df[
                (df.tb_strain == "mdr")
                & (df.tb_date_active >= (now - DateOffset(months=self.repeat)))
                ]
        )

        if new_mdr_cases:
            prop_mdr = new_mdr_cases / new_tb_cases
        else:
            prop_mdr = 0

        # number new mdr cases (0 - 16 years)
        new_mdr_cases_child = len(
            df[
                (df.tb_strain == "mdr")
                & (df.age_years <= 16)
                & (df.tb_date_active >= (now - DateOffset(months=self.repeat)))
                ]
        )

        if new_mdr_cases_child:
            prop_mdr_child = new_mdr_cases_child / new_tb_cases_child
        else:
            prop_mdr_child = 0

        logger.info(
            key="tb_mdr",
            description="Incidence of new active MDR cases and the proportion of TB cases that are MDR",
            data={
                "tbNewActiveMdrCases": new_mdr_cases,
                "tbPropActiveCasesMdr": prop_mdr,
                "tbNewActiveMdrCasesChild": new_mdr_cases_child,
                "tbPropActiveCasesMdrChild": prop_mdr_child,
            },
        )

        # ------------------------------------ CASE NOTIFICATIONS ------------------------------------
        # number diagnoses (new, relapse, reinfection) in last timeperiod
        new_tb_diagnosis = len(
            df[(df.tb_date_diagnosed >= (now - DateOffset(months=self.repeat)))]
        )

        # number diagnoses (new, relapse, reinfection) in last timeperiod for children aged 0-16 years
        new_tb_diagnosis_child = len(
            df[(df.tb_date_diagnosed >= (now - DateOffset(months=self.repeat)))
               & (df.age_years <= 16)]
        )

        # ------------------------------------ TREATMENT ------------------------------------
        # number of tb cases who became active in last timeperiod and initiated treatment
        new_tb_tx = len(
            df[
                (df.tb_date_active >= (now - DateOffset(months=self.repeat)))
                & (df.tb_date_treated >= (now - DateOffset(months=self.repeat)))
                ]
        )

        # treatment coverage: if became active and was treated in last timeperiod
        if new_tb_cases:
            tx_coverage = new_tb_tx / new_tb_cases
            # assert tx_coverage <= 1
        else:
            tx_coverage = 0

        # number of tb cases who became active in last timeperiod and initiated treatment for children aged 0-16 years
        new_tb_tx_child = len(
            df[
                (df.tb_date_active >= (now - DateOffset(months=self.repeat)))
                & (df.tb_date_treated >= (now - DateOffset(months=self.repeat)))
                & (df.age_years <= 16)
                ]
        )

        # treatment coverage: if became active and was treated in last timeperiod for children aged 0-16 years
        if new_tb_cases_child:
            tx_coverage_child = new_tb_tx_child / new_tb_cases_child
            # assert tx_coverage <= 1
        else:
            tx_coverage_child = 0

        logger.info(
            key="tb_treatment",
            description="TB treatment coverage",
            data={
                "tbNewDiagnosis": new_tb_diagnosis,
                "tbNewDiagnosisChild": new_tb_diagnosis_child,
                "tbNewTreatment": new_tb_tx,
                "tbNewTreatmentChild": new_tb_tx_child,
                "tbTreatmentCoverage": tx_coverage,
                "tbTreatmentCoverageChild": tx_coverage_child,
            },
        )

        # ------------------------------------ TREATMENT FAILURE ------------------------------------
        # Number of people that failed treatment
        num_tx_failure = len(
            df[
                (df.tb_date_treated >= (now - DateOffset(months=self.repeat)))
                & df.tb_treatment_failure
                ]
        )

        # Number of children that failed treatment
        num_child_tx_failure = len(
            df[
                (df.age_years <= 16)
                & (df.tb_date_treated >= (now - DateOffset(months=self.repeat)))
                & df.tb_treatment_failure
                ]
        )

        logger.info(
            key="tb_treatment_failure",
            description="TB treatment failure",
            data={
                "tbNumTxFailure": num_tx_failure,
                "tbNumTxFailureChild": num_child_tx_failure,
            }
        )

        # ------------------------------- SCENARIO 4: SHINE TRIAL LOGGER ------------------------------
        # This logger provides all the summary statistics needed to analyse the SHINE Trial health outcomes.
        # Comment out if not running scenario 4, as some statistics are repeated above.

        # (1) Number of new active TB cases (0-16 years)

        num_new_active_tb_cases_child = len(
            df[(df.tb_date_active >= (now - DateOffset(months=self.repeat)))
               & (df.age_years <= 16)]
        )

        # (2) Number of new diagnosed TB cases (0 - 16 years)

        num_new_diagnosed_tb_cases_child = len(
            df[(df.tb_date_diagnosed >= (now - DateOffset(months=self.repeat)))
               & (df.age_years <= 16)]
        )

        # (3) Number of new treated TB cases (0 - 16 years)

        num_new_treated_tb_cases_child = len(
            df[
                (df.tb_date_treated >= (now - DateOffset(months=self.repeat)))
                & (df.age_years <= 16)
                ]
        )

        # (4) Cases detection rate (proportion of new active TB cases that were diagnosed)

        if num_new_active_tb_cases_child:
            case_detection_rate = (num_new_diagnosed_tb_cases_child / num_new_active_tb_cases_child) * 100
        else:
            case_detection_rate = 0

        # (5) Treatment coverage (proportion of new diagnosed TB cases that were treated)

        if num_new_diagnosed_tb_cases_child:
            treatment_coverage = (num_new_treated_tb_cases_child / num_new_diagnosed_tb_cases_child) * 100
        else:
            treatment_coverage = 0

        # The following statistics look at how many patients are eligible for the SHINE trial shorter treatment option.
        # First, we look at the number and proportion of patients who are eligible at the infection stage.
        # Then, we look at how these statistics change at the diagnosis stage.
        # These statistics are for interest, to give an indication as to how well smear-negative cases are identified.

        # (6) Number of new active TB cases eligible for shorter treatment

        num_elig_shorter_tx = len(
            df[(df.tb_date_active >= (now - DateOffset(months=self.repeat)))
               & (df.age_years <= 16)
               & ~df.tb_smear
               & ~df.tb_ever_treated
               & ~df.tb_diagnosed_mdr
               & ~df.is_pregnant]
        )

        # (7) Proportion of new active TB cases eligible for shorter treatment

        if num_new_active_tb_cases_child:
            prop_elig_shorter_tx = (num_elig_shorter_tx / num_new_active_tb_cases_child) * 100
        else:
            prop_elig_shorter_tx = 0

        # (8) Number of new diagnosed TB cases eligible for shorter treatment

        num_elig_shorter_tx_diagnosed_cases = len(
            df[(df.tb_date_diagnosed >= (now - DateOffset(months=self.repeat)))
               & (df.age_years <= 16)
               & ~df.tb_smear
               & ~df.tb_ever_treated
               & ~df.tb_diagnosed_mdr
               & ~df.is_pregnant]
        )

        # (9) Proportion of new diagnosed TB cases eligible for shorter treatment

        if num_new_diagnosed_tb_cases_child:
            prop_elig_shorter_tx_diagnosed_cases = (num_elig_shorter_tx_diagnosed_cases /
                                                    num_new_diagnosed_tb_cases_child) * 100
        else:
            prop_elig_shorter_tx_diagnosed_cases = 0

        logger.info(
            key="tb_shine_data",
            description="Scenario 4:SHINE Trial summary statistics",
            data={
                "NewActiveTBCases": num_new_active_tb_cases_child,
                "NewDiagnosedTBCases": num_new_diagnosed_tb_cases_child,
                "NewTreatedTBCases": num_new_treated_tb_cases_child,
                "TBCaseDetection": case_detection_rate,
                "TBTreatmentCoverage": treatment_coverage,
                "NewActiveTBCasesEligibleShorterTx": num_elig_shorter_tx,
                "PropActiveTBCasesEligibleShorterTx": prop_elig_shorter_tx,
                "NewDiagnosedTBCasesEligibleShorterTx": num_elig_shorter_tx_diagnosed_cases,
                "PropDiagnosedTBCasesEligibleShorterTx": prop_elig_shorter_tx_diagnosed_cases,
            },
        )

        # ------------------------------------ TREATMENT DELAYS ------------------------------------
        # for every person initiated on treatment, record time from onset to treatment
        # each year a series of intervals in days (treatment date - onset date) are recorded
        # convert to list

        # children
        child_tx_idx = df.loc[(df.age_years < 16) &
                              (df.tb_date_treated >= (now - DateOffset(months=self.repeat)))].index
        child_tx_delays = (df.loc[child_tx_idx, "tb_date_treated"] - df.loc[child_tx_idx, "tb_date_active"]).dt.days
        child_tx_delays = child_tx_delays.tolist()

        logger.info(
            key="tb_treatment_delays",
            description="TB time from onset to treatment",
            data={
                "tbTreatmentDelayChildren": child_tx_delays,
            },
        )


class TbTreatmentLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        """produce outputs on the number of patients initiated in each tb treatment regimen"""
        self.repeat = 4  # run this event every 4 months to capture patient initiated on all treatment regimens
        super().__init__(module, frequency=DateOffset(months=self.repeat))

    def apply(self, population):
        # get some summary statistics
        df = population.props
        now = self.sim.date

        # (1) total number initiated on treatment

        num_tb_tx = len(
            df[(df.tb_date_treated >= (now - DateOffset(months=self.repeat)))]
        )

        # (2) number initiated on adult treatment

        num_tb_tx_adult = len(
            df[(df.tb_date_treated >= (now - DateOffset(months=self.repeat)))
               & (df.tb_treatment_regimen == "tb_tx_adult")]
        )

        # (3) number initiated on adult retreatment

        num_tb_retx_adult = len(
            df[(df.tb_date_treated >= (now - DateOffset(months=self.repeat)))
               & (df.tb_treatment_regimen == "tb_retx_adult")]
        )

        # (4) number initiated on child treatment

        num_tb_tx_child = len(
            df[(df.tb_date_treated >= (now - DateOffset(months=self.repeat)))
               & (df.tb_treatment_regimen == "tb_tx_child")]
        )

        # (5) number initiated on shorter child treatment

        num_tb_tx_child_shorter = len(
            df[(df.tb_date_treated >= (now - DateOffset(months=self.repeat)))
               & (df.tb_treatment_regimen == "tb_tx_child_shorter")]
        )

        # (6) number initiated on child retreatment

        num_tb_retx_child = len(
            df[(df.tb_date_treated >= (now - DateOffset(months=self.repeat)))
               & (df.tb_treatment_regimen == "tb_retx_child")]
        )

        # (7) number initiated on mdr treatment

        num_mdr_tx = len(
            df[(df.tb_date_treated >= (now - DateOffset(months=self.repeat)))
               & (df.tb_treatment_regimen == "tb_mdrtx")]
        )

        logger.info(
            key="tb_treatment_regimen",
            description="",
            data={
                "TBTx": num_tb_tx,
                "TBTxChild": num_tb_tx_child,
                "TBTxChildShorter": num_tb_tx_child_shorter,
                "TBRetxChild": num_tb_retx_child,
                "TBTxAdult": num_tb_tx_adult,
                "TBRetxAdult": num_tb_retx_adult,
                "TBTxMdr": num_mdr_tx
            }
        )
