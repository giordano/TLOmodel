from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats

from tlo import DateOffset, Module, Parameter, Property, Types, logging
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.lm import LinearModel
from tlo.methods import Metadata, labour_lm
from tlo.methods.dxmanager import DxTest
from tlo.methods.healthsystem import HSI_Event
from tlo.methods.hiv import HSI_Hiv_TestAndRefer
from tlo.methods.postnatal_supervisor import PostnatalWeekOneEvent
from tlo.util import BitsetHandler

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Labour(Module):
    """This is module is responsible for the the process of labour, birth and the immediate postnatal period (up until
    48hrs post birth). This model has a number of core functions including; initiating the onset of labour for women on
    their pre-determined due date (or prior to this for preterm labour/admission for delivery), applying the incidence
     of a core set of maternal complications ocurring in the intrapartum period and outcomes such as maternal death or
     still birth, scheduling birth for women surviving labour and applying risk of complications and outcomes in the
     postnatal period. Complications explicitly modelled in this module include obstructed labour, antepartum
     haemorrhage, maternal infection and sepsis, progression of hypertensive disorders, uterine rupture and postpartum
      haemorrhage. In addition to the natural history of labour this module manages care seeking for women in labour
      (for delivery or following onset of complications at a home birth) and includes HSIs which represent
      Skilled Birth Attendance at either Basic or Comprehensive level emergency obstetric care facilities."""

    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath

        # First we define dictionaries which will store the current parameters of interest (to allow parameters to
        # change between 2010 and 2020) and the linear models
        self.current_parameters = dict()
        self.la_linear_models = dict()

        # This dictionary will track incidence of complications of labour
        self.labour_tracker = dict()

        # This list contains the individual_ids of women in labour, used for testing
        self.women_in_labour = list()

        # These lists will contain possible complications and are used as checks in assert functions
        self.possible_intrapartum_complications = list()
        self.possible_postpartum_complications = list()

    METADATA = {
        Metadata.DISEASE_MODULE,
        Metadata.USES_HEALTHSYSTEM,
        Metadata.USES_HEALTHBURDEN,
    }

    PARAMETERS = {

        #  PARITY AT BASELINE...
        'intercept_parity_lr2010': Parameter(
            Types.LIST, 'intercept value for linear regression equation predicating womens parity at 2010 baseline'),
        'effect_age_parity_lr2010': Parameter(
            Types.LIST, 'effect of an increase in age by 1 year in the linear regression equation predicating '
                        'womens parity at 2010 baseline'),
        'effect_mar_stat_2_parity_lr2010': Parameter(
            Types.LIST, 'effect of a change in marriage status from comparison (level 1) in the linear '
                        'regression equation predicating womens parity at 2010 baseline'),
        'effect_mar_stat_3_parity_lr2010': Parameter(
            Types.LIST, 'effect of a change in marriage status from comparison (level 1) in the linear '
                        'regression equation predicating womens parity at 2010 baseline'),
        'effect_wealth_lev_5_parity_lr2010': Parameter(
            Types.LIST, 'effect of a change in wealth status from comparison (level 1) in the linear '
                        'regression equation predicating womens parity at 2010 baseline'),
        'effect_wealth_lev_4_parity_lr2010': Parameter(
            Types.LIST, 'effect of an increase in wealth level in the linear regression equation predicating womens '
                        'parity at 2010 base line'),
        'effect_wealth_lev_3_parity_lr2010': Parameter(
            Types.LIST, 'effect of an increase in wealth level in the linear regression equation predicating womens '
                        'parity at 2010 base line'),
        'effect_wealth_lev_2_parity_lr2010': Parameter(
            Types.LIST, 'effect of an increase in wealth level in the linear regression equation predicating womens '
                        'parity at 2010 base line'),
        'effect_wealth_lev_1_parity_lr2010': Parameter(
            Types.LIST, 'effect of an increase in wealth level in the linear regression equation predicating womens '
                        'parity at 2010 base line'),

        # POSTTERM RATE
        'risk_post_term_labour': Parameter(
            Types.LIST, 'risk of remaining pregnant past 42 weeks'),

        # MISC...
        'list_limits_for_defining_term_status': Parameter(
            Types.LIST, 'List of number of days of gestation used to define term, early preterm, late preterm and '
                        'post term delivery'),
        'allowed_interventions': Parameter(
            Types.LIST, 'list of interventions allowed to run, used in analysis'),

        # BIRTH WEIGHT...
        'mean_birth_weights': Parameter(
            Types.LIST, 'list of mean birth weights from gestational age at birth 24-41 weeks'),
        'standard_deviation_birth_weights': Parameter(
            Types.LIST, 'list of standard deviations associated with mean birth weights from gestational age at '
                        'birth 24-41 weeks'),

        # OBSTRUCTED LABOUR....
        'prob_obstruction_cpd': Parameter(
            Types.LIST, 'risk of a woman developing obstructed labour secondary to cephalopelvic disproportion'),
        'rr_obstruction_cpd_stunted_mother': Parameter(
            Types.LIST, 'relative risk of obstruction secondary to CPD in mothers who are stunted'),
        'rr_obstruction_foetal_macrosomia': Parameter(
            Types.LIST, 'relative risk of obstruction secondary to CPD in mothers who are carrying a macrosomic '
                        'foetus'),
        'prob_obstruction_malpos_malpres': Parameter(
            Types.LIST, 'risk of a woman developing obstructed labour secondary to malposition or malpresentation'),
        'prob_obstruction_other': Parameter(
            Types.LIST, 'risk of a woman developing obstructed labour secondary to other causes'),

        # ANTEPARTUM HAEMORRHAGE...
        'prob_placental_abruption_during_labour': Parameter(
            Types.LIST, 'probability of a woman developing placental abruption during labour'),
        'prob_aph_placenta_praevia_labour': Parameter(
            Types.LIST, 'probability of a woman with placenta praevia experiencing an APH during labour'),
        'prob_aph_placental_abruption_labour': Parameter(
            Types.LIST, 'probability of a woman with placental abruption experiencing an APH during labour'),
        'rr_placental_abruption_hypertension': Parameter(
            Types.LIST, 'Relative risk of placental abruption in women with hypertension'),
        'rr_placental_abruption_previous_cs': Parameter(
            Types.LIST, 'Relative risk of placental abruption in women who delivered previously via caesarean section'),
        'severity_maternal_haemorrhage': Parameter(
            Types.LIST, 'probability a maternal hemorrhage is non-severe (<1000mls) or severe (>1000mls)'),
        'cfr_aph': Parameter(
            Types.LIST, 'case fatality rate for antepartum haemorrhage during labour'),

        # MATERNAL INFECTION...
        'prob_sepsis_chorioamnionitis': Parameter(
            Types.LIST, 'risk of sepsis following chorioamnionitis infection'),
        'rr_sepsis_chorio_prom': Parameter(
            Types.LIST, 'relative risk of chorioamnionitis following PROM'),
        'prob_sepsis_endometritis': Parameter(
            Types.LIST, 'risk of sepsis following endometritis'),
        'rr_sepsis_endometritis_post_cs': Parameter(
            Types.LIST, 'relative risk of endometritis following caesarean delivery'),
        'prob_sepsis_urinary_tract': Parameter(
            Types.LIST, 'risk of sepsis following urinary tract infection'),
        'prob_sepsis_skin_soft_tissue': Parameter(
            Types.LIST, 'risk of sepsis following skin or soft tissue infection'),
        'rr_sepsis_sst_post_cs': Parameter(
            Types.LIST, 'relative risk of skin/soft tissue sepsis following caesarean delivery'),
        'cfr_sepsis': Parameter(
            Types.LIST, 'case fatality rate for sepsis during labour'),
        'cfr_pp_sepsis': Parameter(
            Types.LIST, 'case fatality rate for sepsis following delivery'),

        # UTERINE RUPTURE...
        'prob_uterine_rupture': Parameter(
            Types.LIST, 'probability of a uterine rupture during labour'),
        'rr_ur_parity_2': Parameter(
            Types.LIST, 'relative risk of uterine rupture in women who have delivered 2 times previously'),
        'rr_ur_parity_3_or_4': Parameter(
            Types.LIST, 'relative risk of uterine rupture in women who have delivered 3-4 times previously'),
        'rr_ur_parity_5+': Parameter(
            Types.LIST, 'relative risk of uterine rupture in women who have delivered > 5 times previously'),
        'rr_ur_prev_cs': Parameter(
            Types.LIST, 'relative risk of uterine rupture in women who have previously delivered via caesarean '
                        'section'),
        'rr_ur_obstructed_labour': Parameter(
            Types.LIST,
            'relative risk of uterine rupture in women who have been in obstructed labour'),
        'cfr_uterine_rupture': Parameter(
            Types.LIST, 'case fatality rate for uterine rupture in labour'),

        # HYPERTENSIVE DISORDERS...
        'prob_progression_gest_htn': Parameter(
            Types.LIST, 'probability of gestational hypertension progressing to severe gestational hypertension'
                        'during/after labour'),
        'prob_progression_severe_gest_htn': Parameter(
            Types.LIST, 'probability of severe gestational hypertension progressing to severe pre-eclampsia '
                        'during/after labour'),
        'prob_progression_mild_pre_eclamp': Parameter(
            Types.LIST, 'probability of mild pre-eclampsia progressing to severe pre-eclampsia during/after labour'),
        'prob_progression_severe_pre_eclamp': Parameter(
            Types.LIST, 'probability of severe pre-eclampsia progressing to eclampsia during/after labour'),
        'cfr_eclampsia': Parameter(
            Types.LIST, 'case fatality rate for eclampsia during labours'),
        'cfr_severe_pre_eclamp': Parameter(
            Types.LIST, 'case fatality rate for severe pre eclampsia during labour'),
        'cfr_pp_eclampsia': Parameter(
            Types.LIST, 'case fatality rate for eclampsia following delivery'),

        # INTRAPARTUM STILLBIRTH...
        'prob_ip_still_birth': Parameter(
            Types.LIST, 'baseline probability of intrapartum still birth'),
        'rr_still_birth_maternal_death': Parameter(
            Types.LIST, 'relative risk of still birth in mothers who have died during labour'),
        'rr_still_birth_ur': Parameter(
            Types.LIST, 'relative risk of still birth in mothers experiencing uterine rupture'),
        'rr_still_birth_ol': Parameter(
            Types.LIST, 'relative risk of still birth in mothers experiencing obstructed labour'),
        'rr_still_birth_aph': Parameter(
            Types.LIST, 'relative risk of still birth in mothers experiencing antepartum haemorrhage'),
        'rr_still_birth_hypertension': Parameter(
            Types.LIST, 'relative risk of still birth in mothers experiencing hypertension'),
        'rr_still_birth_sepsis': Parameter(
            Types.LIST, 'relative risk of still birth in mothers experiencing intrapartum sepsis'),
        'rr_still_birth_multiple_pregnancy': Parameter(
            Types.LIST, 'relative risk of still birth in mothers pregnant with twins'),
        'rr_still_birth_preterm_labour': Parameter(
            Types.LIST, 'relative risk of still birth in mothers in preterm labour'),
        'prob_both_twins_ip_still_birth': Parameter(
            Types.LIST, 'probability that if this mother will experience still birth, and she is pregnant with twins, '
                        'that neither baby will survive'),

        # todo: shouldnt other RFs from antenatal SB also have effect here

        # POSTPARTUM HAEMORRHAGE...
        'prob_pph_uterine_atony': Parameter(
            Types.LIST, 'risk of pph after experiencing uterine atony'),
        'rr_pph_ua_hypertension': Parameter(
            Types.LIST, 'relative risk risk of pph after secondary to uterine atony in hypertensive women'),
        'rr_pph_ua_multiple_pregnancy': Parameter(
            Types.LIST, 'relative risk risk of pph after secondary to uterine atony in women pregnant with twins'),
        'rr_pph_ua_placental_abruption': Parameter(
            Types.LIST, 'relative risk risk of pph after secondary to uterine atony in women with placental abruption'),
        'rr_pph_ua_macrosomia': Parameter(
            Types.LIST, 'relative risk risk of pph after secondary to uterine atony in women with macrosomic foetus'),
        'rr_pph_ua_diabetes': Parameter(
             Types.LIST, 'risk of pph after experiencing uterine atony'), # TODO: double counting with macrosomia?
        'prob_pph_retained_placenta': Parameter(
            Types.LIST, 'risk of pph after experiencing retained placenta'),
        'prob_pph_other_causes': Parameter(
            Types.LIST, 'risk of pph after experiencing other pph causes'),
        'cfr_pp_pph': Parameter(
            Types.LIST, 'case fatality rate for postpartum haemorrhage'),
        'rr_pph_death_anaemia': Parameter(
            Types.LIST, 'relative risk increase of death in women who are anaemic at time of PPH'),


        # CARE SEEKING FOR HEALTH CENTRE DELIVERY...
        'odds_deliver_in_health_centre': Parameter(
            Types.LIST, 'odds of a woman delivering in a health centre compared to a hospital'),
        'rrr_hc_delivery_age_20_24': Parameter(
            Types.LIST, 'relative risk ratio for a woman aged 20-24 delivering in a health centre compared to a '
                        'hospital'),
        'rrr_hc_delivery_age_25_29': Parameter(
            Types.LIST, 'relative risk ratio for a woman aged 25-29 delivering in a health centre compared to a '
                        'hospital'),
        'rrr_hc_delivery_age_30_34': Parameter(
            Types.LIST, 'relative risk ratio for a woman aged 30-34 delivering in a health centre compared to a '
                        'hospital'),
        'rrr_hc_delivery_age_35_39': Parameter(
            Types.LIST, 'relative risk ratio for a woman aged 35-39 delivering in a health centre compared to a '
                        'hospital'),
        'rrr_hc_delivery_age_40_44': Parameter(
            Types.LIST, 'relative risk ratio for a woman aged 40-44 delivering in a health centre compared to a '
                        'hospital'),
        'rrr_hc_delivery_age_45_49': Parameter(
            Types.LIST, 'relative risk ratio for a woman aged 45-49 delivering in a health centre compared to a '
                        'hospital'),
        'rrr_hc_delivery_tertiary_education': Parameter(
            Types.LIST, ''),
        'rrr_hc_delivery_wealth_4': Parameter(
            Types.LIST, 'relative risk ratio of a woman at wealth level 4 delivering at health centre compared to a '
                        'hospital'),
        'rrr_hc_delivery_wealth_3': Parameter(
            Types.LIST, 'relative risk ratio of a woman at wealth level 3 delivering at health centre compared to a '
                        'hospital'),
        'rrr_hc_delivery_wealth_2': Parameter(
            Types.LIST, 'relative risk ratio of a woman at wealth level 2 delivering at health centre compared to a '
                        'hospital'),
        'rrr_hc_delivery_wealth_1': Parameter(
            Types.LIST, 'relative risk ratio of a woman at wealth level 1 delivering at health centre compared to a '
                        'hospital'),
        'rrr_hc_delivery_parity_3_to_4': Parameter(
            Types.LIST, 'relative risk ratio for a woman with a parity of 3-4 delivering in a health centre compared to'
                        'a hospital'),
        'rrr_hc_delivery_parity_>4': Parameter(
            Types.LIST, 'relative risk ratio of a woman with a parity >4 delivering in health centre compared to a '
                        'hospital'),
        'rrr_hc_delivery_rural': Parameter(
            Types.LIST, 'relative risk ratio of a married woman delivering in a health centre compared to a hospital'),
        'rrr_hc_delivery_married': Parameter(
            Types.LIST, 'relative risk ratio of a married woman delivering in a health centre compared to a hospital'),

        # CARE SEEKING FOR HOME BIRTH...
        'odds_deliver_at_home': Parameter(
            Types.LIST, 'odds of a woman delivering at home compared to a hospital'),
        'rrr_hb_delivery_age_20_24': Parameter(
            Types.LIST, 'relative risk ratio for a woman aged 20-24 delivering at home compared to a '
                        'hospital'),
        'rrr_hb_delivery_age_25_29': Parameter(
            Types.LIST, 'relative risk ratio for a woman aged 25-29 delivering at home compared to a '
                        'hospital'),
        'rrr_hb_delivery_age_30_34': Parameter(
            Types.LIST, 'relative risk ratio for a woman aged 30-34 delivering at home compared to a '
                        'hospital'),
        'rrr_hb_delivery_age_35_39': Parameter(
            Types.LIST, 'relative risk ratio for a woman aged 35-39 delivering at home compared to a '
                        'hospital'),
        'rrr_hb_delivery_age_40_44': Parameter(
            Types.LIST, 'relative risk ratio for a woman aged 40-44 delivering at home compared to a hospital'),
        'rrr_hb_delivery_age_45_49': Parameter(
            Types.LIST, 'relative risk ratio for a woman aged 45-49 delivering at home compared to a hospital'),
        'rrr_hb_delivery_rural': Parameter(
            Types.LIST, 'relative risk ratio of a rural delivering at home compared to a hospital'),
        'rrr_hb_delivery_primary_education': Parameter(
            Types.LIST, 'relative risk ratio of a woman with a primary education delivering at home compared to a '
                        'hospital'),
        'rrr_hb_delivery_secondary_education': Parameter(
            Types.LIST, 'relative risk ratio of a woman with a secondary education delivering at home compared to a '
                        'hospital'),
        'rrr_hb_delivery_tertiary_education': Parameter(
            Types.LIST, 'relative risk ratio of a woman with tertiary delivering at home compared to a '
                        'hospital'),
        'rrr_hb_delivery_wealth_4': Parameter(
            Types.LIST, 'relative risk ratio of a woman at wealth level 4 delivering at home compared to a '
                        'hospital'),
        'rrr_hb_delivery_wealth_3': Parameter(
            Types.LIST, 'relative risk ratio of a woman at wealth level 3 delivering at home compared to a '
                        'hospital'),
        'rrr_hb_delivery_wealth_2': Parameter(
            Types.LIST, 'relative risk ratio of a woman at wealth level 2 delivering at home compared to a '
                        'hospital'),
        'rrr_hb_delivery_wealth_1': Parameter(
            Types.LIST, 'relative risk ratio of a woman at wealth level 1 delivering at home compared to a '
                        'hospital'),
        'rrr_hb_delivery_parity_3_to_4': Parameter(
            Types.LIST, 'relative risk ratio for a woman with a parity of 3-4 delivering at home compared to'
                        'a hospital'),
        'rrr_hb_delivery_parity_>4': Parameter(
            Types.LIST, 'relative risk ratio of a woman with a parity >4 delivering at home compared to a '
                        'hospital'),
        'rrr_hb_delivery_married': Parameter(
            Types.LIST, 'relative risk ratio of a married woman delivering in a home compared to a hospital'),

        # EMERGENCY CARE SEEKING...
        'prob_careseeking_for_complication': Parameter(
            Types.LIST, 'odds of a woman seeking skilled assistance after developing a complication at a home birth'),
        'test_care_seeking_probs': Parameter(
            Types.LIST, 'dummy probabilities of delivery care seeking used in testing'),

        # TREATMENT PARAMETERS...
        'treatment_effect_maternal_infection_clean_delivery': Parameter(
            Types.LIST, 'Effect of clean delivery practices on risk of maternal infection'),
        'treatment_effect_amtsl': Parameter(
            Types.LIST, 'relative risk of severe postpartum haemorrhage following active management of the third '
                        'stage of labour'),
        'prob_haemostatis_uterotonics': Parameter(
            Types.LIST, 'probability of uterotonics stopping a postpartum haemorrhage due to uterine atony '),
        'pph_treatment_effect_uterotonics_md': Parameter(
            Types.LIST, 'effect of uterotonics on maternal death due to postpartum haemorrhage'),
        'prob_successful_manual_removal_placenta': Parameter(
            Types.LIST, 'probability of manual removal of retained products arresting a post partum haemorrhage'),
        'pph_treatment_effect_mrp_md': Parameter(
            Types.LIST, 'effect of uterotonics on maternal death due to postpartum haemorrhage'),
        'success_rate_pph_surgery': Parameter(
            Types.LIST, 'probability of surgery for postpartum haemorrhage being successful'),
        'pph_treatment_effect_surg_md': Parameter(
            Types.LIST, 'effect of surgery on maternal death due to postpartum haemorrhage'),
        'pph_treatment_effect_hyst_md': Parameter(
            Types.LIST, 'effect of hysterectomy on maternal death due to postpartum haemorrhage'),
        'pph_bt_treatment_effect_md': Parameter(
            Types.LIST, 'effect of blood transfusion treatment for postpartum haemorrhage on risk of maternal death'),
        'sepsis_treatment_effect_md': Parameter(
            Types.LIST, 'effect of treatment for sepsis on risk of maternal death'),
        'success_rate_uterine_repair': Parameter(
            Types.LIST, 'probability repairing a ruptured uterus surgically'),
        'prob_successful_assisted_vaginal_delivery': Parameter(
            Types.LIST, 'probability of successful assisted vaginal delivery'),
        'ur_repair_treatment_effect_md': Parameter(
            Types.LIST, 'effect of surgical uterine repair treatment for uterine rupture on risk of maternal death'),
        'ur_treatment_effect_bt_md': Parameter(
            Types.LIST, 'effect of blood transfusion treatment for uterine rupture on risk of maternal death'),
        'ur_hysterectomy_treatment_effect_md': Parameter(
            Types.LIST, 'effect of blood hysterectomy for uterine rupture on risk of maternal death'),
        'eclampsia_treatment_effect_severe_pe': Parameter(
            Types.LIST, 'effect of treatment for severe pre eclampsia on risk of eclampsia'),
        'eclampsia_treatment_effect_md': Parameter(
            Types.LIST, 'effect of treatment for eclampsia on risk of maternal death'),
        'anti_htns_treatment_effect_md': Parameter(
            Types.LIST, 'effect of IV anti hypertensive treatment on risk of death secondary to severe pre-eclampsia/'
                        'eclampsia stillbirth'),
        'anti_htns_treatment_effect_progression': Parameter(
            Types.LIST,
            'effect of IV anti hypertensive treatment on risk of progression from mild to severe gestational'
            ' hypertension'),
        'aph_bt_treatment_effect_md': Parameter(
            Types.LIST, 'effect of blood transfusion treatment for antepartum haemorrhage on risk of maternal death'),
        'aph_cs_treatment_effect_md': Parameter(
            Types.LIST, 'effect of caesarean section for antepartum haemorrhage on risk of maternal death'),
        'treatment_effect_avd_still_birth': Parameter(
            Types.LIST, 'effect of assisted vaginal delivery on risk of intrapartum still birth'),
        'treatment_effect_cs_still_birth': Parameter(
            Types.LIST, 'effect of caesarean section delivery on risk of intrapartum still birth'),

        # ASSESSMENT SENSITIVITIES...
        'sensitivity_of_assessment_of_obstructed_labour_hc': Parameter(
            Types.LIST, 'sensitivity of dx_test assessment by birth attendant for obstructed labour in a health '
                        'centre'),
        'sensitivity_of_assessment_of_obstructed_labour_hp': Parameter(
            Types.LIST, 'sensitivity of dx_test assessment by birth attendant for obstructed labour in a level 1 '
                        'hospital'),
        'sensitivity_of_assessment_of_sepsis_hc': Parameter(
            Types.LIST, 'sensitivity of dx_test assessment by birth attendant for maternal sepsis in a health centre'),
        'sensitivity_of_assessment_of_sepsis_hp': Parameter(
            Types.LIST, 'sensitivity of dx_test assessment by birth attendant for maternal sepsis in a level 1'
                        'hospital'),
        'sensitivity_of_assessment_of_severe_pe_hc': Parameter(
            Types.LIST, 'sensitivity of dx_test assessment by birth attendant for severe pre-eclampsia in a level 1 '
                        'hospital'),
        'sensitivity_of_assessment_of_severe_pe_hp': Parameter(
            Types.LIST, 'sensitivity of dx_test assessment by birth attendant for severe pre-eclampsia in a level 1 '
                        'hospital'),
        'sensitivity_of_assessment_of_hypertension_hc': Parameter(
            Types.LIST, 'sensitivity of dx_test assessment by birth attendant for hypertension in a level 1 health '
                        'centre'),
        'sensitivity_of_assessment_of_hypertension_hp': Parameter(
            Types.LIST, 'sensitivity of dx_test assessment by birth attendant for hypertension in a level 1 hospital'),
        'sensitivity_of_assessment_of_antepartum_haem_hc': Parameter(
            Types.LIST, 'sensitivity of dx_test assessment for referral by birth attendant for antepartum haemorrhage'
                        ' in a health centre'),
        'sensitivity_of_assessment_of_antepartum_haem_hp': Parameter(
            Types.LIST, 'sensitivity of dx_test assessment for treatment by birth attendant for antepartum haemorrhage'
                        ' in a level 1 hospital'),
        'sensitivity_of_assessment_of_uterine_rupture_hc': Parameter(
            Types.LIST, 'sensitivity of dx_test assessment by birth attendant for uterine rupture in a health centre'),
        'sensitivity_of_assessment_of_uterine_rupture_hp': Parameter(
            Types.LIST, 'sensitivity of dx_test assessment by birth attendant for uterine rupture in a level 1 '
                        'hospital'),
        'sensitivity_of_assessment_of_ec_hc': Parameter(
            Types.LIST, 'sensitivity of dx_test assessment by birth attendant for eclampsia in a level 1 hospital'),
        'sensitivity_of_assessment_of_ec_hp': Parameter(
            Types.LIST, 'sensitivity of dx_test assessment by birth attendant for eclampsia in a level 1 '
                        'hospital'),
        'sensitivity_of_assessment_of_pph_hc': Parameter(
            Types.LIST, 'sensitivity of dx_test assessment by birth attendant for postpartum haemorrhage in a level 1 '
                        'hospital'),
        'sensitivity_of_assessment_of_pph_hp': Parameter(
            Types.LIST, 'sensitivity of dx_test assessment by birth attendant for postpartum haemorrhage in a level 1 '
                        'hospital'),

        # SQUEEZE FACTOR THRESHOLDS...
        'squeeze_threshold_proph_ints': Parameter(
            Types.LIST, 'squeeze factor threshold below which prophylactic interventions for birth cant be given'),
        'squeeze_threshold_treatment_spe': Parameter(
            Types.LIST, 'squeeze factor threshold below which treatment for severe pre-eclampsia cant be given'),
        'squeeze_threshold_treatment_ol': Parameter(
            Types.LIST, 'squeeze factor threshold below which treatment for obstructed labour cant be given'),
        'squeeze_threshold_treatment_sep': Parameter(
            Types.LIST, 'squeeze factor threshold below which treatment for maternal sepsis cant be given'),
        'squeeze_threshold_treatment_htn': Parameter(
            Types.LIST, 'squeeze factor threshold below which treatment for hypertension cant be given'),
        'squeeze_threshold_treatment_ec': Parameter(
            Types.LIST, 'squeeze factor threshold below which treatment for eclampsia cant be given'),
        'squeeze_threshold_treatment_ur': Parameter(
            Types.LIST, 'squeeze factor threshold below which treatment for uterine rupture cant be given'),
        'squeeze_threshold_treatment_aph': Parameter(
            Types.LIST, 'squeeze factor threshold below which treatment for antepartum haemorrhage cant be given'),
        'squeeze_threshold_treatment_pph': Parameter(
            Types.LIST, 'squeeze factor threshold below which treatment for antepartum haemorrhage cant be given'),
        'squeeze_threshold_amtsl': Parameter(
            Types.LIST, 'squeeze factor threshold below which treatment for amtsl cant be given'),
    }

    PROPERTIES = {
        'la_due_date_current_pregnancy': Property(Types.DATE, 'The date on which a newly pregnant woman is scheduled to'
                                                              ' go into labour'),
        'la_currently_in_labour': Property(Types.BOOL, 'whether this woman is currently in labour'),
        'la_intrapartum_still_birth': Property(Types.BOOL, 'whether this womans most recent pregnancy has ended '
                                                           'in a stillbirth'),
        'la_parity': Property(Types.REAL, 'total number of previous deliveries'),
        'la_previous_cs_delivery': Property(Types.BOOL, 'whether this woman has ever delivered via caesarean section'),
        'la_has_previously_delivered_preterm': Property(Types.BOOL, 'whether the woman has had a previous preterm '
                                                                    'delivery for any of her previous deliveries'),
        'la_obstructed_labour': Property(Types.BOOL, 'Whether this woman is experiencing obstructed labour'),
        'la_placental_abruption': Property(Types.BOOL, 'whether the woman has experienced placental abruption'),
        'la_antepartum_haem': Property(Types.CATEGORICAL, 'whether the woman has experienced an antepartum haemorrhage'
                                                          ' in this delivery and it severity',
                                       categories=['none', 'mild_moderate', 'severe']),
        'la_antepartum_haem_treatment': Property(Types.BOOL, 'whether this womans antepartum haemorrhage has been '
                                                             'treated'),
        'la_uterine_rupture': Property(Types.BOOL, 'whether the woman has experienced uterine rupture in this '
                                                   'delivery'),
        'la_uterine_rupture_treatment': Property(Types.BOOL, 'whether this womans uterine rupture has been treated'),
        'la_sepsis': Property(Types.BOOL, 'whether this woman has developed sepsis due to an intrapartum infection'),
        'la_sepsis_pp': Property(Types.BOOL, 'whether this woman has developed sepsis due to a postpartum infection'),
        'la_sepsis_treatment': Property(Types.BOOL, 'If this woman has received treatment for maternal sepsis'),
        'la_eclampsia_treatment': Property(Types.BOOL, 'whether this womans eclampsia has been treated'),
        'la_severe_pre_eclampsia_treatment': Property(Types.BOOL, 'whether this woman has been treated for severe '
                                                                  'pre-eclampsia'),
        'la_maternal_hypertension_treatment': Property(Types.BOOL, 'whether this woman has been treated for maternal '
                                                                   'hypertension'),
        'la_postpartum_haem': Property(Types.BOOL, 'whether the woman has experienced an postpartum haemorrhage in this'
                                                   'delivery'),
        'la_postpartum_haem_cause': Property(Types.INT, 'bitset column holding causes of postpartum haemorrhage'),
        'la_postpartum_haem_treatment': Property(Types.INT, ' Treatment for received for postpartum haemorrhage '
                                                            '(bitset)'),
        'la_has_had_hysterectomy': Property(Types.BOOL, 'whether this woman has had a hysterectomy as treatment for a '
                                                        'complication of labour, and therefore is unable to conceive'),
        'la_maternal_death_in_labour': Property(Types.BOOL, ' whether the woman has died as a result of this '
                                                            'pregnancy'),
        'la_maternal_death_in_labour_date': Property(Types.DATE, 'date of death for a date in pregnancy'),
        'la_date_most_recent_delivery': Property(Types.DATE, 'date of on which this mother last delivered'),
        'la_is_postpartum': Property(Types.BOOL, 'Whether a woman is in the postpartum period, from delivery until '
                                                 'day +42 (6 weeks)'),
        'la_iron_folic_acid_postnatal': Property(Types.BOOL, 'Whether a woman is receiving iron and folic acid during '
                                                             'the postnatal period'),
    }

    def read_parameters(self, data_folder):
        parameter_dataframe = pd.read_excel(Path(self.resourcefilepath) / 'ResourceFile_LabourSkilledBirth'
                                                                          'Attendance.xlsx',
                                            sheet_name='parameter_values')
        self.load_parameters_from_dataframe(parameter_dataframe)

        # For the first period (2010-2015) we use the first value in each list as a parameter
        for key, value in self.parameters.items():
            self.current_parameters[key] = self.parameters[key][0]

    def initialise_population(self, population):
        df = population.props
        params = self.current_parameters

        df.loc[df.is_alive, 'la_currently_in_labour'] = False
        df.loc[df.is_alive, 'la_intrapartum_still_birth'] = False
        df.loc[df.is_alive, 'la_parity'] = 0
        df.loc[df.is_alive, 'la_previous_cs_delivery'] = False
        df.loc[df.is_alive, 'la_has_previously_delivered_preterm'] = False
        df.loc[df.is_alive, 'la_due_date_current_pregnancy'] = pd.NaT
        df.loc[df.is_alive, 'la_obstructed_labour'] = False
        df.loc[df.is_alive, 'la_placental_abruption'] = False
        df.loc[df.is_alive, 'la_antepartum_haem'] = 'none'
        df.loc[df.is_alive, 'la_antepartum_haem_treatment'] = False
        df.loc[df.is_alive, 'la_uterine_rupture'] = False
        df.loc[df.is_alive, 'la_uterine_rupture_treatment'] = False
        df.loc[df.is_alive, 'la_sepsis'] = False
        df.loc[df.is_alive, 'la_sepsis_pp'] = False
        df.loc[df.is_alive, 'la_sepsis_treatment'] = False
        df.loc[df.is_alive, 'la_eclampsia_treatment'] = False
        df.loc[df.is_alive, 'la_severe_pre_eclampsia_treatment'] = False
        df.loc[df.is_alive, 'la_maternal_hypertension_treatment'] = False
        df.loc[df.is_alive, 'la_postpartum_haem'] = False
        df.loc[df.is_alive, 'la_postpartum_haem_cause'] = 0
        df.loc[df.is_alive, 'la_postpartum_haem_treatment'] = 0
        df.loc[df.is_alive, 'la_has_had_hysterectomy'] = False
        df.loc[df.is_alive, 'la_maternal_death_in_labour'] = False
        df.loc[df.is_alive, 'la_maternal_death_in_labour_date'] = pd.NaT
        df.loc[df.is_alive, 'la_date_most_recent_delivery'] = pd.NaT
        df.loc[df.is_alive, 'la_is_postpartum'] = False

        #  we store different potential treatments for postpartum haemorrhage via bistet
        self.pph_treatment = BitsetHandler(self.sim.population, 'la_postpartum_haem_treatment',
                                           ['uterotonics', 'manual_removal_placenta', 'surgery', 'hysterectomy'])

        #  ----------------------------ASSIGNING PARITY AT BASELINE --------------------------------------------------
        # This equation predicts the parity of each woman at baseline (who is of reproductive age)
        parity_equation = LinearModel.custom(labour_lm.predict_parity, parameters=params)

        # We assign parity to all women of reproductive age at baseline
        df.loc[df.is_alive & (df.sex == 'F') & (df.age_years > 14), 'la_parity'] = np.ceil(
            parity_equation.predict(df.loc[df.is_alive & (df.sex == 'F') & (df.age_years > 14)])
        )

        assert (df.loc[df.is_alive & (df.sex == 'F') & (df.age_years > 14), 'la_parity'] >= 0).all().all()

    def initialise_simulation(self, sim):

        # We set the LoggingEvent to run a the last day of each year to produce statistics for that year
        sim.schedule_event(LabourLoggingEvent(self), sim.date + DateOffset(years=1))

        # This list contains all the women who are currently in labour and is used for checks/testing
        self.women_in_labour = []

        # This dictionary is the complication tracker used by the logger to output incidence of complications/outcomes
        self.labour_tracker = {'ip_stillbirth': 0, 'maternal_death': 0, 'obstructed_labour': 0,
                               'antepartum_haem': 0, 'antepartum_haem_death': 0, 'sepsis': 0, 'sepsis_death': 0,
                               'eclampsia': 0, 'severe_pre_eclampsia': 0, 'severe_pre_eclamp_death': 0,
                               'eclampsia_death': 0, 'uterine_rupture': 0, 'uterine_rupture_death': 0,
                               'postpartum_haem': 0, 'postpartum_haem_death': 0,
                               'sepsis_pp': 0, 'home_birth': 0, 'health_centre_birth': 0,
                               'hospital_birth': 0, 'caesarean_section': 0, 'early_preterm': 0,
                               'late_preterm': 0, 'post_term': 0, 'term': 0}

        # This list contains all possible complications/outcomes of the intrapartum and postpartum phase- its used in
        # assert functions as a test
        self.possible_intrapartum_complications = ['obstruction_cpd', 'obstruction_malpos_malpres', 'obstruction_other',
                                                   'placental_abruption', 'antepartum_haem', 'sepsis',
                                                   'sepsis_chorioamnionitis', 'uterine_rupture',  'severe_pre_eclamp',
                                                   'eclampsia']

        self.possible_postpartum_complications = ['sepsis_endometritis', 'sepsis_skin_soft_tissue',
                                                  'sepsis_urinary_tract', 'pph_uterine_atony', 'pph_retained_placenta',
                                                  'pph_other', 'severe_pre_eclamp', 'eclampsia', 'postpartum_haem',
                                                  'sepsis']

        # =======================Register dx_tests for complications during labour/postpartum=======================
        # We register all the dx_tests needed within the labour HSI events. For simplicity we use the dx_test here to
        # represent clinical assessment of a woman for a specific complication. If a dx_test assessment returns True a
        # woman will be given treatment (provided their are available consumables and the squeeze is not too high)

        # We vary the 'sensitivity' of these assessment between health centres and hospitals to allow for future
        # calibration to SARA survey data which varies by facility type ...(hp = hospital, hc= health centre)

        p = self.current_parameters
        self.sim.modules['HealthSystem'].dx_manager.register_dx_test(

            # Obstructed Labour diagnosis
            assess_obstructed_labour_hc=DxTest(
                property='la_obstructed_labour',
                sensitivity=p['sensitivity_of_assessment_of_obstructed_labour_hc']),
            assess_obstructed_labour_hp=DxTest(
                property='la_obstructed_labour',
                sensitivity=p['sensitivity_of_assessment_of_obstructed_labour_hp']),

            # Sepsis diagnosis intrapartum...
            # dx_tests for intrapartum and postpartum sepsis only differ in the 'property' variable
            assess_sepsis_hc_ip=DxTest(
                property='la_sepsis',
                sensitivity=p['sensitivity_of_assessment_of_sepsis_hc']),
            assess_sepsis_hp_ip=DxTest(
                property='la_sepsis',
                sensitivity=p['sensitivity_of_assessment_of_sepsis_hp']),

            # Sepsis diagnosis postpartum
            assess_sepsis_hc_pp=DxTest(
                property='la_sepsis_pp',
                sensitivity=p['sensitivity_of_assessment_of_sepsis_hc']),
            assess_sepsis_hp_pp=DxTest(
                property='la_sepsis_pp',
                sensitivity=p['sensitivity_of_assessment_of_sepsis_hp']),

            # Hypertension diagnosis
            assess_hypertension_hc_ip=DxTest(
                property='ps_htn_disorders', target_categories=['gest_htn', 'severe_gest_htn',
                                                                'mild_pre_eclamp', 'severe_pre_eclamp', 'eclampsia'],
                sensitivity=p['sensitivity_of_assessment_of_hypertension_hc']),
            assess_hypertension_hp_ip=DxTest(
                property='ps_htn_disorders', target_categories=['gest_htn', 'severe_gest_htn',
                                                                'mild_pre_eclamp', 'severe_pre_eclamp', 'eclampsia'],
                sensitivity=p['sensitivity_of_assessment_of_hypertension_hp']),

            assess_hypertension_hc_pp=DxTest(
                property='pn_htn_disorders', target_categories=['gest_htn', 'severe_gest_htn',
                                                                'mild_pre_eclamp', 'severe_pre_eclamp', 'eclampsia'],
                sensitivity=p['sensitivity_of_assessment_of_hypertension_hc']),
            assess_hypertension_hp_pp=DxTest(
                property='pn_htn_disorders', target_categories=['gest_htn', 'severe_gest_htn',
                                                                'mild_pre_eclamp', 'severe_pre_eclamp', 'eclampsia'],
                sensitivity=p['sensitivity_of_assessment_of_hypertension_hp']),

            # severe pre-eclampsia diagnosis
            assess_severe_pe_hc_ip=DxTest(
                property='ps_htn_disorders', target_categories=['severe_pre_eclamp'],
                sensitivity=p['sensitivity_of_assessment_of_severe_pe_hc']),
            assess_severe_pe_hp_ip=DxTest(
                property='ps_htn_disorders', target_categories=['severe_pre_eclamp'],
                sensitivity=p['sensitivity_of_assessment_of_severe_pe_hp']),

            assess_severe_pe_hc_pp=DxTest(
                property='pn_htn_disorders', target_categories=['severe_pre_eclamp'],
                sensitivity=p['sensitivity_of_assessment_of_severe_pe_hc']),
            assess_severe_pe_hp_pp=DxTest(
                property='pn_htn_disorders', target_categories=['severe_pre_eclamp'],
                sensitivity=p['sensitivity_of_assessment_of_severe_pe_hp']),


            # Eclampsia diagnosis
            assess_eclampsia_hc_ip=DxTest(
                property='ps_htn_disorders', target_categories=['eclampsia'],
                sensitivity=p['sensitivity_of_assessment_of_ec_hc']),
            assess_eclampsia_hp_ip=DxTest(
                property='ps_htn_disorders', target_categories=['eclampsia'],
                sensitivity=p['sensitivity_of_assessment_of_ec_hp']),

            assess_eclampsia_hc_pp=DxTest(
                property='pn_htn_disorders', target_categories=['eclampsia'],
                sensitivity=p['sensitivity_of_assessment_of_ec_hc']),
            assess_eclampsia_hp_pp=DxTest(
                property='pn_htn_disorders', target_categories=['eclampsia'],
                sensitivity=p['sensitivity_of_assessment_of_ec_hp']),

            # Antepartum Haemorrhage
            assess_aph_hc=DxTest(
                property='la_antepartum_haem', target_categories=['mild_moderate', 'severe'],
                sensitivity=p['sensitivity_of_assessment_of_antepartum_haem_hc']),
            assess_aph_hp=DxTest(
                property='la_antepartum_haem', target_categories=['mild_moderate', 'severe'],
                sensitivity=p['sensitivity_of_assessment_of_antepartum_haem_hc']),

            # Uterine Rupture
            assess_uterine_rupture_hc=DxTest(
                property='la_uterine_rupture',
                sensitivity=p['sensitivity_of_assessment_of_uterine_rupture_hc']),
            assess_uterine_rupture_hp=DxTest(
                property='la_uterine_rupture',
                sensitivity=p['sensitivity_of_assessment_of_uterine_rupture_hp']),

            # Postpartum haemorrhage
            assess_pph_hc=DxTest(
                property='la_postpartum_haem',
                sensitivity=p['sensitivity_of_assessment_of_pph_hc']),
            assess_pph_hp=DxTest(
                property='la_postpartum_haem',
                sensitivity=p['sensitivity_of_assessment_of_pph_hp']),
        )

        # ======================================= LINEAR MODEL EQUATIONS ==============================================
        # Here we define the equations that will be used throughout this module using the linear
        # model and stored them as a parameter

        params = self.current_parameters
        self.la_linear_models = {

            # This equation predicts the parity of each woman at baseline (who is of reproductive age)
            'parity': LinearModel.custom(labour_lm.predict_parity, parameters=params),

            # This equation is used to calculate a womans risk of obstructed labour. As we assume obstructed labour can
            # only occur following on of three preceding causes, this model is additive
            'obstruction_cpd_ip': LinearModel.custom(labour_lm.predict_obstruction_cpd_ip, parameters=params),

            # This equation is used to calculate a womans risk of developing sepsis due to chorioamnionitis infection
            # during the intrapartum phase of labour and is mitigated by clean delivery
            'sepsis_chorioamnionitis_ip': LinearModel.custom(labour_lm.predict_sepsis_chorioamnionitis_ip,
                                                             parameters=params),

            # This equation is used to calculate a womans risk of developing sepsis due to endometritis infection
            # during the postpartum phase of labour and is mitigated by clean delivery
            'sepsis_endometritis_pp': LinearModel.custom(labour_lm.predict_sepsis_endometritis_pp, parameters=params),

            # This equation is used to calculate a womans risk of developing sepsis due to skin or soft tissue
            # infection during the
            # postpartum phase of labour and is mitigated by clean delivery
            'sepsis_skin_soft_tissue_pp': LinearModel.custom(labour_lm.predict_sepsis_skin_soft_tissue_pp,
                                                             parameters=params),

            # This equation is used to calculate a womans risk of developing a urinary tract infection during the
            # postpartum phase of labour and is mitigated by clean delivery
            'sepsis_urinary_tract_pp': LinearModel.custom(labour_lm.predict_sepsis_urinary_tract_pp,
                                                          parameters=params),

            # This equation is used to calculate a womans risk of death following sepsis during labour and is mitigated
            # by treatment
            'sepsis_death': LinearModel.custom(labour_lm.predict_sepsis_death, parameters=params),

            # This equation is used to calculate a womans risk of death following postpartum sepsis and is mitigated
            # by treatment
            'sepsis_pp_death': LinearModel.custom(labour_lm.predict_sepsis_pp_death, parameters=params),

            # This equation is used to calculate a womans risk of death following eclampsia and is mitigated
            # by treatment delivered either immediately prior to admission for delivery or during labour
            'eclampsia_death': LinearModel.custom(labour_lm.predict_eclampsia_death, parameters=params),

            # This equation is used to calculate a womans risk of death following eclampsia and is mitigated
            # by treatment delivered either immediately prior to admission for delivery or during labour
            'eclampsia_pp_death': LinearModel.custom(labour_lm.predict_eclampsia_pp_death, parameters=params),

            # This equation is used to calculate a womans risk of death following eclampsia and is mitigated
            # by treatment delivered either immediately prior to admission for delivery or during labour
            'severe_pre_eclamp_death': LinearModel.custom(labour_lm.predict_severe_pre_eclamp_death, parameters=params),
            'severe_pre_eclamp_pp_death': LinearModel.custom(labour_lm.predict_severe_pre_eclamp_death,
                                                             parameters=params),

            # This equation is used to calculate a womans risk of placental abruption in labour
            'placental_abruption_ip': LinearModel.custom(labour_lm.predict_placental_abruption_ip, parameters=params),

            # This equation is used to calculate a womans risk of antepartum haemorrhage. We assume APH can only occur
            # in the presence of a preceding placental causes (abruption/praevia) therefore this model is additive
            'antepartum_haem_ip': LinearModel.custom(labour_lm.predict_antepartum_haem_ip, parameters=params),

            # This equation is used to calculate a womans risk of death following antepartum haemorrhage. Risk is
            # mitigated by treatment
            'antepartum_haem_death': LinearModel.custom(labour_lm.predict_antepartum_haem_death, parameters=params),

            # This equation is used to calculate a womans risk of postpartum haemorrhage due to uterine atony
            'pph_uterine_atony_pp': LinearModel.custom(labour_lm.predict_pph_uterine_atony_pp, parameters=params),

            # This equation is used to calculate a womans risk of postpartum haemorrhage due to retained placenta
            'pph_retained_placenta_pp': LinearModel.custom(labour_lm.predict_pph_retained_placenta_pp,
                                                           parameters=params),

            # This equation is used to calculate a womans risk of death following postpartum haemorrhage. Risk is
            # mitigated by treatment
            'postpartum_haem_pp_death': LinearModel.custom(labour_lm.predict_postpartum_haem_pp_death, module=self),

            # This equation is used to calculate a womans risk of uterine rupture
            'uterine_rupture_ip': LinearModel.custom(labour_lm.predict_uterine_rupture_ip, parameters=params),

            # This equation is used to calculate a womans risk of death following uterine rupture. Risk if reduced by
            # treatment
            'uterine_rupture_death': LinearModel.custom(labour_lm.predict_uterine_rupture_death, parameters=params),

            # This equation is used to calculate a womans risk of still birth during the intrapartum period. Assisted
            # vaginal delivery and caesarean delivery are assumed to significantly reduce risk
            'intrapartum_still_birth': LinearModel.custom(labour_lm.predict_intrapartum_still_birth,
                                                          parameters=params),

            # This regression equation uses data from the DHS to predict a womans probability of choosing to deliver in
            # a health centre
            'probability_delivery_health_centre': LinearModel.custom(
                labour_lm.predict_probability_delivery_health_centre, parameters=params),

            # This regression equation uses data from the DHS to predict a womans probability of choosing to deliver in
            # at home
            'probability_delivery_at_home': LinearModel.custom(
                labour_lm.predict_probability_delivery_at_home, parameters=params),

            # This equation calculates a womans probability of seeking care following a complication during labour or
            # immediately after birth
            'care_seeking_for_complication': LinearModel.custom(
                labour_lm.predict_care_seeking_for_complication, parameters=params),
        }

    def on_birth(self, mother_id, child_id):
        df = self.sim.population.props

        df.at[child_id, 'la_due_date_current_pregnancy'] = pd.NaT
        df.at[child_id, 'la_currently_in_labour'] = False
        df.at[child_id, 'la_intrapartum_still_birth'] = False
        df.at[child_id, 'la_parity'] = 0
        df.at[child_id, 'la_previous_cs_delivery'] = False
        df.at[child_id, 'la_has_previously_delivered_preterm'] = False
        df.at[child_id, 'la_obstructed_labour'] = False
        df.at[child_id, 'la_placental_abruption'] = False
        df.at[child_id, 'la_antepartum_haem'] = 'none'
        df.at[child_id, 'la_antepartum_haem_treatment'] = False
        df.at[child_id, 'la_uterine_rupture'] = False
        df.at[child_id, 'la_uterine_rupture_treatment'] = False
        df.at[child_id, 'la_sepsis'] = False
        df.at[child_id, 'la_sepsis_pp'] = False
        df.at[child_id, 'la_sepsis_treatment'] = False
        df.at[child_id, 'la_eclampsia_treatment'] = False
        df.at[child_id, 'la_severe_pre_eclampsia_treatment'] = False
        df.at[child_id, 'la_maternal_hypertension_treatment'] = False
        df.at[child_id, 'la_postpartum_haem'] = False
        df.at[child_id, 'la_postpartum_haem_cause'] = 0
        df.at[child_id, 'la_postpartum_haem_treatment'] = 0
        df.at[child_id, 'la_has_had_hysterectomy'] = False
        df.at[child_id, 'la_maternal_death_in_labour'] = False
        df.at[child_id, 'la_maternal_death_in_labour_date'] = pd.NaT
        df.at[child_id, 'la_date_most_recent_delivery'] = pd.NaT
        df.at[child_id, 'la_is_postpartum'] = False

    def further_on_birth_labour(self, mother_id, child_id):
        """
        This function is called by the on_birth function of NewbornOutcomes module. This function contains additional
        code related to the labour module that should be ran on_birth for all births - it has been
        parcelled into functions to ensure each modules (pregnancy,antenatal care, labour, newborn, postnatal) on_birth
        code is ran in the correct sequence (as this can vary depending on how modules are registered)
        :param mother_id: mothers individual id
        """
        df = self.sim.population.props
        mother = df.loc[mother_id]

        # If a mothers labour has resulted in an intrapartum still birth her child is still generated by the simulation
        # but the death is recorded through the InstantaneousDeath function

        # Store only live births to a mother parity
        if ~mother.la_intrapartum_still_birth:
            df.at[mother_id, 'la_parity'] += 1  # Only live births contribute to parity
            logger.info(key='live_birth', data={'mother': mother_id, 'child': child_id})

        if mother.la_intrapartum_still_birth:
            self.sim.modules['Demography'].do_death(individual_id=child_id, cause='intrapartum stillbirth',
                                                    originating_module=self.sim.modules['Labour'])

        # We use this variable in the postnatal supervisor module to track postpartum women
        df.at[mother_id, 'la_is_postpartum'] = True
        df.at[mother_id, 'la_date_most_recent_delivery'] = self.sim.date

    def on_hsi_alert(self, person_id, treatment_id):
        """ This is called whenever there is an HSI event commissioned by one of the other disease modules."""
        logger.debug(key='message', data=f'This is Labour, being alerted about a health system interaction '
                                         f'person {person_id}for: {treatment_id}')

    def report_daly_values(self):
        logger.debug(key='message', data='This is Labour reporting my health values')
        df = self.sim.population.props  # shortcut to population properties data frame

        daly_series = pd.Series(data=0, index=df.index[df.is_alive])

        return daly_series

    # ===================================== HELPER AND TESTING FUNCTIONS ==============================================
    def set_date_of_labour(self, individual_id):
        """
        This function is called by contraception.py within the events 'PregnancyPoll' and 'Fail' for women who are
        allocated to become pregnant during a simulation run. This function schedules the onset of labour between 37
        and 44 weeks gestational age (not foetal age) to ensure all women who become pregnant will go into labour.
        Women may enter labour before the due date set in this function either due to pre-term labour or induction/
        caesarean section.
        :param individual_id: individual_id
        """
        df = self.sim.population.props
        params = self.current_parameters
        logger.debug(key='message', data=f'person {individual_id} is having their labour scheduled on date '
                                         f'{self.sim.date}', )

        # Check only alive newly pregnant women are scheduled to this function
        assert df.at[individual_id, 'is_alive'] and df.at[individual_id, 'is_pregnant']
        assert df.at[individual_id, 'date_of_last_pregnancy'] == self.sim.date

        # At the point of conception we schedule labour to onset for all women between after 37 weeks gestation - first
        # we determine if she will go into labour post term (41+ weeks)
        if self.rng.random_sample() < params['risk_post_term_labour']:
            df.at[individual_id, 'la_due_date_current_pregnancy'] = \
                (df.at[individual_id, 'date_of_last_pregnancy'] + pd.DateOffset(
                    days=(7 * 39) + self.rng.randint(0, 7 * 4)))

        else:
            df.at[individual_id, 'la_due_date_current_pregnancy'] = \
                (df.at[individual_id, 'date_of_last_pregnancy'] + pd.DateOffset(
                    days=(7 * 35) + self.rng.randint(0, 7 * 4)))

        self.sim.schedule_event(LabourOnsetEvent(self, individual_id),
                                df.at[individual_id, 'la_due_date_current_pregnancy'])

        # Here we check that no one is scheduled to go into labour before 37 gestational age (35 weeks foetal age,
        # ensuring all preterm labour comes from the pregnancy supervisor module
        days_until_labour = df.at[individual_id, 'la_due_date_current_pregnancy'] - self.sim.date
        assert days_until_labour >= pd.Timedelta(245, unit='d')

    def predict(self, eq, person_id):
        """
        This function compares the result of a specific linear equation with a random draw providing a boolean for
        the outcome under examination
        :param eq: Linear model equation
        :param person_id: individual_id
        """
        df = self.sim.population.props
        mni = self.sim.modules['PregnancySupervisor'].mother_and_newborn_info
        person = df.loc[[person_id]]

        # We define specific external variables used as predictors in the equations defined below
        has_rbt = mni[person_id]['received_blood_transfusion']
        mode_of_delivery = mni[person_id]['mode_of_delivery']
        received_clean_delivery = mni[person_id]['clean_birth_practices']
        received_abx_for_prom = mni[person_id]['abx_for_prom_given']
        amtsl_given = mni[person_id]['amtsl_given']
        if df.at[person_id, 'ps_gestational_age_in_weeks'] < 37:
            preterm_labour = True
        else:
            preterm_labour = False
        if mni[person_id]['birth_weight'] == 'macrosomia':
            macrosomia = True
        else:
            macrosomia = False

        # We run a random draw and return the outcome
        return self.rng.random_sample() < eq.predict(person,
                                                     received_clean_delivery=received_clean_delivery,
                                                     received_abx_for_prom=received_abx_for_prom,
                                                     mode_of_delivery=mode_of_delivery,
                                                     received_blood_transfusion=has_rbt,
                                                     preterm_labour=preterm_labour,
                                                     amtsl_given=amtsl_given,
                                                     macrosomia=macrosomia)[person_id]

    def reset_due_date(self, ind_or_df, id_or_index, new_due_date):
        """
        This function is called at various points in the PregnancySupervisor module to reset the due-date of women who
        may have experience pregnancy loss or will now go into pre-term labour on new due-date
        :param ind_or_df: (STR) Is this function being use on an individual row or slice of the data frame
         'individual'/'data_frame'
        :param id_or_index: The individual id OR dataframe slice that this change will be made for
        :param new_due_date: (DATE) the new due-date
        """
        df = self.sim.population.props

        if ind_or_df == 'individual':
            change = df.at
        else:
            change = df.loc

        change[id_or_index, 'la_due_date_current_pregnancy'] = new_due_date

    def check_labour_can_proceed(self, individual_id):
        """
        This function is called by the LabourOnsetEvent to evaluate if labour can proceed for the woman who has arrived
         at the event
        :param individual_id: individual_id
        :returns True/False if labour can proceed
        """
        df = self.sim.population.props
        person = df.loc[individual_id]

        # If the mother has died OR has lost her pregnancy OR is already in labour then the labour events wont run
        if ~person.is_alive or ~person.is_pregnant or person.la_currently_in_labour:
            logger.debug(key='message', data=f'person {individual_id} has just reached LabourOnsetEvent on '
                                             f'{self.sim.date}, however this is event is no longer relevant for this '
                                             f'individual and will not run')
            return False

        # If she is alive, pregnant, not in labour AND her due date is today then the event will run
        elif person.is_alive and person.is_pregnant and (person.la_due_date_current_pregnancy == self.sim.date) \
                and ~person.la_currently_in_labour:

            # If the woman in not currently an inpatient then we assume this is her normal labour
            if person.ac_admitted_for_immediate_delivery == 'none':
                logger.debug(key='message', data=f'person {individual_id} has just reached LabourOnsetEvent on '
                                                 f'{self.sim.date} and will now go into labour at gestation '
                                                 f'{person.ps_gestational_age_in_weeks}')

            # Otherwise she may have gone into labour whilst admitted as an inpatient and is awaiting induction/
            # caesarean when she is further along in her pregnancy, in that case labour can proceed via the method she
            # was admitted for
            else:
                logger.debug(key='message', data=f'person {individual_id}, who is currently admitted and awaiting '
                                                 f'delivery, has just gone into spontaneous labour and reached '
                                                 f'LabourOnsetEvent on {self.sim.date} - she will now go into labour '
                                                 f'at gestation {person.ps_gestational_age_in_weeks}')
            return True

        # If she is alive, pregnant, not in labour BUT her due date is not today, however shes been admitted then we
        # labour can progress as she requires early delivery
        elif person.is_alive and person.is_pregnant and ~person.la_currently_in_labour and \
            (person.la_due_date_current_pregnancy != self.sim.date) and (person.ac_admitted_for_immediate_delivery !=
                                                                         'none'):

            logger.debug(key='message', data=f'person {individual_id} has just reached LabourOnsetEvent on '
                                             f'{self.sim.date}- they have been admitted for delivery due to '
                                             f'complications in the antenatal period and will now progress into the '
                                             f'labour event at gestation {person.ps_gestational_age_in_weeks}')

            # We set her due date to today so she the event will run properly
            df.at[individual_id, 'la_due_date_current_pregnancy'] = self.sim.date
            return True

        else:
            return False

    def set_intrapartum_complications(self, individual_id, complication):
        """This function is called either during a LabourAtHomeEvent OR HSI_Labour_ReceivesSkilledBirthAttendanceDuring
        Labour for all women during labour (home birth vs facility delivery). The function is used to apply risk of
        complications which have been passed ot it including the preceding causes of obstructed labour
        (malposition, malpresentation and cephalopelvic disproportion), obstructed labour, uterine rupture, placental
        abruption, antepartum haemorrhage,  infections (chorioamnionitis/other) and sepsis. Properties in the dataframe
         are set accordingly including properties which map to disability weights to capture DALYs
        :param individual_id: individual_id
        :param complication: (STR) the complication passed to the function which is being evaluated
        ['obstruction_cpd', 'obstruction_malpos_malpres', 'obstruction_other', 'placental_abruption',
        'antepartum_haem', 'sepsis_chorioamnionitis', 'uterine_rupture']
        """
        df = self.sim.population.props
        params = self.current_parameters
        mni = self.sim.modules['PregnancySupervisor'].mother_and_newborn_info

        # First we run check to ensure only women who have started the labour process are passed to this function
        assert mni[individual_id]['delivery_setting'] != 'none'

        # Then we check that only complications from the master complication list are passed to the function (to ensure
        # any typos for string variables are caught)
        assert complication in self.possible_intrapartum_complications

        # Women may have been admitted for delivery from the antenatal ward because they have developed a complication
        # in pregnancy requiring delivery. Here we make sure women admitted due to these complications do not experience
        # the same complication again when this code runs
        if df.at[individual_id, 'ac_admitted_for_immediate_delivery'] != 'none':

            # Both 'la_antepartum_haem' and 'ps_antepartum_haem' will trigger treatment if identified
            if (complication == 'antepartum_haem') and (df.at[individual_id, 'ps_antepartum_haemorrhage'] != 'none'):
                return

            # Onset of placental abruption antenatally or intrapartum can lead to APH in linear model
            if (complication == 'placental_abruption') and df.at[individual_id, 'ps_placental_abruption']:
                return

            # Women admitted with histological chorioamnionitis from the community are more at risk of sepsis
            if (complication == 'sepsis_chorioamnionitis') and (df.at[individual_id, 'ps_chorioamnionitis'] ==
                                                                'histological'):
                return

            # Women admitted with clinical chorioamnionitis from the community are assumed to be septic in labour
            if (complication == 'sepsis') and (df.at[individual_id, 'ps_chorioamnionitis'] == 'clinical'):
                df.at[individual_id, f'la_{complication}'] = True
                return

        # For the preceding complications that can cause obstructed labour, we apply risk using a set probability
        if (complication == 'obstruction_malpos_malpres') or (complication == 'obstruction_other'):
            result = self.rng.random_sample() < params[f'prob_{complication}']

        # Otherwise we use the linear model to predict likelihood of a complication
        else:
            result = self.predict(self.la_linear_models[f'{complication}_ip'], individual_id)

        # --------------------------------------- COMPLICATION ------------------------------------------------------
        # If 'result' == True, this woman will experience the complication passed to the function
        if result:
            logger.debug(key='message', data=f'person {individual_id} has developed {complication} during birth on date'
                                             f'{self.sim.date}')

            # For 'complications' stored in a biset property - they are set here
            if (complication == 'obstruction_cpd') or (complication == 'obstruction_malpres_malpos') or \
              (complication == 'obstruction_other'):

                df.at[individual_id, 'la_obstructed_labour'] = True
                self.sim.modules['PregnancySupervisor'].store_dalys_in_mni(individual_id, 'obstructed_labour_onset')

                if complication == 'obstruction_cpd':
                    mni[individual_id]['cpd'] = True

            # TODO: this WILL double count
            # self.labour_tracker[f'{complication}'] += 1

            # Otherwise they are stored as individual properties (women with undiagnosed placental abruption may present
            # to labour)
            elif complication == 'placental_abruption':
                df.at[individual_id, 'la_placental_abruption'] = True

            elif complication == 'antepartum_haem':
                random_choice = self.rng.choice(['mild_moderate', 'severe'],
                                                p=params['severity_maternal_haemorrhage'])
                df.at[individual_id, f'la_{complication}'] = random_choice
                if random_choice != 'severe':
                    self.sim.modules['PregnancySupervisor'].store_dalys_in_mni(individual_id, 'mild_mod_aph_onset')
                else:
                    self.sim.modules['PregnancySupervisor'].store_dalys_in_mni(individual_id, 'severe_aph_onset')

            elif complication == 'sepsis_chorioamnionitis':
                df.at[individual_id, 'la_sepsis'] = True
                mni[individual_id]['chorio_lab'] = True
                self.labour_tracker['sepsis'] += 1
                self.sim.modules['PregnancySupervisor'].store_dalys_in_mni(individual_id, 'sepsis_onset')

            else:
                df.at[individual_id, 'la_uterine_rupture'] = True
                self.labour_tracker['uterine_rupture'] += 1
                self.sim.modules['PregnancySupervisor'].store_dalys_in_mni(individual_id, f'{complication}_onset')

    def set_postpartum_complications(self, individual_id, complication):
        """
        This function is called either during a PostpartumLabourAtHomeEvent OR HSI_Labour_ReceivesSkilledBirthAttendance
        FollowingLabour for all women following labour and birth (home birth vs facility delivery). The function is
        used to apply risk of complications which have been passed ot it including the preceding causes of postpartum
        haemorrhage (uterine atony, retained placenta, lacerations, other), postpartum haemorrhage, preceding infections
         to sepsis (endometritis, skin/soft tissue infection, urinary tract, other), sepsis. Properties in the dataframe
         are set accordingly including properties which map to disability weights to capture DALYs
        :param individual_id: individual_id
        :param complication: (STR) the complication passed to the function which is being evaluated [
        'sepsis_endometritis', 'sepsis_skin_soft_tissue', 'sepsis_urinary_tract', 'pph_uterine_atony',
        'pph_retained_placenta', 'pph_other']
        """
        df = self.sim.population.props
        mni = self.sim.modules['PregnancySupervisor'].mother_and_newborn_info
        params = self.current_parameters

        # This function follows a roughly similar pattern as set_intrapartum_complications
        assert mni[individual_id]['delivery_setting'] != 'none'
        assert complication in self.possible_postpartum_complications

        #  We determine if this woman has experienced any of the other potential preceding causes of PPH
        if complication == 'pph_other':
            result = self.rng.random_sample() < params['prob_pph_other_causes']

        # For the other complications which can be passed to this function we use the linear model to return a womans
        # risk and compare that to a random draw
        else:
            result = self.predict(self.la_linear_models[f'{complication}_pp'], individual_id)

        # ------------------------------------- COMPLICATION ---------------------------------------------------------
        # If result == True the complication has happened and the appropriate changes to the data frame are made
        if result:
            logger.debug(key='message', data=f'person {individual_id} has developed {complication} during the'
                                             f' postpartum phase of a birth on date {self.sim.date}')

            if complication == 'sepsis_endometritis' or complication == 'sepsis_urinary_tract' or \
               complication == 'sepsis_skin_soft_tissue':

                df.at[individual_id, 'la_sepsis_pp'] = True
                # TODO: this would cause double counting
                # self.labour_tracker['sepsis_pp'] += 1
                self.sim.modules['PregnancySupervisor'].store_dalys_in_mni(individual_id, 'sepsis_onset')

            if complication == 'pph_uterine_atony' or complication == 'pph_retained_placenta' or \
               complication == 'pph_other':
                # Set primary complication to true
                df.at[individual_id, 'la_postpartum_haem'] = True
                # self.labour_tracker[f'{complication}'] += 1 # TODO: same issue r.e. double counting

                # Store mni variables used during treatment
                if complication == 'pph_uterine_atony':
                    mni[individual_id]['uterine_atony'] = True
                if complication == 'pph_retained_placenta':
                    mni[individual_id]['retained_placenta'] = True

                # We set the severity to map to DALY weights
                if pd.isnull(mni[individual_id]['mild_mod_pph_onset']) and pd.isnull(mni[individual_id]['severe_pph_'
                                                                                                        'onset']):

                    random_choice = self.rng.choice(['non_severe', 'severe'], size=1,
                                                    p=params['severity_maternal_haemorrhage'])

                    if random_choice == 'non_severe':
                        self.sim.modules['PregnancySupervisor'].store_dalys_in_mni(individual_id, 'mild_mod_pph_onset')
                    else:
                        self.sim.modules['PregnancySupervisor'].store_dalys_in_mni(individual_id, 'severe_pph_onset')

    def progression_of_hypertensive_disorders(self, individual_id, property_prefix):
        """
        This function is called during LabourAtHomeEvent/PostpartumLabourAtHomeEvent or HSI_Labour_Receives
        SkilledBirthAttendanceDuring/FollowingLabour to determine if a woman with a hypertensive disorder will
        experience progression to a more severe state of disease during labour or the immediate postpartum period.
        We do not allow for new onset of  hypertensive disorders during this module - only progression of
        exsisting disease.
        :param individual_id: individual_id
        :param property_prefix: (STR) 'pn' or 'ps'
        """
        df = self.sim.population.props
        params = self.current_parameters

        # n.b. on birth women whose hypertension will continue into the postnatal period have their disease state stored
        # in a new property therefore antenatal/intrapartum hypertension is 'ps_htn_disorders' and postnatal is
        # 'pn_htn_disorders' hence the use of property prefix variable (as this function is called before and after
        # birth)

        # Women can progress from severe pre-eclampsia to eclampsia
        if df.at[individual_id, f'{property_prefix}_htn_disorders'] == 'severe_pre_eclamp':

            risk_ec = params['prob_progression_severe_pre_eclamp']

            # Risk of progression from severe pre-eclampsia to eclampsia during labour is mitigated by administration of
            # magnesium sulfate in women with severe pre-eclampsia (this may have been delivered on admission or in the
            # antenatal ward)
            if df.at[individual_id, 'la_severe_pre_eclampsia_treatment'] or \
                (df.at[individual_id, 'ac_mag_sulph_treatment'] and
                 (df.at[individual_id, 'ac_admitted_for_immediate_delivery'] != 'none')):
                risk_progression_spe_ec = risk_ec * params['eclampsia_treatment_effect_severe_pe']
            else:
                risk_progression_spe_ec = risk_ec

            if risk_progression_spe_ec > self.rng.random_sample():
                df.at[individual_id, f'{property_prefix}_htn_disorders'] = 'eclampsia'
                self.sim.modules['PregnancySupervisor'].store_dalys_in_mni(individual_id, 'eclampsia_onset')
                self.labour_tracker['eclampsia'] += 1

                logger.debug(key='msg', data=f'Mother {individual_id} has developed eclampsia_{property_prefix}')

        # Or from mild to severe gestational hypertension, risk reduced by treatment
        if df.at[individual_id, f'{property_prefix}_htn_disorders'] == 'gest_htn':
            if df.at[individual_id, 'la_maternal_hypertension_treatment']:
                risk_prog_gh_sgh = params['prob_progression_gest_htn'] * params[
                    'anti_htns_treatment_effect_progression']
            else:
                risk_prog_gh_sgh = params['prob_progression_gest_htn']
            if risk_prog_gh_sgh > self.rng.random_sample():
                df.at[individual_id, f'{property_prefix}_htn_disorders'] = 'severe_gest_htn'
                logger.debug(key='msg', data=f'Mother {individual_id} has developed severe_gest_htn_{property_prefix}')

        # Or from severe gestational hypertension to severe pre-eclampsia...
        if df.at[individual_id, f'{property_prefix}_htn_disorders'] == 'severe_gest_htn':
            if params['prob_progression_severe_gest_htn'] > self.rng.random_sample():
                df.at[individual_id, f'{property_prefix}_htn_disorders'] = 'severe_pre_eclamp'
                self.labour_tracker['severe_pre_eclampsia'] += 1
                logger.debug(key='msg', data=f'Mother {individual_id} has developed severe_pre_eclamp_'
                                             f'{property_prefix}')

        # Or from mild pre-eclampsia to severe pre-eclampsia...
        if df.at[individual_id, f'{property_prefix}_htn_disorders'] == 'mild_pre_eclamp':
            if params['prob_progression_mild_pre_eclamp'] > self.rng.random_sample():
                df.at[individual_id, f'{property_prefix}_htn_disorders'] = 'severe_pre_eclamp'
                self.labour_tracker['severe_pre_eclampsia'] += 1
                logger.debug(key='msg', data=f'Mother {individual_id} has developed severe_pre_eclamp_'
                                             f'{property_prefix}')

    def set_maternal_death_status_intrapartum(self, individual_id, cause):
        """
        This function is called by the LabourDeathEvent. A 'cause' of death is passed to this function and a linear
        model is used to calculate if the cause will contribute to this womans death during labour
        :param individual_id: individual_id
        :param cause: (STR) complication which may cause death
        """
        df = self.sim.population.props
        mni = self.sim.modules['PregnancySupervisor'].mother_and_newborn_info
        params = self.current_parameters

        # Check that only the correct causes are being passed to the function
        assert cause in self.possible_intrapartum_complications

        # todo: only allow death from severe haemorrhage (IP and PP)

        # We determine if this woman will die of the complication defined in the function
        if self.predict(self.la_linear_models[f'{cause}_death'], individual_id):
            logger.debug(key='message', data=f'{cause} has contributed to person {individual_id} death during labour')

            mni[individual_id]['death_in_labour'] = True
            mni[individual_id]['cause_of_death_in_labour'].append(cause)

            # This information is passed to the event where the InstantaneousDeathEvent is scheduled - this allows for
            # the application of multiple risk of death from multiple complications
            df.at[individual_id, 'la_maternal_death_in_labour'] = True
            df.at[individual_id, 'la_maternal_death_in_labour_date'] = self.sim.date
            self.labour_tracker[f'{cause}_death'] += 1

        else:
            # As eclampsia is a transient acute event, if women survive we reset their disease state to severe
            # pre-eclampsia
            if cause == 'eclampsia':
                df.at[individual_id, 'ps_htn_disorders'] = 'severe_pre_eclamp'

    def set_maternal_death_status_postpartum(self, individual_id, cause):
        """
        This function is called by from within the apply_risk_of_early_postpartum_death function (below).
        A 'cause' of death is passed to this function and a linear model is used to calculate if the cause will
        contribute to this womans death following labour
        :param individual_id: individual_id
        :param cause: (STR) complication which may cause death
        """
        df = self.sim.population.props
        mni = self.sim.modules['PregnancySupervisor'].mother_and_newborn_info
        params = self.current_parameters

        assert cause in self.possible_postpartum_complications

        if self.predict(self.la_linear_models[f'{cause}_pp_death'], individual_id):
            logger.debug(key='message',
                         data=f'{cause} has contributed to person {individual_id} death following labour')

            self.labour_tracker[f'{cause}_death'] += 1
            mni[individual_id]['death_postpartum'] = True
            mni[individual_id]['cause_of_death_in_labour'].append(f'{cause}_postpartum')
            df.at[individual_id, 'la_maternal_death_in_labour'] = True
            df.at[individual_id, 'la_maternal_death_in_labour_date'] = self.sim.date

        else:
            if cause == 'eclampsia':
                df.at[individual_id, 'pn_htn_disorders'] = 'severe_pre_eclamp'

    def apply_risk_of_early_postpartum_death(self, individual_id):
        """
        This function is called for all women who have survived labour. This function is called at various points in
        the model depending on a womans pathway through labour and includes PostpartumLabourAtHomeEvent,
        HSI_Labour_ReceivesSkilledBirthAttendanceFollowingLabour, HSI_Labour_ReceivesComprehensiveEmergencyObstetric
        Care and HSI_Labour_ReceivesCareFollowingCaesareanSection. The function cycles through each complication to
        determine if that will contribute to a womans death and then schedules InstantaneousDeathEvent accordingly.
        For women who survive their properties from the labour module are reset and they are scheduled to
        PostnatalWeekOneEvent
        :param individual_id: individual_id
        """
        df = self.sim.population.props
        mni = self.sim.modules['PregnancySupervisor'].mother_and_newborn_info

        # We ensure that this function is only being applied to the correct women
        self.postpartum_characteristics_checker(individual_id)

        # todo: SPE should cause death here too? (or will this leead to really high death in SPE?)

        # We then move through each complication to calculate risk of death
        if df.at[individual_id, 'pn_htn_disorders'] == 'severe_pre_eclamp':
            self.set_maternal_death_status_postpartum(individual_id, cause='eclampsia')

        if df.at[individual_id, 'pn_htn_disorders'] == 'eclampsia':
            self.set_maternal_death_status_postpartum(individual_id, cause='eclampsia')

        if df.at[individual_id, 'la_postpartum_haem']:
            self.set_maternal_death_status_postpartum(individual_id, cause='postpartum_haem')

        if df.at[individual_id, 'la_sepsis_pp']:
            self.set_maternal_death_status_postpartum(individual_id, cause='sepsis')

        if mni[individual_id]['death_postpartum']:
            self.labour_tracker['maternal_death'] += 1

            logger.info(key='direct_maternal_death', data={'person': individual_id, 'preg_state': 'postnatal',
                                                           'year': self.sim.date.year})

            self.sim.modules['Demography'].do_death(individual_id=individual_id, cause='maternal',
                                                    originating_module=self.sim.modules['Labour'])

            logger.debug(key='message', data=f'Mother {individual_id} has died due to postpartum complications')

        # For women who have survived we first reset all the appropriate variables as this is the last function ran
        # within the module
        elif not mni[individual_id]['death_postpartum']:
            # ================================ RESET LABOUR MODULE VARIABLES =========================================
            # Reset labour variable
            df.at[individual_id, 'la_currently_in_labour'] = False
            df.at[individual_id, 'la_due_date_current_pregnancy'] = pd.NaT

            #  complication variables
            df.at[individual_id, 'la_intrapartum_still_birth'] = False
            df.at[individual_id, 'la_postpartum_haem'] = False
            df.at[individual_id, 'la_obstructed_labour'] = False
            df.at[individual_id, 'la_placental_abruption'] = False
            df.at[individual_id, 'la_antepartum_haem'] = 'none'
            df.at[individual_id, 'la_uterine_rupture'] = False
            df.at[individual_id, 'la_sepsis'] = False
            df.at[individual_id, 'la_sepsis_pp'] = False
            df.at[individual_id, 'la_postpartum_haem'] = False

            # Treatment variables
            df.at[individual_id, 'la_antepartum_haem_treatment'] = False
            df.at[individual_id, 'la_uterine_rupture_treatment'] = False
            df.at[individual_id, 'la_severe_pre_eclampsia_treatment'] = False
            df.at[individual_id, 'la_maternal_hypertension_treatment'] = False
            df.at[individual_id, 'la_eclampsia_treatment'] = False
            df.at[individual_id, 'la_sepsis_treatment'] = False
            self.pph_treatment.unset(
                [individual_id], 'uterotonics', 'manual_removal_placenta', 'surgery', 'hysterectomy')

        # todo: This event looks at the mother and child  pair an applies risk of complications- therefore I cant
        #  condition on the mother being alive for the event to be schedule (otherwise newborns of mothers who died in
        #  labour or after labour dont get risk applied) - should be fine as event wont run on mothers who arent alive

        # ================================ SCHEDULE POSTNATAL WEEK ONE EVENT =====================================
        # For women who have survived first 24 hours after birth we scheduled them to attend the first event in the
        # PostnatalSupervisorModule - PostnatalWeekOne Event

        # This event determines if women/newborns will develop complications in week one. We stagger when women
        # arrive at this event to simulate bunching of complications in the first few days after birth

        days_post_birth_td = self.sim.date - df.at[individual_id, 'la_date_most_recent_delivery']
        days_post_birth_int = int(days_post_birth_td / np.timedelta64(1, 'D'))

        assert days_post_birth_int < 6

        # change to parameter
        day_for_event = int(self.rng.choice([0, 1, 2, 3, 4, 5], p=[0.3, 0.3, 0.2, 0.1, 0.05, 0.05]))

        # Ensure no women go to this event after week 1
        if day_for_event + days_post_birth_int > 5:
            day_for_event = 0

        self.sim.schedule_event(PostnatalWeekOneEvent(self.sim.modules['PostnatalSupervisor'], individual_id),
                                self.sim.date + DateOffset(days=day_for_event))

        # Here we remove all women (dead and alive) who have passed through the labour events from the checker list
        self.women_in_labour.remove(individual_id)

    def labour_characteristics_checker(self, individual_id):
        """This function is called at multiples points in the module to ensure women of the right characteristics are
        in labour. This function doesnt check for a woman being pregnant or alive, as some events will still run despite
        those variables being set to false
        :param individual_id: individual_id
        """
        df = self.sim.population.props
        mother = df.loc[individual_id]

        assert individual_id in self.women_in_labour
        assert mother.sex == 'F'
        assert mother.age_years > 14
        assert mother.age_years < 51
        assert mother.la_currently_in_labour
        assert mother.ps_gestational_age_in_weeks >= 22
        assert not pd.isnull(mother.la_due_date_current_pregnancy)

    def postpartum_characteristics_checker(self, individual_id):
        """This function is called at multiples points in the module to ensure women of the right characteristics are
        in the period following labour. This function doesnt check for a woman being pregnant or alive, as some events
        will still run despite those variables being set to false
        :param individual_id: individual_id
        """
        df = self.sim.population.props
        mother = df.loc[individual_id]

        assert individual_id in self.women_in_labour
        assert mother.sex == 'F'
        assert mother.age_years > 14
        assert mother.age_years < 51
        assert mother.la_currently_in_labour

    # ============================================== HSI FUNCTIONS ====================================================
    # Management of each complication is housed within its own function, defined here in the module, and all follow a
    # similar pattern ...
    #                   a.) The required consumables for the intervention(s) are defined
    #                   b.) The woman is assessed for a complication using the dx_test function. Specificity of
    #                       assessment varies between facility type (hospital or health centre)
    #                   c.) If she has the complication and it is correctly identified by HCWs, they check
    #                       consumables are available
    #                   d.) If the consumables are available- she will receive treatment

    # The function is only called if the squeeze factor of the HSI calling the function is below a set 'threshold' for
    # each intervention. Thresholds will vary between intervention

    def prophylactic_labour_interventions(self, hsi_event):
        """
        This function houses prophylactic interventions delivered by a Skilled Birth Attendant to women in labour.
        It is called by HSI_Labour_PresentsForSkilledBirthAttendanceInLabour
        :param hsi_event: HSI event in which the function has been called:
        """
        df = self.sim.population.props
        params = self.current_parameters
        mni = self.sim.modules['PregnancySupervisor'].mother_and_newborn_info
        person_id = hsi_event.target
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']

        # params['allowed_interventions'] contains a list of interventions delivered in this module. Removal of
        # interventions from this list within test/analysis will stop this intervention from running
        if 'prophylactic_labour_interventions' not in params['allowed_interventions']:
            return
        else:
            # ----------------------------------CLEAN DELIVERY PRACTICES ---------------------------------------------
            # The first in this suite of interventions is clean delivery practices. We assume clean birth practices are
            # delivered if a clean birth kit is available at the facility
            pkg_code_clean_delivery_kit = pd.unique(
                consumables.loc[consumables['Intervention_Pkg'] == 'Clean practices and immediate essential newborn '
                                                                   'care (in facility)', 'Intervention_Pkg_Code'])[0]

            all_available = hsi_event.get_all_consumables(
                pkg_codes=[pkg_code_clean_delivery_kit])

            # If available we store this intervention in the mni dictionary. Clean delivery will reduce a mothers risk
            # of intrapartum infections
            if all_available:
                mni[person_id]['clean_birth_practices'] = True
                logger.debug(key='message', data=f'Mother {person_id} will be able to experience clean birth practices '
                                                 f'during her delivery as the consumables are available')
            else:
                logger.debug(key='message', data=f'Mother {person_id} will not be able to experience clean birth '
                                                 f'practices during her delivery as the consumables arent available')

            # --------------------------------- ANTIBIOTICS FOR PROM/PPROM -------------------------------------------
            # Next we determine if the HCW will administer antibiotics for women with premature rupture of membranes
            if df.at[person_id, 'ps_premature_rupture_of_membranes']:

                # The mother may have received these antibiotics already if she presented to the antenatal ward from the
                # community following PROM. We store this in the mni dictionary
                if df.at[person_id, 'ac_received_abx_for_prom']:
                    mni[person_id]['abx_for_prom_given'] = True

                else:
                    # If she has not already receive antibiotics, we check for consumables
                    pkg_code_pprom = pd.unique(
                        consumables.loc[consumables['Intervention_Pkg'] == 'Antibiotics for pPRoM',
                                        'Intervention_Pkg_Code'])[0]

                    all_available = hsi_event.get_all_consumables(
                        pkg_codes=[pkg_code_pprom])

                    # TODO: review antibiotics in this package, add equipment to deliver IV

                    # And provide if available. Antibiotics for from reduce risk of newborn sepsis within the first
                    # week of life
                    if all_available:
                        mni[person_id]['abx_for_prom_given'] = True
                        logger.debug(key='message', data=f'This facility has provided antibiotics for mother '
                                                         f'{person_id} who is a risk of sepsis due to PROM')
                    else:
                        logger.debug(key='message', data='This facility has no antibiotics for the treatment of PROM.')

            # ------------------------------ STEROIDS FOR PRETERM LABOUR -------------------------------
            # Next we see if women in pre term labour will receive antenatal corticosteroids
            if mni[person_id]['labour_state'] == 'early_preterm_labour' or \
               mni[person_id]['labour_state'] == 'late_preterm_labour':

                item_code_steroids_prem_dexamethasone = pd.unique(
                    consumables.loc[consumables['Items'] == 'Dexamethasone 5mg/ml, 5ml_each_CMST', 'Item_Code'])[0]
                item_code_steroids_prem_betamethasone = pd.unique(
                    consumables.loc[consumables['Items'] == 'Betamethasone, 12 mg injection', 'Item_Code'])[0]

                # todo: add consumables for delivering IV drugs (there is a consumables pkg we could ammend)

                consumables_steriod_preterm = {
                    'Intervention_Package_Code': {},
                    'Item_Code': {item_code_steroids_prem_dexamethasone: 1, item_code_steroids_prem_betamethasone: 1}}

                outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
                    hsi_event=hsi_event,
                    cons_req_as_footprint=consumables_steriod_preterm)

                # If available they are given. Antenatal steroids reduce a preterm newborns chance of developing
                # respiratory distress syndrome and of death associated with prematurity
                if (outcome_of_request_for_consumables['Item_Code'][
                    item_code_steroids_prem_dexamethasone]) and \
                   (outcome_of_request_for_consumables['Item_Code'][item_code_steroids_prem_betamethasone]):

                    mni[person_id]['corticosteroids_given'] = True
                    logger.debug(key='message', data=f'This facility has provided corticosteroids for mother '
                                                     f'{person_id} who is in preterm labour')

                else:
                    logger.debug(key='message', data='This facility has no steroids for women in preterm labour.')

    def assessment_and_treatment_of_severe_pre_eclampsia_mgso4(self, hsi_event, facility_type, labour_stage):
        """This function represents the diagnosis and management of severe pre-eclampsia during labour. This function
        defines the required consumables, uses the dx_test to determine a woman with severe pre-eclampsia is correctly
        identified, and administers the intervention if so. The intervention is intravenous magnesium sulphate.
        It is called by HSI_Labour_PresentsForSkilledBirthAttendanceInLabour
        :param hsi_event: HSI event in which the function has been called:
        :param facility_type: type of facility this intervention is being delivered in
        (STR) 'hc' == health centre, 'hp' == hospital
        :param labour_stage: intrapartum or postpartum period of labour (STR) 'ip' or 'pp':
        """
        df = self.sim.population.props
        params = self.current_parameters
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        person_id = hsi_event.target

        # Women who have been admitted for delivery due to severe pre-eclampsia AND have already received magnesium
        # before moving to the labour ward do not receive the intervention again
        if (df.at[person_id, 'ac_admitted_for_immediate_delivery'] != 'none') and df.at[person_id,
                                                                                        'ac_mag_sulph_treatment']:

            logger.debug(key='msg', data=f'Mother {person_id} has already received magnesium therapy following '
                                         f' to the antenatal ward prior to onset of labour ')
            return

        if ('assessment_and_treatment_of_severe_pre_eclampsia' not in params['allowed_interventions']) or \
            df.at[person_id, 'la_severe_pre_eclampsia_treatment']:
            return

        else:
            # Define the required consumables
            pkg_code_severe_pre_eclampsia = pd.unique(
                consumables.loc[consumables['Intervention_Pkg'] == 'Management of eclampsia',
                                'Intervention_Pkg_Code'])[0]
            all_available = hsi_event.get_all_consumables(
                pkg_codes=[pkg_code_severe_pre_eclampsia])

            if labour_stage == 'ip':
                prefix = 'ps'
            else:
                prefix = 'pn'

            # Here we run a dx_test function to determine if the birth attendant will correctly identify this womans
            # severe pre-eclampsia, and therefore administer treatment
            if self.sim.modules['HealthSystem'].dx_manager.run_dx_test(
                dx_tests_to_run=f'assess_severe_pe_{facility_type}_{labour_stage}', hsi_event=hsi_event):

                # If so, and the consumables are available - the intervention is delivered. IV magnesium reduces the
                # probability that a woman with severe pre-eclampsia will experience eclampsia in labour
                if all_available:
                    df.at[person_id, 'la_severe_pre_eclampsia_treatment'] = True
                    logger.debug(key='message', data=f'mother {person_id} has has their severe pre-eclampsia '
                                                     f'identified during delivery. As consumables are available '
                                                     f'they will receive treatment')

                elif df.at[person_id, f'{prefix}_htn_disorders'] == 'severe_pre_eclamp':
                    logger.debug(key='message', data=f'mother {person_id} has not had their severe pre-eclampsia '
                                                     f'identified during delivery and will not be treated')

    def assessment_and_treatment_of_hypertension(self, hsi_event, facility_type, labour_stage):
        """
        This function represents the diagnosis and management of hypertension during labour. This function
        defines the required consumables, uses the dx_test to determine a woman with hypertension is correctly
        identified, and administers the intervention if so. The intervention is intravenous antihypertensives.
        It is called by HSI_Labour_PresentsForSkilledBirthAttendanceInLabour
        :param hsi_event: HSI event in which the function has been called:
        :param facility_type: type of facility this intervention is being delivered in
        (STR) 'hc' == health centre, 'hp' == hospital
        :param labour_stage: intrapartum or postpartum period of labour (STR) 'ip' or 'pp':
        """
        df = self.sim.population.props
        person_id = hsi_event.target
        params = self.current_parameters
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']

        if ('assessment_and_treatment_of_hypertension' not in params['allowed_interventions']) or \
           df.at[person_id, 'ac_iv_anti_htn_treatment'] or df.at[person_id, 'la_maternal_hypertension_treatment']:
            return
        else:
            item_code_hydralazine = pd.unique(
                consumables.loc[consumables['Items'] == 'Hydralazine, powder for injection, 20 mg ampoule',
                                'Item_Code'])[0]
            item_code_wfi = pd.unique(
                consumables.loc[consumables['Items'] == 'Water for injection, 10ml_Each_CMST', 'Item_Code'])[0]
            item_code_needle = pd.unique(
                consumables.loc[consumables['Items'] == 'Syringe, needle + swab', 'Item_Code'])[0]
            item_code_gloves = pd.unique(
                consumables.loc[consumables['Items'] == 'Gloves, exam, latex, disposable, pair', 'Item_Code'])[0]

            consumables_gest_htn_treatment = {
                'Intervention_Package_Code': {},
                'Item_Code': {item_code_hydralazine: 1,
                              item_code_wfi: 1,
                              item_code_gloves: 1,
                              item_code_needle: 1}}

            # Then query if these consumables are available during this HSI
            outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
                hsi_event=hsi_event,
                cons_req_as_footprint=consumables_gest_htn_treatment)

            if labour_stage == 'ip':
                prefix = 'ps'
            else:
                prefix = 'pn'

            if self.sim.modules['HealthSystem'].dx_manager.run_dx_test(dx_tests_to_run=f'assess_'
                                                                                       f'hypertension_{facility_type}_'
                                                                                       f'{labour_stage}',
                                                                       hsi_event=hsi_event):

                # If they are available then the woman is started on treatment. Intravenous antihypertensive reduce a
                # womans risk of progression from mild to severe gestational hypertension ANd reduce risk of death for
                # women with severe pre-eclampsia and eclampsia
                if outcome_of_request_for_consumables['Item_Code'][item_code_hydralazine]:
                    df.at[person_id, 'la_maternal_hypertension_treatment'] = True
                    logger.debug(key='message', data=f'mother {person_id} has has their hypertension identified during '
                                                     f'delivery. As consumables are available they will receive'
                                                     f' treatment')

            elif df.at[person_id, f'{prefix}_htn_disorders'] != 'none':
                logger.debug(key='message', data=f'mother {person_id} has not had their hypertension identified during '
                                                 f'delivery and will not be treated')

    def assessment_and_treatment_of_eclampsia(self, hsi_event, facility_type, labour_stage):
        """
        This function represents the diagnosis and management of eclampsia during or following labour. This function
        defines the required consumables, uses the dx_test to determine a woman with eclampsia is correctly
        identified, and administers the intervention if so. The intervention is intravenous magnesium sulphate.
        It is called by either HSI_Labour_PresentsForSkilledBirthAttendanceInLabour
        or HSI_Labour_ReceivesCareForPostpartumPeriod
        :param hsi_event: HSI event in which the function has been called:
        :param facility_type: type of facility this intervention is being delivered in
        (STR) 'hc' == health centre, 'hp' == hospital
        :param labour_stage: intrapartum or postpartum period of labour (STR) 'ip' or 'pp':
        """
        df = self.sim.population.props
        person_id = hsi_event.target
        params = self.current_parameters
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']

        if ('assessment_and_treatment_of_eclampsia' not in params['allowed_interventions']) or \
           df.at[person_id, 'la_eclampsia_treatment']:
            return

        else:
            pkg_code_eclampsia = pd.unique(
                consumables.loc[consumables['Intervention_Pkg'] == 'Management of eclampsia',
                                'Intervention_Pkg_Code'])[0]

            all_available = hsi_event.get_all_consumables(
                pkg_codes=[pkg_code_eclampsia])

            if labour_stage == 'ip':
                prefix = 'ps'
            else:
                prefix = 'pn'

            if self.sim.modules['HealthSystem'].dx_manager.run_dx_test(dx_tests_to_run=f'assess_'
                                                                                       f'eclampsia_{facility_type}_'
                                                                                       f'{labour_stage}',
                                                                       hsi_event=hsi_event):

                if all_available:
                    # Treatment with magnesium reduces a womans risk of death from eclampsia
                    df.at[person_id, 'la_eclampsia_treatment'] = True
                    logger.debug(key='message', data=f'mother {person_id} has has their eclampsia identified during '
                                                     f'delivery. As consumables are available they will receive '
                                                     f'treatment')

                elif df.at[person_id, f'{prefix}_htn_disorders'] == 'eclampsia':
                    logger.debug(key='message', data=f'mother {person_id} has not had their eclampsia identified '
                                                     f'during delivery and will not be treated')

    def assessment_and_treatment_of_obstructed_labour_via_avd(self, hsi_event, facility_type):
        """
        This function represents the diagnosis and management of obstructed labour during labour. This function
        defines the required consumables, uses the dx_test to determine a woman with obstructed labour is correctly
        identified, and administers the intervention if so. The intervention in this function is assisted vaginal
        delivery. It is called by either HSI_Labour_PresentsForSkilledBirthAttendanceInLabour
        :param hsi_event: HSI event in which the function has been called:
        :param facility_type: type of facility this intervention is being delivered in
        (STR) 'hc' == health centre, 'hp' == hospital
        """
        df = self.sim.population.props
        mni = self.sim.modules['PregnancySupervisor'].mother_and_newborn_info
        params = self.current_parameters
        person_id = hsi_event.target
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']

        if 'assessment_and_treatment_of_obstructed_labour' not in params['allowed_interventions']:
            return
        else:
            pkg_code_obstructed_labour = pd.unique(
                consumables.loc[consumables['Intervention_Pkg'] == 'Management of obstructed labour',
                                'Intervention_Pkg_Code'])[0]

            item_code_forceps = pd.unique(consumables.loc[consumables['Items'] == 'Forceps, obstetric', 'Item_Code'])[0]
            item_code_vacuum = pd.unique(consumables.loc[consumables['Items'] == 'Vacuum, obstetric', 'Item_Code'])[0]

            consumables_obstructed_labour = {'Intervention_Package_Code': {pkg_code_obstructed_labour: 1},
                                             'Item_Code': {item_code_forceps: 1, item_code_vacuum: 1}}

            # todo: review the contents of this consumables pkg

            outcome_of_request_for_consumables_ol = self.sim.modules['HealthSystem'].request_consumables(
                hsi_event=hsi_event,
                cons_req_as_footprint=consumables_obstructed_labour)

            if self.sim.modules['HealthSystem'].dx_manager.run_dx_test(dx_tests_to_run=f'assess_'
                                                                                       f'obstructed_'
                                                                                       f'labour_{facility_type}',
                                                                       hsi_event=hsi_event):

                # If the general package is available AND the facility has the correct tools to carry out the delivery
                # then it can occur
                if (outcome_of_request_for_consumables_ol['Intervention_Package_Code'][pkg_code_obstructed_labour]) \
                    and ((outcome_of_request_for_consumables_ol['Item_Code'][item_code_forceps]) or
                         (outcome_of_request_for_consumables_ol['Item_Code'][item_code_vacuum])):

                    logger.debug(key='message', data=f'mother {person_id} has had her obstructed labour identified'
                                                     f'during delivery. Staff will attempt an assisted vaginal delivery'
                                                     f'as the equipment is available')

                    # We assume women with CPD cannot be delivered via AVD and will require a caesarean
                    if mni[person_id]['cpd']:
                        treatment_success = False
                    else:
                        treatment_success = params['prob_successful_assisted_vaginal_delivery'] > \
                                            self.rng.random_sample()

                    # If AVD was successful then we record the mode of delivery. We use this variable to reduce risk of
                    # intrapartum still birth when applying risk in the death event
                    if treatment_success:
                        mni[person_id]['mode_of_delivery'] = 'instrumental'

                    # Otherwise if the delivery is unsuccessful woman will be referred for a caesarean
                    else:
                        logger.debug(key='message', data=f'Following a failed assisted vaginal delivery other '
                                                         f'{person_id} will need additional treatment')

                        mni[person_id]['referred_for_cs'] = True

            elif df.at[person_id, 'la_obstructed_labour']:
                logger.debug(key='message', data=f'mother {person_id} has not had their obstructed labour identified '
                                                 f'during delivery and will not be treated')

    def assessment_and_treatment_of_maternal_sepsis(self, hsi_event, facility_type, labour_stage):
        """
        This function represents the diagnosis and management of maternal sepsis during or following labour. This
        function defines the required consumables, uses the dx_test to determine a woman with sepsis is correctly
        identified, and administers the intervention if so. The intervention in this function is maternal sepsis case
        management. It is called by either HSI_Labour_PresentsForSkilledBirthAttendanceInLabour
        or HSI_Labour_ReceivesCareForPostpartumPeriod
        :param hsi_event: HSI event in which the function has been called:
        :param facility_type: type of facility this intervention is being delivered in
        (STR) 'hc' == health centre, 'hp' == hospital
        :param labour_stage: intrapartum or postpartum period of labour (STR) 'ip' or 'pp':
        """
        df = self.sim.population.props
        params = self.current_parameters
        person_id = hsi_event.target
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']

        if 'assessment_and_treatment_of_maternal_sepsis' not in params['allowed_interventions']:
            return
        else:

            pkg_code_sepsis = pd.unique(
                consumables.loc[consumables['Intervention_Pkg'] == 'Maternal sepsis case management',
                                'Intervention_Pkg_Code'])[0]

            # todo: review consumables pkg, 5+ abx for a single case

            all_available = hsi_event.get_all_consumables(
                pkg_codes=[pkg_code_sepsis])

            if self.sim.modules['HealthSystem'].dx_manager.run_dx_test(dx_tests_to_run=f'assess_'
                                                                                       f'sepsis_{facility_type}_'
                                                                                       f'{labour_stage}',
                                                                       hsi_event=hsi_event):

                # If delivered this intervention reduces a womans risk of dying from sepsis
                if all_available:
                    logger.debug(key='message', data=f'mother {person_id} has has their sepsis identified during '
                                                     f'delivery. As consumables are available they will receive '
                                                     f'treatment')

                    df.at[person_id, 'la_sepsis_treatment'] = True

            elif df.at[person_id, 'la_sepsis'] or df.at[person_id, 'la_sepsis_pp']:
                logger.debug(key='message', data=f'mother {person_id} has not had their sepsis identified during '
                                                 f'delivery and will not be treated')

    def assessment_and_plan_for_antepartum_haemorrhage(self, hsi_event, facility_type):
        """
        This function represents the diagnosis of antepartum haemorrhage during  labour. This
        function uses the dx_test to determine a woman with antepartum haemorrhage is correctly identified, and ensure
        that woman is referred for comprehensive care via caesarean section and blood transfusion.
        It is called by either HSI_Labour_PresentsForSkilledBirthAttendanceInLabour
        :param hsi_event: HSI event in which the function has been called:
        :param facility_type: type of facility this intervention is being delivered in
        (STR) 'hc' == health centre, 'hp' == hospital
        """
        df = self.sim.population.props
        params = self.current_parameters
        mni = self.sim.modules['PregnancySupervisor'].mother_and_newborn_info
        person_id = hsi_event.target

        if 'assessment_and_plan_for_referral_antepartum_haemorrhage' not in params['allowed_interventions']:
            return
        else:
            # We assume that any woman who has been referred from antenatal inpatient care due to haemorrhage are
            # automatically scheduled for blood transfusion
            if (df.at[person_id, 'ps_antepartum_haemorrhage'] != 'none') and (df.at[person_id,
                                                                                    'ac_admitted_for_immediate_'
                                                                                    'delivery'] != 'none'):

                mni[person_id]['referred_for_blood'] = True
                logger.debug(key='message', data=f'mother {person_id} who was admitted for treatment following an '
                                                 f'antepartum haemorrhage will be referred for treatment ')

            else:
                # Otherwise the same format is followed, women are evaluated via the dx_test function. If bleeding is
                # determined then they will be referred for caesarean delivery

                # Caesarean delivery reduces the risk of intrapartum still birth and blood transfusion reduces the risk
                # of maternal death due to bleeding
                if self.sim.modules['HealthSystem'].dx_manager.run_dx_test(dx_tests_to_run=f'assess_'
                                                                                           f'aph_{facility_type}',
                                                                           hsi_event=hsi_event):
                    mni[person_id]['referred_for_cs'] = True
                    mni[person_id]['referred_for_blood'] = True

                    logger.debug(key='message', data=f'mother {person_id} has has their antepartum haemorrhage '
                                                     f'identified during delivery. They will now be referred for '
                                                     f'additional treatment')

                elif df.at[person_id, 'la_antepartum_haem'] != 'none':
                    logger.debug(key='message', data=f'mother {person_id} has not had their antepartum haemorrhage '
                                                     f'identified during delivery and will not be referred for '
                                                     f'treatment')

    def assessment_for_referral_uterine_rupture(self, hsi_event, facility_type):
        """
        This function represents the diagnosis of uterine rupture during  labour. This
        function uses the dx_test to determine a woman with uterine rupture is correctly identified, and ensure
        that woman is referred for comprehensive care via caesarean section, surgical repair and blood transfusion.
        It is called by either HSI_Labour_PresentsForSkilledBirthAttendanceInLabour
        :param hsi_event: HSI event in which the function has been called:
        :param facility_type: type of facility this intervention is being delivered in
        (STR) 'hc' == health centre, 'hp' == hospital
        """
        df = self.sim.population.props
        params = self.current_parameters
        mni = self.sim.modules['PregnancySupervisor'].mother_and_newborn_info
        person_id = hsi_event.target

        if 'assessment_and_plan_for_referral_uterine_rupture' not in params['allowed_interventions']:
            return
        else:
            if self.sim.modules['HealthSystem'].dx_manager.run_dx_test(dx_tests_to_run=f'assess_'
                                                                                       f'uterine_'
                                                                                       f'rupture_{facility_type}',
                                                                       hsi_event=hsi_event):
                logger.debug(key='message', data='mother %d has has their uterine rupture identified during delivery. '
                                                 'They will now be referred for additional treatment')

                mni[person_id]['referred_for_surgery'] = True
                mni[person_id]['referred_for_cs'] = True
                mni[person_id]['referred_for_blood'] = True

            elif df.at[person_id, 'la_uterine_rupture']:
                logger.debug(key='message', data=f'mother {person_id} has not had their uterine_rupture identified '
                                                 f'during delivery and will not be referred for treatment')

    def active_management_of_the_third_stage_of_labour(self, hsi_event):
        """
        This function represents the administration of active management of the third stage of labour. This
        function checks the availibility of consumables and delivers the intervention accordingly. It is called by
        HSI_Labour_PresentsForSkilledBirthAttendanceFollowingLabour
        :param hsi_event: HSI event in which the function has been called:
        """
        mni = self.sim.modules['PregnancySupervisor'].mother_and_newborn_info
        params = self.current_parameters
        person_id = hsi_event.target
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']

        if 'active_management_of_the_third_stage_of_labour' not in params['allowed_interventions']:
            return
        else:
            item_code_oxytocin = pd.unique(consumables.loc[consumables['Items'] == 'Oxytocin, injection, '
                                                                                   '10 IU in 1 ml ampoule',
                                                                                   'Item_Code'])[0]
            item_code_needle = pd.unique(
                consumables.loc[consumables['Items'] == 'Syringe, needle + swab', 'Item_Code'])[0]

            consumables_amtsl = {'Intervention_Package_Code': {},
                                 'Item_Code': {item_code_oxytocin: 1, item_code_needle: 1}}

            outcome_of_request_for_consumables_amtsl = self.sim.modules['HealthSystem'].request_consumables(
                hsi_event=hsi_event,
                cons_req_as_footprint=consumables_amtsl)

            # This treatment reduces a womans risk of developing uterine atony AND retained placenta, both of which are
            # preceding causes of postpartum haemorrhage
            if outcome_of_request_for_consumables_amtsl['Item_Code'][item_code_oxytocin]:
                logger.debug(key='message', data=f'mother {person_id} did not receive active management of the third '
                                                 f'stage of labour')
                mni[person_id]['amtsl_given'] = True
            else:
                logger.debug(key='message', data=f'mother {person_id} did not receive active management of the third '
                                                 f'stage of labour')

    def assessment_and_treatment_of_pph_uterine_atony(self, hsi_event, facility_type):
        """
        This function represents the diagnosis and management of postpartum haemorrhage secondary to uterine atony
        following labour. This function defines the required consumables, uses the dx_test to determine a woman with
        PPH/UA is correctly identified, and administers the intervention if so. The intervention in this function is
        intravenous uterotonics followed by referral for further care in the event of continued haemorrhage.
         It is called by HSI_Labour_ReceivesCareForPostpartumPeriod
        :param hsi_event: HSI event in which the function has been called:
        :param facility_type: type of facility this intervention is being delivered in
        (STR) 'hc' == health centre, 'hp' == hospital
        """
        mni = self.sim.modules['PregnancySupervisor'].mother_and_newborn_info
        params = self.current_parameters
        person_id = hsi_event.target
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']

        if 'assessment_and_treatment_of_pph_uterine_atony' not in params['allowed_interventions']:
            return
        else:

            pkg_code_pph = pd.unique(
                consumables.loc[consumables['Intervention_Pkg'] == 'Treatment of postpartum hemorrhage',
                                'Intervention_Pkg_Code'])[0]

            consumables_needed_pph = {'Intervention_Package_Code': {pkg_code_pph: 1}, 'Item_Code': {}}

            outcome_of_request_for_consumables_pph = self.sim.modules['HealthSystem'].request_consumables(
                hsi_event=hsi_event,
                cons_req_as_footprint=consumables_needed_pph)

            if self.sim.modules['HealthSystem'].dx_manager.run_dx_test(dx_tests_to_run=f'assess_pph_{facility_type}',
                                                                       hsi_event=hsi_event):
                if outcome_of_request_for_consumables_pph:

                    # We apply a probability that this treatment will stop a womans bleeding in the first instance
                    # meaning she will not require further treatment
                    if params['prob_haemostatis_uterotonics'] > self.rng.random_sample():
                        logger.debug(key='msg', data=f'mother {person_id} received uterotonics for her PPH which has '
                                                     f'resolved')

                        # We store the treatment in a bitset column, where it will be used to reduce risk of death
                        # after PPH. Additionally she is referred for blood which also reduces risk of death
                        self.pph_treatment.set([person_id], 'uterotonics')
                        mni[person_id]['referred_for_blood'] = True

                    # If uterotonics do not stop bleeding the woman is referred for additional treatment
                    else:
                        logger.debug(key='msg',
                                     data=f'mother {person_id} received uterotonics for her PPH which has not'
                                          f' resolved and she will need additional treatment')
                        mni[person_id]['referred_for_surgery'] = True
                        mni[person_id]['referred_for_blood'] = True
                        return True

    def assessment_and_treatment_of_pph_retained_placenta(self, hsi_event, facility_type):
        """
        This function represents the diagnosis and management of postpartum haemorrhage secondary to retained placenta
        following labour. This function defines the required consumables, uses the dx_test to determine a woman with
        PPH/RP is correctly identified, and administers the intervention if so. The intervention in this function is
        manual removal of placenta (bedside) followed by referral for further care in the event of continued
        haemorrhage. It is called by HSI_Labour_ReceivesCareForPostpartumPeriod
        :param hsi_event: HSI event in which the function has been called:
        :param facility_type: type of facility this intervention is being delivered in
        (STR) 'hc' == health centre, 'hp' == hospital
        """
        mni = self.sim.modules['PregnancySupervisor'].mother_and_newborn_info
        params = self.current_parameters
        person_id = hsi_event.target
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']

        if 'assessment_and_treatment_of_pph_retained_placenta' not in params['allowed_interventions']:
            return
        else:
            pkg_code_pph = pd.unique(consumables.loc[consumables['Intervention_Pkg'] == 'Treatment of postpartum '
                                                                                        'hemorrhage',
                                                     'Intervention_Pkg_Code'])[0]

            all_available = hsi_event.get_all_consumables(
                pkg_codes=[pkg_code_pph])

            if self.sim.modules['HealthSystem'].dx_manager.run_dx_test(dx_tests_to_run=f'assess_pph_{facility_type}',
                                                                       hsi_event=hsi_event):
                if all_available:

                    # Similar to uterotonics we apply a probability that this intervention will successfully stop
                    # bleeding to ensure some women go on to require further care
                    if params['prob_successful_manual_removal_placenta'] > self.rng.random_sample():
                        logger.debug(key='msg', data=f'mother {person_id} undergone MRP due to retained placenta and '
                                                     f'her PPH has resolved')
                        self.pph_treatment.set([person_id], 'manual_removal_placenta')
                        mni[person_id]['referred_for_blood'] = True

                    else:
                        logger.debug(key='msg',
                                     data=f'mother {person_id} undergone MRP due to retained placenta and her PPH has '
                                          f'not resolved- she will need further treatment')
                        mni[person_id]['referred_for_surgery'] = True
                        mni[person_id]['referred_for_blood'] = True

    def surgical_management_of_pph(self, hsi_event):
        """
        This function represents the surgical management of postpartum haemorrhage (all-cause) following labour. This
        function is either called during HSI_Labour_ReceivesComprehensiveEmergencyObstetricCare or
        HSI_Labour_ReceivesCareFollowingCaesareanSection for women who have PPH and medical management has failed.
        :param hsi_event: HSI event in which the function has been called
        """
        person_id = hsi_event.target
        df = self.sim.population.props
        params = self.current_parameters
        mni = self.sim.modules['PregnancySupervisor'].mother_and_newborn_info

        if df.at[person_id, 'la_postpartum_haem'] and mni[person_id]['uterine_atony'] \
            and not self.pph_treatment.has_all(person_id, 'uterotonics'):

            # We apply a probability that surgical techniques will be effective
            treatment_success_pph = params['success_rate_pph_surgery'] > self.rng.random_sample()

            # And store the treatment which will dramatically reduce risk of death
            if treatment_success_pph:
                logger.debug(key='msg',
                                 data=f'mother {person_id} undergone surgery to manage her PPH which resolved')
                self.pph_treatment.set(person_id, 'surgery')

            # If the treatment is unsuccessful then women will require a hysterectomy to stop the bleeding
            elif ~treatment_success_pph:
                logger.debug(key='msg', data=f'mother {person_id} undergone surgery to manage her PPH, she required'
                                             f' a hysterectomy to stop the bleeding')

                self.pph_treatment.set(person_id, 'hysterectomy')
                df.at[person_id, 'la_has_had_hysterectomy'] = True

        # Next we apply the effect of surgical treatment for women with retained placenta
        if df.at[person_id, 'la_postpartum_haem'] and mni[person_id]['retained_placenta'] \
            and not self.pph_treatment.has_all(person_id, 'manual_removal_placenta'):

            self.pph_treatment.set(person_id, 'surgery')
            logger.debug(key='msg',
                         data=f'mother {person_id} undergone surgical removal of a retained placenta ')

    def blood_transfusion(self, hsi_event):
        """
        This function represents the blood transfusion during or after labour. This
        function is either called during HSI_Labour_ReceivesComprehensiveEmergencyObstetricCare or
        HSI_Labour_ReceivesCareFollowingCaesareanSection for women who have experience blood loss due to antepartum
        haemorrhage, postpartum haemorrhage or uterine rupture
        :param hsi_event: HSI event in which the function has been called
        """
        person_id = hsi_event.target
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        mni = self.sim.modules['PregnancySupervisor'].mother_and_newborn_info

        item_code_bt1 = pd.unique(consumables.loc[consumables['Items'] == 'Blood, one unit', 'Item_Code'])[0]
        item_code_bt2 = \
            pd.unique(consumables.loc[consumables['Items'] == 'Lancet, blood, disposable', 'Item_Code'])[0]
        item_code_bt3 = pd.unique(consumables.loc[consumables['Items'] == 'Test, hemoglobin', 'Item_Code'])[0]
        item_code_bt4 = pd.unique(consumables.loc[consumables['Items'] == 'IV giving/infusion set, with needle',
                                                  'Item_Code'])[0]

        consumables_needed_bt = {'Intervention_Package_Code': {}, 'Item_Code': {item_code_bt1: 2, item_code_bt2: 1,
                                                                                item_code_bt3: 1, item_code_bt4: 2}}

        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=hsi_event, cons_req_as_footprint=consumables_needed_bt)

        # If they're available, the event happens
        if outcome_of_request_for_consumables['Item_Code'][item_code_bt1]:
            mni[person_id]['received_blood_transfusion'] = True
            logger.debug(key='message', data=f'Mother {person_id} has received a blood transfusion due following a'
                                             f' maternal haemorrhage')
        else:
            logger.debug(key='message', data=f'Mother {person_id} was unable to receive a blood transfusion due to '
                                             f'insufficient consumables')

    def interventions_delivered_pre_discharge(self, hsi_event):
        """
        This function contains the interventions that are delivered to women prior to discharge. This are considered
        part of essential postnatal care and currently include testing for HIV and postnatal iron and folic acid
        supplementation. It is called by HSI_Labour_ReceivesCareForPostpartumPeriod
        :param hsi_event: HSI event in which the function has been called:
        """
        df = self.sim.population.props
        person_id = int(hsi_event.target)
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']

        if 'Hiv' in self.sim.modules.keys():
            if ~df.at[person_id, 'hv_diagnosed']:
                self.sim.modules['HealthSystem'].schedule_hsi_event(
                    HSI_Hiv_TestAndRefer(person_id=person_id, module=self.sim.modules['Hiv']),
                    topen=self.sim.date,
                    tclose=None,
                    priority=0)

        # ------------------------------- Postnatal iron and folic acid ---------------------------------------------
        item_code_iron_folic_acid = pd.unique(
            consumables.loc[consumables['Items'] == 'Ferrous Salt + Folic Acid, tablet, 200 + 0.25 mg', 'Item_Code'])[0]

        consumables_iron = {
            'Intervention_Package_Code': {},
            'Item_Code': {item_code_iron_folic_acid: 93}}

        # Check there availability
        outcome_of_request_for_consumables = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=hsi_event,
            cons_req_as_footprint=consumables_iron)

        # Women are started on iron and folic acid for the next three months which reduces risk of anaemia in the
        # postnatal period
        if outcome_of_request_for_consumables['Item_Code'][item_code_iron_folic_acid]:
            df.at[person_id, 'la_iron_folic_acid_postnatal'] = True

        # TODO: link up with Tara and EPI module to code postnatal immunisation


class LabourOnsetEvent(Event, IndividualScopeEventMixin):
    """
    This is the LabourOnsetEvent. It is scheduled by the set_date_of_labour function for all women who are newly
    pregnant. It represents the start of a womans labour and is the first event all woman who are going to give birth
    pass through - regardless of mode of delivery or if they are already an inpatient. This event performs a number of
    different functions including populating the mni dictionary to store additional variables important to labour
    and HSIs, determining if and where a woman will seek care for delivery, schedules the LabourAtHome event and the
    HSI_Labour_PresentsForSkilledAttendance at birth (depending on care seeking), the BirthEvent and the
    LabourDeathEvent.
    """

    def __init__(self, module, individual_id):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props
        params = self.module.current_parameters
        mni = self.sim.modules['PregnancySupervisor'].mother_and_newborn_info

        # First we use this check to determine if labour can precede (includes checks on is_alive)
        if self.module.check_labour_can_proceed(individual_id):

            # We indicate this woman is now in labour using this property, and by adding her individual ID to our
            # labour list (for testing)
            df.at[individual_id, 'la_currently_in_labour'] = True
            self.module.women_in_labour.append(individual_id)

            # We then run the labour_characteristics_checker as a final check that only appropriate women are here
            self.module.labour_characteristics_checker(individual_id)

            # Append labor specific variables to the mni
            labour_variables = {'labour_state': None,
                                # Term Labour (TL), Early Preterm (EPTL), Late Preterm (LPTL) or Post Term (POTL)
                                'birth_weight': 'normal_birth_weight',
                                'birth_size': 'average_for_gestational_age',
                                'delivery_setting': None,  # home_birth, health_centre, hospital
                                'twin_count': 0,
                                'twin_one_comps': False,
                                'sought_care_for_twin_one': False,
                                'bf_status_twin_one': 'none',
                                'eibf_status_twin_one': False,
                                'delayed_pp_infection': False,
                                'onset_of_delayed_inf': 0,
                                'corticosteroids_given': False,
                                'clean_birth_practices': False,
                                'abx_for_prom_given': False,
                                'abx_for_pprom_given': False,
                                'chorio_lab': False,
                                'retained_placenta': False,
                                'uterine_atony': False,
                                'amtsl_given': False,
                                'cpd': False,
                                'mode_of_delivery': 'vaginal_delivery',
                                # vaginal_delivery, instrumental, caesarean_section
                                'squeeze_to_high_for_hsi': False,  # True (T) or False (F)
                                'squeeze_to_high_for_hsi_pp': False,  # True (T) or False (F)
                                'sought_care_for_complication': False,  # True (T) or False (F)
                                'sought_care_labour_phase': 'none',  # none, intrapartum, postpartum
                                'referred_for_cs': False,  # True (T) or False (F)
                                'referred_for_blood': False,  # True (T) or False (F)
                                'received_blood_transfusion': False,  # True (T) or False (F)
                                'referred_for_surgery': False,  # True (T) or False (F)'
                                'death_in_labour': False,  # True (T) or False (F)
                                'cause_of_death_in_labour': [],
                                'single_twin_still_birth': False,  # True (T) or False (F)
                                'death_postpartum': False,  # True (T) or False (F)
                                }

            mni[individual_id].update(labour_variables)

            # ===================================== LABOUR STATE  =====================================================
            # Next we categories each woman according to her gestation age at delivery. These categories include term
            # (37-42 weeks gestational age), post term (42 weeks plus), early preterm (24-33 weeks) and late preterm
            # (34-36 weeks)

            # First we calculate foetal age - days from conception until todays date and then add 2 weeks to calculate
            # gestational age
            foetal_age_in_days = (self.sim.date - df.at[individual_id, 'date_of_last_pregnancy']).days
            gestational_age_in_days = foetal_age_in_days + 14

            # We use parameters containing the upper and lower limits, in days, that a mothers pregnancy has to be to be
            # categorised accordingly
            if params['list_limits_for_defining_term_status'][0] <= gestational_age_in_days <= \
                params['list_limits_for_defining_term_status'][1]:

                self.module.labour_tracker['term'] += 1
                mni[individual_id]['labour_state'] = 'term_labour'

            # Here we allow a woman to go into early preterm labour with a gestational age of 23 (limit is 24) to
            # account for PregnancySupervisor only updating weekly
            elif params['list_limits_for_defining_term_status'][2] <= gestational_age_in_days <= \
                params['list_limits_for_defining_term_status'][3]:

                mni[individual_id]['labour_state'] = 'early_preterm_labour'
                self.module.labour_tracker['early_preterm'] += 1
                df.at[individual_id, 'la_has_previously_delivered_preterm'] = True

            elif params['list_limits_for_defining_term_status'][4] <= gestational_age_in_days <= \
                params['list_limits_for_defining_term_status'][5]:

                mni[individual_id]['labour_state'] = 'late_preterm_labour'
                self.module.labour_tracker['late_preterm'] += 1
                df.at[individual_id, 'la_has_previously_delivered_preterm'] = True

            elif gestational_age_in_days >= params['list_limits_for_defining_term_status'][6]:

                mni[individual_id]['labour_state'] = 'postterm_labour'
                self.module.labour_tracker['post_term'] += 1

            # We check all women have had their labour state set
            assert mni[individual_id]['labour_state'] is not None
            labour_state = mni[individual_id]['labour_state']
            logger.debug(key='message', data=f'This is LabourOnsetEvent, person {individual_id} has now gone into '
                                             f'{labour_state} on date {self.sim.date}')

            # ----------------------------------- FOETAL WEIGHT/BIRTH WEIGHT ----------------------------------------
            # Here we determine the weight of the foetus being carried by this mother, this is calculated here to allow
            # the size of the baby to effect risk of certain maternal complications (mostly obstructed labour)
            if df.at[individual_id, 'ps_gestational_age_in_weeks'] < 24:
                mean_birth_weight_list_location = 1
            else:
                mean_birth_weight_list_location = int(min(41, df.at[individual_id, 'ps_gestational_age_in_weeks']) - 24)

            standard_deviation = params['standard_deviation_birth_weights'][mean_birth_weight_list_location]

            # We randomly draw this newborns weight from a normal distribution around the mean for their gestation
            birth_weight = self.module.rng.normal(loc=params['mean_birth_weights'][mean_birth_weight_list_location],
                                                  scale=standard_deviation)

            # Then we calculate the 10th and 90th percentile, these are the case definition for 'small for gestational
            # age and 'large for gestational age'
            small_for_gestational_age_cutoff = scipy.stats.norm.ppf(
                0.1, loc=params['mean_birth_weights'][mean_birth_weight_list_location], scale=standard_deviation)

            large_for_gestational_age_cutoff = scipy.stats.norm.ppf(
                0.9, loc=params['mean_birth_weights'][mean_birth_weight_list_location], scale=standard_deviation)

            # Make the appropriate changes to the mni dictionary (both are stored as property of the newborn on birth)
            if birth_weight >= 4000:
                mni[individual_id]['birth_weight'] = 'macrosomia'
            elif birth_weight >= 2500:
                mni[individual_id]['birth_weight'] = 'normal_birth_weight'
            elif 1500 <= birth_weight < 2500:
                mni[individual_id]['birth_weight'] = 'low_birth_weight'
            elif 1000 <= birth_weight < 1500:
                mni[individual_id]['birth_weight'] = 'very_low_birth_weight'
            elif birth_weight < 1000:
                mni[individual_id]['birth_weight'] = 'extremely_low_birth_weight'

            if birth_weight < small_for_gestational_age_cutoff:
                mni[individual_id]['birth_size'] = 'small_for_gestational_age'
            elif birth_weight > large_for_gestational_age_cutoff:
                mni[individual_id]['birth_size'] = 'large_for_gestational_age'
            else:
                mni[individual_id]['birth_size'] = 'average_for_gestational_age'

            # ===================================== CARE SEEKING AND DELIVERY SETTING ================================
            # Next we determine if women who are now in labour will seek care for delivery. We assume women who have
            # been admitted antenatally for delivery will be delivering in hospital and that is scheduled accordingly

            if df.at[individual_id, 'ac_admitted_for_immediate_delivery'] == 'none':

                # Here we calculate this womans predicted risk of home birth and health centre birth
                pred_hb_delivery = self.module.la_linear_models['probability_delivery_at_home'].predict(
                    df.loc[[individual_id]])[individual_id]
                pred_hc_delivery = self.module.la_linear_models['probability_delivery_health_centre'].predict(
                    df.loc[[individual_id]])[individual_id]

                # The denominator is calculated
                denom = 1 + pred_hb_delivery + pred_hc_delivery

                # Followed by the probability of each of the three outcomes - home birth, health centre birth or
                # hospital birth
                prob_hb = pred_hb_delivery / denom
                prob_hc = pred_hc_delivery / denom
                prob_hp = 1 / denom

                # And a probability weighted random draw is used to determine where the woman will deliver
                facility_types = ['home_birth', 'health_centre', 'hospital']

                # This allows us to simply manipulate care seeking during labour test file
                if mni[individual_id]['test_run']:
                    probabilities = params['test_care_seeking_probs']
                else:
                    probabilities = [prob_hb, prob_hc, prob_hp]

                mni[individual_id]['delivery_setting'] = self.module.rng.choice(facility_types, p=probabilities)

            else:
                # We assume all women with complications will deliver in a hospital
                mni[individual_id]['delivery_setting'] = 'hospital'

            # Check all women's 'delivery setting' is set
            assert mni[individual_id]['delivery_setting'] is not None

            # Women delivering at home move the the LabourAtHomeEvent as they will not receive skilled birth attendance
            if mni[individual_id]['delivery_setting'] == 'home_birth':
                self.sim.schedule_event(LabourAtHomeEvent(self.module, individual_id), self.sim.date)

                logger.info(key='message', data=f'This is LabourOnsetEvent, person {individual_id} as they has chosen '
                                                f'not to seek care at a health centre for delivery and will give birth '
                                                f'at home on date {self.sim.date}')

            # Otherwise the appropriate HSI is scheduled
            elif mni[individual_id]['delivery_setting'] == 'health_centre':
                health_centre_delivery = HSI_Labour_ReceivesSkilledBirthAttendanceDuringLabour(
                    self.module, person_id=individual_id, facility_level_of_this_hsi=1)
                self.sim.modules['HealthSystem'].schedule_hsi_event(health_centre_delivery, priority=0,
                                                                    topen=self.sim.date,
                                                                    tclose=self.sim.date + DateOffset(days=1))

                logger.info(key='message', data=f'This is LabourOnsetEvent, scheduling '
                                                f'HSI_Labour_PresentsForSkilledAttendanceInLabour on date '
                                                f'{self.sim.date} for person {individual_id} as they have chosen to '
                                                f'seek care at a health centre for delivery')

            # TODO: hospital care could be delivered at level 1,2 or 3 but at the moment it made sense to limit to
            #  1 and 2. need to look at the data to make sure the right number of women are at district hospitals
            #  versus higher level hospitals

            elif mni[individual_id]['delivery_setting'] == 'hospital':
                facility_level = int(self.module.rng.choice([1, 2]))
                hospital_delivery = HSI_Labour_ReceivesSkilledBirthAttendanceDuringLabour(
                    self.module, person_id=individual_id, facility_level_of_this_hsi=facility_level)
                self.sim.modules['HealthSystem'].schedule_hsi_event(hospital_delivery, priority=0,
                                                                    topen=self.sim.date,
                                                                    tclose=self.sim.date + DateOffset(days=1))
                logger.info(key='message', data=f'This is LabourOnsetEvent, scheduling '
                                                f'HSI_Labour_PresentsForSkilledAttendanceInLabour on date '
                                                f'{self.sim.date} for person {individual_id} as they have chosen to '
                                                f'seek care at a hospital for delivery')

            # ======================================== SCHEDULING BIRTH AND DEATH EVENTS ============================
            # We schedule all women to move through both the death and birth event.

            # The death event is scheduled to happen after a woman has received care OR delivered at home to allow for
            # any treatment effects to mitigate risk of poor outcomes
            self.sim.schedule_event(LabourDeathAndStillBirthEvent(self.module, individual_id), self.sim.date +
                                    DateOffset(days=4))

            logger.debug(key='message', data=f'This is LabourOnsetEvent scheduling a potential death for mother '
                                             f'{individual_id} which will occur on'
                                             f' {self.sim.date + DateOffset(days=4)}')

            # After the death event women move to the Birth Event where, for surviving women and foetus, birth occurs
            # in the simulation
            self.sim.schedule_event(BirthEvent(self.module, individual_id), self.sim.date + DateOffset(days=5))

            logger.debug(key='message', data=f'This is LabourOnsetEvent scheduling a birth on date to mother'
                                             f' {individual_id} which will occur on '
                                             f'{self.sim.date + DateOffset(days=5)}')


class LabourAtHomeEvent(Event, IndividualScopeEventMixin):
    """
    This is the LabourAtHomeEvent. It is scheduled by the LabourOnsetEvent for women who will not
    seek care at a health facility. This event applies the probability that women delivering at home will experience
    complications associated with the intrapartum phase of labour and makes the appropriate changes to the data frame.
     Additionally this event applies a probability that women who develop complications during a home birth may choose
     to seek care from at a health facility. In that case the appropriate HSI is scheduled
     """

    def __init__(self, module, individual_id):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props
        mni = self.sim.modules['PregnancySupervisor'].mother_and_newborn_info
        params = self.module.current_parameters

        if not df.at[individual_id, 'is_alive']:
            return

        # Check only women delivering at home pass through this event and that the right characteristics are present
        assert mni[individual_id]['delivery_setting'] == 'home_birth'
        self.module.labour_characteristics_checker(individual_id)

        logger.debug(key='message', data=f'Mother {individual_id}  is now going to deliver at home')
        self.module.labour_tracker['home_birth'] += 1

        # ===================================  APPLICATION OF COMPLICATIONS ===========================================
        # Using the complication_application function we loop through each complication and determine if a woman
        # will experience any of these if she has delivered at home
        for complication in ['obstruction_cpd', 'obstruction_malpos_malpres', 'obstruction_other',
                             'placental_abruption', 'antepartum_haem', 'sepsis_chorioamnionitis', 'uterine_rupture']:
            self.module.set_intrapartum_complications(individual_id, complication=complication)

        # And we determine if any existing hypertensive disorders would worsen
        self.module.progression_of_hypertensive_disorders(individual_id, property_prefix='ps')

        # ==============================  CARE SEEKING FOLLOWING COMPLICATIONS ========================================
        # Next we determine if women who develop a complication during a home birth will seek care

        # (Women who have been scheduled a home birth after seeking care at a facility that didnt have capacity to
        # deliver the HSI will not try to seek care if they develop a complication)
        if not mni[individual_id]['squeeze_to_high_for_hsi']:

            if df.at[individual_id, 'la_obstructed_labour'] or \
                (df.at[individual_id, 'la_antepartum_haem'] != 'none') or \
                df.at[individual_id, 'la_sepsis'] or \
                (df.at[individual_id, 'ps_htn_disorders'] == 'eclampsia') or \
                (df.at[individual_id, 'ps_htn_disorders'] == 'severe_pre_eclamp') or \
               df.at[individual_id, 'la_uterine_rupture']:

                if self.module.predict(self.module.la_linear_models['care_seeking_for_complication'], individual_id):

                    mni[individual_id]['sought_care_for_complication'] = True
                    mni[individual_id]['sought_care_labour_phase'] = 'intrapartum'

                    # We assume women present to the health system through the generic a&e appointment
                    from tlo.methods.hsi_generic_first_appts import (
                        HSI_GenericEmergencyFirstApptAtFacilityLevel1,
                    )

                    event = HSI_GenericEmergencyFirstApptAtFacilityLevel1(
                        module=self.module,
                        person_id=individual_id)

                    logger.debug(key='message', data=f'mother {individual_id} will now seek care for a complication'
                                                     f' that has developed during labour')
                    self.sim.modules['HealthSystem'].schedule_hsi_event(event,
                                                                        priority=0,
                                                                        topen=self.sim.date,
                                                                        tclose=self.sim.date + DateOffset(days=1))
                else:
                    logger.debug(key='message', data=f'mother {individual_id} will not seek care following a '
                                                     f'complication that has developed during labour')


class BirthEvent(Event, IndividualScopeEventMixin):
    """This is the BirthEvent. It is scheduled by LabourOnsetEvent. For women who survived labour, the appropriate
    variables are reset/updated and the function do_birth is executed. This event schedules PostPartumLabourEvent for
    those women who have survived"""

    def __init__(self, module, mother_id):
        super().__init__(module, person_id=mother_id)

    def apply(self, mother_id):
        df = self.sim.population.props
        person = df.loc[mother_id]
        mni = self.sim.modules['PregnancySupervisor'].mother_and_newborn_info

        # This event tells the simulation that the woman's pregnancy is over and generates the new child in the
        # data frame
        logger.info(key='message', data=f'mother {mother_id} at birth event')

        # Check the correct amount of time has passed between labour onset and birth event and that women at the event
        # have the right characteristics present

        assert (self.sim.date - df.at[mother_id, 'la_due_date_current_pregnancy']) == pd.to_timedelta(5, unit='D')
        self.module.labour_characteristics_checker(mother_id)

        # =============================================== BIRTH ====================================================
        # If the mother is alive and still pregnant we generate a  child and the woman is scheduled to move to the
        # postpartum event to determine if she experiences any additional complications (intrapartum stillbirths still
        # trigger births for monitoring purposes)

        if person.is_alive and person.is_pregnant:
            logger.info(key='message', data=f'A Birth is now occurring, to mother {mother_id}')

            # If the mother is pregnant with twins, we call the do_birth function twice and then link the twins
            # (via sibling id) in the newborn module
            if person.ps_multiple_pregnancy:
                child_one = self.sim.do_birth(mother_id)
                child_two = self.sim.do_birth(mother_id)
                logger.debug(key='message', data=f'Mother {mother_id} will now deliver twins {child_one} & {child_two}')
                self.sim.modules['NewbornOutcomes'].link_twins(child_one, child_two, mother_id)
            else:
                self.sim.do_birth(mother_id)

            # ====================================== SCHEDULING POSTPARTUM EVENTS ======================================
            # Women who have birth at home will next pass to this event, again applying risk of complications
            if mni[mother_id]['delivery_setting'] == 'home_birth':
                self.sim.schedule_event(PostpartumLabourAtHomeEvent(self.module, mother_id), self.sim.date)
                logger.debug(key='message', data=f'This is BirthEvent scheduling PostpartumLabourAtHomeEvent for '
                                                 f'person {mother_id} on date {self.sim.date}')

            # Otherwise we scheduled the next HSI event that manages care immediately after birth
            else:
                # Women who have delivered vaginally pass to HSI_Labour_ReceivesSkilledBirthAttendanceFollowingLabour
                # for management of the postpartum period
                if mni[mother_id]['mode_of_delivery'] == 'vaginal_delivery' or \
                  mni[mother_id]['mode_of_delivery'] == 'instrumental':

                    health_centre_care = HSI_Labour_ReceivesSkilledBirthAttendanceFollowingLabour(
                        self.module, person_id=mother_id, facility_level_of_this_hsi=1)

                    all_facility_care = HSI_Labour_ReceivesSkilledBirthAttendanceFollowingLabour(
                        self.module, person_id=mother_id, facility_level_of_this_hsi=int(
                            self.module.rng.choice([1, 2])))

                    if mni[mother_id]['delivery_setting'] == 'health_centre':
                        logger.debug(key='message', data='This is BirthEvent scheduling HSI_Labour_ReceivesCareFor'
                                                         f'PostpartumPeriod for person {mother_id} on date'
                                                         f' {self.sim.date}')
                        self.sim.modules['HealthSystem'].schedule_hsi_event(health_centre_care,
                                                                            priority=0,
                                                                            topen=self.sim.date,
                                                                            tclose=self.sim.date + DateOffset(days=1))

                    elif mni[mother_id]['delivery_setting'] == 'hospital':
                        logger.info(key='message', data='This is BirthEvent scheduling '
                                                        'HSI_Labour_ReceivesCareForPostpartumPeriod for person '
                                                        f'{mother_id}')

                        self.sim.modules['HealthSystem'].schedule_hsi_event(all_facility_care,
                                                                            priority=0,
                                                                            topen=self.sim.date,
                                                                            tclose=self.sim.date + DateOffset(days=1))
                else:
                    # Women who delivered via c-section are scheduled to a different HSI
                    post_cs_care = HSI_Labour_ReceivesCareFollowingCaesareanSection(
                        self.module, person_id=mother_id, facility_level_of_this_hsi=int(
                            self.module.rng.choice([1, 2])))

                    logger.info(key='message', data='This is BirthEvent scheduling '
                                                    f'HSI_Labour_ReceivesCareFollowingCaesareanSection for person '
                                                    f'{mother_id}who gave birth via caesarean section  on date '
                                                    f'{self.sim.date}')
                    self.sim.modules['HealthSystem'].schedule_hsi_event(post_cs_care,
                                                                        priority=0,
                                                                        topen=self.sim.date,
                                                                        tclose=self.sim.date + DateOffset(days=1))

        # If the mother has died during childbirth the child is still generated with is_alive=false to monitor
        # stillbirth rates. She will not pass through the postpartum complication events
        if ~person.is_alive and ~person.la_intrapartum_still_birth and person.la_maternal_death_in_labour:
            logger.info(key='message', data=f'A Birth is now occurring, to mother {mother_id} who died in childbirth '
                                            f'but her child survived')

            if person.ps_multiple_pregnancy:
                child_one = self.sim.do_birth(mother_id)
                child_two = self.sim.do_birth(mother_id)
                self.sim.modules['NewbornOutcomes'].link_twins(child_one, child_two, mother_id)
            else:
                self.sim.do_birth(mother_id)


class PostpartumLabourAtHomeEvent(Event, IndividualScopeEventMixin):
    """
    This is PostpartumLabourAtHomeEvent. This event is scheduled by BirthEvent for women whose laboured and gave
    birth at home OR HSI_Labour_ReceivesCareForPostpartumPeriod for women who couldnt receive in-facility postpartum
    care due to high squeeze factor. This event applies risk of postpartum complications and determines if women
    experiencing complications will seek care. Finally is calls the apply_risk_of_early_postpartum_death function to
     determine if women experiencing the postpartum phase of labour at home will die.
     """

    def __init__(self, module, individual_id):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props
        mni = self.sim.modules['PregnancySupervisor'].mother_and_newborn_info
        params = self.module.current_parameters

        assert (self.sim.date - df.at[individual_id, 'la_due_date_current_pregnancy']) == pd.to_timedelta(5, unit='D')
        self.module.postpartum_characteristics_checker(individual_id)

        # Event should only run if woman is still alive
        if not df.at[individual_id, 'is_alive']:
            return
        else:
            logger.debug(key='msg', data=f'Mother {individual_id} has arrived at PostpartumLabourAtHomeEvent and it '
                                         f'will run')

        # We first determine if this woman will experience any complications immediately following/ or in the days after
        # birth
        for complication in ['sepsis_endometritis', 'sepsis_urinary_tract', 'sepsis_skin_soft_tissue',
                             'pph_uterine_atony', 'pph_retained_placenta', 'pph_other']:
            self.module.set_postpartum_complications(individual_id, complication=complication)

        self.module.progression_of_hypertensive_disorders(individual_id, property_prefix='pn')

        # For women who experience a complication at home immediately following birth we use a care seeking equation to
        # determine if they will now seek additional care for management of this complication

        # Women who have come home, following a facility delivery, due to high squeeze will not try and seek care
        # for any complications

        if df.at[individual_id, 'la_sepsis_pp'] or (df.at[individual_id, 'ps_htn_disorders'] == 'eclampsia') or \
            (df.at[individual_id, 'ps_htn_disorders'] == 'severe_pre_eclamp') or df.at[individual_id,
                                                                                       'la_postpartum_haem']:

            if not mni[individual_id]['squeeze_to_high_for_hsi_pp'] and\
              (self.module.predict(self.module.la_linear_models['care_seeking_for_complication'], individual_id)):

                # If this woman choses to seek care, she will present to the health system via the generic emergency
                # system and be referred on to receive specific care
                mni[individual_id]['sought_care_for_complication'] = True
                mni[individual_id]['sought_care_labour_phase'] = 'postpartum'

                from tlo.methods.hsi_generic_first_appts import (
                    HSI_GenericEmergencyFirstApptAtFacilityLevel1,
                )

                event = HSI_GenericEmergencyFirstApptAtFacilityLevel1(
                    module=self.module, person_id=individual_id)

                self.sim.modules['HealthSystem'].schedule_hsi_event(
                        event,
                        priority=0,
                        topen=self.sim.date,
                        tclose=self.sim.date + DateOffset(days=1))

                logger.debug(key='message',
                             data=f'mother {individual_id} will now seek care for a complication that'
                                  f'has developed following labour on date {self.sim.date}')
            else:
                logger.debug(key='message', data=f'mother {individual_id} will not seek care for a complication that'
                                                 f'has developed following labour on date {self.sim.date}')

                # For women who dont seek care for complications following birth we immediately apply risk of death
                self.module.apply_risk_of_early_postpartum_death(individual_id)

        else:
            # Women without complications still pass through this event
            self.module.apply_risk_of_early_postpartum_death(individual_id)


class LabourDeathAndStillBirthEvent(Event, IndividualScopeEventMixin):
    """
    This is the LabourDeathAndStillBirthEvent. It is scheduled by the LabourOnsetEvent for all women in the labour
    module following the application of complications (and possibly treatment) for women who have given birth at home
    OR in a facility . This event determines if women who have experienced complications in labour will die or
    experience an intrapartum stillbirth.
    """

    def __init__(self, module, individual_id):
        super().__init__(module, person_id=individual_id)

    def apply(self, individual_id):
        df = self.sim.population.props
        params = self.module.current_parameters
        mni = self.sim.modules['PregnancySupervisor'].mother_and_newborn_info

        if not df.at[individual_id, 'is_alive']:
            return

        # Check the correct amount of time has passed between labour onset and postpartum event
        assert (self.sim.date - df.at[individual_id, 'la_due_date_current_pregnancy']) == pd.to_timedelta(4, unit='D')
        self.module.labour_characteristics_checker(individual_id)

        # We cycle through each complication and apply risk of death using the set_maternal_death_status_intrapartum
        # described above
        if df.at[individual_id, 'ps_htn_disorders'] == 'severe_pre_eclamp':
            self.module.set_maternal_death_status_intrapartum(individual_id, cause='severe_pre_eclamp')

        if df.at[individual_id, 'ps_htn_disorders'] == 'eclampsia':
            self.module.set_maternal_death_status_intrapartum(individual_id, cause='eclampsia')

        if (df.at[individual_id, 'la_antepartum_haem'] == 'none') or \
            ((df.at[individual_id, 'ps_antepartum_haemorrhage'] != 'none') and
             (df.at[individual_id, 'ac_admitted_for_immediate_delivery'] != 'none')):

            self.module.set_maternal_death_status_intrapartum(individual_id, cause='antepartum_haem')

        if df.at[individual_id, 'la_sepsis']:
            self.module.set_maternal_death_status_intrapartum(individual_id, cause='sepsis')

        if df.at[individual_id, 'la_uterine_rupture']:
            self.module.set_maternal_death_status_intrapartum(individual_id, cause='uterine_rupture')

        # Next we determine if this woman will experience an intrapartum still birth. We assume any women who have had
        # to deliver before 24 weeks will experience stillbirth

        # Otherwise all woman have a risk of still birth applied which is increased by complications in labour.
        # Treatment modelled to reduce risk of intrapartum stillbirth include assisted vaginal delivery and
        # caesarean section

        outcome_of_still_birth_equation = self.module.predict(self.module.la_linear_models['intrapartum_still_birth'],
                                                              individual_id)

        if (df.at[individual_id, 'ps_gestational_age_in_weeks'] < 24) or outcome_of_still_birth_equation:
            logger.debug(key='message', data=f'person {individual_id} has experienced an intrapartum still birth')

            random_draw = self.module.rng.random_sample()

            # If this woman will experience a stillbirth and she was not pregnant with twins OR she was pregnant with
            # twins but both twins have died during labour we reset/set the appropriate variables
            if ~df.at[individual_id, 'ps_multiple_pregnancy'] or (df.at[individual_id, 'ps_multiple_pregnancy'] and
                                                                  (random_draw < params['prob_both_twins_ip_still_'
                                                                                        'birth'])):

                df.at[individual_id, 'la_intrapartum_still_birth'] = True
                # This variable is therefore only ever true when the pregnancy has ended in stillbirth
                df.at[individual_id, 'ps_prev_stillbirth'] = True
                df.at[individual_id, 'is_pregnant'] = False

            # If one twin survives we store this as a property of the MNI which is reference on_birth of the newborn
            # outcomes to ensure this twin pregnancy only leads to one birth
            elif (df.at[individual_id, 'ps_multiple_pregnancy'] and (random_draw > params['prob_both_twins_ip_still_'
                                                                                          'birth'])):
                df.at[individual_id, 'ps_prev_stillbirth'] = True
                mni[individual_id]['single_twin_still_birth'] = True

        # For a woman who die (due to the effect of one or more of the above complications) we schedule the death event
        if mni[individual_id]['death_in_labour']:
            logger.info(key='direct_maternal_death', data={'person': individual_id, 'preg_state': 'labour',
                                                           'year': self.sim.date.year})

            self.module.labour_tracker['maternal_death'] += 1
            self.sim.modules['Demography'].do_death(individual_id=individual_id, cause='maternal',
                                                    originating_module=self.sim.modules['Labour'])

            # Log the maternal death
            logger.info(key='message', data=f'This is LabourDeathEvent scheduling a death for person {individual_id} on'
                                            f' date {self.sim.date} who died due to intrapartum complications')

            labour_complications = {'person_id': individual_id,
                                    'death_date': self.sim.date,
                                    'labour_profile': mni[individual_id]}

            logger.info(key='labour_complications', data=labour_complications, description='mni dictionary for a woman '
                                                                                           'who has died in labour')

            if mni[individual_id]['death_in_labour'] and df.at[individual_id, 'la_intrapartum_still_birth']:
                # We delete the mni dictionary if both mother and baby have died in labour, if the mother has died but
                # the baby has survived we delete the dictionary following the on_birth function of NewbornOutcomes
                del mni[individual_id]

        if df.at[individual_id, 'la_intrapartum_still_birth'] or mni[individual_id]['single_twin_still_birth']:
            self.module.labour_tracker['ip_stillbirth'] += 1
            logger.info(key='message', data=f'A Still Birth has occurred, to mother {individual_id}')
            still_birth = {'mother_id': individual_id,
                           'date_of_ip_stillbirth': self.sim.date}

            logger.info(key='intrapartum_stillbirth', data=still_birth, description='record of intrapartum stillbirth')


class HSI_Labour_ReceivesSkilledBirthAttendanceDuringLabour(HSI_Event, IndividualScopeEventMixin):
    """
    This is the HSI_Labour_ReceivesSkilledBirthAttendanceDuringLabour. This event is the first HSI for women who have
    chosen to deliver in a health care facility. Broadly this HSI represents care provided by a skilled birth attendant
    during labour. This event...

    1.) Determines if women will benefit from prophylactic interventions and delivers the interventions
    2.) Applies risk of intrapartum complications
    3.) Determines if women who have experience complications will benefit from treatment interventions and delivers
        the interventions
    4.) Schedules additional comprehensive emergency obstetric care for women who need it. (Comprehensive
        interventions (blood transfusion, caeasarean section and surgery) are housed within a different HSI.)

    Only interventions that can be delivered in BEmONC facilities are delivered in this HSI. These include intravenous
    antibiotics, intravenous anticonvulsants and assisted vaginal delivery. Additionally women may receive
    antihypertensives in line with Malawi's EHP. Interventions will only be attempted to be delivered if the squeeze
    factor of the HSI is below a predetermined threshold of each intervention. CEmONC level interventions are managed
    within HSI_Labour_ReceivesComprehensiveEmergencyObstetricCare
    """

    def __init__(self, module, person_id, facility_level_of_this_hsi):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Labour)

        self.TREATMENT_ID = 'Labour_ReceivesSkilledBirthAttendanceDuringLabour'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'NormalDelivery': 1})
        self.ALERT_OTHER_DISEASES = []
        self.ACCEPTED_FACILITY_LEVEL = facility_level_of_this_hsi
        self.BEDDAYS_FOOTPRINT = self.make_beddays_footprint({'general_bed': 1})

    def apply(self, person_id, squeeze_factor):
        mni = self.sim.modules['PregnancySupervisor'].mother_and_newborn_info
        df = self.sim.population.props
        params = self.module.current_parameters
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']

        if not df.at[person_id, 'is_alive']:
            return

        logger.info(key='message', data=f'This is HSI_Labour_PresentsForSkilledAttendanceInLabour: Mother {person_id} '
                                        f'has presented to a health facility on date {self.sim.date} following the '
                                        f'onset of her labour')

        # First we capture women who have presented to this event during labour at home. Currently we just set these
        # women to be delivering at a health centre (this will need to be randomised to match any availble data)
        if mni[person_id]['delivery_setting'] == 'home_birth' and mni[person_id]['sought_care_for_complication']:
            mni[person_id]['delivery_setting'] = 'health_centre'

        # Next delivery setting is captured in the labour_tracker, processed by the logger and reset yearly
        if mni[person_id]['delivery_setting'] == 'health_centre':
            self.module.labour_tracker['health_centre_birth'] += 1

        elif mni[person_id]['delivery_setting'] == 'hospital':
            self.module.labour_tracker['hospital_birth'] += 1

        logger.info(key='facility_delivery', data={'mother': person_id,
                                                   'facility_type': mni[person_id]['delivery_setting']})

        # Next we check this woman has the right characteristics to be at this event
        self.module.labour_characteristics_checker(person_id)
        assert mni[person_id]['delivery_setting'] != 'home_birth'
        assert (self.sim.date - df.at[person_id, 'la_due_date_current_pregnancy']) < pd.to_timedelta(3, unit='D')

        # We use this variable to prevent repetition when calling functions in this event
        if mni[person_id]['delivery_setting'] == 'health_centre':
            facility_type_code = 'hc'
        else:
            facility_type_code = 'hp'

        if (df.at[person_id, 'ac_admitted_for_immediate_delivery'] == 'caesarean_now') or \
                (df.at[person_id, 'ac_admitted_for_immediate_delivery'] == 'caesarean_later'):
            mni[person_id]['referred_for_cs'] = True

        # LOG CONSUMABLES FOR DELIVERY...
        # We assume all deliveries require this basic package of consumables but we do not condition the event running
        # on their availability
        pkg_code_delivery = pd.unique(
            consumables.loc[consumables['Intervention_Pkg'] == 'Vaginal delivery - skilled attendance',
                            'Intervention_Pkg_Code'])[0]

        self.get_all_consumables(
            pkg_codes=[pkg_code_delivery])

        # ===================================== PROPHYLACTIC CARE ===================================================
        # The following function manages the consumables and administration of prophylactic interventions in labour
        # (clean delivery practice, antibiotics for PROM, steroids for preterm labour). This intervention, like all
        # other in the event will only occur if the squeeze factor is below a preset threshold

        if squeeze_factor < params['squeeze_threshold_proph_ints']:
            self.module.prophylactic_labour_interventions(self)
        else:
            # Otherwise she receives no benefit of prophylaxis
            logger.debug(key='message', data=f'mother {person_id} did not receive prophylactic labour interventions due'
                                             f'to high squeeze')

        # ================================= PROPHYLACTIC MANAGEMENT PRE-ECLAMPSIA  ==============================
        # Next we see if women with severe pre-eclampsia will be identified and treated, reducing their risk of
        # eclampsia
        if squeeze_factor < params['squeeze_threshold_treatment_spe']:
            self.module.assessment_and_treatment_of_severe_pre_eclampsia_mgso4(self, facility_type_code, 'ip')
        else:
            logger.debug(key='message', data=f'mother {person_id} did not receive assessment or treatment of severe '
                                             f'pre-eclampsia due to high squeeze')

        # ===================================== APPLYING COMPLICATION INCIDENCE =======================================
        # Following administration of prophylaxis we assess if this woman will develop any complications during labour.
        # Women who have sought care because of complication have already had these risk applied so it doesnt happen
        # again

        if not mni[person_id]['sought_care_for_complication']:
            for complication in ['obstruction_cpd', 'obstruction_malpos_malpres', 'obstruction_other',
                                 'placental_abruption', 'antepartum_haem', 'sepsis_chorioamnionitis']:

                # Uterine rupture is the only complication we dont apply the risk of here due to the causal link
                # between obstructed labour and uterine rupture. Therefore we want interventions for obstructed labour
                # to reduce the risk of uterine rupture

                self.module.set_intrapartum_complications(person_id, complication=complication)

        self.module.progression_of_hypertensive_disorders(person_id, property_prefix='ps')

        # ======================================= COMPLICATION MANAGEMENT ==========================
        # Next, women in labour are assessed for complications and treatment delivered if a need is identified and
        # consumables are available

        if squeeze_factor < params['squeeze_threshold_treatment_ol']:
            self.module.assessment_and_treatment_of_obstructed_labour_via_avd(self, facility_type_code)

        if squeeze_factor < params['squeeze_threshold_treatment_sep']:
            self.module.assessment_and_treatment_of_maternal_sepsis(self, facility_type_code, 'ip')

        if squeeze_factor < params['squeeze_threshold_treatment_htn']:
            self.module.assessment_and_treatment_of_hypertension(self, facility_type_code, 'ip')

        if squeeze_factor < params['squeeze_threshold_treatment_aph']:
            self.module.assessment_and_plan_for_antepartum_haemorrhage(self, facility_type_code)

        if squeeze_factor < params['squeeze_threshold_treatment_ec']:
            self.module.assessment_and_treatment_of_eclampsia(self, facility_type_code, 'ip')

        # Now we apply the risk of uterine rupture to all women who will deliver vaginally
        if mni[person_id]['mode_of_delivery'] == 'vaginal_delivery' and not mni[person_id]['referred_for_cs']:
            self.module.set_intrapartum_complications(
                person_id, complication='uterine_rupture')

        # Uterine rupture follows the same pattern as antepartum haemorrhage
        if squeeze_factor < params['squeeze_threshold_treatment_ur']:
            self.module.assessment_for_referral_uterine_rupture(self, facility_type_code)

        # ========================================== SCHEDULING CEMONC CARE =========================================
        # Finally women who require additional treatment have the appropriate HSI scheduled to deliver further care

        if mni[person_id]['referred_for_cs'] or \
            mni[person_id]['referred_for_surgery'] or \
           mni[person_id]['referred_for_blood']:

            surgical_management = HSI_Labour_ReceivesComprehensiveEmergencyObstetricCare(
                self.module, person_id=person_id, facility_level_of_this_hsi=self.ACCEPTED_FACILITY_LEVEL,
                timing='intrapartum')
            self.sim.modules['HealthSystem'].schedule_hsi_event(surgical_management,
                                                                priority=0,
                                                                topen=self.sim.date,
                                                                tclose=self.sim.date + DateOffset(days=1))

        # If a this woman has experienced a complication the appointment footprint is changed from normal to
        # complicated
        actual_appt_footprint = self.EXPECTED_APPT_FOOTPRINT

        if df.at[person_id, 'la_sepsis'] or (df.at[person_id, 'la_antepartum_haem'] != 'none') or \
            df.at[person_id, 'la_obstructed_labour'] or df.at[person_id, 'la_uterine_rupture'] \
            or df.at[person_id, 'ps_htn_disorders'] == 'eclampsia' \
           or df.at[person_id, 'ps_htn_disorders'] == 'severe_pre_eclamp':
            actual_appt_footprint['NormalDelivery'] = actual_appt_footprint['CompDelivery']  # todo: is this right?

        return actual_appt_footprint

    def did_not_run(self):
        person_id = self.target
        mni = self.sim.modules['PregnancySupervisor'].mother_and_newborn_info

        # If a woman has chosen to deliver in a facility from the onset of labour, but the squeeze factor is too high,
        # she will be forced to return home to deliver
        if not mni[person_id]['sought_care_for_complication']:
            logger.debug(key='message', data=f'squeeze factor is too high for this event to run for mother {person_id} '
                                             f'on date {self.sim.date} and she will now deliver at home')

            mni[person_id]['delivery_setting'] = 'home_birth'
            mni[person_id]['squeeze_to_high_for_hsi'] = True
            self.sim.schedule_event(LabourAtHomeEvent(self.module, person_id), self.sim.date)

        # If a woman has presented to this event during labour due to a complication, she will not receive any treatment
        if mni[person_id]['sought_care_for_complication']:
            logger.debug(key='message', data=f'squeeze factor is too high for this event to run for mother {person_id} '
                                             f'on date {self.sim.date} and she could not receive care for the '
                                             f'complications developed during her home birth')

        return False

    def not_available(self):
        """This is called when the HSI is passed to the health system scheduler but not scheduled as the TREATMENT_ID
        is not allowed under the 'services_available' parameter of the health system.
        Note that this called at the time of the event being passed to the Health System at schedule_hsi_event(...) and
        not at the time when the HSI is intended to be run (as specified by the 'topen' parameter in that call)"""
        person_id = self.target
        mni = self.sim.modules['PregnancySupervisor'].mother_and_newborn_info
        # If a woman has chosen to deliver in a facility but this event isnt allowed with the set service configuration
        # then she will deliver at home
        logger.debug(key='message', data=f'This event is not in the allowed service availability and therefore cannot '
                                         f'run for mother {person_id} on date {self.sim.date}, she will now deliver at '
                                         f'home')

        if not mni[person_id]['sought_care_for_complication']:
            mni[person_id]['delivery_setting'] = 'home_birth'
            self.sim.schedule_event(LabourAtHomeEvent(self.module, person_id), self.sim.date)


class HSI_Labour_ReceivesSkilledBirthAttendanceFollowingLabour(HSI_Event, IndividualScopeEventMixin):
    """
    This is the HSI HSI_Labour_ReceivesSkilledBirthAttendanceFollowingLabour. This event is scheduled by the
    PostpartumLabourEvent and broadly represents care delivered by a skilled birth attendant following vaginal delivery
    in a BEmONC facility. Care following caesarean delivery is managed in another event. The structure and flow of this
    event is largely similar to HSI_Labour_ReceivesSkilledBirth AttendanceDuringLabour as described above.
    """
    def __init__(self, module, person_id, facility_level_of_this_hsi):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Labour)

        self.TREATMENT_ID = 'Labour_ReceivesSkilledBirthAttendanceFollowingLabour'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'InpatientDays': 1})
        self.ALERT_OTHER_DISEASES = []
        self.ACCEPTED_FACILITY_LEVEL = facility_level_of_this_hsi
        self.BEDDAYS_FOOTPRINT = self.make_beddays_footprint({'general_bed': 2})

    def apply(self, person_id, squeeze_factor):
        mni = self.sim.modules['PregnancySupervisor'].mother_and_newborn_info
        df = self.sim.population.props
        params = self.module.current_parameters

        logger.info(key='message', data='This is HSI_Labour_ReceivesCareForPostpartumPeriodFacilityLevel1: Providing '
                                        f'skilled attendance following birth for person {person_id}')

        if not df.at[person_id, 'is_alive']:
            return

        assert mni[person_id]['mode_of_delivery'] == 'vaginal_delivery' or \
            mni[person_id]['mode_of_delivery'] == 'instrumental'

        if mni[person_id]['delivery_setting'] == 'home_birth' and mni[person_id]['sought_care_for_complication']:
            mni[person_id]['delivery_setting'] = 'health_centre'

        # We run similar checks as the labour HSI
        self.module.postpartum_characteristics_checker(person_id)
        assert mni[person_id]['delivery_setting'] != 'home_birth'

        # Define what type of facility this woman is in
        if mni[person_id]['delivery_setting'] == 'health_centre':
            facility_type_code = 'hc'
        else:
            facility_type_code = 'hp'

        # -------------------------- Active Management of the third stage of labour ----------------------------------
        # Prophylactic treatment to prevent postpartum bleeding is applied
        if not mni[person_id]['sought_care_for_complication'] and squeeze_factor < params['squeeze_threshold_amtsl']:
            self.module.active_management_of_the_third_stage_of_labour(self)
        if not mni[person_id]['sought_care_for_complication'] and squeeze_factor < params['squeeze_threshold_'\
                                                                                          'treatment_spe']:
            self.module.assessment_and_treatment_of_severe_pre_eclampsia_mgso4(self, facility_type_code, 'pp')

        # ===================================== APPLYING COMPLICATION INCIDENCE =======================================
        # Again we use the mothers individual risk of each complication to determine if she will experience any
        # complications using the set_complications_during_facility_birth function.
        if not mni[person_id]['sought_care_for_complication']:

            for complication in ['sepsis_endometritis', 'sepsis_urinary_tract', 'sepsis_skin_soft_tissue',
                                 'pph_uterine_atony', 'pph_retained_placenta', 'pph_other']:
                self.module.set_postpartum_complications(person_id, complication=complication)
            self.module.progression_of_hypertensive_disorders(person_id, property_prefix='pn')

        # ======================================= COMPLICATION MANAGEMENT =============================================
        # And then determine if interventions will be delivered...

        if squeeze_factor < params['squeeze_threshold_treatment_sep']:
            self.module.assessment_and_treatment_of_maternal_sepsis(self, facility_type_code, 'pp')

        if squeeze_factor < params['squeeze_threshold_treatment_htn']:
            self.module.assessment_and_treatment_of_hypertension(self, facility_type_code, 'pp')
            # todo should this come before complications are applied too

        if squeeze_factor < params['squeeze_threshold_treatment_pph']:
            self.module.assessment_and_treatment_of_pph_retained_placenta(self, facility_type_code)
            self.module.assessment_and_treatment_of_pph_uterine_atony(self, facility_type_code)
        if squeeze_factor < params['squeeze_threshold_treatment_ec']:
            self.module.assessment_and_treatment_of_eclampsia(self, facility_type_code, 'pp')

        self.module.interventions_delivered_pre_discharge(self)

        # ========================================== SCHEDULING CeMONC CARE ==========================================
        if mni[person_id]['referred_for_surgery'] or mni[person_id]['referred_for_blood']:
            surgical_management = HSI_Labour_ReceivesComprehensiveEmergencyObstetricCare(
                self.module, person_id=person_id, facility_level_of_this_hsi=self.ACCEPTED_FACILITY_LEVEL,
                timing='postpartum')
            self.sim.modules['HealthSystem'].schedule_hsi_event(surgical_management,
                                                                priority=0,
                                                                topen=self.sim.date,
                                                                tclose=self.sim.date + DateOffset(days=1))

        # ====================================== APPLY RISK OF DEATH===================================================
        # For women who experience a complication following birth in a facility, but do not require additional care,
        # we apply risk of death considering the treatment effect applied (women referred on have this risk applied
        # later)
        if not mni[person_id]['referred_for_surgery'] and not mni[person_id]['referred_for_blood']:
            self.module.apply_risk_of_early_postpartum_death(person_id)

        actual_appt_footprint = self.EXPECTED_APPT_FOOTPRINT
        return actual_appt_footprint

    def did_not_run(self):
        person_id = self.target
        mni = self.sim.modules['PregnancySupervisor'].mother_and_newborn_info

        logger.debug(key='message', data='HSI_Labour_ReceivesCareForPostpartumPeriod: did not run as the squeeze factor'
                                         f'is too high, mother {person_id} will return home on date {self.sim.date}')

        # Women who delivered at a facility, but can receive no more care due to high squeeze, will go home for the
        # immediate period after birth- where there risk of complications is applied
        if mni[person_id]['delivery_setting'] != 'home_birth':
            mni[person_id]['squeeze_to_high_for_hsi_pp'] = True
            self.sim.schedule_event(PostpartumLabourAtHomeEvent(self.module, person_id), self.sim.date)

        return False

    def not_available(self):
        person_id = self.target
        logger.debug(key='message', data='This event is not in the allowed service availability and therefore cannot '
                                         f'run for mother {person_id} on date {self.sim.date}, she will now deliver at '
                                         f'home')

        self.sim.schedule_event(PostpartumLabourAtHomeEvent(self.module, person_id), self.sim.date)


class HSI_Labour_ReceivesComprehensiveEmergencyObstetricCare(HSI_Event, IndividualScopeEventMixin):
    """
    This is HSI_Labour_ReceivesComprehensiveEmergencyObstetricCare. This event houses all the interventions that are
    required to be delivered at a CEmONC level facility including caesarean section, blood transfusion and surgery
    during or following labour Currently we assume that if this even runs and the consumables are available then the
    intervention is delivered i.e. we dont apply squeeze factor threshold.
    """

    def __init__(self, module, person_id, facility_level_of_this_hsi, timing):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Labour)

        self.TREATMENT_ID = 'Labour_ReceivesComprehensiveEmergencyObstetricCare'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'MajorSurg': 1})
        self.ACCEPTED_FACILITY_LEVEL = facility_level_of_this_hsi
        self.ALERT_OTHER_DISEASES = []
        self.BEDDAYS_FOOTPRINT = self.make_beddays_footprint({'general_bed': 2})

        self.timing = timing

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        mni = self.sim.modules['PregnancySupervisor'].mother_and_newborn_info
        params = self.module.current_parameters
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']

        logger.debug(key='msg', data='This is HSI_Labour_ReceivesComprehensiveEmergencyObstetricCare running for '
                                     f'mother {person_id}')

        # We use the variable self.timing to differentiate between women sent to this event during labour and women
        # sent after labour

        # ========================================== CAESAREAN SECTION ===============================================
        # For women arriving to this event during labour who have been referred for caesarean the intervention is
        # delivered
        if mni[person_id]['referred_for_cs'] and self.timing == 'intrapartum':
            self.module.labour_tracker['caesarean_section'] += 1

            pkg_code_cs = pd.unique(
                consumables.loc[
                    consumables['Intervention_Pkg'] == 'Cesearian Section with indication (with complication)',
                    'Intervention_Pkg_Code'])[0]

            all_available = self.get_all_consumables(
                pkg_codes=[pkg_code_cs])

            # We check if the consumables are available but we dont condition the event happening on the result of the
            # check
            if all_available:
                logger.debug(key='message',
                             data='All the required consumables are available and will can be used for this'
                                  f'caesarean delivery for mother {person_id}.')
            else:
                logger.debug(key='message', data='The required consumables are not available for this caesarean '
                                                 f'delivery for mother {person_id}')

            # The appropriate variables in the MNI and dataframe are stored. Current caesarean section reduces risk of
            # intrapartum still birth and death due to antepartum haemorrhage
            mni[person_id]['mode_of_delivery'] = 'caesarean_section'
            mni[person_id]['amtsl_given'] = True
            df.at[person_id, 'la_previous_cs_delivery'] = True

        # ================================ SURGICAL MANAGEMENT OF RUPTURED UTERUS =====================================
        # Women referred after the labour HSI following correct identification of ruptured uterus will also need to
        # undergo surgical repair of this complication
        # TODO: women who have a caesarean due to UTERINE rupture will be treated, i dont know if this should be
        #  seperated (i.e. no woman would have a CS and not have her uterus repaired/removed)

        if mni[person_id]['referred_for_surgery'] and self.timing == 'intrapartum' and df.at[person_id,
                                                                                             'la_uterine_rupture']:

            # We dont have a specific package code for general surgery...
            dummy_surg_pkg_code = pd.unique(consumables.loc[consumables['Intervention_Pkg'] ==
                                                            'Cesearian Section with indication (with complication)',
                                                            'Intervention_Pkg_Code'])[0]

            # Again we check for the consumables but dont condition on consumables being avaible for the event to
            # happen
            all_available = self.get_all_consumables(
                pkg_codes=[dummy_surg_pkg_code])

            if all_available:
                logger.debug(key='message',
                             data='Consumables required for surgery are available and therefore have been '
                                  'used')
            else:
                logger.debug(key='message',
                             data='Consumables required for surgery are unavailable and therefore have not '
                                  'been used')

            # We apply a probability that repair surgery will be successful which will reduce risk of death from
            # uterine rupture
            treatment_success_ur = params['success_rate_uterine_repair'] > self.module.rng.random_sample()

            if treatment_success_ur:
                df.at[person_id, 'la_uterine_rupture_treatment'] = True
            # Unsuccessful repair will lead to this woman requiring a hysterectomy. Hysterectomy will also reduce risk
            # of death from uterine rupture but leads to permanent infertility in the simulation
            elif ~treatment_success_ur:
                df.at[person_id, 'la_has_had_hysterectomy'] = True

        # ============================= SURGICAL MANAGEMENT OF POSTPARTUM HAEMORRHAGE==================================
        # Women referred for surgery immediately following labour will need surgical management of postpartum bleeding
        # Treatment is varied accordingly to underlying cause of bleeding

        if mni[person_id]['referred_for_surgery'] and self.timing == 'postpartum' and df.at[person_id,
                                                                                            'la_postpartum_haem']:
            self.module.surgical_management_of_pph(self)

        # =========================================== BLOOD TRANSFUSION ===============================================
        # Women referred for blood transfusion alone or in conjunction with one of the above interventions will receive
        # that here
        if mni[person_id]['referred_for_blood']:
            self.module.blood_transfusion(self)

        # Women who have passed through the postpartum SBA HSI have not yet had their risk of death calculated because
        # they required interventions delivered via this event. We now determine if these women will survive
        if self.timing == 'postpartum':
            logger.debug(key='msg', data=f'{person_id} apply_risk_of_early_postpartum_death')
            self.module.apply_risk_of_early_postpartum_death(person_id)

        actual_appt_footprint = self.EXPECTED_APPT_FOOTPRINT

        # Here we edit the appointment footprint so only women receiving surgery require the surgical footprint
        if mni[person_id]['referred_for_surgery'] or mni[person_id]['referred_for_cs']:
            actual_appt_footprint['MajorSurg'] = actual_appt_footprint['MajorSurg']

        elif (not mni[person_id]['referred_for_surgery'] and not mni[person_id]['referred_for_cs']) and\
                mni[person_id]['referred_for_blood']:
            actual_appt_footprint['MajorSurg'] = actual_appt_footprint['InpatientDays']

    def did_not_run(self):
        person_id = self.target
        mni = self.sim.modules['PregnancySupervisor'].mother_and_newborn_info

        logger.debug(key='message', data='HSI_Labour_ReceivesComprehensiveEmergencyObstetricCare: did not run as the '
                                         f'squeeze factor is too high, mother {person_id} did not receive care')

        # For women who cant deliver via caesarean we apply risk of uterine rupture (which was blocked before)
        if mni[person_id]['referred_for_cs'] and self.timing == 'intrapartum':
            logger.debug(key='message', data=f'squeeze factor is too high for this event to run for mother'
                                             f' {person_id} on date {self.sim.date} and she is unable to deliver via '
                                             f'caesarean section')
            self.module.set_intrapartum_complications(
                person_id, complication='uterine_rupture')

        # For women referred to this event after the postnatal SBA HSI we apply risk of death (as if should have been
        # applied in this event if it ran)
        elif self.timing == 'postpartum':
            logger.debug(key='msg', data=f'{person_id} apply_risk_of_early_postpartum_death')
            self.module.apply_risk_of_early_postpartum_death(person_id)

        return False

    def not_available(self):
        person_id = self.target
        mni = self.sim.modules['PregnancySupervisor'].mother_and_newborn_info

        logger.debug(key='message', data='HSI_Labour_ReceivesComprehensiveEmergencyObstetricCare is not in the '
                                         f'allowed service availability and therefore cannot run for mother {person_id}'
                                         f'meaning she will not receive CEmONC interventions she requires')

        if mni[person_id]['referred_for_cs'] and self.timing == 'intrapartum':
            self.module.set_intrapartum_complications(
                person_id, complication='uterine_rupture')

        elif self.timing == 'postpartum':
            logger.debug(key='msg', data=f'{person_id} apply_risk_of_early_postpartum_death')
            self.module.apply_risk_of_early_postpartum_death(person_id)


class HSI_Labour_ReceivesCareFollowingCaesareanSection(HSI_Event, IndividualScopeEventMixin):
    """
    This is HSI_Labour_ReceivesCareFollowingCaesareanSection. It is scheduled via the BirthEvent for all women who have
    delivered via caesarean section and represents the care provided by a skilled birth attendant to women post
    section.  The structure and flow of this event is largely similar to
    HSI_Labour_ReceivesSkilledBirthAttendanceFollowingLabour as described above.
    """

    def __init__(self, module, person_id, facility_level_of_this_hsi):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Labour)

        self.TREATMENT_ID = 'Labour_ReceivesCareFollowingCaesareanSection'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'InpatientDays': 1})
        self.ACCEPTED_FACILITY_LEVEL = facility_level_of_this_hsi
        self.ALERT_OTHER_DISEASES = []
        self.BEDDAYS_FOOTPRINT = self.make_beddays_footprint({'general_bed': 2})

    def apply(self, person_id, squeeze_factor):
        mni = self.sim.modules['PregnancySupervisor'].mother_and_newborn_info
        df = self.sim.population.props
        params = self.module.current_parameters

        logger.debug(key='message', data='Labour_ReceivesCareFollowingCaesareanSection is firing')

        if not df.at[person_id, 'is_alive']:
            return

        assert mni[person_id]['mode_of_delivery'] == 'caesarean_section'
        assert mni[person_id]['referred_for_cs']

        # This event represents care women receive after delivering via caesarean section
        # Women pass through different 'post delivery' events depending on mode of delivery due to how risk and
        # treatment of certain complications, such as postpartum haemorrhage, are managed

        if squeeze_factor < params['squeeze_threshold_treatment_spe']:
            self.module.assessment_and_treatment_of_severe_pre_eclampsia_mgso4(self, 'hp', 'pp')

        # ===================================== APPLYING COMPLICATION INCIDENCE =======================================
        # Here we apply the risk that this woman will develop and infection or experience worsening hypertension after
        # her caesarean
        for complication in ['sepsis_endometritis', 'sepsis_urinary_tract', 'sepsis_skin_soft_tissue',
                             'pph_uterine_atony', 'pph_retained_placenta', 'pph_other']:
            self.module.set_postpartum_complications(person_id, complication=complication)
        self.module.progression_of_hypertensive_disorders(person_id, property_prefix='pn')

        # ======================================= PPH MANAGEMENT =============================================
        outcome_of_pph_management = self.module.assessment_and_treatment_of_pph_uterine_atony(self, 'hp')
        if outcome_of_pph_management:
            self.module.surgical_management_of_pph(self)
            self.module.blood_transfusion(self)

        # ======================================= COMPLICATION MANAGEMENT =============================================
        # Next we apply the effect of any treatments
        if squeeze_factor < params['squeeze_threshold_treatment_sep']:
            self.module.assessment_and_treatment_of_maternal_sepsis(self, 'hp', 'pp')
        if squeeze_factor < params['squeeze_threshold_treatment_htn']:
            self.module.assessment_and_treatment_of_hypertension(self, 'hp', 'pp')
        if squeeze_factor < params['squeeze_threshold_treatment_ec']:
            self.module.assessment_and_treatment_of_eclampsia(self, 'hp', 'pp')

        self.module.interventions_delivered_pre_discharge(self)

        # ====================================== APPLY RISK OF DEATH===================================================
        logger.debug(key='msg', data=f'{person_id} apply_risk_of_early_postpartum_death')
        self.module.apply_risk_of_early_postpartum_death(person_id)

    def did_not_run(self):
        person_id = self.target
        logger.debug(key='message', data=f'squeeze factor is too high for this event to run for mother {person_id} on '
                                         f'date {self.sim.date} and she is unable to receive post caesarean care')

        # For simulation runs where the squeeze is set too high for the event to run we apply risk of complications and
        # death as this would have happened within the HSI
        for complication in ['sepsis_endometritis', 'sepsis_urinary_tract', 'sepsis_skin_soft_tissue']:
            self.module.set_postpartum_complications(person_id, complication=complication)

        self.module.progression_of_hypertensive_disorders(person_id, property_prefix='pn')
        self.module.apply_risk_of_early_postpartum_death(person_id)

        return False

    def not_available(self):
        person_id = self.target

        logger.debug(key='message', data=f'HSI_Labour_ReceivesCareFollowingCaesareanSection is not allowed in the '
                                         f'current service availability, for mother {person_id} '
                                         f'and she is unable to receive care after caesarean section')

        for complication in ['sepsis_endometritis', 'sepsis_urinary_tract', 'sepsis_skin_soft_tissue']:
            self.module.set_postpartum_complications(person_id, complication=complication)
        self.module.progression_of_hypertensive_disorders(person_id, property_prefix='pn')
        self.module.apply_risk_of_early_postpartum_death(person_id)


class LabourLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    """This is the LabourLoggingEvent. It uses the data frame and the labour_tracker to produce summary statistics which
    are processed and presented by different analysis scripts """

    def __init__(self, module):
        self.repeat = 1
        super().__init__(module, frequency=DateOffset(years=self.repeat))

    def apply(self, population):
        df = self.sim.population.props

        # Previous Year...
        one_year_prior = self.sim.date - np.timedelta64(1, 'Y')
        total_births_last_year = len(df.index[(df.date_of_birth > one_year_prior) & (df.date_of_birth < self.sim.date)])

        # Denominators...
        total_ip_maternal_deaths_last_year = len(df.index[df.la_maternal_death_in_labour & (
            df.la_maternal_death_in_labour_date > one_year_prior) & (df.la_maternal_death_in_labour_date <
                                                                     self.sim.date)])
        if total_ip_maternal_deaths_last_year == 0:
            total_ip_maternal_deaths_last_year = 1

        if total_births_last_year == 0:
            total_births_last_year = 1

        # yearly number of complications
        deaths = self.module.labour_tracker['maternal_death']
        still_births = self.module.labour_tracker['ip_stillbirth']
        ol = self.module.labour_tracker['obstructed_labour']
        aph = self.module.labour_tracker['antepartum_haem']
        ur = self.module.labour_tracker['uterine_rupture']
        ec = self.module.labour_tracker['eclampsia']
        spe = self.module.labour_tracker['severe_pre_eclampsia']
        pph = self.module.labour_tracker['postpartum_haem']
        sep = self.module.labour_tracker['sepsis']
        sep_pp = self.module.labour_tracker['sepsis_pp']

        # yearly number deliveries by setting
        home = self.module.labour_tracker['home_birth']
        hospital = self.module.labour_tracker['health_centre_birth']
        health_centre = self.module.labour_tracker['hospital_birth']
        cs_deliveries = self.module.labour_tracker['caesarean_section']

        ept = self.module.labour_tracker['early_preterm']
        lpt = self.module.labour_tracker['late_preterm']
        pt = self.module.labour_tracker['post_term']
        t = self.module.labour_tracker['term']

        # TODO: division by zero crashes code on small runs

        dict_for_output = {
            'total_births_last_year': total_births_last_year,
            'maternal_deaths_checker': deaths,
            'maternal_deaths_df': total_ip_maternal_deaths_last_year,
            'still_births': still_births,
            'sbr': still_births / total_births_last_year * 100,
            'intrapartum_mmr': total_ip_maternal_deaths_last_year / total_births_last_year * 100000,
            'home_births_crude': home,
            'home_births_prop': home / total_births_last_year * 100,
            'health_centre_births': health_centre / total_births_last_year * 100,
            'hospital_births': hospital / total_births_last_year * 100,
            'cs_delivery_rate': cs_deliveries / total_births_last_year * 100,
            'ol_incidence': ol / total_births_last_year * 100,
            'aph_incidence': aph / total_births_last_year * 100,
            'ur_incidence': ur / total_births_last_year * 100,
            'ec_incidence': ec / total_births_last_year * 100,
            'spe_incidence': spe / total_births_last_year * 100,
            'sep_incidence': sep + sep_pp / total_births_last_year * 100,
            'sep_incidence_pp': sep_pp / total_births_last_year * 100,
            'pph_incidence': pph / total_births_last_year * 100,
        }

        dict_crude_cases = {'intrapartum_mmr': total_ip_maternal_deaths_last_year,
                            'ol_cases': ol,
                            'aph_cases': aph,
                            'ur_cases': ur,
                            'ec_cases': ec,
                            'spe_cases': spe,
                            'sep_cases': sep,
                            'sep_cases_pp': sep_pp,
                            'pph_cases': pph}

        deliveries = {'ept': ept / total_births_last_year * 100,
                      'lpt': lpt / total_births_last_year * 100,
                      'term': t / total_births_last_year * 100,
                      'post_term': pt / total_births_last_year * 100}

        # deaths
        sep_d = self.module.labour_tracker['sepsis_death']
        ur_d = self.module.labour_tracker['uterine_rupture_death']
        aph_d = self.module.labour_tracker['antepartum_haem_death']
        ec_d = self.module.labour_tracker['eclampsia_death']
        pph_d = self.module.labour_tracker['postpartum_haem_death']

        deaths = {'sepsis': sep_d,
                  'uterine_rupture': ur_d,
                  'aph': aph_d,
                  'eclampsia': ec_d,
                  'postpartum_haem': pph_d,
                  }

        logger.info(key='labour_summary_stats_incidence', data=dict_for_output, description='Yearly incidence summary '
                                                                                            'statistics output from '
                                                                                            'the labour module')
        logger.info(key='labour_summary_stats_crude_cases', data=dict_crude_cases, description='Yearly crude '
                                                                                               'summary statistics '
                                                                                               'output from the labour '
                                                                                               'module')
        logger.info(key='labour_summary_stats_delivery', data=deliveries, description='Yearly delivery summary '
                                                                                      'statistics output from the '
                                                                                      'labour module')
        logger.info(key='labour_summary_stats_death', data=deaths, description='Yearly death summary statistics '
                                                                               'output from the labour module')

        # Reset the EventTracker
        for k in self.module.labour_tracker:
            self.module.labour_tracker[k] = 0
