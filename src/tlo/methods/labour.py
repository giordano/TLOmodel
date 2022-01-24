from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats

from tlo import DateOffset, Module, Parameter, Property, Types, logging
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.lm import LinearModel
from tlo.methods import Metadata, labour_lm
from tlo.methods.causes import Cause
from tlo.methods.dxmanager import DxTest
from tlo.methods.healthsystem import HSI_Event
from tlo.methods.hiv import HSI_Hiv_TestAndRefer
from tlo.methods.postnatal_supervisor import PostnatalWeekOneMaternalEvent
from tlo.util import BitsetHandler

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Labour(Module):
    """This is module is responsible for the the process of labour, birth and the immediate postnatal period (up until
    48hrs post birth). This model has a number of core functions including; initiating the onset of labour for women on
    their pre-determined due date (or prior to this for preterm labour/admission for delivery), applying the incidence
     of a core set of maternal complications occurring in the intrapartum period and outcomes such as maternal death or
     still birth, scheduling birth for women surviving labour and applying risk of complications and outcomes in the
     postnatal period. Complications explicitly modelled in this module include obstructed labour, antepartum
     haemorrhage, maternal infection and sepsis, progression of hypertensive disorders, uterine rupture and postpartum
      haemorrhage. In addition to the natural history of labour this module manages care seeking for women in labour
      (for delivery or following onset of complications at a home birth) and includes HSIs which represent
      Skilled Birth Attendance at either Basic or Comprehensive level emergency obstetric care facilities.
      Following birth this module manages postnatal care delivered via HSI_Labour_ReceivesPostnatalCheck and schedules
      PostnatalWeekOneMaternalEvent which represents the start of a womans postnatal period.
      """

    def __init__(self, name=None, resourcefilepath=None):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath

        # First we define dictionaries which will store the current parameters of interest (to allow parameters to
        # change between 2010 and 2020) and the linear models
        self.current_parameters = dict()
        self.la_linear_models = dict()

        # This list contains the individual_ids of women in labour, used for testing
        self.women_in_labour = list()

        # These lists will contain possible complications and are used as checks in assert functions
        self.possible_intrapartum_complications = list()
        self.possible_postpartum_complications = list()

        # Finally define a dictionary which will hold the required consumables for each intervention
        self.item_codes_lab_consumables = dict()

    INIT_DEPENDENCIES = {
        'Demography', 'PostnatalSupervisor', 'CareOfWomenDuringPregnancy', 'Lifestyle', 'PregnancySupervisor',
        'HealthSystem'
    }

    ADDITIONAL_DEPENDENCIES = {
        'Contraception',
        'NewbornOutcomes',
        'Hiv'
    }

    METADATA = {
        Metadata.DISEASE_MODULE,
        Metadata.USES_HEALTHSYSTEM,
        Metadata.USES_HEALTHBURDEN,
    }
    # Declare Causes of Death
    CAUSES_OF_DEATH = {
        'uterine_rupture': Cause(gbd_causes='Maternal disorders', label='Maternal Disorders'),
        'intrapartum_sepsis': Cause(gbd_causes='Maternal disorders', label='Maternal Disorders'),
        'antepartum_haemorrhage': Cause(gbd_causes='Maternal disorders', label='Maternal Disorders'),
        'postpartum_sepsis': Cause(gbd_causes='Maternal disorders', label='Maternal Disorders'),
        'postpartum_haemorrhage': Cause(gbd_causes='Maternal disorders', label='Maternal Disorders'),
        'secondary_postpartum_haemorrhage': Cause(gbd_causes='Maternal disorders', label='Maternal Disorders'),
        'severe_pre_eclampsia': Cause(gbd_causes='Maternal disorders', label='Maternal Disorders'),
        'eclampsia': Cause(gbd_causes='Maternal disorders', label='Maternal Disorders')}

    # Declare Causes of Disability
    CAUSES_OF_DISABILITY = {
        'maternal': Cause(gbd_causes='Maternal disorders', label='Maternal Disorders')
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
                        'regression equation predicating womans parity at 2010 baseline'),
        'effect_mar_stat_3_parity_lr2010': Parameter(
            Types.LIST, 'effect of a change in marriage status from comparison (level 1) in the linear '
                        'regression equation predicating womans parity at 2010 baseline'),
        'effect_wealth_lev_4_parity_lr2010': Parameter(
            Types.LIST, 'effect of an increase in wealth level in the linear regression equation predicating womans '
                        'parity at 2010 base line'),
        'effect_wealth_lev_3_parity_lr2010': Parameter(
            Types.LIST, 'effect of an increase in wealth level in the linear regression equation predicating womans '
                        'parity at 2010 base line'),
        'effect_wealth_lev_2_parity_lr2010': Parameter(
            Types.LIST, 'effect of an increase in wealth level in the linear regression equation predicating womans '
                        'parity at 2010 base line'),
        'effect_wealth_lev_1_parity_lr2010': Parameter(
            Types.LIST, 'effect of an increase in wealth level in the linear regression equation predicating womans '
                        'parity at 2010 base line'),
        'effect_edu_lev_2_parity_lr2010': Parameter(
            Types.LIST, 'effect of an increase in education level in the linear regression equation predicating womans '
                        'parity at 2010 base line'),
        'effect_edu_lev_3_parity_lr2010': Parameter(
            Types.LIST, 'effect of an increase in education level in the linear regression equation predicating womans '
                        'parity at 2010 base line'),
        'effect_rural_parity_lr2010': Parameter(
            Types.LIST, 'effect of rural living in the linear regression equation predicating womans parity at 2010 '
                        'base line'),
        'prob_previous_caesarean_at_baseline': Parameter(
            Types.LIST, 'probability of previously having delivered via caesarean section at baseline'),

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
        'residual_prob_of_macrosomia': Parameter(
            Types.LIST, 'probability that those allocated to be normal birth weight and above will be macrosomic '
                        '(>4kg)'),

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
        'prob_both_twins_ip_still_birth': Parameter(
            Types.LIST, 'probability that if this mother will experience still birth, and she is pregnant with twins, '
                        'that neither baby will survive'),

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
             Types.LIST, 'risk of pph after experiencing uterine atony'),
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

        'probability_delivery_hospital': Parameter(
            Types.LIST, 'probability of delivering in a hospital'),

        # PNC CHECK...
        'prob_timings_pnc': Parameter(
            Types.LIST, 'probabilities that a woman who will receive a PNC check will receive care <48hrs post birth '
                        'or > 48hrs post birth'),
        'odds_will_attend_pnc': Parameter(
            Types.LIST, 'baseline odss a woman will seek PNC for her and her newborn following delivery'),
        'or_pnc_age_30_35': Parameter(
            Types.LIST, 'odds ratio for women aged 30-35 attending PNC'),
        'or_pnc_age_>35': Parameter(
            Types.LIST, 'odds ratio for women aged > 35 attending PNC'),
        'or_pnc_rural': Parameter(
            Types.LIST, 'odds ratio for women who live rurally attending PNC'),
        'or_pnc_wealth_level_1': Parameter(
            Types.LIST, 'odds ratio for women from the highest wealth level attending PNC'),
        'or_pnc_anc4+': Parameter(
            Types.LIST, 'odds ratio for women who attended ANC4+ attending PNC'),
        'or_pnc_caesarean_delivery': Parameter(
            Types.LIST, 'odds ratio for women who delivered by caesarean attending PNC'),
        'or_pnc_facility_delivery': Parameter(
            Types.LIST, 'odds ratio for women who delivered in a health facility attending PNC'),
        'or_pnc_parity_>4': Parameter(
            Types.LIST, 'odds ratio for women with a parity of >4 attending PNC'),
        'probs_of_attending_pn_event_by_day': Parameter(
            Types.LIST, 'probabilities used in a weighted random draw to determine when a woman will attend the '
                        'postnatal event'),

        # EMERGENCY CARE SEEKING...
        'prob_careseeking_for_complication': Parameter(
            Types.LIST, 'odds of a woman seeking skilled assistance after developing a complication at a home birth'),
        'test_care_seeking_probs': Parameter(
            Types.LIST, 'dummy probabilities of delivery care seeking used in testing'),

        # TREATMENT PARAMETERS...
        'prob_delivery_modes_ec': Parameter(
            Types.LIST, 'probabilities that a woman admitted with eclampsia will deliver normally, via caesarean or '
                        'via assisted vaginal delivery'),
        'prob_delivery_modes_spe': Parameter(
            Types.LIST, 'probabilities that a woman admitted with severe pre-eclampsia will deliver normally, via '
                        'caesarean or via assisted vaginal delivery'),
        'residual_prob_avd': Parameter(
            Types.LIST, 'probabilities that a woman will deliver via assisted vaginal delivery for an indication not '
                        'explicitly modelled'),
        'residual_prob_caesarean': Parameter(
            Types.LIST, 'probabilities that a woman will deliver via caesarean section for an indication not '
                        'explicitly modelled'),
        'prob_adherent_ifa': Parameter(
            Types.LIST, 'probability that a woman started on postnatal IFA will be adherent'),
        'effect_of_ifa_for_resolving_anaemia': Parameter(
            Types.LIST, 'relative effect of iron and folic acid on anaemia status'),
        'treatment_effect_maternal_infection_clean_delivery': Parameter(
            Types.LIST, 'Effect of clean delivery practices on risk of maternal infection'),
        'treatment_effect_maternal_chorio_abx_prom': Parameter(
            Types.LIST, 'Relative effect of antibiotics for premature rupture of membranes on maternal risk of '
                        'chorioamnionitis prior to birth'),
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
        'treatment_effect_blood_transfusion_anaemia': Parameter(
            Types.LIST, 'effect of blood transfusion treatment for severe anaemia'),
        'aph_cs_treatment_effect_md': Parameter(
            Types.LIST, 'effect of caesarean section for antepartum haemorrhage on risk of maternal death'),
        'treatment_effect_avd_still_birth': Parameter(
            Types.LIST, 'effect of assisted vaginal delivery on risk of intrapartum still birth'),
        'treatment_effect_cs_still_birth': Parameter(
            Types.LIST, 'effect of caesarean section delivery on risk of intrapartum still birth'),

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
        'prob_intervention_delivered_sep_assessment_pnc': Parameter(
            Types.LIST, 'probability a woman will be assessed for infection during PNC given that the HSI has ran '
                        'and the consumables are available (proxy for clinical quality)'),
        'prob_intervention_delivered_bp_pnc': Parameter(
            Types.LIST, 'probability a woman will have their BP checked during PNC given that the HSI has ran '
                        'and the consumables are available (proxy for clinical quality)'),
        'prob_intervention_delivered_anaemia_assessment_pnc': Parameter(
            Types.LIST, 'probability a woman will have their Hb levels checked during PNC given that the HSI has ran '
                        'and the consumables are available (proxy for clinical quality)'),
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
        'la_gest_htn_on_treatment': Property(Types.BOOL, 'whether this woman has is receiving regular '
                                                         'antihypertensives'),
        'la_postpartum_haem': Property(Types.BOOL, 'whether the woman has experienced an postpartum haemorrhage in this'
                                                   'delivery'),
        'la_postpartum_haem_cause': Property(Types.INT, 'bitset column holding causes of postpartum haemorrhage'),
        'la_postpartum_haem_treatment': Property(Types.INT, ' Treatment for received for postpartum haemorrhage '
                                                            '(bitset)'),
        'la_has_had_hysterectomy': Property(Types.BOOL, 'whether this woman has had a hysterectomy as treatment for a '
                                                        'complication of labour, and therefore is unable to conceive'),
        'la_date_most_recent_delivery': Property(Types.DATE, 'date of on which this mother last delivered'),
        'la_is_postpartum': Property(Types.BOOL, 'Whether a woman is in the postpartum period, from delivery until '
                                                 'day +42 (6 weeks)'),
        'la_pn_checks_maternal': Property(Types.INT, 'Number of postnatal checks this woman has received'),
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
        df.loc[df.is_alive, 'la_gest_htn_on_treatment'] = False
        df.loc[df.is_alive, 'la_postpartum_haem'] = False
        df.loc[df.is_alive, 'la_postpartum_haem_cause'] = 0
        df.loc[df.is_alive, 'la_postpartum_haem_treatment'] = 0
        df.loc[df.is_alive, 'la_has_had_hysterectomy'] = False
        df.loc[df.is_alive, 'la_date_most_recent_delivery'] = pd.NaT
        df.loc[df.is_alive, 'la_is_postpartum'] = False
        df.loc[df.is_alive, 'la_pn_checks_maternal'] = 0
        df.loc[df.is_alive, 'la_iron_folic_acid_postnatal'] = False

        #  we store different potential treatments for postpartum haemorrhage via bistet
        self.pph_treatment = BitsetHandler(self.sim.population, 'la_postpartum_haem_treatment',
                                           ['manual_removal_placenta', 'surgery', 'hysterectomy'])

        #  ----------------------------ASSIGNING PARITY AT BASELINE --------------------------------------------------
        # This equation predicts the parity of each woman at baseline (who is of reproductive age)
        parity_equation = LinearModel.custom(labour_lm.predict_parity, parameters=params)

        # We assign parity to all women of reproductive age at baseline
        df.loc[df.is_alive & (df.sex == 'F') & (df.age_years > 14), 'la_parity'] = \
            parity_equation.predict(df.loc[df.is_alive & (df.sex == 'F') & (df.age_years > 14)])

        assert (df.loc[df.is_alive & (df.sex == 'F') & (df.age_years > 14), 'la_parity'] >= 0).all().all()

        #  ----------------------- ASSIGNING PREVIOUS CS DELIVERY AT BASELINE -----------------------------------------
        # This equation determines the proportion of women at baseline who have previously delivered via caesarean
        # section
        reproductive_age_women = df.is_alive & (df.sex == 'F') & (df.age_years > 14) & (df.age_years < 50)

        previous_cs = pd.Series(
            self.rng.random_sample(len(reproductive_age_women.loc[reproductive_age_women])) <
            self.current_parameters['prob_previous_caesarean_at_baseline'],
            index=reproductive_age_women.loc[reproductive_age_women].index)

        df.loc[previous_cs.loc[previous_cs].index, 'la_previous_cs_delivery'] = True

    def get_and_store_labour_item_codes(self):
        """
        This function defines the required consumables for each intervention delivered during this module and stores
        them in a module level dictionary called within HSIs
         """
        get_item_code_from_name = self.sim.modules['HealthSystem'].get_item_code_from_item_name
        get_item_code_from_pkg = self.sim.modules['HealthSystem'].get_item_codes_from_package_name

        # -------------------------------------------- DELIVERY ------------------------------------------------------
        # assuming CDK has blade, soap, cord tie
        self.item_codes_lab_consumables['delivery'] = \
            [get_item_code_from_name('Clean delivery kit')] + \
            [get_item_code_from_name('Chlorhexidine 1.5% solution_5_CMST')] + \
            [get_item_code_from_name('Cannula iv  (winged with injection pot) 20_each_CMST')] + \
            [get_item_code_from_name('Apron, disposable, polythene_100_CMST')] + \
            [get_item_code_from_name('Gloves, exam, latex, disposable, pair')] + \
            [get_item_code_from_name('Gauze, swabs 8-ply 10cm x 10cm_100_CMST')] + \
            [get_item_code_from_name('Cotton swab')] + \
            [get_item_code_from_name('Paracetamol 500mg_1000_CMST')]

        # -------------------------------------------- CAESAREAN DELIVERY ------------------------------------------
        # TODO: this package is incomplete?
        # todo: replace- halothane with thiopental
        self.item_codes_lab_consumables['caesarean_delivery'] = \
            [get_item_code_from_name('Halothane (fluothane)_250ml_CMST')] + \
            [get_item_code_from_name('Ceftriaxone 1g, PFR_each_CMST')] + \
            [get_item_code_from_name('Metronidazole 200mg_1000_CMST')] + \
            [get_item_code_from_name('Cannula iv  (winged with injection pot) 20_each_CMST')] + \
            [get_item_code_from_name('Paracetamol 500mg _1000_CMST')] + \
            [get_item_code_from_name('Declofenac injection_each_CMST')] + \
            [get_item_code_from_name('Pethidine, 50 mg/ml, 2 ml ampoule')] + \
            [get_item_code_from_name('Foley catheter')] + \
            [get_item_code_from_name('Bag, urine, collecting, 2000 ml')] + \
            [get_item_code_from_name("Sodium lactate injection (Ringer's), 500 ml, with giving set")] + \
            [get_item_code_from_name('Sodium chloride, injectable solution, 0,9 %, 500 ml')] + \
            [get_item_code_from_name("Giving set iv administration + needle 15 drops/ml_each_CMST")] + \
            [get_item_code_from_name("Chlorhexidine 1.5% solution_5_CMST")]

        # -------------------------------------------- OBSTETRIC SURGERY ----------------------------------------------
        # TODO: this package may not be accurate yet
        self.item_codes_lab_consumables['obstetric_surgery'] = \
            [get_item_code_from_name('Halothane (fluothane)_250ml_CMST')] + \
            [get_item_code_from_name('Ceftriaxone 1g, PFR_each_CMST')] + \
            [get_item_code_from_name('Metronidazole 200mg_1000_CMST')] + \
            [get_item_code_from_name('Cannula iv  (winged with injection pot) 20_each_CMST')] + \
            [get_item_code_from_name('Paracetamol 500mg _1000_CMST')] + \
            [get_item_code_from_name('Declofenac injection_each_CMST')] + \
            [get_item_code_from_name('Pethidine, 50 mg/ml, 2 ml ampoule')] + \
            [get_item_code_from_name('Foley catheter')] + \
            [get_item_code_from_name('Bag, urine, collecting, 2000 ml')] + \
            [get_item_code_from_name("Sodium lactate injection (Ringer's), 500 ml, with giving set")] + \
            [get_item_code_from_name('Sodium chloride, injectable solution, 0,9 %, 500 ml')] + \
            [get_item_code_from_name("Giving set iv administration + needle 15 drops/ml_each_CMST")]

        # -------------------------------------------- ABX FOR PROM -------------------------------------------------
        self.item_codes_lab_consumables['abx_for_prom'] = \
            [get_item_code_from_name('Benzathine benzylpenicillin, powder for injection, 2.4 million IU')] + \
            [get_item_code_from_name('Water for injection, 10ml_Each_CMST')] + \
            [get_item_code_from_name('Cannula iv  (winged with injection pot) 20_each_CMST')] + \
            [get_item_code_from_name('Syringe, needle + swab')] + \
            [get_item_code_from_name('Gloves, exam, latex, disposable, pair')]

        # -------------------------------------------- ANTENATAL STEROIDS ---------------------------------------------
        self.item_codes_lab_consumables['antenatal_steroids'] = \
            [get_item_code_from_name('Dexamethasone 5mg/ml, 5ml_each_CMST')] + \
            [get_item_code_from_name('Betamethasone, 12 mg injection')] + \
            [get_item_code_from_name('Cannula iv  (winged with injection pot) 20_each_CMST')] + \
            [get_item_code_from_name('Water for injection, 10ml_Each_CMST')] + \
            [get_item_code_from_name('Syringe, needle + swab')] + \
            [get_item_code_from_name('Gloves, exam, latex, disposable, pair')]

        # -------------------------------------  INTRAVENOUS ANTIHYPERTENSIVES ---------------------------------------
        self.item_codes_lab_consumables['iv_antihypertensives'] = \
            [get_item_code_from_name('Hydralazine, powder for injection, 20 mg ampoule')] + \
            [get_item_code_from_name('Cannula iv  (winged with injection pot) 20_each_CMST')] + \
            [get_item_code_from_name('Water for injection, 10ml_Each_CMST')] + \
            [get_item_code_from_name('Syringe, needle + swab')] + \
            [get_item_code_from_name('Gloves, exam, latex, disposable, pair')]

        # --------------------------------------- ORAL ANTIHYPERTENSIVES ---------------------------------------------
        self.item_codes_lab_consumables['oral_antihypertensives'] = \
            [get_item_code_from_name('Methyldopa 250mg_1000_CMST')]

        # ----------------------------------  SEVERE PRE-ECLAMPSIA/ECLAMPSIA  -----------------------------------------
        self.item_codes_lab_consumables['severe_pre_eclampsia'] = \
            [get_item_code_from_name('Magnesium sulfate, injection, 500 mg/ml in 10-ml ampoule')] + \
            [get_item_code_from_name('Lidocaine HCl (in dextrose 7.5%), ampoule 2 ml')] + \
            [get_item_code_from_name('Sodium chloride, injectable solution, 0,9 %, 500 ml')] + \
            [get_item_code_from_name('Cannula iv  (winged with injection pot) 20_each_CMST')] + \
            [get_item_code_from_name('Giving set iv administration + needle 15 drops/ml_each_CMST')] + \
            [get_item_code_from_name('Foley catheter')] + \
            [get_item_code_from_name('Bag, urine, collecting, 2000 ml')] + \
            [get_item_code_from_name('Syringe, needle + swab')] + \
            [get_item_code_from_name('Gloves, exam, latex, disposable, pair')]

        # -------------------------------------  OBSTRUCTED LABOUR  ---------------------------------------------------
        # todo: this package is not finished
        self.item_codes_lab_consumables['obstructed_labour'] = get_item_code_from_pkg('Management of obstructed labour')

        # -------------------------------------  OBSTETRIC FORCEPS  ---------------------------------------------------
        self.item_codes_lab_consumables['forceps'] = get_item_code_from_name('Forceps, obstetric')

        # -------------------------------------  OBSTETRIC VACUUM   ---------------------------------------------------
        self.item_codes_lab_consumables['vacuum'] = get_item_code_from_name('Vacuum, obstetric')

        # -------------------------------------  MATERNAL SEPSIS  -----------------------------------------------------
        self.item_codes_lab_consumables['maternal_sepsis'] = \
            [get_item_code_from_name('Ampicillin, powder for injection, 500 mg, vial')] + \
            [get_item_code_from_name('Gentamycin, injection, 40 mg/ml in 2 ml vial')] + \
            [get_item_code_from_name('Metronidazole, injection, 500 mg in 100 ml vial')] + \
            [get_item_code_from_name('Cannula iv  (winged with injection pot) 20_each_CMST')] + \
            [get_item_code_from_name('Oxygen, 1000 liters, primarily with oxygen cylinders')] + \
            [get_item_code_from_name('Paracetamol, tablet, 500 mg')] + \
            [get_item_code_from_name('Giving set iv administration + needle 15 drops/ml_each_CMST')] + \
            [get_item_code_from_name('Foley catheter')] + \
            [get_item_code_from_name('Bag, urine, collecting, 2000 ml')] + \
            [get_item_code_from_name('Syringe, needle + swab')] + \
            [get_item_code_from_name('Gloves, exam, latex, disposable, pair')]

        # -------------------------------------  ACTIVE MANAGEMENT THIRD STAGE  ---------------------------------------
        self.item_codes_lab_consumables['amtsl'] = \
            get_item_code_from_pkg('Active management of the 3rd stage of labour')

        # -------------------------------------  POSTPARTUM HAEMORRHAGE  ---------------------------------------
        self.item_codes_lab_consumables['pph'] = \
            [get_item_code_from_name('Oxytocin, injection, 10 IU in 1 ml ampoule')] + \
            [get_item_code_from_name('Misoprostol, tablet, 200 mcg')] + \
            [get_item_code_from_name('Pethidine, 50 mg/ml, 2 ml ampoule')] + \
            [get_item_code_from_name('Oxygen, 1000 liters, primarily with oxygen cylinders')] + \
            [get_item_code_from_name('Cannula iv  (winged with injection pot) 20_each_CMST')] + \
            [get_item_code_from_name('Bag, urine, collecting, 2000 ml')] + \
            [get_item_code_from_name('Foley catheter')] + \
            [get_item_code_from_name('Giving set iv administration + needle 15 drops/ml_each_CMST')] + \
            [get_item_code_from_name('Syringe, needle + swab')] + \
            [get_item_code_from_name('Gloves, exam, latex, disposable, pair')]

        # -------------------------------------  BLOOD TRANSFUSION  ---------------------------------------
        self.item_codes_lab_consumables['blood_transfusion'] = \
            [get_item_code_from_name('Blood, one unit')] + \
            [get_item_code_from_name('IV giving/infusion set, with needle')] + \
            [get_item_code_from_name('Cannula iv  (winged with injection pot) 20_each_CMST')] + \
            [get_item_code_from_name('Gloves, exam, latex, disposable, pair')]

        # ------------------------------------------ FULL BLOOD COUNT -------------------------------------------------
        self.item_codes_lab_consumables['full_blood_count'] = \
            [get_item_code_from_name('Complete blood count')] + \
            [get_item_code_from_name('Blood collecting tube, 5 ml')] + \
            [get_item_code_from_name('Syringe, needle + swab')] + \
            [get_item_code_from_name('Gloves, exam, latex, disposable, pair')]

        # ---------------------------------- IRON AND FOLIC ACID ------------------------------------------------------
        self.item_codes_lab_consumables['iron_folic_acid'] = \
            [get_item_code_from_name('Ferrous Salt + Folic Acid, tablet, 200 + 0.25 mg')]

    def initialise_simulation(self, sim):

        # We call the following function to store the required consumables for the simulation run within the appropriate
        # dictionary
        self.get_and_store_labour_item_codes()

        # We set the LoggingEvent to run a the last day of each year to produce statistics for that year
        sim.schedule_event(LabourLoggingEvent(self), sim.date + DateOffset(days=1))

        # This list contains all the women who are currently in labour and is used for checks/testing
        self.women_in_labour = []

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
        # define any dx_tests
        self.sim.modules['HealthSystem'].dx_manager.register_dx_test(
            full_blood_count_hb_pn=DxTest(
                property='pn_anaemia_following_pregnancy',
                target_categories=['mild', 'moderate', 'severe'],
                item_codes=self.item_codes_lab_consumables['full_blood_count'],
                sensitivity=1.0),
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
            'intrapartum_sepsis_death': LinearModel.custom(labour_lm.predict_sepsis_death, parameters=params),
            'postpartum_sepsis_death': LinearModel.custom(labour_lm.predict_sepsis_death, parameters=params),


            # This equation is used to calculate a womans risk of death following eclampsia and is mitigated
            # by treatment delivered either immediately prior to admission for delivery or during labour
            'eclampsia_death': LinearModel.custom(labour_lm.predict_eclampsia_death, parameters=params),

            # This equation is used to calculate a womans risk of death following eclampsia and is mitigated
            # by treatment delivered either immediately prior to admission for delivery or during labour
            'severe_pre_eclampsia_death': LinearModel.custom(labour_lm.predict_severe_pre_eclamp_death,
                                                             parameters=params),

            # This equation is used to calculate a womans risk of placental abruption in labour
            'placental_abruption_ip': LinearModel.custom(labour_lm.predict_placental_abruption_ip, parameters=params),

            # This equation is used to calculate a womans risk of antepartum haemorrhage. We assume APH can only occur
            # in the presence of a preceding placental causes (abruption/praevia) therefore this model is additive
            'antepartum_haem_ip': LinearModel.custom(labour_lm.predict_antepartum_haem_ip, parameters=params),

            # This equation is used to calculate a womans risk of death following antepartum haemorrhage. Risk is
            # mitigated by treatment
            'antepartum_haemorrhage_death': LinearModel.custom(labour_lm.predict_antepartum_haem_death,
                                                               parameters=params),

            # This equation is used to calculate a womans risk of postpartum haemorrhage due to uterine atony
            'pph_uterine_atony_pp': LinearModel.custom(labour_lm.predict_pph_uterine_atony_pp, parameters=params),

            # This equation is used to calculate a womans risk of postpartum haemorrhage due to retained placenta
            'pph_retained_placenta_pp': LinearModel.custom(labour_lm.predict_pph_retained_placenta_pp,
                                                           parameters=params),

            # This equation is used to calculate a womans risk of death following postpartum haemorrhage. Risk is
            # mitigated by treatment
            'postpartum_haemorrhage_death': LinearModel.custom(labour_lm.predict_postpartum_haem_pp_death, module=self),

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

            # This equation is used to determine the probability that a woman will seek care for PNC after delivery
            'postnatal_check': LinearModel.custom(
                labour_lm.predict_postnatal_check, parameters=params),
        }

        # Here we create a dict with all the models to be scaled and the 'target' rate parameter
        mod = self.la_linear_models
        models_to_be_scaled = {'ur': [mod['uterine_rupture_ip'], 'prob_uterine_rupture'],
                               'pn': [mod['postnatal_check'], 'odds_will_attend_pnc'],
                               'hb': [mod['probability_delivery_at_home'], 'odds_deliver_at_home'],
                               'hc': [mod['probability_delivery_health_centre'], 'odds_deliver_in_health_centre']}

        # Scale all models updating the parameter used as the intercept of the linear models
        for k in models_to_be_scaled:
            self.sim.modules['PregnancySupervisor'].scale_linear_model_at_initialisation(
                module_of_interest=self, model=models_to_be_scaled[k][0], parameter_key=models_to_be_scaled[k][1])

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
        df.at[child_id, 'la_gest_htn_on_treatment'] = False
        df.at[child_id, 'la_postpartum_haem'] = False
        df.at[child_id, 'la_postpartum_haem_cause'] = 0
        df.at[child_id, 'la_postpartum_haem_treatment'] = 0
        df.at[child_id, 'la_has_had_hysterectomy'] = False
        df.at[child_id, 'la_date_most_recent_delivery'] = pd.NaT
        df.at[child_id, 'la_is_postpartum'] = False
        df.at[child_id, 'la_pn_checks_maternal'] = 0
        df.at[child_id, 'la_iron_folic_acid_postnatal'] = False

    def further_on_birth_labour(self, mother_id):
        """
        This function is called by the on_birth function of NewbornOutcomes module. This function contains additional
        code related to the labour module that should be ran on_birth for all births - it has been
        parcelled into functions to ensure each modules (pregnancy,antenatal care, labour, newborn, postnatal) on_birth
        code is ran in the correct sequence (as this can vary depending on how modules are registered)
        :param mother_id: mothers individual id
        """
        df = self.sim.population.props
        mni = self.sim.modules['PregnancySupervisor'].mother_and_newborn_info

        # log delivery setting
        logger.info(key='delivery_setting_and_mode', data={'mother': mother_id,
                                                           'facility_type': mni[mother_id]['delivery_setting'],
                                                           'mode': mni[mother_id]['mode_of_delivery']})

        # Store only live births to a mother parity
        if not df.at[mother_id, 'la_intrapartum_still_birth']:
            df.at[mother_id, 'la_parity'] += 1  # Only live births contribute to parity

        # Currently we assume women who received antihypertensive in the antenatal period will continue to use them
        if df.at[mother_id, 'ac_gest_htn_on_treatment']:
            df.at[mother_id, 'la_gest_htn_on_treatment'] = True

    def on_hsi_alert(self, person_id, treatment_id):
        """ This is called whenever there is an HSI event commissioned by one of the other disease modules."""
        logger.debug(key='message', data=f'This is Labour, being alerted about a health system interaction '
                                         f'person {person_id}for: {treatment_id}')

    def report_daly_values(self):
        logger.debug(key='message', data='This is Labour reporting my health values')
        df = self.sim.population.props  # shortcut to population properties data frame

        # All dalys related to maternal outcome are outputted by the pregnancy supervisor module...
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
        delivery_setting = mni[person_id]['delivery_setting']

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
                                                     amtsl_given=amtsl_given,
                                                     macrosomia=macrosomia,
                                                     delivery_setting=delivery_setting)[person_id]

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
            update = df.at
        else:
            update = df.loc

        update[id_or_index, 'la_due_date_current_pregnancy'] = new_due_date

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

            # Women admitted with clinical chorioamnionitis from the community are assumed to be septic in labour
            if (complication == 'sepsis_chorioamnionitis') and df.at[individual_id, 'ps_chorioamnionitis']:
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

            # Otherwise they are stored as individual properties (women with undiagnosed placental abruption may present
            # to labour)
            elif complication == 'placental_abruption':
                df.at[individual_id, 'la_placental_abruption'] = True
                logger.info(key='maternal_complication', data={'person': individual_id,
                                                               'type': 'placental_abruption',
                                                               'timing': 'intrapartum'})

            elif complication == 'antepartum_haem':
                random_choice = self.rng.choice(['mild_moderate', 'severe'],
                                                p=params['severity_maternal_haemorrhage'])
                df.at[individual_id, f'la_{complication}'] = random_choice

                if random_choice != 'severe':
                    self.sim.modules['PregnancySupervisor'].store_dalys_in_mni(individual_id, 'mild_mod_aph_onset')
                    logger.info(key='maternal_complication', data={'person': individual_id,
                                                                   'type': 'mild_mod_antepartum_haemorrhage',
                                                                   'timing': 'intrapartum'})

                else:
                    self.sim.modules['PregnancySupervisor'].store_dalys_in_mni(individual_id, 'severe_aph_onset')
                    logger.info(key='maternal_complication', data={'person': individual_id,
                                                                   'type': 'severe_antepartum_haemorrhage',
                                                                   'timing': 'intrapartum'})

            elif complication == 'sepsis_chorioamnionitis':
                df.at[individual_id, 'la_sepsis'] = True
                mni[individual_id]['chorio_in_preg'] = True
                self.sim.modules['PregnancySupervisor'].store_dalys_in_mni(individual_id, 'sepsis_onset')
                logger.info(key='maternal_complication', data={'person': individual_id,
                                                               'type': 'sepsis',
                                                               'timing': 'intrapartum'})

            elif complication == 'uterine_rupture':
                df.at[individual_id, 'la_uterine_rupture'] = True
                self.sim.modules['PregnancySupervisor'].store_dalys_in_mni(individual_id, f'{complication}_onset')
                logger.info(key='maternal_complication', data={'person': individual_id,
                                                               'type': 'uterine_rupture',
                                                               'timing': 'intrapartum'})

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
                self.sim.modules['PregnancySupervisor'].store_dalys_in_mni(individual_id, 'sepsis_onset')

                if complication == 'sepsis_endometritis':
                    mni[individual_id]['endo_pp'] = True

            if complication == 'pph_uterine_atony' or complication == 'pph_retained_placenta' or \
               complication == 'pph_other':
                # Set primary complication to true
                df.at[individual_id, 'la_postpartum_haem'] = True

                logger.info(key='maternal_complication', data={'person': individual_id,
                                                               'type': f'{complication}',
                                                               'timing': 'intrapartum'})

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
        mni = self.sim.modules['PregnancySupervisor'].mother_and_newborn_info

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

                logger.info(key='maternal_complication', data={'person': individual_id,
                                                               'type': 'eclampsia',
                                                               'timing': 'intrapartum'})

        # Or from mild to severe gestational hypertension, risk reduced by treatment
        if df.at[individual_id, f'{property_prefix}_htn_disorders'] == 'gest_htn':
            if df.at[individual_id, 'la_maternal_hypertension_treatment']:
                risk_prog_gh_sgh = params['prob_progression_gest_htn'] * params[
                    'anti_htns_treatment_effect_progression']
            else:
                risk_prog_gh_sgh = params['prob_progression_gest_htn']

            if risk_prog_gh_sgh > self.rng.random_sample():
                df.at[individual_id, f'{property_prefix}_htn_disorders'] = 'severe_gest_htn'

                logger.info(key='maternal_complication', data={'person': individual_id,
                                                               'type': 'severe_gest_htn',
                                                               'timing': 'intrapartum'})

        # Or from severe gestational hypertension to severe pre-eclampsia...
        if df.at[individual_id, f'{property_prefix}_htn_disorders'] == 'severe_gest_htn':
            if params['prob_progression_severe_gest_htn'] > self.rng.random_sample():
                df.at[individual_id, f'{property_prefix}_htn_disorders'] = 'severe_pre_eclamp'
                mni[individual_id]['new_onset_spe'] = True

                logger.info(key='maternal_complication', data={'person': individual_id,
                                                               'type': 'severe_pre_eclamp',
                                                               'timing': 'intrapartum'})

        # Or from mild pre-eclampsia to severe pre-eclampsia...
        if df.at[individual_id, f'{property_prefix}_htn_disorders'] == 'mild_pre_eclamp':
            if params['prob_progression_mild_pre_eclamp'] > self.rng.random_sample():
                df.at[individual_id, f'{property_prefix}_htn_disorders'] = 'severe_pre_eclamp'
                mni[individual_id]['new_onset_spe'] = True

                logger.info(key='maternal_complication', data={'person': individual_id,
                                                               'type': 'severe_pre_eclamp',
                                                               'timing': 'intrapartum'})

    def get_potential_causes_of_death(self, individual_id, timing):
        """
        This function is called during LabourDeathAndStillBirthEvent and apply_risk_of_early_postnatal_death function
        and returns a list of any 'death-causing' complications a woman has experience during or after labour
        :param individual_id: individual_id
        :param timing: (STR) 'intrapartum' or 'postpartum'
        """
        df = self.sim.population.props
        mni = self.sim.modules['PregnancySupervisor'].mother_and_newborn_info
        causes = list()

        # Cycle through key complications and add to a list (which matches with the pre-registered causes of death
        # declared in this modules metadata) those which may lead a woman to be at risk of death
        if timing == 'intrapartum':
            if (df.at[individual_id, 'ps_htn_disorders'] == 'severe_pre_eclamp') and \
              mni[individual_id]['new_onset_spe']:
                causes.append('severe_pre_eclampsia')

            if df.at[individual_id, 'ps_htn_disorders'] == 'eclampsia':
                causes.append('eclampsia')

            if df.at[individual_id, 'la_sepsis'] or\
                (df.at[individual_id, 'ps_chorioamnionitis'] and
                 (df.at[individual_id, 'ac_admitted_for_immediate_delivery'] != 'none')):
                causes.append('intrapartum_sepsis')

            if df.at[individual_id, 'la_uterine_rupture']:
                causes.append('uterine_rupture')

            if (df.at[individual_id, 'la_antepartum_haem'] != 'none') or \
                ((df.at[individual_id, 'ps_antepartum_haemorrhage'] != 'none') and
                 (df.at[individual_id, 'ac_admitted_for_immediate_delivery'] != 'none')):
                causes.append('antepartum_haemorrhage')

        if timing == 'postpartum':
            if (df.at[individual_id, 'pn_htn_disorders'] == 'severe_pre_eclamp') and \
               mni[individual_id]['new_onset_spe']:
                causes.append('severe_pre_eclampsia')
            if df.at[individual_id, 'pn_htn_disorders'] == 'eclampsia':
                causes.append('eclampsia')
            if df.at[individual_id, 'la_sepsis_pp'] or df.at[individual_id, 'pn_sepsis_late_postpartum']:
                causes.append('postpartum_sepsis')
            if df.at[individual_id, 'la_postpartum_haem'] or df.at[individual_id, 'pn_postpartum_haem_secondary']:
                causes.append('postpartum_haemorrhage')
            if df.at[individual_id, 'pn_postpartum_haem_secondary']:
                causes.append('secondary_postpartum_haemorrhage')

        return causes

    def apply_risk_of_death(self, individual_id, causes):
        """
        This function is called during LabourDeathAndStillBirthEvent and apply_risk_of_early_postnatal_death function
        and calculates the overall risk of death from one or more complications AND in the instance of multiple
        complications returns the primary cause
        :param individual_id: individual_id
        """
        df = self.sim.population.props
        mni = self.sim.modules['PregnancySupervisor'].mother_and_newborn_info

        risks = dict()

        # Cycle through each 'cause' in the list past through the function, appending onto a dictionary a womans risk of
        # from each complication she has - as calculated from the linear model
        for cause in causes:
            if cause == 'secondary_postpartum_haemorrhage':
                risk = {f'{cause}': self.la_linear_models['postpartum_haemorrhage_death'].predict(
                    df.loc[[individual_id]],
                    received_blood_transfusion=mni[individual_id]['received_blood_transfusion'],)[individual_id]}
            else:
                risk = {f'{cause}': self.la_linear_models[f'{cause}_death'].predict(
                    df.loc[[individual_id]],
                    received_blood_transfusion=mni[individual_id]['received_blood_transfusion'],
                    mode_of_delivery=mni[individual_id]['mode_of_delivery'],
                    chorio_in_preg=mni[individual_id]['chorio_in_preg'])[individual_id]}

            risks.update(risk)

        # Next calculate the overall risk of death from one or more complications
        result = 1

        for cause in risks:
            result *= (1 - risks[cause])

        # If random draw is less that the total risk of death, she will die and the primary cause is then determined
        if self.rng.random_sample() < (1 - result):
            denominator = sum(risks.values())

            probs = list()

            # Cycle over each cause in the dictionary and divide by the sum of the probabilities
            for cause in risks:
                risks[cause] = risks[cause] / denominator
                probs.append(risks[cause])

            # Log the death (eventually this can be removed)
            cause_of_death = self.rng.choice(causes, p=probs)
            # logger.info(key='cause_specific_mortality_fractions', data=risks)  # todo: fix

            # And enact the death via demography
            self.sim.modules['Demography'].do_death(individual_id=individual_id, cause=f'{cause_of_death}',
                                                    originating_module=self.sim.modules['Labour'])
            mni[individual_id]['death_in_labour'] = True
            return True
        else:
            return False

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
        params = self.current_parameters

        # Check the right women are at risk of death
        self.postpartum_characteristics_checker(individual_id)

        # This function cycles through key 'death-causing' complications of the postnatal period. If this woman is
        # experiencing any they are returned in a list
        causes = self.get_potential_causes_of_death(individual_id, timing='postpartum')

        # If the list is not empty she is at risk of death
        if causes:

            # This function then calculates and applies her risk of death in the context of one or more causes
            # (scheduling her death in demography as storing the primary cause of death
            death_eq_outcome = self.apply_risk_of_death(individual_id, causes=causes)

            # If she hasn't died from any complications, we reset some key properties that resolve after risk of death
            # has been applied
            if not death_eq_outcome:

                if df.at[individual_id, 'pn_htn_disorders'] == 'eclampsia':
                    df.at[individual_id, 'pn_htn_disorders'] = 'severe_pre_eclamp'

                if df.at[individual_id, 'pn_htn_disorders'] == 'severe_pre_eclamp':
                    mni[individual_id]['new_onset_spe'] = False

                if df.at[individual_id, 'pn_postpartum_haem_secondary']:
                    df.at[individual_id, 'pn_postpartum_haem_secondary'] = False

                if df.at[individual_id, 'pn_sepsis_late_postpartum']:
                    df.at[individual_id, 'pn_sepsis_late_postpartum'] = False

                df.at[individual_id, 'la_severe_pre_eclampsia_treatment'] = False
                df.at[individual_id, 'la_maternal_hypertension_treatment'] = False
                df.at[individual_id, 'la_eclampsia_treatment'] = False
                df.at[individual_id, 'la_sepsis_treatment'] = False

                mni[individual_id]['retained_placenta'] = False
                mni[individual_id]['uterine_atony'] = False
                self.pph_treatment.unset(
                    [individual_id], 'manual_removal_placenta', 'surgery', 'hysterectomy')

        # ================================ SCHEDULE POSTNATAL WEEK ONE EVENT =====================================
        # For women who have survived first 24 hours after birth we reset all the key labour variables and
        # scheduled them to attend the first event in the PostnatalSupervisorModule - PostnatalWeekOne Event

        if not mni[individual_id]['passed_through_week_one']:

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

            # Labour specific treatment variables
            df.at[individual_id, 'la_antepartum_haem_treatment'] = False
            df.at[individual_id, 'la_uterine_rupture_treatment'] = False

            # This event determines if women will develop complications in week one. We stagger when women
            # arrive at this event to simulate bunching of complications in the first few days after birth
            days_post_birth_td = self.sim.date - df.at[individual_id, 'la_date_most_recent_delivery']
            days_post_birth_int = int(days_post_birth_td / np.timedelta64(1, 'D'))

            assert days_post_birth_int < 6

            day_for_event = int(self.rng.choice([2, 3, 4, 5, 6], p=params['probs_of_attending_pn_event_by_day']))

            # Ensure no women go to this event after week 1
            if day_for_event + days_post_birth_int > 6:
                day_for_event = 1

            self.sim.schedule_event(
                PostnatalWeekOneMaternalEvent(self.sim.modules['PostnatalSupervisor'], individual_id),
                self.sim.date + DateOffset(days=day_for_event))

        if mni[individual_id]['passed_through_week_one']:
            assert individual_id not in self.women_in_labour
            assert ~df.at[individual_id, 'la_currently_in_labour']

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
        mni = self.sim.modules['PregnancySupervisor'].mother_and_newborn_info

        assert individual_id in mni
        if not mni[individual_id]['passed_through_week_one']:
            assert individual_id in self.women_in_labour
        assert df.at[individual_id, 'sex'] == 'F'
        assert df.at[individual_id, 'age_years'] > 14
        assert df.at[individual_id, 'age_years'] < 51
        assert df.at[individual_id, 'la_is_postpartum']

    # ============================================== HSI FUNCTIONS ====================================================
    # Management of each complication is housed within its own function, defined here in the module, and all follow a
    # similar pattern in which consumables are requested and the intervention is delivered if they are available

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

        # params['allowed_interventions'] contains a list of interventions delivered in this module. Removal of
        # interventions from this list within test/analysis will stop this intervention from running
        if 'prophylactic_labour_interventions' not in params['allowed_interventions']:
            return
        else:
            #  We determine if the HCW will administer antibiotics for women with premature rupture of membranes
            if df.at[person_id, 'ps_premature_rupture_of_membranes']:

                # The mother may have received these antibiotics already if she presented to the antenatal ward from the
                # community following PROM. We store this in the mni dictionary
                if df.at[person_id, 'ac_received_abx_for_prom']:
                    mni[person_id]['abx_for_prom_given'] = True

                else:
                    # If she has not already receive antibiotics, we check for consumables
                    avail = hsi_event.get_consumables(item_codes=self.item_codes_lab_consumables['abx_for_prom'])

                    # Then query if these consumables are available during this HSI And provide if available.
                    # Antibiotics for from reduce risk of newborn sepsis within the first
                    # week of life
                    if avail:
                        mni[person_id]['abx_for_prom_given'] = True

            # ------------------------------ STEROIDS FOR PRETERM LABOUR -------------------------------
            # Next we see if women in pre term labour will receive antenatal corticosteroids
            if mni[person_id]['labour_state'] == 'early_preterm_labour' or \
               mni[person_id]['labour_state'] == 'late_preterm_labour':

                avail = hsi_event.get_consumables(item_codes=self.item_codes_lab_consumables['antenatal_steroids'])

                # If available they are given. Antenatal steroids reduce a preterm newborns chance of developing
                # respiratory distress syndrome and of death associated with prematurity
                if avail:
                    mni[person_id]['corticosteroids_given'] = True

    def determine_delivery_mode_in_spe_or_ec(self, person_id, complication):
        """
        This function is called following treatment for either severe pre-eclampsia or eclampsia during labour and
        determines if women with these conditions will undergo vaginal, assisted vaginal or caesarean section due to
         their complication
        :param person_id: mothers individual id
        :param complication: (STR) severe_pre_eclamp OR eclampsia
        """
        params = self.current_parameters
        mni = self.sim.modules['PregnancySupervisor'].mother_and_newborn_info

        # We use a weighted random draw to determine the delivery mode of this woman depending on her complication
        # (eclampsia, the more severe of the two conditions, is assumed to be more likely to lead to caesarean section)
        delivery_modes = ['vaginal', 'avd', 'cs']
        mode = self.rng.choice(delivery_modes, p=params[f'prob_delivery_modes_{complication}'])

        if mode == 'avd':
            mni[person_id]['mode_of_delivery'] = 'instrumental'

        elif mode == 'cs':
            mni[person_id]['referred_for_cs'] = True
            mni[person_id]['cs_indication'] = 'spe_ec'

    def assessment_and_treatment_of_severe_pre_eclampsia_mgso4(self, hsi_event, labour_stage):
        """This function represents the diagnosis and management of severe pre-eclampsia during labour. This function
        defines the required consumables and administers the intervention if available. The intervention is
        intravenous magnesium sulphate.  It is called by HSI_Labour_PresentsForSkilledBirthAttendanceInLabour or
        HSI_Labour_ReceivesPostnatalCheck
        :param hsi_event: HSI event in which the function has been called:
        (STR) 'hc' == health centre, 'hp' == hospital
        :param labour_stage: intrapartum or postpartum period of labour (STR) 'ip' or 'pp':
        """
        df = self.sim.population.props
        params = self.current_parameters
        person_id = hsi_event.target

        # Women who have been admitted for delivery due to severe pre-eclampsia AND have already received magnesium
        # before moving to the labour ward do not receive the intervention again
        if (df.at[person_id, 'ac_admitted_for_immediate_delivery'] != 'none') and \
           df.at[person_id, 'ac_mag_sulph_treatment']:
            return

        if ('assessment_and_treatment_of_severe_pre_eclampsia' not in params['allowed_interventions']) or \
           (df.at[person_id, 'la_severe_pre_eclampsia_treatment'] and (labour_stage == 'pp')):
            return

        if (df.at[person_id, 'ps_htn_disorders'] == 'severe_pre_eclamp') or \
           (df.at[person_id, 'pn_htn_disorders'] == 'severe_pre_eclamp'):

            # Determine if this person will deliver vaginally or via caesarean
            if (df.at[person_id, 'ac_admitted_for_immediate_delivery'] == 'none') and (labour_stage == 'ip'):
                self.determine_delivery_mode_in_spe_or_ec(person_id, 'spe')

            # Define the required consumables
            consumables = hsi_event.get_consumables(item_codes=self.item_codes_lab_consumables['severe_pre_eclampsia'],
                                                    return_individual_results=True)
            key_consumable_avail = list(consumables.values())[0]

            # If the consumables are available - the intervention is delivered. IV magnesium reduces the
            # probability that a woman with severe pre-eclampsia will experience eclampsia in labour
            if key_consumable_avail:
                df.at[person_id, 'la_severe_pre_eclampsia_treatment'] = True

    def assessment_and_treatment_of_hypertension(self, hsi_event):
        """
        This function represents the diagnosis and management of hypertension during labour. This function
        defines the required consumable  and administers the intervention if available. The intervention is
        intravenous magnesium sulphate.  It is called by HSI_Labour_PresentsForSkilledBirthAttendanceInLabour or
        HSI_Labour_ReceivesPostnatalCheck
        :param hsi_event: HSI event in which the function has been called:
        (STR) 'hc' == health centre, 'hp' == hospital
        """
        df = self.sim.population.props
        person_id = hsi_event.target
        params = self.current_parameters

        # If the treatment is not allowed to be delivered or it has already been delivered the function won't run
        if ('assessment_and_treatment_of_hypertension' not in params['allowed_interventions']) or \
           df.at[person_id, 'ac_iv_anti_htn_treatment'] or df.at[person_id, 'la_maternal_hypertension_treatment']:
            return
        else:

            if (df.at[person_id, 'ps_htn_disorders'] != 'none') or (df.at[person_id, 'pn_htn_disorders'] != 'none'):

                # Then query if these consumables are available during this HSI
                avail = hsi_event.get_consumables(item_codes=self.item_codes_lab_consumables['iv_antihypertensives'])

                # If they are available then the woman is started on treatment. Intravenous antihypertensive reduce a
                # womans risk of progression from mild to severe gestational hypertension ANd reduce risk of death for
                # women with severe pre-eclampsia and eclampsia
                if avail:
                    df.at[person_id, 'la_maternal_hypertension_treatment'] = True

                    avail = hsi_event.get_consumables(
                        item_codes=self.item_codes_lab_consumables['oral_antihypertensives'])

                    if avail:
                        df.at[person_id, 'la_gest_htn_on_treatment'] = True

    def assessment_and_treatment_of_eclampsia(self, hsi_event, labour_stage):
        """
        This function represents the diagnosis and management of eclampsia during or following labour. This function
        defines the required consumables and administers the intervention if available. The intervention is
        intravenous magnesium sulphate.  It is called by HSI_Labour_PresentsForSkilledBirthAttendanceInLabour or
        HSI_Labour_ReceivesPostnatalCheck
        :param hsi_event: HSI event in which the function has been called:
        (STR) 'hc' == health centre, 'hp' == hospital
        :param labour_stage: intrapartum or postpartum period of labour (STR) 'ip' or 'pp':
        """
        df = self.sim.population.props
        person_id = hsi_event.target
        params = self.current_parameters

        if 'assessment_and_treatment_of_eclampsia' not in params['allowed_interventions']:
            return

        elif (df.at[person_id, 'ps_htn_disorders'] == 'eclampsia') or (df.at[person_id, 'pn_htn_disorders'] ==
                                                                       'eclampsia'):

            consumables = hsi_event.get_consumables(item_codes=self.item_codes_lab_consumables['severe_pre_eclampsia'],
                                                    return_individual_results=True)
            key_consumable_avail = list(consumables.values())[0]

            if (labour_stage == 'ip') and (df.at[person_id, 'ac_admitted_for_immediate_delivery'] == 'none'):
                self.determine_delivery_mode_in_spe_or_ec(person_id, 'ec')

            if key_consumable_avail:
                # Treatment with magnesium reduces a womans risk of death from eclampsia
                df.at[person_id, 'la_eclampsia_treatment'] = True

    def assessment_and_treatment_of_obstructed_labour_via_avd(self, hsi_event):
        """
        This function represents the diagnosis and management of obstructed labour during labour. This function
        defines the required consumables and administers the intervention if available. The intervention in this
        function is assisted vaginal delivery. It is called by either HSI_Labour_PresentsForSkilledBirthAttendanceIn
        Labour
        :param hsi_event: HSI event in which the function has been called:
        (STR) 'hc' == health centre, 'hp' == hospital
        """
        df = self.sim.population.props
        mni = self.sim.modules['PregnancySupervisor'].mother_and_newborn_info
        params = self.current_parameters
        person_id = hsi_event.target
        # consumables = self.sim.modules['HealthSystem'].parameters['Consumables']

        if ('assessment_and_treatment_of_obstructed_labour' not in params['allowed_interventions']) or \
            (df.at[person_id, 'ac_admitted_for_immediate_delivery'] == 'caesarean_now') or \
           (df.at[person_id, 'ac_admitted_for_immediate_delivery'] == 'caesarean_future'):
            return

        elif df.at[person_id, 'la_obstructed_labour']:
            avail_pkg = hsi_event.get_consumables(item_codes=self.item_codes_lab_consumables['obstructed_labour'])
            avail_forceps = hsi_event.get_consumables(item_codes=self.item_codes_lab_consumables['forceps'])
            avail_vacuum = hsi_event.get_consumables(item_codes=self.item_codes_lab_consumables['vacuum'])

            # We assume women with CPD cannot be delivered via AVD and will require a caesarean
            if not mni[person_id]['cpd']:
                # If the general package is available AND the facility has the correct tools to carry out the
                # delivery then it can occur
                if avail_pkg and (avail_forceps or avail_vacuum):

                    # If AVD was successful then we record the mode of delivery. We use this variable to reduce
                    # risk of intrapartum still birth when applying risk in the death event
                    if params['prob_successful_assisted_vaginal_delivery'] > self.rng.random_sample():
                        mni[person_id]['mode_of_delivery'] = 'instrumental'

                    else:
                        # If unsuccessful, this woman will require a caesarean section
                        mni[person_id]['referred_for_cs'] = True
                        mni[person_id]['cs_indication'] = 'ol_failed_avd'
                        logger.info(key='cs_ol', data={'person': person_id})

            else:
                mni[person_id]['referred_for_cs'] = True
                mni[person_id]['cs_indication'] = 'ol'
                logger.info(key='cs_ol', data={'person': person_id})

    def assessment_and_treatment_of_maternal_sepsis(self, hsi_event, labour_stage):
        """
        This function represents the diagnosis and management of maternal sepsis during or following labour. This
        function defines the required consumables and administers the intervention if they are available. The
        intervention in this function is maternal sepsis case management. It is called by either
         HSI_Labour_PresentsForSkilledBirthAttendanceInLabour OR  HSI_Labour_ReceivesPostnatalCheck
        :param hsi_event: HSI event in which the function has been called:
        (STR) 'hc' == health centre, 'hp' == hospital
        :param labour_stage: intrapartum or postpartum period of labour (STR) 'ip' or 'pp':
        """
        df = self.sim.population.props
        params = self.current_parameters
        person_id = hsi_event.target

        if 'assessment_and_treatment_of_maternal_sepsis' not in params['allowed_interventions']:
            return
        else:
            if (
                df.at[person_id, 'la_sepsis'] or
                df.at[person_id, 'la_sepsis_pp'] or
                ((labour_stage == 'ip') and df.at[person_id, 'ps_chorioamnionitis'] and
                 (df.at[person_id, 'ac_admitted_for_immediate_delivery'] != 'none')) or
               (labour_stage == 'pp' and df.at[person_id, 'pn_sepsis_late_postpartum'])):

                consumables = hsi_event.get_consumables(item_codes=self.item_codes_lab_consumables['maternal_sepsis'],
                                                        return_individual_results=True)
                abx_1_avail = list(consumables.values())[0]
                abx_2_avail = list(consumables.values())[1]

                # If delivered this intervention reduces a womans risk of dying from sepsis
                if abx_1_avail and abx_2_avail:
                    df.at[person_id, 'la_sepsis_treatment'] = True

    def assessment_and_plan_for_antepartum_haemorrhage(self, hsi_event):
        """
        This function represents the diagnosis of antepartum haemorrhage during  labour. This
        function ensures that woman is referred for comprehensive care via caesarean section and blood transfusion.
        It is called by  HSI_Labour_PresentsForSkilledBirthAttendanceInLabour
        :param hsi_event: HSI event in which the function has been called:
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

            elif df.at[person_id, 'la_antepartum_haem'] != 'none':

                # Caesarean delivery reduces the risk of intrapartum still birth and blood transfusion reduces the risk
                # of maternal death due to bleeding
                mni[person_id]['referred_for_cs'] = True
                mni[person_id]['cs_indication'] = 'la_aph'
                mni[person_id]['referred_for_blood'] = True

    def assessment_for_referral_uterine_rupture(self, hsi_event):
        """
        This function represents the diagnosis of uterine rupture during  labour and ensures
        that a woman is referred for comprehensive care via caesarean section, surgical repair and blood transfusion.
        It is called by either HSI_Labour_PresentsForSkilledBirthAttendanceInLabour
        :param hsi_event: HSI event in which the function has been called:
        (STR) 'hc' == health centre, 'hp' == hospital
        """
        df = self.sim.population.props
        params = self.current_parameters
        mni = self.sim.modules['PregnancySupervisor'].mother_and_newborn_info
        person_id = hsi_event.target

        if 'assessment_and_plan_for_referral_uterine_rupture' not in params['allowed_interventions']:
            return

        elif df.at[person_id, 'la_uterine_rupture']:
            mni[person_id]['referred_for_surgery'] = True
            mni[person_id]['referred_for_cs'] = True
            mni[person_id]['cs_indication'] = 'ur'
            mni[person_id]['referred_for_blood'] = True

    def active_management_of_the_third_stage_of_labour(self, hsi_event):
        """
        This function represents the administration of active management of the third stage of labour. This
        function checks the availability of consumables and delivers the intervention accordingly. It is called by
        HSI_Labour_PresentsForSkilledBirthAttendanceFollowingLabour
        :param hsi_event: HSI event in which the function has been called:
        """
        mni = self.sim.modules['PregnancySupervisor'].mother_and_newborn_info
        params = self.current_parameters
        person_id = hsi_event.target

        if 'active_management_of_the_third_stage_of_labour' not in params['allowed_interventions']:
            return
        else:
            avail = hsi_event.get_consumables(item_codes=self.item_codes_lab_consumables['amtsl'])

            # This treatment reduces a womans risk of developing uterine atony AND retained placenta, both of which are
            # preceding causes of postpartum haemorrhage
            if avail:
                mni[person_id]['amtsl_given'] = True

    def assessment_and_treatment_of_pph_uterine_atony(self, hsi_event):
        """
        This function represents the diagnosis and management of postpartum haemorrhage secondary to uterine atony
        following labour. This function defines the required consumables and administers the intervention if they are
        available. The intervention in this function is  intravenous uterotonics followed by referral for further care
        in the event of continued haemorrhage. It is called by HSI_Labour_ReceivesPostnatalCheck
        :param hsi_event: HSI event in which the function has been called
        """
        df = self.sim.population.props
        mni = self.sim.modules['PregnancySupervisor'].mother_and_newborn_info
        params = self.current_parameters
        person_id = hsi_event.target

        if 'assessment_and_treatment_of_pph_uterine_atony' not in params['allowed_interventions']:
            return

        elif df.at[person_id, 'la_postpartum_haem'] and ~mni[person_id]['retained_placenta']:

            consumables = hsi_event.get_consumables(item_codes=self.item_codes_lab_consumables['pph'],
                                                    return_individual_results=True)
            uto_1_avail = list(consumables.values())[0]
            uto_2_avail = list(consumables.values())[1]

            if uto_1_avail and uto_2_avail:

                # We apply a probability that this treatment will stop a womans bleeding in the first instance
                # meaning she will not require further treatment
                if params['prob_haemostatis_uterotonics'] > self.rng.random_sample():

                    # Bleeding has stopped, this woman will not be at risk of death
                    df.at[person_id, 'la_postpartum_haem'] = False
                    mni[person_id]['uterine_atony'] = False

                # If uterotonics do not stop bleeding the woman is referred for additional treatment
                else:
                    mni[person_id]['referred_for_surgery'] = True
                    mni[person_id]['referred_for_blood'] = True
                    return True

    def assessment_and_treatment_of_pph_retained_placenta(self, hsi_event):
        """
        This function represents the diagnosis and management of postpartum haemorrhage secondary to retained placenta
        following labour. This function defines the required consumables and administers the intervention if they are
        available. The intervention in this function is manual removal of placenta (bedside) followed by referral for
        further care in the event of continued haemorrhage. It is called by HSI_Labour_ReceivesPostnatalCheck
        :param hsi_event: HSI event in which the function has been called:
        """
        df = self.sim.population.props
        mni = self.sim.modules['PregnancySupervisor'].mother_and_newborn_info
        params = self.current_parameters
        person_id = hsi_event.target

        if 'assessment_and_treatment_of_pph_retained_placenta' not in params['allowed_interventions']:
            return

        elif (
            (df.at[person_id, 'la_postpartum_haem'] and mni[person_id]['retained_placenta']) or
            df.at[person_id, 'pn_postpartum_haem_secondary']
             ):

            # Log the consumables but dont condition the treatment on their availability - the primary mechanism of this
            # intervention doesnt require consumables
            hsi_event.get_consumables(item_codes=self.item_codes_lab_consumables['pph'])

            # Similar to uterotonics we apply a probability that this intervention will successfully stop
            # bleeding to ensure some women go on to require further care
            if params['prob_successful_manual_removal_placenta'] > self.rng.random_sample():

                df.at[person_id, 'la_postpartum_haem'] = False
                mni[person_id]['retained_placenta'] = False

                if df.at[person_id, 'pn_postpartum_haem_secondary']:
                    df.at[person_id, 'pn_postpartum_haem_secondary'] = False

            else:
                mni[person_id]['referred_for_surgery'] = True
                mni[person_id]['referred_for_blood'] = True

    def surgical_management_of_pph(self, hsi_event):
        """
        This function represents the surgical management of postpartum haemorrhage (all-cause) following labour. This
        function is called during HSI_Labour_ReceivesComprehensiveEmergencyObstetricCare for women who have PPH and
        medical management has failed.
        :param hsi_event: HSI event in which the function has been called
        """
        person_id = hsi_event.target
        df = self.sim.population.props
        params = self.current_parameters
        mni = self.sim.modules['PregnancySupervisor'].mother_and_newborn_info

        if not mni[person_id]['retained_placenta']:

            # We apply a probability that surgical techniques will be effective
            treatment_success_pph = params['success_rate_pph_surgery'] > self.rng.random_sample()

            # We log the log the required consumables and condition the surgery happening on the availability of the
            # first consumable in this package, the anaesthetic required for the surgery
            consumables = hsi_event.get_consumables(item_codes=self.item_codes_lab_consumables['obstetric_surgery'],
                                                    return_individual_results=True)
            key_consumable_avail = list(consumables.values())[0]

            # And store the treatment which will dramatically reduce risk of death
            if treatment_success_pph and key_consumable_avail:
                logger.debug(key='msg',
                                 data=f'mother {person_id} undergone surgery to manage her PPH which resolved')
                self.pph_treatment.set(person_id, 'surgery')

            # If the treatment is unsuccessful then women will require a hysterectomy to stop the bleeding
            elif ~treatment_success_pph and key_consumable_avail:
                logger.debug(key='msg', data=f'mother {person_id} undergone surgery to manage her PPH, she required'
                                             f' a hysterectomy to stop the bleeding')

                self.pph_treatment.set(person_id, 'hysterectomy')
                df.at[person_id, 'la_has_had_hysterectomy'] = True

        # Next we apply the effect of surgical treatment for women with retained placenta
        elif mni[person_id]['retained_placenta'] and not self.pph_treatment.has_all(person_id, 'manual_removal_'
                                                                                               'placenta'):

            self.pph_treatment.set(person_id, 'surgery')
            logger.debug(key='msg',
                         data=f'mother {person_id} undergone surgical removal of a retained placenta ')

    def blood_transfusion(self, hsi_event):
        """
        This function represents the blood transfusion during or after labour. This function is called during
        HSI_Labour_ReceivesComprehensiveEmergencyObstetricCare for women who have experience blood loss due to
        antepartum haemorrhage, postpartum haemorrhage or uterine rupture
        :param hsi_event: HSI event in which the function has been called
        """
        person_id = hsi_event.target
        mni = self.sim.modules['PregnancySupervisor'].mother_and_newborn_info
        params = self.current_parameters
        df = self.sim.population.props

        # Check consumables
        avail = hsi_event.get_consumables(item_codes=self.item_codes_lab_consumables['blood_transfusion'])

        if avail:
            mni[person_id]['received_blood_transfusion'] = True

            # We assume that anaemia is corrected by blood transfusion
            if df.at[person_id, 'pn_anaemia_following_pregnancy'] != 'none':
                if params['treatment_effect_blood_transfusion_anaemia'] > self.rng.random_sample():

                    self.sim.modules['PregnancySupervisor'].store_dalys_in_mni(
                        person_id, 'severe_anaemia_resolution')
                    df.at[person_id, 'pn_anaemia_following_pregnancy'] = 'none'

    def assessment_and_treatment_of_anaemia(self, hsi_event):
        """
        This function represents the management of postnatal anaemia including iron and folic acid administration and
        blood testing. Women with severe anaemia are scheduled to receive blood. It is called during
        HSI_Labour_ReceivesPostnatalCheck
        :param hsi_event: HSI event in which the function has been called
        """
        df = self.sim.population.props
        person_id = hsi_event.target
        params = self.current_parameters
        mother = df.loc[person_id]
        mni = self.sim.modules['PregnancySupervisor'].mother_and_newborn_info

        # Use dx_test function to assess anaemia status
        test_result = self.sim.modules['HealthSystem'].dx_manager.run_dx_test(
            dx_tests_to_run='full_blood_count_hb_pn', hsi_event=hsi_event)

        # Check consumables
        if test_result:
            # Start iron and folic acid supplementation for women not already receiving this
            if ~mother.la_iron_folic_acid_postnatal:

                days = int((6 - df.at[person_id, 'pn_postnatal_period_in_weeks']) * 7)
                cons = {_i: days for _i in self.item_codes_lab_consumables['iron_folic_acid']}
                avail = hsi_event.get_consumables(item_codes=cons)

                # Start iron and folic acid treatment
                if avail:
                    df.at[person_id, 'la_iron_folic_acid_postnatal'] = True

                    if self.rng.random_sample() < params['effect_of_ifa_for_resolving_anaemia']:
                        # Store date of resolution for daly calculations
                        self.sim.modules['PregnancySupervisor'].store_dalys_in_mni(
                            person_id, f'{df.at[person_id, "pn_anaemia_following_pregnancy"]}_anaemia_'
                                       f'resolution')

                        df.at[person_id, 'pn_anaemia_following_pregnancy'] = 'none'

        # Severe anaemia would require treatment via blood transfusion
            if mother.pn_anaemia_following_pregnancy == 'severe':
                mni[person_id]['referred_for_blood'] = True

    def assessment_for_depression(self, hsi_event):
        """
        This function schedules depression screening for women who attend postnatal care via the Depression module. It
        is called within HSI_Labour_ReceivesPostnatalCheck.
        :param hsi_event: HSI event in which the function has been called
        """
        df = self.sim.population.props
        person_id = hsi_event.target

        if 'Depression' in self.sim.modules.keys():
            if not df.at[person_id, 'de_ever_diagnosed_depression']:
                self.sim.modules['Depression'].do_when_suspected_depression(person_id, hsi_event)

    def interventions_delivered_pre_discharge(self, hsi_event):
        """
        This function contains the interventions that are delivered to women prior to discharge. This are considered
        part of essential postnatal care and currently include testing for HIV and postnatal iron and folic acid
        supplementation. It is called by HSI_Labour_ReceivesPostnatalCheck
        :param hsi_event: HSI event in which the function has been called:
        """
        df = self.sim.population.props
        person_id = int(hsi_event.target)
        params = self.current_parameters

        # HIV testing occurs within the HIV module for women who havent already been diagnosed
        if 'Hiv' in self.sim.modules.keys():
            if ~df.at[person_id, 'hv_diagnosed']:
                self.sim.modules['HealthSystem'].schedule_hsi_event(
                    HSI_Hiv_TestAndRefer(person_id=person_id, module=self.sim.modules['Hiv']),
                    topen=self.sim.date,
                    tclose=None,
                    priority=0)

        # ------------------------------- Postnatal iron and folic acid ---------------------------------------------
        cons = {_i: 93 for _i in self.item_codes_lab_consumables['iron_folic_acid']}
        avail = hsi_event.get_consumables(item_codes=cons)

        # Women are started on iron and folic acid for the next three months which reduces risk of anaemia in the
        # postnatal period
        if avail and (self.rng.random_sample() < params['prob_adherent_ifa']):
            df.at[person_id, 'la_iron_folic_acid_postnatal'] = True

    def set_labour_mni_variables(self, individual_id):
        """
        This function updates the mother_and_newborn_info entry for each woman with new variables that store key pieces
        of information required for the labour module to run
        :param individual_id: womans individual id
        """
        df = self.sim.population.props
        mni = self.sim.modules['PregnancySupervisor'].mother_and_newborn_info

        # Append labor specific variables to the mni
        labour_variables = {'labour_state': None,
                            # Term Labour (TL), Early Preterm (EPTL), Late Preterm (LPTL) or Post Term (POTL)
                            'birth_weight': 'normal_birth_weight',
                            'birth_size': 'average_for_gestational_age',
                            'delivery_setting': None,  # home_birth, health_centre, hospital
                            'twins': df.at[individual_id, 'ps_multiple_pregnancy'],
                            'twin_count': 0,
                            'twin_one_comps': False,
                            'pnc_twin_one': 'none',
                            'bf_status_twin_one': 'none',
                            'eibf_status_twin_one': False,
                            'an_placental_abruption': df.at[individual_id, 'ps_placental_abruption'],
                            'corticosteroids_given': False,
                            'clean_birth_practices': False,
                            'abx_for_prom_given': False,
                            'abx_for_pprom_given': False,
                            'endo_pp': False,
                            'retained_placenta': False,
                            'uterine_atony': False,
                            'amtsl_given': False,
                            'cpd': False,
                            'mode_of_delivery': 'vaginal_delivery',
                            # vaginal_delivery, instrumental, caesarean_section
                            'hsi_cant_run': False,  # True (T) or False (F)
                            'sought_care_for_complication': False,  # True (T) or False (F)
                            'sought_care_labour_phase': 'none',
                            'referred_for_cs': False,  # True (T) or False (F)
                            'referred_for_blood': False,  # True (T) or False (F)
                            'received_blood_transfusion': False,  # True (T) or False (F)
                            'referred_for_surgery': False,  # True (T) or False (F)'
                            'death_in_labour': False,  # True (T) or False (F)
                            'cause_of_death_in_labour': [],
                            'single_twin_still_birth': False,  # True (T) or False (F)
                            'will_receive_pnc': 'none',
                            'passed_through_week_one': False
                            }

        mni[individual_id].update(labour_variables)

    def run_if_receives_skilled_birth_attendance_cant_run(self, hsi_event):
        """
        This function is called by HSI_Labour_PresentsForSkilledBirthAttendanceFollowingLabour if the HSI is unable to
        run on the date it has been scheduled for. For these women who cannot deliver in a facility despite presenting
         for care, they will be scheduled to undergo a home birth via LabourAtHomeEvent and will continue with the
         correct schedule of labour events
        :param hsi_event: HSI event in which the function has been called:
        """
        person_id = hsi_event.target
        mni = self.sim.modules['PregnancySupervisor'].mother_and_newborn_info
        logger.info(key='message', data=f'HSI_Labour_ReceivesSkilledBirthAttendanceDuringLabour will not run for '
                                        f'{person_id}')

        # Women may have presented to HSI_Labour_PresentsForSkilledBirthAttendanceFollowingLabour following
        # complications at a home birth. Those women should not be scheduled to LabourAtHomeEvent
        if not mni[person_id]['sought_care_for_complication']:
            mni[person_id]['delivery_setting'] = 'home_birth'
            self.sim.schedule_event(LabourAtHomeEvent(self, person_id), self.sim.date)
            mni[person_id]['hsi_cant_run'] = True

    def run_if_receives_postnatal_check_cant_run(self, hsi_event):
        """
        This function is called by HSI_Labour_ReceivesPostnatalCheck if the HSI is unable to
        run on the date it has been scheduled for. For these women who have may have experienced life threatening
        complications we apply risk of death in this function, as it would have been applied in the HSI which can not
        run
        :param hsi_event: HSI event in which the function has been called:
        """
        person_id = hsi_event.target
        logger.info(key='message', data=f'HSI_Labour_ReceivesPostnatalCheck will not run for {person_id}')
        self.apply_risk_of_early_postpartum_death(person_id)

    def run_if_receives_comprehensive_emergency_obstetric_care_cant_run(self, hsi_event):
        """
        This function is called by HSI_Labour_ReceivesComprehensiveEmergencyObstetricCare if the HSI is unable to
        run on the date it has been scheduled for. For these women who have may have experienced life threatening
        complications we apply risk of death in this function, as it would have been applied in the HSI which can not
        run
        :param hsi_event: HSI event in which the function has been called:
        """
        person_id = hsi_event.target
        logger.debug(key='message', data=f'HSI_Labour_ReceivesComprehensiveEmergencyObstetricCare will not run for'
                                         f' {person_id}')

        # For women referred to this event after the postnatal SBA HSI we apply risk of death (as if should have been
        # applied in this event if it ran)
        if hsi_event.timing == 'postpartum':
            self.apply_risk_of_early_postpartum_death(person_id)


class LabourOnsetEvent(Event, IndividualScopeEventMixin):
    """
    This is the LabourOnsetEvent. It is scheduled by the set_date_of_labour function for all women who are newly
    pregnant. In addition it is scheduled via the Pregnancy Supervisor module for women who are going into preterm
    labour and by the Care of Women During Pregnancy module for women who require emergency delivery as an intervention
    to treat a pregnancy-threatening complication.

    It represents the start of a womans labour and is the first event all woman who are going to give birth
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

            # Update the mni with new variables
            self.module.set_labour_mni_variables(individual_id)

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

                mni[individual_id]['labour_state'] = 'term_labour'

            # Here we allow a woman to go into early preterm labour with a gestational age of 23 (limit is 24) to
            # account for PregnancySupervisor only updating weekly
            elif params['list_limits_for_'
                        'defining_term_status'][2] <= gestational_age_in_days <= params['list_limits_for_'
                                                                                        'defining_term_status'][3]:

                mni[individual_id]['labour_state'] = 'early_preterm_labour'
                df.at[individual_id, 'la_has_previously_delivered_preterm'] = True

                logger.info(key='maternal_complication', data={'person': individual_id,
                                                               'type': 'early_preterm_labour',
                                                               'timing': 'intrapartum'})

            elif params['list_limits_for_defining_term_status'][4] <= gestational_age_in_days <= params['list_limits'
                                                                                                        '_for_defining'
                                                                                                        '_term_'
                                                                                                        'status'][5]:

                mni[individual_id]['labour_state'] = 'late_preterm_labour'
                df.at[individual_id, 'la_has_previously_delivered_preterm'] = True

                logger.info(key='maternal_complication', data={'person': individual_id,
                                                               'type': 'late_preterm_labour',
                                                               'timing': 'intrapartum'})

            elif gestational_age_in_days >= params['list_limits_for_defining_term_status'][6]:

                mni[individual_id]['labour_state'] = 'postterm_labour'

                logger.info(key='maternal_complication', data={'person': individual_id,
                                                               'type': 'post_term_labour',
                                                               'timing': 'intrapartum'})

            # We check all women have had their labour state set
            assert mni[individual_id]['labour_state'] is not None
            labour_state = mni[individual_id]['labour_state']
            logger.info(key='message', data=f'This is LabourOnsetEvent, person {individual_id} has now gone into '
                                            f'{labour_state} on date {self.sim.date}')

            # ----------------------------------- FOETAL WEIGHT/BIRTH WEIGHT ----------------------------------------
            # Here we determine the weight of the foetus being carried by this mother, this is calculated here to allow
            # the size of the baby to effect risk of certain maternal complications (i.e. obstructed labour)
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
                if self.module.rng.random_sample() < params['residual_prob_of_macrosomia']:
                    mni[individual_id]['birth_weight'] = 'macrosomia'
                else:
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
                pred_hp_delivery = params['probability_delivery_hospital']

                # The denominator is calculated
                denom = pred_hp_delivery + pred_hb_delivery + pred_hc_delivery

                # Followed by the probability of each of the three outcomes - home birth, health centre birth or
                # hospital birth
                prob_hb = pred_hb_delivery / denom
                prob_hc = pred_hc_delivery / denom
                prob_hp = pred_hp_delivery / denom

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
                    self.module, person_id=individual_id, facility_level_of_this_hsi='1a')
                self.sim.modules['HealthSystem'].schedule_hsi_event(health_centre_delivery, priority=0,
                                                                    topen=self.sim.date,
                                                                    tclose=self.sim.date + DateOffset(days=1))

                logger.debug(key='message', data=f'This is LabourOnsetEvent, scheduling '
                                                 f'HSI_Labour_PresentsForSkilledAttendanceInLabour on date '
                                                 f'{self.sim.date} for person {individual_id} as they have chosen to '
                                                 f'seek care at a health centre for delivery')

            elif mni[individual_id]['delivery_setting'] == 'hospital':
                facility_level = self.module.rng.choice(['1a', '1b'])
                hospital_delivery = HSI_Labour_ReceivesSkilledBirthAttendanceDuringLabour(
                    self.module, person_id=individual_id, facility_level_of_this_hsi=facility_level)
                self.sim.modules['HealthSystem'].schedule_hsi_event(hospital_delivery, priority=0,
                                                                    topen=self.sim.date,
                                                                    tclose=self.sim.date + DateOffset(days=1))

                logger.debug(key='message', data=f'This is LabourOnsetEvent, scheduling '
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
            self.sim.schedule_event(BirthAndPostnatalOutcomesEvent(self.module, individual_id), self.sim.date +
                                    DateOffset(days=5))

            logger.debug(key='message', data=f'This is LabourOnsetEvent scheduling a birth on date to mother'
                                             f' {individual_id} which will occur on '
                                             f'{self.sim.date + DateOffset(days=5)}')


class LabourAtHomeEvent(Event, IndividualScopeEventMixin):
    """
    This is the LabourAtHomeEvent. It is scheduled by the LabourOnsetEvent for women who will not seek delivery care at
    a  health facility. This event applies the probability that women delivering at home will experience
    complications associated with the intrapartum phase of labour and makes the appropriate changes to the data frame.
     Additionally this event applies a probability that women who develop complications during a home birth may choose
     to seek care from at a health facility. In that case the appropriate HSI is scheduled.
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

        # ===================================  APPLICATION OF COMPLICATIONS ===========================================
        # Using the complication_application function we loop through each complication and determine if a woman
        # will experience any of these if she has delivered at home
        for complication in ['obstruction_cpd', 'obstruction_malpos_malpres', 'obstruction_other',
                             'placental_abruption', 'antepartum_haem', 'sepsis_chorioamnionitis', 'uterine_rupture']:
            self.module.set_intrapartum_complications(individual_id, complication=complication)

        if df.at[individual_id, 'la_obstructed_labour']:
            logger.info(key='maternal_complication', data={'mother': individual_id,
                                                           'type': 'obstructed_labour',
                                                           'timing': 'intrapartum'})

        # And we determine if any existing hypertensive disorders would worsen
        self.module.progression_of_hypertensive_disorders(individual_id, property_prefix='ps')

        # ==============================  CARE SEEKING FOLLOWING COMPLICATIONS ========================================
        # Next we determine if women who develop a complication during a home birth will seek care

        # (Women who have been scheduled a home birth after seeking care at a facility that didnt have capacity to
        # deliver the HSI will not try to seek care if they develop a complication)
        if not mni[individual_id]['hsi_cant_run']:

            if df.at[individual_id, 'la_obstructed_labour'] or \
                (df.at[individual_id, 'la_antepartum_haem'] != 'none') or \
                df.at[individual_id, 'la_sepsis'] or \
                (df.at[individual_id, 'ps_htn_disorders'] == 'eclampsia') or \
                ((df.at[individual_id, 'ps_htn_disorders'] == 'severe_pre_eclamp') and
                 mni[individual_id]['new_onset_spe']) or df.at[individual_id, 'la_uterine_rupture']:

                if self.module.rng.random_sample() < params['prob_careseeking_for_complication']:

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
        mni = self.sim.modules['PregnancySupervisor'].mother_and_newborn_info
        params = self.module.current_parameters

        if not df.at[individual_id, 'is_alive']:
            return

        # Check the correct amount of time has passed between labour onset and postpartum event
        assert (self.sim.date - df.at[individual_id, 'la_due_date_current_pregnancy']) == pd.to_timedelta(4, unit='D')
        self.module.labour_characteristics_checker(individual_id)

        outcome_death_equations = False

        # This function cycles through key 'death-causing' complications of the intrapartum period. If this woman is
        # experiencing any they are returned in a list
        causes = self.module.get_potential_causes_of_death(individual_id, timing='intrapartum')

        # If the list is not empty she is at risk of death
        if causes:

            # This function then calculates and applies her risk of death in the context of one or more causes
            # (scheduling her death in demography as storing the primary cause of death
            outcome_death_equations = self.module.apply_risk_of_death(individual_id, causes=causes)

        # Next we determine if she will experience a stillbirth during her delivery
        outcome_of_still_birth_equation = self.module.predict(self.module.la_linear_models['intrapartum_still_birth'],
                                                              individual_id)

        # We also assume that if a womans labour has started prior to 24 weeks the baby would not survive and we class
        # this as a stillbirth
        if (df.at[individual_id, 'ps_gestational_age_in_weeks'] < 24) or outcome_of_still_birth_equation:
            logger.debug(key='message', data=f'person {individual_id} has experienced an intrapartum still birth')

            random_draw = self.module.rng.random_sample()

            # If this woman will experience a stillbirth and she was not pregnant with twins OR she was pregnant with
            # twins but both twins have died during labour we reset/set the appropriate variables
            if ~df.at[individual_id, 'ps_multiple_pregnancy'] or \
                (df.at[individual_id, 'ps_multiple_pregnancy'] and (random_draw < params['prob_both_twins_ip_still_'
                                                                                         'birth'])):

                df.at[individual_id, 'la_intrapartum_still_birth'] = True
                # This variable is therefore only ever true when the pregnancy has ended in stillbirth
                df.at[individual_id, 'ps_prev_stillbirth'] = True

                # Next reset pregnancy and update contraception
                self.sim.modules['Contraception'].end_pregnancy(individual_id)

            # If one twin survives we store this as a property of the MNI which is reference on_birth of the newborn
            # outcomes to ensure this twin pregnancy only leads to one birth
            elif (df.at[individual_id, 'ps_multiple_pregnancy'] and (random_draw > params['prob_both_twins_ip_still_'
                                                                                          'birth'])):
                df.at[individual_id, 'ps_prev_stillbirth'] = True
                mni[individual_id]['single_twin_still_birth'] = True
                logger.debug(key='msg', data=f'single twin stillbirth for {individual_id}')

        if outcome_death_equations and df.at[individual_id, 'la_intrapartum_still_birth']:
            # We delete the mni dictionary if both mother and baby have died in labour, if the mother has died but
            # the baby has survived we delete the dictionary following the on_birth function of NewbornOutcomes
            del mni[individual_id]

        if df.at[individual_id, 'la_intrapartum_still_birth'] or mni[individual_id]['single_twin_still_birth']:
            logger.info(key='intrapartum_stillbirth', data={'mother_id': individual_id,
                                                            'date_of_ip_stillbirth': self.sim.date})


class BirthAndPostnatalOutcomesEvent(Event, IndividualScopeEventMixin):
    """
    This is BirthAndPostnatalOutcomesEvent. It is scheduled by LabourOnsetEvent when women go into labour. This event
    calls the do_birth function for all women who have gone into labour to generate a newborn within the simulation.
    Additionally this event applies the incidence of complications immediately following birth and determines if each
    woman will receive a full postnatal check-up and when.
    """
    def __init__(self, module, mother_id):
        super().__init__(module, person_id=mother_id)

    def apply(self, mother_id):
        df = self.sim.population.props
        person = df.loc[mother_id]
        mni = self.sim.modules['PregnancySupervisor'].mother_and_newborn_info
        params = self.module.current_parameters

        # This event tells the simulation that the woman's pregnancy is over and generates the new child in the
        # data frame - Check the correct amount of time has passed between labour onset and birth event and that women
        # at the event have the right characteristics present

        assert (self.sim.date - df.at[mother_id, 'la_due_date_current_pregnancy']) == pd.to_timedelta(5, unit='D')
        self.module.labour_characteristics_checker(mother_id)

        if ~person.is_alive and person.la_intrapartum_still_birth:
            return

        # =============================================== BIRTH ====================================================
        # If the mother is alive and still pregnant OR has died but the foetus has survived we generate a child.

        # Live women are scheduled to move to the postpartum event to determine if they experiences any additional
        # complications

        if (person.is_alive and person.is_pregnant and ~person.la_intrapartum_still_birth) or \
           (~person.is_alive and mni[mother_id]['death_in_labour'] and ~person.la_intrapartum_still_birth):
            logger.info(key='message', data=f'A Birth is now occurring, to mother {mother_id}')

            # If the mother is pregnant with twins, we call the do_birth function twice and then link the twins
            # (via sibling id) in the newborn module
            if person.ps_multiple_pregnancy:
                child_one = self.sim.do_birth(mother_id)

                if not mni[mother_id]['single_twin_still_birth']:
                    child_two = self.sim.do_birth(mother_id)
                    logger.debug(key='message', data=f'Mother {mother_id} will now deliver twins {child_one} & '
                                                     f'{child_two}')
                    self.sim.modules['NewbornOutcomes'].link_twins(child_one, child_two, mother_id)
            else:
                self.sim.do_birth(mother_id)

        df = self.sim.population.props

        # If the mother survived labour but experienced a stillbirth we reset all the relevant pregnancy variables now
        if df.at[mother_id, 'is_alive'] and df.at[mother_id, 'la_intrapartum_still_birth']:
            self.sim.modules['Labour'].further_on_birth_labour(mother_id)
            self.sim.modules['PregnancySupervisor'].further_on_birth_pregnancy_supervisor(mother_id)
            self.sim.modules['PostnatalSupervisor'].further_on_birth_postnatal_supervisor(mother_id)
            self.sim.modules['CareOfWomenDuringPregnancy'].further_on_birth_care_of_women_in_pregnancy(mother_id)

        # Next we apply the risk of complications following delivery
        if df.at[mother_id, 'is_alive']:

            df.at[mother_id, 'la_is_postpartum'] = True
            df.at[mother_id, 'la_date_most_recent_delivery'] = self.sim.date

            for complication in ['sepsis_endometritis', 'sepsis_urinary_tract', 'sepsis_skin_soft_tissue',
                                 'pph_uterine_atony', 'pph_retained_placenta', 'pph_other']:
                self.module.set_postpartum_complications(mother_id, complication=complication)

            if df.at[mother_id, 'la_sepsis_pp']:
                logger.info(key='maternal_complication', data={'person': mother_id,
                                                               'type': 'sepsis_postnatal',
                                                               'timing': 'postnatal'})
            if df.at[mother_id, 'la_postpartum_haem']:
                logger.info(key='maternal_complication', data={'person': mother_id,
                                                               'type': 'primary_postpartum_haemorrhage',
                                                               'timing': 'postnatal'})

            self.module.progression_of_hypertensive_disorders(mother_id, property_prefix='pn')

            # Following this we determine if this woman will receive/seek care for postnatal care after delivery
            if(df.at[mother_id, 'la_sepsis_pp'] or
                df.at[mother_id, 'la_postpartum_haem'] or
                ((df.at[mother_id, 'pn_htn_disorders'] == 'severe_pre_eclamp') and mni[mother_id]['new_onset_spe']) or
               (df.at[mother_id, 'pn_htn_disorders'] == 'eclampsia')):

                # Women with complications have a higher baseline probability of seeking postnatal care
                prob_pnc = params['prob_careseeking_for_complication']
                has_comps = True

            else:
                # We use a linear model to determine if women without complications will receive any postnatal care
                prob_pnc = self.module.la_linear_models['postnatal_check'].predict(
                    df.loc[[mother_id]],
                    mode_of_delivery=pd.Series(mni[mother_id]['mode_of_delivery'], index=df.loc[[mother_id]].index),
                    delivery_setting=pd.Series(mni[mother_id]['delivery_setting'], index=df.loc[[mother_id]].index)
                )[mother_id]
                has_comps = False

            # If she will receive PNC, we determine if this will happen less than 48 hours from birth or later
            if self.module.rng.random_sample() < prob_pnc:
                timing = self.module.rng.choice(['<48', '>48'], p=params['prob_timings_pnc'])

                if timing == '<48' or has_comps:
                    mni[mother_id]['will_receive_pnc'] = 'early'

                    early_event = HSI_Labour_ReceivesPostnatalCheck(
                        module=self.module, person_id=mother_id)

                    self.sim.modules['HealthSystem'].schedule_hsi_event(
                        early_event,
                        priority=0,
                        topen=self.sim.date,
                        tclose=self.sim.date + DateOffset(days=1))

                else:
                    # For women who do not have prompt PNC, we determine if they will die from complications occurring
                    # following birth that have not been treated
                    mni[mother_id]['will_receive_pnc'] = 'late'
                    self.module.apply_risk_of_early_postpartum_death(mother_id)
            else:
                # Similarly for women who will not receive PNC at all we apply risk of death
                self.module.apply_risk_of_early_postpartum_death(mother_id)


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

        if not df.at[person_id, 'is_alive']:
            return

        logger.info(key='message', data=f'This is HSI_Labour_PresentsForSkilledAttendanceInLabour: Mother {person_id} '
                                        f'has presented to a health facility on date {self.sim.date} following the '
                                        f'onset of her labour')

        # First we capture women who have presented to this event during labour at home. Currently we just set these
        # women to be delivering at a health centre (this will need to be randomised to match any available data)
        if mni[person_id]['delivery_setting'] == 'home_birth' and mni[person_id]['sought_care_for_complication']:
            mni[person_id]['delivery_setting'] = 'health_centre'

        # Next we check this woman has the right characteristics to be at this event
        self.module.labour_characteristics_checker(person_id)
        assert mni[person_id]['delivery_setting'] != 'home_birth'

        # Here we ensure that women who were admitted via the antenatal ward for assisted/caesarean delivery have the
        # correct variables updated leading to referral for delivery
        if (df.at[person_id, 'ac_admitted_for_immediate_delivery'] == 'caesarean_now') or \
                (df.at[person_id, 'ac_admitted_for_immediate_delivery'] == 'caesarean_future'):
            mni[person_id]['referred_for_cs'] = True

        elif df.at[person_id, 'ac_admitted_for_immediate_delivery'] == 'avd_now':
            mni[person_id]['mode_of_delivery'] = 'instrumental'

        # LOG CONSUMABLES FOR DELIVERY...
        # We assume all deliveries require this basic package of consumables
        consumables = self.get_consumables(item_codes=self.module.item_codes_lab_consumables['delivery'],
                                           return_individual_results=True)
        key_consumable_avail = list(consumables.values())[0]

        # If the clean delivery kit consumable is available, we assume women benefit from clean delivery
        if key_consumable_avail:
            mni[person_id]['clean_birth_practices'] = True

        # ===================================== PROPHYLACTIC CARE ===================================================
        # The following function manages the consumables and administration of prophylactic interventions in labour
        # (clean delivery practice, antibiotics for PROM, steroids for preterm labour). This intervention, like all
        # other in the event will only occur if the squeeze factor is below a preset threshold

        if squeeze_factor < params['squeeze_threshold_proph_ints']:
            self.module.prophylactic_labour_interventions(self)

        # ================================= PROPHYLACTIC MANAGEMENT PRE-ECLAMPSIA  ==============================
        # Next we see if women with severe pre-eclampsia will be identified and treated, reducing their risk of
        # eclampsia
        if squeeze_factor < params['squeeze_threshold_treatment_spe']:
            self.module.assessment_and_treatment_of_severe_pre_eclampsia_mgso4(self, 'ip')

        # ===================================== APPLYING COMPLICATION INCIDENCE =======================================
        # Following administration of prophylaxis we assess if this woman will develop any complications during labour.
        # Women who have sought care because of complication have already had these risk applied so it doesnt happen
        # again

        if not mni[person_id]['sought_care_for_complication']:
            for complication in ['obstruction_cpd', 'obstruction_malpos_malpres', 'obstruction_other',
                                 'placental_abruption', 'antepartum_haem', 'sepsis_chorioamnionitis',
                                 'uterine_rupture']:

                # Uterine rupture is the only complication we dont apply the risk of here due to the causal link
                # between obstructed labour and uterine rupture. Therefore we want interventions for obstructed labour
                # to reduce the risk of uterine rupture

                self.module.set_intrapartum_complications(person_id, complication=complication)
            self.module.progression_of_hypertensive_disorders(person_id, property_prefix='ps')

            if df.at[person_id, 'la_obstructed_labour']:
                logger.info(key='maternal_complication', data={'mother': person_id,
                                                               'type': 'obstructed_labour',
                                                               'timing': 'intrapartum'})

        # ======================================= COMPLICATION MANAGEMENT ==========================
        # Next, women in labour are assessed for complications and treatment delivered if a need is identified and
        # consumables are available

        if squeeze_factor < params['squeeze_threshold_treatment_ol']:
            self.module.assessment_and_treatment_of_obstructed_labour_via_avd(self)

        if squeeze_factor < params['squeeze_threshold_treatment_sep']:
            self.module.assessment_and_treatment_of_maternal_sepsis(self, 'ip')

        if squeeze_factor < params['squeeze_threshold_treatment_htn']:
            self.module.assessment_and_treatment_of_hypertension(self)

        if squeeze_factor < params['squeeze_threshold_treatment_aph']:
            self.module.assessment_and_plan_for_antepartum_haemorrhage(self)

        if squeeze_factor < params['squeeze_threshold_treatment_ec']:
            self.module.assessment_and_treatment_of_eclampsia(self, 'ip')

        # Uterine rupture follows the same pattern as antepartum haemorrhage
        if squeeze_factor < params['squeeze_threshold_treatment_ur']:
            self.module.assessment_for_referral_uterine_rupture(self)

        # -------------------------- Active Management of the third stage of labour ----------------------------------
        # Prophylactic treatment to prevent postpartum bleeding is applied
        if not mni[person_id]['sought_care_for_complication'] and (squeeze_factor < params['squeeze_threshold_amtsl']):
            self.module.active_management_of_the_third_stage_of_labour(self)

        # -------------------------- Caesarean section/AVD for un-modelled reason ------------------------------------
        # We apply a probability to women who have not already been allocated to undergo assisted/caesarean delivery
        # that they will require assisted/caesarean delivery to capture indications which are not explicitly modelled
        if not mni[person_id]['referred_for_cs'] and (not mni[person_id]['mode_of_delivery'] == 'instrumental'):
            if self.module.rng.random_sample() < params['residual_prob_caesarean']:
                mni[person_id]['referred_for_cs'] = True
                mni[person_id]['cs_indication'] = 'other'
            elif self.module.rng.random_sample() < params['residual_prob_avd']:
                mni[person_id]['mode_of_delivery'] = 'instrumental'

        # ========================================== SCHEDULING CEMONC CARE =========================================
        # Finally women who require additional treatment have the appropriate HSI scheduled to deliver further care

        if mni[person_id]['referred_for_cs'] or \
            mni[person_id]['referred_for_surgery'] or \
           mni[person_id]['referred_for_blood']:

            surgical_management = HSI_Labour_ReceivesComprehensiveEmergencyObstetricCare(
                self.module, person_id=person_id, timing='intrapartum')
            self.sim.modules['HealthSystem'].schedule_hsi_event(surgical_management,
                                                                priority=0,
                                                                topen=self.sim.date,
                                                                tclose=self.sim.date + DateOffset(days=1))

        # If a this woman has experienced a complication the appointment footprint is changed from normal to
        # complicated
        if (
            df.at[person_id, 'la_sepsis']
            or df.at[person_id, 'la_antepartum_haem'] != 'none'
            or df.at[person_id, 'la_obstructed_labour']
            or df.at[person_id, 'la_uterine_rupture']
            or df.at[person_id, 'ps_htn_disorders'] == 'eclampsia'
            or df.at[person_id, 'ps_htn_disorders'] == 'severe_pre_eclamp'
        ):
            return self.make_appt_footprint({'CompDelivery': 1})

    def never_ran(self):
        self.module.run_if_receives_skilled_birth_attendance_cant_run(self)

    def did_not_run(self):
        self.module.run_if_receives_skilled_birth_attendance_cant_run(self)
        return False

    def not_available(self):
        self.module.run_if_receives_skilled_birth_attendance_cant_run(self)


class HSI_Labour_ReceivesPostnatalCheck(HSI_Event, IndividualScopeEventMixin):
    """
    This is HSI_Labour_ReceivesPostnatalCheck. It is scheduled by BirthAndPostnatalOutcomesEvent for all women who
    will receive full postnatal checkup after birth . This event represents the postpartum care contact after
    delivery and includes assessment and treatment of severe pre-eclampsia, hypertension, sepsis and postpartum
    bleeding. In addition woman are scheduled HIV screening if appropriate and started on postnatal iron tablets
    """
    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Labour)

        self.TREATMENT_ID = 'Labour_ReceivesPostnatalCheck'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'Over5OPD': 1})
        self.ALERT_OTHER_DISEASES = []
        self.ACCEPTED_FACILITY_LEVEL = '1a'
        self.BEDDAYS_FOOTPRINT = self.make_beddays_footprint({'general_bed': 2})

    def apply(self, person_id, squeeze_factor):
        mni = self.sim.modules['PregnancySupervisor'].mother_and_newborn_info
        df = self.sim.population.props
        params = self.module.current_parameters

        if pd.isnull(df.at[person_id, 'la_date_most_recent_delivery']):
            return
        if not df.at[person_id, 'is_alive']:
            return

        # Ensure that women who were scheduled to receive early PNC have received care prior to passing through
        # PostnatalWeekOneMaternalEvent
        if (mni[person_id]['will_receive_pnc'] == 'early') and not mni[person_id]['passed_through_week_one']:
            assert self.sim.date < (df.at[person_id, 'la_date_most_recent_delivery'] + pd.DateOffset(days=2))
            assert df.at[person_id, 'la_pn_checks_maternal'] == 0

        elif mni[person_id]['will_receive_pnc'] == 'late' and not mni[person_id]['passed_through_week_one']:
            assert self.sim.date >= (df.at[person_id, 'la_date_most_recent_delivery'] + pd.DateOffset(days=2))
            assert df.at[person_id, 'la_pn_checks_maternal'] == 0

        # Run checks
        self.module.postpartum_characteristics_checker(person_id)

        # log the PNC visit
        logger.info(key='postnatal_check', data={'person_id': person_id,
                                                 'delivery_setting': mni[person_id]['delivery_setting'],
                                                 'visit_number':  df.at[person_id, 'la_pn_checks_maternal'],
                                                 'timing':  mni[person_id]['will_receive_pnc']})

        df.at[person_id, 'la_pn_checks_maternal'] += 1

        # Perform assessments and treatment for each of the major complications that can occur after birth
        self.module.assessment_and_treatment_of_eclampsia(self, 'pp')
        self.module.assessment_and_treatment_of_pph_retained_placenta(self)
        self.module.assessment_and_treatment_of_pph_uterine_atony(self)

        if self.module.rng.random_sample() < params['prob_intervention_delivered_sep_assessment_pnc']:
            self.module.assessment_and_treatment_of_maternal_sepsis(self, 'pp')

        if self.module.rng.random_sample() < params['prob_intervention_delivered_bp_pnc']:
            self.module.assessment_and_treatment_of_severe_pre_eclampsia_mgso4(self, 'pp')
            self.module.assessment_and_treatment_of_hypertension(self)

        if self.module.rng.random_sample() < params['prob_intervention_delivered_anaemia_assessment_pnc']:
            self.module.assessment_and_treatment_of_anaemia(self)

        self.module.assessment_for_depression(self)
        self.module.interventions_delivered_pre_discharge(self)

        # Schedule higher level care for women requiring comprehensive treatment
        if mni[person_id]['referred_for_surgery'] or mni[person_id]['referred_for_blood']:
            surgical_management = HSI_Labour_ReceivesComprehensiveEmergencyObstetricCare(
                self.module, person_id=person_id, timing='postpartum')
            self.sim.modules['HealthSystem'].schedule_hsi_event(surgical_management,
                                                                priority=0,
                                                                topen=self.sim.date,
                                                                tclose=self.sim.date + DateOffset(days=1))

        # ====================================== APPLY RISK OF DEATH===================================================
        if not mni[person_id]['referred_for_surgery'] and not mni[person_id]['referred_for_blood']:
            self.module.apply_risk_of_early_postpartum_death(person_id)

        actual_appt_footprint = self.EXPECTED_APPT_FOOTPRINT
        return actual_appt_footprint

    def never_ran(self):
        self.module.run_if_receives_postnatal_check_cant_run(self)

    def did_not_run(self):
        self.module.run_if_receives_postnatal_check_cant_run(self)
        return False

    def not_available(self):
        self.module.run_if_receives_postnatal_check_cant_run(self)


class HSI_Labour_ReceivesComprehensiveEmergencyObstetricCare(HSI_Event, IndividualScopeEventMixin):
    """
    This is HSI_Labour_ReceivesComprehensiveEmergencyObstetricCare. This event houses all the interventions that are
    required to be delivered at a CEmONC level facility including caesarean section, blood transfusion and surgery
    during or following labour Currently we assume that if this even runs and the consumables are available then the
    intervention is delivered i.e. we dont apply squeeze factor threshold.
    """
    def __init__(self, module, person_id, timing):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, Labour)

        self.TREATMENT_ID = 'Labour_ReceivesComprehensiveEmergencyObstetricCare'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'MajorSurg': 1})
        self.ACCEPTED_FACILITY_LEVEL = '1b'
        self.ALERT_OTHER_DISEASES = []
        self.BEDDAYS_FOOTPRINT = self.make_beddays_footprint({'general_bed': 2})

        self.timing = timing

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        mni = self.sim.modules['PregnancySupervisor'].mother_and_newborn_info
        params = self.module.current_parameters

        logger.debug(key='msg', data='This is HSI_Labour_ReceivesComprehensiveEmergencyObstetricCare running for '
                                     f'mother {person_id}')

        # We use the variable self.timing to differentiate between women sent to this event during labour and women
        # sent after labour

        # ========================================== CAESAREAN SECTION ===============================================
        # For women arriving to this event during labour who have been referred for caesarean the intervention is
        # delivered
        if mni[person_id]['referred_for_cs'] and self.timing == 'intrapartum':

            # We log the log the required consumables and condition the caesarean happening on the availability of the
            # first consumable in this package, the anaesthetic required for the surgery
            consumables = self.get_consumables(item_codes=self.module.item_codes_lab_consumables['caesarean_delivery'],
                                               return_individual_results=True)
            key_consumable_avail = list(consumables.values())[0]

            if key_consumable_avail:
                person = df.loc[person_id]
                logger.info(key='caesarean_delivery', data=person.to_dict())
                logger.info(key='cs_indications', data={'id': person_id,
                                                        'indication': mni[person_id]['cs_indication']})

                # The appropriate variables in the MNI and dataframe are stored. Current caesarean section reduces risk
                # of intrapartum still birth and death due to antepartum haemorrhage
                mni[person_id]['mode_of_delivery'] = 'caesarean_section'
                mni[person_id]['amtsl_given'] = True
                df.at[person_id, 'la_previous_cs_delivery'] = True

        # ================================ SURGICAL MANAGEMENT OF RUPTURED UTERUS =====================================
        # Women referred after the labour HSI following correct identification of ruptured uterus will also need to
        # undergo surgical repair of this complication
        if mni[person_id]['referred_for_surgery'] and self.timing == 'intrapartum' and df.at[person_id,
                                                                                             'la_uterine_rupture']:

            # We log the log the required consumables and condition the surgery happening on the availability of the
            # first consumable in this package, the anaesthetic required for the surgery
            consumables = self.get_consumables(item_codes=self.module.item_codes_lab_consumables['obstetric_surgery'],
                                               return_individual_results=True)
            key_consumable_avail = list(consumables.values())[0]

            # We apply a probability that repair surgery will be successful which will reduce risk of death from
            # uterine rupture
            treatment_success_ur = params['success_rate_uterine_repair'] > self.module.rng.random_sample()

            if key_consumable_avail and treatment_success_ur:
                df.at[person_id, 'la_uterine_rupture_treatment'] = True

            # Unsuccessful repair will lead to this woman requiring a hysterectomy. Hysterectomy will also reduce risk
            # of death from uterine rupture but leads to permanent infertility in the simulation
            elif key_consumable_avail and ~treatment_success_ur:
                df.at[person_id, 'la_has_had_hysterectomy'] = True

        # ============================= SURGICAL MANAGEMENT OF POSTPARTUM HAEMORRHAGE==================================
        # Women referred for surgery immediately following labour will need surgical management of postpartum bleeding
        # Treatment is varied accordingly to underlying cause of bleeding

        if (mni[person_id]['referred_for_surgery'] and
            (self.timing == 'postpartum') and
           (df.at[person_id, 'la_postpartum_haem'] or df.at[person_id, 'pn_postpartum_haem_secondary'])):
            self.module.surgical_management_of_pph(self)

        # =========================================== BLOOD TRANSFUSION ===============================================
        # Women referred for blood transfusion alone or in conjunction with one of the above interventions will receive
        # that here
        if mni[person_id]['referred_for_blood']:
            self.module.blood_transfusion(self)

        # Women who have passed through the postpartum SBA HSI have not yet had their risk of death calculated because
        # they required interventions delivered via this event. We now determine if these women will survive
        if self.timing == 'postpartum':
            self.module.apply_risk_of_early_postpartum_death(person_id)

        actual_appt_footprint = self.EXPECTED_APPT_FOOTPRINT

        # Here we edit the appointment footprint so only women receiving surgery require the surgical footprint
        if mni[person_id]['referred_for_surgery'] or mni[person_id]['referred_for_cs']:
            actual_appt_footprint['MajorSurg'] = actual_appt_footprint['MajorSurg']

        elif (not mni[person_id]['referred_for_surgery'] and not mni[person_id]['referred_for_cs']) and\
                mni[person_id]['referred_for_blood']:
            actual_appt_footprint['MajorSurg'] = actual_appt_footprint['InpatientDays']

    def never_ran(self):
        self.module.run_if_receives_comprehensive_emergency_obstetric_care_cant_run(self)

    def did_not_run(self):
        self.module.run_if_receives_comprehensive_emergency_obstetric_care_cant_run(self)
        return False

    def not_available(self):
        self.module.run_if_receives_comprehensive_emergency_obstetric_care_cant_run(self)


class LabourLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    """This is the LabourLoggingEvent. It uses the data frame and the labour_tracker to produce summary statistics which
    are processed and presented by different analysis scripts """

    def __init__(self, module):
        self.repeat = 1
        super().__init__(module, frequency=DateOffset(years=self.repeat))

    def apply(self, population):
        df = self.sim.population.props
        repro_women = df.is_alive & (df.sex == 'F') & (df.age_years > 14) & (df.age_years < 50)

        hysterectomy = df.is_alive & (df.sex == 'F') & df.la_has_had_hysterectomy & (df.age_years > 14) & (df.age_years
                                                                                                           < 50)
        labour = df.is_alive & (df.sex == 'F') & df.la_currently_in_labour & (df.age_years > 14) & (df.age_years
                                                                                                    < 50)
        postnatal = df.is_alive & (df.sex == 'F') & df.la_is_postpartum & (df.age_years > 14) & (df.age_years
                                                                                                 < 50)
        inpatient = df.is_alive & (df.sex == 'F') & df.hs_is_inpatient & (df.age_years > 14) & (df.age_years
                                                                                                < 50)

        prop_hyst = (len(hysterectomy.loc[hysterectomy]) / len(repro_women.loc[repro_women])) * 100
        prop_in_labour = (len(labour.loc[labour]) / len(repro_women.loc[repro_women])) * 100
        prop_pn = (len(postnatal.loc[postnatal]) / len(repro_women.loc[repro_women])) * 100
        prop_ip = (len(inpatient.loc[inpatient]) / len(repro_women.loc[repro_women])) * 100

        logger.info(key='women_data_debug', data={'hyst': prop_hyst,
                                                  'labour': prop_in_labour,
                                                  'pn': prop_pn,
                                                  'ip': prop_ip})
