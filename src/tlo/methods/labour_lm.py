"""Module contains functions to be passed to LinearModel.custom function

The following template can be used for implementing:

def predict_for_individual(self, df, rng=None, **externals):
    # this is a single row dataframe. get the individual record.
    person = df.iloc[0]
    params = self.parameters
    result = 0.0  # or other intercept value
    # ...implement model here, adjusting result...
    # caller expects a series to be returned
    return pd.Series(data=[result], index=df.index)

or

def predict_for_dataframe(self, df, rng=None, **externals):
    params = self.parameters
    result = pd.Series(data=params['some_intercept'], index=df.index)
    # result series has same index as dataframe, update as required
    # e.g. result[df.age == 5.0] += params['some_value']
    return result
"""
import pandas as pd


def predict_parity(self, df, rng=None, **externals):
    """population level"""
    params = self.parameters
    result = pd.Series(data=params['intercept_parity_lr2010'], index=df.index)
    result += df.age_years * 0.22
    result[df.li_mar_stat == 2] += params['effect_mar_stat_2_parity_lr2010']
    result[df.li_mar_stat == 3] += params['effect_mar_stat_3_parity_lr2010']
    result[df.li_wealth == 1] += params['effect_wealth_lev_1_parity_lr2010']
    result[df.li_wealth == 2] += params['effect_wealth_lev_2_parity_lr2010']
    result[df.li_wealth == 3] += params['effect_wealth_lev_3_parity_lr2010']
    result[df.li_wealth == 4] += params['effect_wealth_lev_4_parity_lr2010']
    result[df.li_wealth == 5] += params['effect_wealth_lev_5_parity_lr2010']

    return result


def predict_obstruction_cpd_ip(self, df, rng=None, **externals):
    """individual level"""
    # person = df.iloc[0]
    params = self.parameters
    result = params['prob_obstruction_cpd']

    # TODO: update with stunting and macrosomia properties

    # if person['SOME_STUNTING_PROPERTY']:
    #    result *= params['rr_obstruction_cpd_stunted_mother']
    if externals['macrosomia']:
        result *= params['rr_obstruction_foetal_macrosomia']
    # caller expects a series to be returned
    return pd.Series(data=[result], index=df.index)


def predict_sepsis_chorioamnionitis_ip(self, df, rng=None, **externals):
    """individual level"""
    person = df.iloc[0]
    params = self.parameters
    result = params['prob_sepsis_chorioamnionitis']

    if person['ps_premature_rupture_of_membranes']:
        result *= params['rr_sepsis_chorio_prom']
    if externals['received_clean_delivery']:
        result *= params['treatment_effect_maternal_infection_clean_delivery']
    # caller expects a series to be returned
    return pd.Series(data=[result], index=df.index)


def predict_sepsis_endometritis_pp(self, df, rng=None, **externals):
    """individual level"""
    params = self.parameters
    result = params['prob_sepsis_endometritis']

    if externals['mode_of_delivery'] == 'caesarean_section':
        result *= params['rr_sepsis_endometritis_post_cs']
    if externals['received_clean_delivery']:
        result *= params['treatment_effect_maternal_infection_clean_delivery']
    # caller expects a series to be returned
    return pd.Series(data=[result], index=df.index)


def predict_sepsis_skin_soft_tissue_pp(self, df, rng=None, **externals):
    """individual level"""
    params = self.parameters
    result = params['prob_sepsis_skin_soft_tissue']

    if externals['mode_of_delivery'] == 'caesarean_section':
        result *= params['rr_sepsis_sst_post_cs']
    if externals['received_clean_delivery']:
        result *= params['treatment_effect_maternal_infection_clean_delivery']
    # caller expects a series to be returned
    return pd.Series(data=[result], index=df.index)


def predict_sepsis_urinary_tract_pp(self, df, rng=None, **externals):
    """individual level"""
    params = self.parameters
    result = params['prob_sepsis_urinary_tract']
    if externals['received_clean_delivery']:
        result *= params['treatment_effect_maternal_infection_clean_delivery']
    # caller expects a series to be returned
    return pd.Series(data=[result], index=df.index)


def predict_sepsis_death(self, df, rng=None, **externals):
    """individual level"""
    person = df.iloc[0]
    params = self.parameters
    result = params['cfr_sepsis']

    if person['la_sepsis_treatment'] or person['ac_received_abx_for_chorioamnionitis']:
        result *= params['sepsis_treatment_effect_md']

    # caller expects a series to be returned
    return pd.Series(data=[result], index=df.index)


def predict_sepsis_pp_death(self, df, rng=None, **externals):
    """individual level"""
    person = df.iloc[0]
    params = self.parameters
    result = params['cfr_sepsis']

    if person['la_sepsis_treatment']:
        result *= params['sepsis_treatment_effect_md']

    # caller expects a series to be returned
    return pd.Series(data=[result], index=df.index)


def predict_eclampsia_death(self, df, rng=None, **externals):
    """individual level"""
    person = df.iloc[0]
    params = self.parameters
    result = params['cfr_eclampsia']

    if person['la_eclampsia_treatment']:
        result *= params['eclampsia_treatment_effect_md']
    # Both these predictors represent intravenous antihypertensives- both will not be true for the same
    # woman
    if person['la_maternal_hypertension_treatment'] or person['ac_iv_anti_htn_treatment']:
        result *= params['anti_htns_treatment_effect_md']

    # caller expects a series to be returned
    return pd.Series(data=[result], index=df.index)


def predict_eclampsia_pp_death(self, df, rng=None, **externals):
    """individual level"""
    person = df.iloc[0]
    params = self.parameters
    result = params['cfr_pp_eclampsia']  # todo: collapse as one eclampsia death equation

    if person['la_eclampsia_treatment']:
        result *= params['eclampsia_treatment_effect_md']
    # Both these predictors represent intravenous antihypertensives- both will not be true for the same
    # woman
    if person['la_maternal_hypertension_treatment'] or person['ac_iv_anti_htn_treatment'] :
        result *= params['anti_htns_treatment_effect_md']

    # caller expects a series to be returned
    return pd.Series(data=[result], index=df.index)


def predict_severe_pre_eclamp_death(self, df, rng=None, **externals):
    """individual level"""
    person = df.iloc[0]
    params = self.parameters
    result = params['cfr_severe_pre_eclamp']

    if person['la_maternal_hypertension_treatment']:
        result *= params['anti_htns_treatment_effect_md']
    if person['ac_iv_anti_htn_treatment']:
        result *= params['anti_htns_treatment_effect_md']

    # caller expects a series to be returned
    return pd.Series(data=[result], index=df.index)


def predict_placental_abruption_ip(self, df, rng=None, **externals):
    """individual level"""
    person = df.iloc[0]
    params = self.parameters
    result = params['prob_placental_abruption_during_labour']

    if person['la_previous_cs_delivery']:
        result *= params['rr_placental_abruption_previous_cs']
    if person['ps_htn_disorders'] == 'mild_pre_eclamp':
        result *= params['rr_placental_abruption_hypertension']
    if person['ps_htn_disorders'] == 'gest_htn':
        result *= params['rr_placental_abruption_hypertension']
    if person['ps_htn_disorders'] == 'severe_gest_htn':
        result *= params['rr_placental_abruption_hypertension']
    if person['ps_htn_disorders'] == 'severe_pre_eclamp':
        result *= params['rr_placental_abruption_hypertension']

    # caller expects a series to be returned
    return pd.Series(data=[result], index=df.index)


def predict_antepartum_haem_ip(self, df, rng=None, **externals):
    """individual level"""
    person = df.iloc[0]
    params = self.parameters
    result = 0.0

    if person['ps_placenta_praevia']:
        result += params['prob_aph_placenta_praevia_labour']
    if person['ps_placental_abruption']:
        result += params['prob_aph_placental_abruption_labour']
    if person['la_placental_abruption']:
        result += params['prob_aph_placental_abruption_labour']

    return pd.Series(data=[result], index=df.index)


def predict_antepartum_haem_death(self, df, rng=None, **externals):
    """individual level"""
    params = self.parameters
    result = params['cfr_aph']

    if externals['received_blood_transfusion']:
        result *= params['aph_bt_treatment_effect_md']
    if externals['mode_of_delivery'] == 'caesarean_section':
        result *= params['aph_cs_treatment_effect_md']

    # caller expects a series to be returned
    return pd.Series(data=[result], index=df.index)


def predict_pph_uterine_atony_pp(self, df, rng=None, **externals):
    """individual level"""
    person = df.iloc[0]
    params = self.parameters
    result = params['prob_pph_uterine_atony']

    if externals['amtsl_given']:
        result *= params['treatment_effect_amtsl']
    if person['pn_htn_disorders'] == 'mild_pre_eclamp':
        result *= params['rr_pph_ua_hypertension']
    if person['pn_htn_disorders'] == 'mild_pre_eclamp':
        result *= params['rr_pph_ua_hypertension']
    if person['pn_htn_disorders'] == 'mild_pre_eclamp':
        result *= params['rr_pph_ua_hypertension']
    if person['pn_htn_disorders'] == 'mild_pre_eclamp':
        result *= params['rr_pph_ua_hypertension']
    if person['ps_multiple_pregnancy']:
        # TODO: replace with MNI property as this will be reset by the time this eq is called
        result *= params['rr_pph_ua_multiple_pregnancy']
    if person['la_placental_abruption']:
        result *= params['rr_pph_ua_placental_abruption']
    if person['ps_placental_abruption']:
        # TODO: replace with MNI property as this will be reset by the time this eq is called
        result *= params['rr_pph_ua_placental_abruption']

    if externals['macrosomia']:
        result *= params['rr_pph_ua_macrosomia']

    return pd.Series(data=[result], index=df.index)


def predict_pph_retained_placenta_pp(self, df, rng=None, **externals):
    """individual level"""
    params = self.parameters
    result = params['cfr_aph']

    if externals['amtsl_given']:
        result *= params['treatment_effect_amtsl']

    return pd.Series(data=[result], index=df.index)


def predict_postpartum_haem_pp_death(self, df, rng=None, **externals):
    """individual level"""
    person = df.iloc[0]
    params = self.module.parameters
    treatment = self.module.pph_treatment.to_strings(person.la_postpartum_haem_treatment)
    result = params['cfr_pp_pph']

    if 'uterotonics' in treatment:
        result *= params['pph_treatment_effect_uterotonics_md']
    if 'manual_removal_placenta' in treatment:
        result *= params['pph_treatment_effect_mrp_md']
    if 'surgery' in treatment:
        result *= params['pph_treatment_effect_surg_md']
    if 'hysterectomy' in treatment:
        result *= params['pph_treatment_effect_hyst_md']
    if externals['received_blood_transfusion']:
        result *= params['pph_bt_treatment_effect_md']
    # if person['ps_anaemia_in_pregnancy']:
    #     result *= params['rr_pph_death_anaemia']

    return pd.Series(data=[result], index=df.index)


def predict_uterine_rupture_ip(self, df, rng=None, **externals):
    """individual level"""
    person = df.iloc[0]
    params = self.parameters
    result = params['prob_uterine_rupture']

    if person['la_parity'] == 2:
        result *= params['rr_ur_parity_2']
    if person['la_parity'] == 3:
        result *= params['rr_ur_parity_3_or_4']
    if person['la_parity'] == 4:
        result *= params['rr_ur_parity_3_or_4']
    if person['la_parity'] >= 5:
        result *= params['rr_ur_parity_5+']
    if person['la_previous_cs_delivery']:
        result *= params['rr_ur_prev_cs']
    if person['la_obstructed_labour']:
        result *= params['rr_ur_obstructed_labour']

    return pd.Series(data=[result], index=df.index)


def predict_uterine_rupture_death(self, df, rng=None, **externals):
    """individual level"""
    person = df.iloc[0]
    params = self.parameters
    result = params['cfr_uterine_rupture']

    if person['la_uterine_rupture_treatment']:
        result *= params['ur_repair_treatment_effect_md']
    if person['la_has_had_hysterectomy']:
        result *= params['ur_hysterectomy_treatment_effect_md']
    if externals['received_blood_transfusion']:
        result *= params['ur_treatment_effect_bt_md']

    return pd.Series(data=[result], index=df.index)


def predict_intrapartum_still_birth(self, df, rng=None, **externals):
    """individual level"""
    person = df.iloc[0]
    params = self.parameters
    result = params['prob_ip_still_birth']

    if person['la_maternal_death_in_labour']:
        result *= params['rr_still_birth_maternal_death']
    if person['la_uterine_rupture']:
        result *= params['rr_still_birth_ur']
    if person['la_obstructed_labour']:
        result *= params['rr_still_birth_ol']
    if person['la_antepartum_haem'] != 'none':
        result *= params['rr_still_birth_aph']
    if person['ps_antepartum_haemorrhage'] != 'none':
        result *= params['rr_still_birth_aph']
    # todo: risk should modify with severity- placeholder

    if person['ps_htn_disorders'] == 'mild_pre_eclamp':
        result *= params['rr_still_birth_hypertension']
    if person['ps_htn_disorders'] == 'gest_htn':
        result *= params['rr_still_birth_hypertension']
    if person['ps_htn_disorders'] == 'severe_gest_htn':
        result *= params['rr_still_birth_hypertension']
    if person['ps_htn_disorders'] == 'severe_pre_eclamp':
        result *= params['rr_still_birth_hypertension']

    if person['la_sepsis'] or (person['ps_chorioamnionitis'] != 'none'):
        result *= params['rr_still_birth_sepsis']
    if person['ps_multiple_pregnancy']:
        result *= params['rr_still_birth_multiple_pregnancy']
    if externals['preterm_labour']:
        result *= params['rr_still_birth_preterm_labour']

    if externals['mode_of_delivery'] == 'caesarean_section':
        result *= params['treatment_effect_cs_still_birth']
    if externals['mode_of_delivery'] == 'instrumental':
        result *= params['treatment_effect_avd_still_birth']

    return pd.Series(data=[result], index=df.index)


def predict_probability_delivery_health_centre(self, df, rng=None, **externals):
    """individual level"""
    person = df.iloc[0]
    params = self.parameters
    result = params['odds_deliver_in_health_centre']

    if 19 < person['age_years'] < 25:
        result *= params['rrr_hc_delivery_age_20_24_2010']
    if 24 < person['age_years'] < 30:
        result *= params['rrr_hc_delivery_age_25_29_2010']
    if 29 < person['age_years'] < 35:
        result *= params['rrr_hc_delivery_age_30_34_2010']
    if 34 < person['age_years'] < 40:
        result *= params['rrr_hc_delivery_age_35_39_2010']
    if 39 < person['age_years'] < 45:
        result *= params['rrr_hc_delivery_age_40_44_2010']
    if 44 < person['age_years'] < 50:
        result *= params['rrr_hc_delivery_age_45_49_2010']

    # TODO: level 3 is secondary and tertiarty combined...
    # if person['li_ed_lev'] == 3:
    #    result *= params['rrr_hc_delivery_tertiary_education_2010']

    if person['li_wealth'] == 1:
        result *= params['rrr_hc_delivery_wealth_1_2010']
    if person['li_wealth'] == 2:
        result *= params['rrr_hc_delivery_wealth_2_2010']
    if person['li_wealth'] == 3:
        result *= params['rrr_hc_delivery_wealth_3_2010']
    if person['li_wealth'] == 4:
        result *= params['rrr_hc_delivery_wealth_4_2010']

    if 2 < person['la_parity'] < 5:
        result *= params['rrr_hc_delivery_parity_3_to_4_2010']
    if person['la_parity'] > 4:
        result *= params['rrr_hc_delivery_parity_>4_2010']

    if ~person['li_urban']:
        result *= params['rrr_hc_delivery_rural_2010']

    if person['li_mar_stat'] == 2:
        result *= params['rrr_hc_delivery_married_2010']

    result = result / (1 + result)
    return pd.Series(data=[result], index=df.index)


def predict_probability_delivery_at_home(self, df, rng=None, **externals):
    """individual level"""
    person = df.iloc[0]
    params = self.parameters
    result = params['odds_deliver_at_home']

    if 19 < person['age_years'] < 25:
        result *= params['rrr_hb_delivery_age_20_24_2010']
    if 29 < person['age_years'] < 35:
        result *= params['rrr_hb_delivery_age_30_34_2010']
    if 34 < person['age_years'] < 40:
        result *= params['rrr_hb_delivery_age_35_39_2010']
    if 39 < person['age_years'] < 45:
        result *= params['rrr_hb_delivery_age_40_44_2010']
    if 44 < person['age_years'] < 50:
        result *= params['rrr_hb_delivery_age_45_49_2010']

    if ~person['li_urban']:
        result *= params['rrr_hb_delivery_rural_2010']

    if 2 < person['la_parity'] < 5:
        result *= params['rrr_hb_delivery_parity_3_to_4_2010']
    if person['la_parity'] > 4:
        result *= params['rrr_hb_delivery_parity_>4_2010']

    if person['li_ed_lev'] == 2:
       result *= params['rrr_hb_delivery_primary_education_2010']
    # if person['li_ed_lev'] == 3:
    #   result *= params['rrr_hb_delivery_secondary_education_2010']
    # if person['li_ed_lev'] == 3:
    #    result *= params['rrr_hb_delivery_tertiary_education_2010']

    if person['li_wealth'] == 1:
        result *= params['rrr_hb_delivery_wealth_1_2010']
    if person['li_wealth'] == 2:
        result *= params['rrr_hb_delivery_wealth_2_2010']
    if person['li_wealth'] == 3:
        result *= params['rrr_hb_delivery_wealth_3_2010']
    if person['li_wealth'] == 4:
        result *= params['rrr_hb_delivery_wealth_4_2010']

    if 2 < person['la_parity'] < 5:
        result *= params['rrr_hb_delivery_parity_3_to_4_2010']
    if person['la_parity'] > 4:
        result *= params['rrr_hb_delivery_parity_>4_2010']

    if person['li_mar_stat'] == 3:
        result *= params['rrr_hb_delivery_married_2010']

    result = result / (1 + result)
    return pd.Series(data=[result], index=df.index)


def predict_care_seeking_for_complication(self, df, rng=None, **externals):
    """individual level"""
    #  person = df.iloc[0]
    params = self.parameters
    result = params['prob_careseeking_for_complication']

    return pd.Series(data=[result], index=df.index)
