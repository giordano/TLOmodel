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

from tlo.util import BitsetHandler

# n.b. all LMs here are coded as population level


def predict_obstetric_fistula(self, df, rng=None, **externals):
    """individual level"""
    person = df.iloc[0]
    params = self.parameters
    result = params['prob_obstetric_fistula']

    if person['la_obstructed_labour']:
        result *= params['rr_obstetric_fistula_obstructed_labour']

    return pd.Series(data=[result], index=df.index)


def predict_secondary_postpartum_haem(self, df, rng=None, **externals):
    """population level"""
    params = self.parameters
    result = pd.Series(data=params['prob_secondary_pph'], index=df.index)

    result[externals['endometritis']] *= params['rr_secondary_pph_endometritis']

    # if externals['endometritis']:
    #    result *= params['rr_secondary_pph_endometritis']

    return result


def predict_sepsis_endometritis_late_postpartum(self, df, rng=None, **externals):
    """population level"""
    params = self.parameters
    result = pd.Series(data=params['prob_late_sepsis_endometritis'], index=df.index)

    result[externals['mode_of_delivery'] == 'caesarean_section'] *= params['rr_sepsis_endometritis_post_cs']

    # if externals['mode_of_delivery'] == 'caesarean_section':
    #    result *= params['rr_sepsis_endometritis_post_cs']
    return result


def predict_sepsis_sst_late_postpartum(self, df, rng=None, **externals):
    """population level"""
    params = self.parameters
    result = pd.Series(data=params['prob_late_sepsis_skin_soft_tissue'], index=df.index)
    result[externals['mode_of_delivery'] == 'caesarean_section'] *= params['rr_sepsis_sst_post_cs']

    #if externals['mode_of_delivery'] == 'caesarean_section':
    #    result *= params['rr_sepsis_sst_post_cs']
    return result


def predict_gest_htn_pn(self, df, rng=None, **externals):
    """population level"""
    params = self.parameters
    result = pd.Series(data=params['weekly_prob_gest_htn_pn'], index=df.index)
    result[df.li_bmi == 4] *= params['rr_gest_htn_obesity']
    result[df.li_bmi == 5] *= params['rr_gest_htn_obesity']

    return result


def predict_pre_eclampsia_pn(self, df, rng=None, **externals):
    """population level"""
    params = self.parameters
    result = pd.Series(data=params['weekly_prob_pre_eclampsia_pn'], index=df.index)
    result[df.li_bmi == 4] *= params['rr_gest_htn_obesity']
    result[df.li_bmi == 5] *= params['rr_gest_htn_obesity']
    result[df.nc_hypertension] *= params['rr_gest_htn_obesity']
    result[df.nc_diabetes] *= params['rr_gest_htn_obesity']

    return result


def predict_death_from_hypertensive_disorder_pn(self, df, rng=None, **externals):
    """population level"""
    params = self.parameters
    result = pd.Series(data=params['cfr_severe_htn_pn'], index=df.index)

    return result


def predict_anaemia_after_pregnancy(self, df, rng=None, **externals):
    """population level"""
    m = self.module
    p = m.current_parameters
    deficiencies_following_pregnancy = m.deficiencies_following_pregnancy

    result = pd.Series(data=p['baseline_prob_anaemia_per_week'], index=df.index)

    result[deficiencies_following_pregnancy.has_any(df.index, 'iron')] *= p['rr_anaemia_if_iron_deficient_pn']
    result[deficiencies_following_pregnancy.has_any(df.index, 'folate')] *= p['rr_anaemia_if_folate_deficient_pn']
    result[deficiencies_following_pregnancy.has_any(df.index, 'b12')] *= p['rr_anaemia_if_b12_deficient_pn']
    result[df.hv_inf & (df.hv_art != "not")] *= p['rr_anaemia_hiv_no_art']
    result[df.ma_is_infected] *= p['rr_anaemia_maternal_malaria']
    result[df.la_iron_folic_acid_postnatal] *= p['treatment_effect_iron_folic_acid_anaemia']

    return result


def predict_early_onset_neonatal_sepsis_week_1(self, df, rng=None, **externals):
    """population level"""
    params = self.parameters
    result = pd.Series(data=params['prob_early_onset_neonatal_sepsis_week_1'], index=df.index)

    result[df.nb_clean_birth] *= params['treatment_effect_clean_birth']
    result[df.nb_received_cord_care] *= params['treatment_effect_cord_care']
    result[df.nb_early_init_breastfeeding] *= params['treatment_effect_early_init_bf']
    result[df.nb_early_preterm] *= params['rr_eons_preterm_neonate']
    result[df.nb_late_preterm] *= params['rr_eons_preterm_neonate']

    result[externals['received_abx_for_prom']] *= params['treatment_effect_abx_prom']
    result[externals['maternal_chorioamnionitis']] *= params['rr_eons_maternal_chorio']
    result[externals['maternal_prom']] *= params['rr_eons_maternal_prom']

    #if externals['received_abx_for_prom']:
    #    result *= params['treatment_effect_abx_prom']
    #if externals['maternal_chorioamnionitis']:
    #    result *= params['rr_eons_maternal_chorio']
    #if externals['maternal_prom']:
    #    result *= params['rr_eons_maternal_prom']

    return result


def predict_late_onset_neonatal_sepsis(self, df, rng=None, **externals):
    """population level"""
    params = self.parameters
    result = pd.Series(data=params['prob_late_onset_neonatal_sepsis'], index=df.index)
    result[df.nb_early_init_breastfeeding] *= params['treatment_effect_early_init_bf']
    return result


def predict_care_seeking_for_first_pnc_visit(self, df, rng=None, **externals):
    """population level"""
    params = self.parameters
    result = pd.Series(data=params['odds_will_attend_pnc'], index=df.index)

    result[(df.age_years > 29) & (df.age_years < 36)] *= params['or_pnc_age_30_35']
    result[df.age_years >= 36] *= params['or_pnc_age_>35']
    result[~df.li_urban] *= params['or_pnc_rural']
    result[df.li_wealth == 1] *= params['or_pnc_wealth_level_1']
    result[df.la_parity > 4] *= params['or_pnc_parity_>4']

    result[externals['mode_of_delivery'] == 'caesarean_section'] *= params['or_pnc_caesarean_delivery']
    result[externals['delivery_setting'] != 'home_birth'] *= params['or_pnc_facility_delivery']

    # if externals['caesarean_delivery']:
    #    result *= params['or_pnc_caesarean_delivery']
    #if externals['facility_delivery']:
    #    result *= params['or_pnc_facility_delivery']

    result = result / (1 + result)
    return result


def predict_care_seeking_postnatal_complication_mother(self, df, rng=None, **externals):
    """population level"""
    params = self.parameters
    result = pd.Series(data=params['prob_care_seeking_postnatal_emergency'], index=df.index)

    return result


def predict_care_seeking_postnatal_complication_neonate(self, df, rng=None, **externals):
    """population level"""
    params = self.parameters
    result = pd.Series(data=params['prob_care_seeking_postnatal_emergency_neonate'], index=df.index)

    return result


def predict_care_seeking_for_fistula_repair(self, df, rng=None, **externals):
    """population level"""
    params = self.parameters
    result = pd.Series(data=params['odds_care_seeking_fistula_repair'], index=df.index)
    result[df.age_years.between(14, 20)] *= params['aor_cs_fistula_age_15_19']
    result[df.li_ed_lev == 1] *= params['aor_cs_fistula_age_lowest_education']

    result = result / (1 + result)
    return result
