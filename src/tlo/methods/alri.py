"""
Childhood Acute Lower Respiratory Infection Module

Overview
--------
Individuals are exposed to the risk of infection by a pathogen (and potentially also with a secondary bacterial
infection) that can cause one of several type of acute lower respiratory infection (Alri).
The disease is manifested as either Pneumonia, or Bronchiolitis/other ALRI.

During an episode (prior to recovery - either naturally or cured with treatment), symptom are manifest and there may be
complications (e.g. local pulmonary complication: pleural effusuion, empyema, lung abscess, pneumothorax; and/or
systemic complications: sepsis, and hypoxaemia). All these complication are set at onset of disease, for simplicity.

The individual may recover naturally or die. The risk of death is applied to those who develop complications.

Health care seeking is prompted by the onset of the symptom. The individual can be treated; if successful the risk of
death is lowered and they are cured (symptom resolved) some days later.

Outstanding issues
------------------
* All HSI events
* Follow-up appointments for initial HSI events.
* Double check parameters and consumables codes for the HSI events.
* Duration of Alri Event is not informed by data

"""

from collections import defaultdict
from itertools import chain
from pathlib import Path

import numpy as np
import pandas as pd

from tlo import DateOffset, Module, Parameter, Property, Types, logging
from tlo.events import Event, IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent
from tlo.lm import LinearModel, LinearModelType, Predictor
from tlo.methods import Metadata
from tlo.methods.causes import Cause
from tlo.methods.healthsystem import HSI_Event
from tlo.methods.symptommanager import Symptom
from tlo.util import random_date, sample_outcome

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------------------------------------
#   MODULE DEFINITION
# ---------------------------------------------------------------------------------------------------------

class Alri(Module):
    """This is the disease module for Acute Lower Respiratory Infections."""

    INIT_DEPENDENCIES = {
        'Demography',
        'Hiv',
        'Lifestyle',
        # 'NewbornOutcomes',
        'SymptomManager',
        # 'Wasting',
    }

    ADDITIONAL_DEPENDENCIES = {'Epi'}

    OPTIONAL_INIT_DEPENDENCIES = {'HealthBurden'}

    # Declare Metadata
    METADATA = {
        Metadata.DISEASE_MODULE,
        Metadata.USES_SYMPTOMMANAGER,
        Metadata.USES_HEALTHSYSTEM,
        Metadata.USES_HEALTHBURDEN
    }

    pathogens = {
        'viral': [
            'RSV',
            'Rhinovirus',
            'HMPV',
            'Parainfluenza',
            'Influenza',
            'other_viral_pathogens'
            # <-- Coronaviruses NL63, 229E OC43 and HKU1, Cytomegalovirus, Parechovirus/Enterovirus
        ],
        'bacterial': [
            'Strep_pneumoniae_PCV13',      # <--  risk of acquisition is affected by the pneumococcal vaccine
            'Strep_pneumoniae_non_PCV13',  # <--  risk of acquisition is not affected by the pneumococcal vaccine
            'Hib',
            'H.influenzae_non_type_b',
            'Staph_aureus',
            'Enterobacteriaceae',  # includes E. coli, Enterobacter species, and Klebsiella species
            'other_Strepto_Enterococci',  # includes Streptococcus pyogenes and Enterococcus faecium
            'other_bacterial_pathogens'
            # <-- includes Bordetella pertussis, Chlamydophila pneumoniae,
            # Legionella species, Mycoplasma pneumoniae, Moraxella catarrhalis, Non-fermenting gram-negative
            # rods (Acinetobacter species and Pseudomonas species), Neisseria meningitidis
        ],
        'fungal/other': [
            'P.jirovecii',
            'other_pathogens_NoS'
        ]
    }

    # Make set of all pathogens combined:
    all_pathogens = set(chain.from_iterable(pathogens.values()))

    # Declare Causes of Death
    CAUSES_OF_DEATH = {
        f"ALRI_{path}": Cause(gbd_causes={'Lower respiratory infections'}, label='Lower respiratory infections')
        for path in all_pathogens
    }

    # Declare Causes of Disability
    CAUSES_OF_DISABILITY = {
        f"ALRI_{path}": Cause(gbd_causes={'Lower respiratory infections'}, label='Lower respiratory infections')
        for path in all_pathogens
    }

    # Declare the disease types:
    disease_types = {
        'pneumonia', 'bronchiolitis/other_alri'
    }

    # Declare the Alri complications:
    complications = {'pneumothorax',
                     'pleural_effusion',
                     'empyema',
                     'lung_abscess',
                     'sepsis',
                     'hypoxaemia'  # <-- Low implies Sp02<93%'
                     }

    PARAMETERS = {
        # Incidence rate by pathogens  -----
        'base_inc_rate_ALRI_by_RSV':
            Parameter(Types.LIST,
                      'baseline incidence rate of Alri caused by RSV in age groups 0-11, 12-23, 24-59 months, '
                      'per child per year'
                      ),
        'base_inc_rate_ALRI_by_Rhinovirus':
            Parameter(Types.LIST,
                      'baseline incidence of Alri caused by Rhinovirus in age groups 0-11, 12-23, 24-59 months, '
                      'per child per year'
                      ),
        'base_inc_rate_ALRI_by_HMPV':
            Parameter(Types.LIST,
                      'baseline incidence of Alri caused by HMPV in age groups 0-11, 12-23, 24-59 months, '
                      'per child per year'
                      ),
        'base_inc_rate_ALRI_by_Parainfluenza':
            Parameter(Types.LIST,
                      'baseline incidence of Alri caused by Parainfluenza 1-4 in age groups 0-11, 12-23, 24-59 months, '
                      'per child per year'
                      ),
        'base_inc_rate_ALRI_by_Strep_pneumoniae_PCV13':
            Parameter(Types.LIST,
                      'baseline incidence of Alri caused by Streptoccocus pneumoniae PCV13 serotype '
                      'in age groups 0-11, 12-23, 24-59 months, per child per year'
                      ),
        'base_inc_rate_ALRI_by_Strep_pneumoniae_non_PCV13':
            Parameter(Types.LIST,
                      'baseline incidence of Alri caused by Streptoccocus pneumoniae non PCV13 serotype '
                      'in age groups 0-11, 12-23, 24-59 months, per child per year'
                      ),
        'base_inc_rate_ALRI_by_Hib':
            Parameter(Types.LIST,
                      'baseline incidence of Alri caused by Haemophilus influenzae type b '
                      'in age groups 0-11, 12-23, 24-59 months, per child per year'
                      ),
        'base_inc_rate_ALRI_by_H.influenzae_non_type_b':
            Parameter(Types.LIST,
                      'baseline incidence of Alri caused by Haemophilus influenzae non-type b '
                      'in age groups 0-11, 12-23, 24-59 months, per child per year'
                      ),
        'base_inc_rate_ALRI_by_Enterobacteriaceae':
            Parameter(Types.LIST,
                      'baseline incidence of Alri caused by Enterobacteriaceae in age groups 0-11, 12-23, 24-59 months,'
                      ' per child per year'
                      ),
        'base_inc_rate_ALRI_by_other_Strepto_Enterococci':
            Parameter(Types.LIST,
                      'baseline incidence of Alri caused by other streptococci and Enterococci including '
                      'Streptococcus pyogenes and Enterococcus faecium in age groups 0-11, 12-23, 24-59 months,'
                      ' per child per year'
                      ),
        'base_inc_rate_ALRI_by_Staph_aureus':
            Parameter(Types.LIST,
                      'baseline incidence of Alri caused by Staphylococcus aureus '
                      'in age groups 0-11, 12-23, 24-59 months, per child per year'
                      ),
        'base_inc_rate_ALRI_by_Influenza':
            Parameter(Types.LIST,
                      'baseline incidence of Alri caused by Influenza type A, B, and C '
                      'in age groups 0-11, 12-23, 24-59 months, per child per year'
                      ),
        'base_inc_rate_ALRI_by_P.jirovecii':
            Parameter(Types.LIST,
                      'baseline incidence of Alri caused by P. jirovecii in age groups 0-11, 12-59 months, '
                      'per child per year'
                      ),
        'base_inc_rate_ALRI_by_other_viral_pathogens':
            Parameter(Types.LIST,
                      'baseline incidence of Alri caused by other viral pathogens in age groups 0-11, 12-59 months, '
                      'per child per year'
                      ),
        'base_inc_rate_ALRI_by_other_bacterial_pathogens':
            Parameter(Types.LIST,
                      'baseline incidence of Alri caused by other viral pathogens in age groups 0-11, 12-59 months, '
                      'per child per year'
                      ),
        'base_inc_rate_ALRI_by_other_pathogens_NoS':
            Parameter(Types.LIST,
                      'baseline incidence of Alri caused by other pathogens not otherwise specified'
                      ' in age groups 0-11, 12-59 months, per child per year'
                      ),

        # Proportions of what disease type (pneumonia/ bronchiolitis/ other alri) -----
        'proportion_pneumonia_in_RSV_ALRI':
            Parameter(Types.LIST,
                      'proportion of RSV-attributed ALRI manifesting as pneumonia by age, '
                      '(based on PERCH CXR+ results)'
                      ),
        'proportion_pneumonia_in_Rhinovirus_ALRI':
            Parameter(Types.LIST,
                      'proportion of Rhinovirus-attributed ALRI manifesting as pneumonia by age, '
                      '(based on PERCH CXR+ results)'
                      ),
        'proportion_pneumonia_in_HMPV_ALRI':
            Parameter(Types.LIST,
                      'proportion of HMPV-attributed ALRI manifesting as pneumonia by age, '
                      '(based on PERCH CXR+ results)'
                      ),
        'proportion_pneumonia_in_Parainfluenza_ALRI':
            Parameter(Types.LIST,
                      'proportion of Parainfluenza-attributed ALRI manifesting as pneumonia by age, '
                      '(based on PERCH CXR+ results)'
                      ),
        'proportion_pneumonia_in_Strep_pneumoniae_PCV13_ALRI':
            Parameter(Types.LIST,
                      'proportion of S. pneumoniae PCV13-type-attributed ALRI manifesting as pneumonia by age, '
                      '(based on PERCH CXR+ results)'
                      ),
        'proportion_pneumonia_in_Strep_pneumoniae_non_PCV13_ALRI':
            Parameter(Types.LIST,
                      'proportion of S. pneumoniae non PCV13-type-attributed ALRI manifesting as pneumonia by age, '
                      '(based on PERCH CXR+ results)'
                      ),
        'proportion_pneumonia_in_Hib_ALRI':
            Parameter(Types.LIST,
                      'proportion of Hib ALRI-attributed ALRI manifesting as pneumonia by age, '
                      '(based on PERCH CXR+ results)'
                      ),
        'proportion_pneumonia_in_H.influenzae_non_type_b_ALRI':
            Parameter(Types.LIST,
                      'proportion of H.influenzae non type-b-attributed ALRI manifesting as pneumonia by age, '
                      '(based on PERCH CXR+ results)'
                      ),
        'proportion_pneumonia_in_Staph_aureus_ALRI':
            Parameter(Types.LIST,
                      'proportion of S. aureus-attributed ALRI manifesting as pneumonia by age, '
                      '(based on PERCH CXR+ results)'
                      ),
        'proportion_pneumonia_in_Enterobacteriaceae_ALRI':
            Parameter(Types.LIST,
                      'proportion of Enterobacteriaceae-attributed ALRI manifesting as pneumonia by age, '
                      '(based on PERCH CXR+ results)'
                      ),
        'proportion_pneumonia_in_other_Strepto_Enterococci_ALRI':
            Parameter(Types.LIST,
                      'proportion of other Streptococci- and Enterococci-attributed ALRI manifesting as pneumonia'
                      'by age, (based on PERCH CXR+ results)'
                      ),
        'proportion_pneumonia_in_Influenza_ALRI':
            Parameter(Types.LIST,
                      'proportion of Influenza-attributed ALRI manifesting as pneumonia by age, '
                      '(based on PERCH CXR+ results)'
                      ),
        'proportion_pneumonia_in_P.jirovecii_ALRI':
            Parameter(Types.LIST,
                      'proportion of P. jirovecii-attributed ALRI manifesting as pneumonia by age, '
                      '(based on PERCH CXR+ results)'
                      ),
        'proportion_pneumonia_in_other_viral_pathogens_ALRI':
            Parameter(Types.LIST,
                      'proportion of other viral pathogens-attributed ALRI manifesting as pneumonia by age, '
                      '(based on PERCH CXR+ results)'
                      ),
        'proportion_pneumonia_in_other_bacterial_pathogens_ALRI':
            Parameter(Types.LIST,
                      'proportion of other bacterial pathogens-attributed ALRI manifesting as pneumonia by age, '
                      '(based on PERCH CXR+ results)'
                      ),
        'proportion_pneumonia_in_other_pathogens_NoS_ALRI':
            Parameter(Types.LIST,
                      'proportion of other pathogens NoS-attributed ALRI manifesting as pneumonia by age, '
                      '(based on PERCH CXR+ results)'
                      ),

        # Risk factors for incidence infection -----
        'rr_ALRI_HIV/AIDS':
            Parameter(Types.REAL,
                      'relative rate of acquiring Alri for children with HIV+/AIDS '
                      ),
        'rr_ALRI_incomplete_measles_immunisation':
            Parameter(Types.REAL,
                      'relative rate of acquiring Alri for children with incomplete measles immunisation'
                      ),
        'rr_ALRI_low_birth_weight':
            Parameter(Types.REAL,
                      'relative rate of acquiring Alri for infants with low birth weight'
                      ),
        'rr_ALRI_non_exclusive_breastfeeding':
            Parameter(Types.REAL,
                      'relative rate of acquiring Alri for not exclusive breastfeeding upto 6 months'
                      ),
        'rr_ALRI_indoor_air_pollution':
            Parameter(Types.REAL,
                      'relative rate of acquiring Alri for indoor air pollution'
                      ),
        'rr_ALRI_crowding':
            Parameter(Types.REAL,
                      'relative rate of acquiring Alri for children living in crowed households (>7 pph)'
                      ),  # TODO: change to wealth?
        'rr_ALRI_underweight':
            Parameter(Types.REAL,
                      'relative rate of acquiring Alri for underweight children'
                      ),  # TODO: change to SAM/MAM?

        # Probability of bacterial co- / secondary infection -----
        'prob_viral_pneumonia_bacterial_coinfection':
            Parameter(Types.REAL,
                      'probability of primary viral pneumonia having a bacterial co-infection'
                      ),

        # Probability of complications -----
        'overall_progression_to_severe_ALRI':
            Parameter(Types.REAL,
                      'probability of progression to severe ALRI'
                      ),
        'prob_pulmonary_complications_in_pneumonia':
            Parameter(Types.REAL,
                      'probability of pulmonary complications in (CXR+) pneumonia'
                      ),
        'prob_pleural_effusion_in_pulmonary_complicated_pneumonia':
            Parameter(Types.REAL,
                      'probability of pleural effusion in pneumonia with pulmonary complications'
                      ),
        'prob_empyema_in_pulmonary_complicated_pneumonia':
            Parameter(Types.REAL,
                      'probability of empyema in pneumonia with pulmonary complications'
                      ),
        'prob_lung_abscess_in_pulmonary_complicated_pneumonia':
            Parameter(Types.REAL,
                      'probability of lung abscess in pneumonia with pulmonary complications'
                      ),
        'prob_pneumothorax_in_pulmonary_complicated_pneumonia':
            Parameter(Types.REAL,
                      'probability of pneumothorax in pneumonia with pulmonary complications'
                      ),
        'prob_hypoxaemia_in_pneumonia':
            Parameter(Types.REAL,
                      'probability of hypoxaemia in pneumonia cases'
                      ),
        'prob_hypoxaemia_in_bronchiolitis/other_alri':
            Parameter(Types.REAL,
                      'probability of hypoxaemia in bronchiolitis and other alri cases'
                      ),
        'prob_bacteraemia_in_pneumonia':
            Parameter(Types.REAL,
                      'probability of bacteraemia in pneumonia'
                      ),
        'prob_progression_to_sepsis_with_bacteraemia':
            Parameter(Types.REAL,
                      'probability of progression to sepsis from bactereamia'
                      ),
        'proportion_hypoxaemia_with_SpO2<90%':
            Parameter(Types.REAL,
                      'proportion of hypoxaemic children with SpO2 <90%'
                      ),

        # Risk of death parameters -----
        'overall_CFR_ALRI':
            Parameter(Types.REAL,
                      'overall case-fatality rate of ALRI (calibration value)'
                      ),
        'baseline_odds_alri_death':
            Parameter(Types.REAL,
                      'baseline odds of alri death, no risk factors'
                      ),
        'or_death_ALRI_age<2mo':
            Parameter(Types.REAL,
                      'odds ratio of ALRI death for infants aged less than 2 months'
                      ),
        'or_death_ALRI_P.jirovecii':
            Parameter(Types.REAL,
                      'odds ratio of ALRI death for P.jirovecii infection'
                      ),
        'or_death_ALRI_HIV/AIDS':
            Parameter(Types.REAL,
                      'odds ratio of ALRI death for HIV/AIDS children'
                      ),
        'or_death_ALRI_SAM':
            Parameter(Types.REAL,
                      'odds ratio of ALRI death for SAM'
                      ),
        'or_death_ALRI_MAM':
            Parameter(Types.REAL,
                      'odds ratio of ALRI death for MAM'
                      ),
        'or_death_ALRI_male':
            Parameter(Types.REAL,
                      'odds ratio of ALRI death for male children'
                      ),
        'or_death_ALRI_hc_referral':
            Parameter(Types.REAL,
                      'odds ratio of ALRI death for children referred from a health center'
                      ),
        'or_death_ALRI_SpO2<=92%':
            Parameter(Types.REAL,
                      'odds ratio of ALRI death for SpO2<=92%'
                      ),
        'or_death_ALRI_severe_underweight':
            Parameter(Types.REAL,
                      'odds ratio of ALRI death for severely underweight children'
                      ),
        'or_death_ALRI_danger_signs':
            Parameter(Types.REAL,
                      'odds ratio of ALRI death for very severe pneumonia (presenting danger signs)'
                      ),

        # Probability of symptom development -----
        'prob_cough_in_pneumonia':
            Parameter(Types.REAL,
                      'probability of cough in pneumonia'
                      ),
        'prob_difficult_breathing_in_pneumonia':
            Parameter(Types.REAL,
                      'probability of difficulty breathing in pneumonia'
                      ),
        'prob_chest_indrawing_in_pneumonia':
            Parameter(Types.REAL,
                      'probability of chest indrawing in pneumonia'
                      ),
        'prob_tachypnoea_in_pneumonia':
            Parameter(Types.REAL,
                      'probability of tachypnoea in pneumonia'
                      ),
        'prob_cough_in_bronchiolitis/other_alri':
            Parameter(Types.REAL,
                      'probability of cough in bronchiolitis or other alri'
                      ),
        'prob_difficult_breathing_in_bronchiolitis/other_alri':
            Parameter(Types.REAL,
                      'probability of difficulty breathing in bronchiolitis or other alri'
                      ),
        'prob_tachypnoea_in_bronchiolitis/other_alri':
            Parameter(Types.REAL,
                      'probability of tachypnoea in bronchiolitis or other alri'
                      ),
        'prob_chest_indrawing_in_bronchiolitis/other_alri':
            Parameter(Types.REAL,
                      'probability of chest wall indrawing in bronchiolitis or other alri'
                      ),
        'prob_danger_signs_in_sepsis':
            Parameter(Types.REAL,
                      'probability of any danger signs in complicated ALRI'
                      ),
        'prob_danger_signs_in_SpO2<90%':
            Parameter(Types.REAL,
                      'probability of any danger signs in children with SpO2 <90%'
                      ),
        'prob_danger_signs_in_SpO2_90-92%':
            Parameter(Types.REAL,
                      'probability of any danger signs in children with SpO2 between 90-92%'
                      ),
        'prob_chest_indrawing_in_SpO2<90%':
            Parameter(Types.REAL,
                      'probability of chest indrawing in children with SpO2 <90%'
                      ),
        'prob_chest_indrawing_in_SpO2_90-92%':
            Parameter(Types.REAL,
                      'probability of chest indrawing in children with SpO2 between 90-92%'
                      ),
        'prob_danger_signs_in_pulmonary_complications':
            Parameter(Types.REAL,
                      'probability of danger signs in children with pulmonary complications'
                      ),
        'prob_chest_indrawing_in_pulmonary_complications':
            Parameter(Types.REAL,
                      'probability of chest indrawing in children with pulmonary complications'
                      ),


        # Parameters governing the effects of vaccine ----------------
        'rr_Strep_pneum_VT_ALRI_with_PCV13_age<2y':
            Parameter(Types.REAL,
                      'relative rate of acquiring S. pneumoniae vaccine-type Alri '
                      'for children under 2 years of age immunised wth PCV13'
                      ),
        'rr_Strep_pneum_VT_ALRI_with_PCV13_age2to5y':
            Parameter(Types.REAL,
                      'relative rate of acquiring S. pneumoniae vaccine-type Alri '
                      'for children aged 2 to 5 immunised wth PCV13'
                      ),
        'rr_all_strains_Strep_pneum_ALRI_with_PCV13':
            Parameter(Types.REAL,
                      'relative rate of acquiring S. pneumoniae all types Alri '
                      'for children immunised wth PCV13'
                      ),
        'effectiveness_Hib_vaccine_on_Hib_strains':
            Parameter(Types.REAL,
                      'effectiveness of Hib vaccine against H. influenzae typ-b ALRI'
                      ),
        'rr_Hib_ALRI_with_Hib_vaccine':
            Parameter(Types.REAL,
                      'relative rate of acquiring H. influenzae type-b Alri '
                      'for children immunised wth Hib vaccine'
                      ),

        # Parameters governing treatment effectiveness and assoicated behaviours ----------------
        'days_between_treatment_and_cure':
            Parameter(Types.INT, 'number of days between any treatment being given in an HSI and the cure occurring.'
                      ),
        'prob_of_cure_for_uncomplicated_pneumonia_given_IMCI_pneumonia_treatment':
            Parameter(Types.REAL,
                      'probability of cure for uncomplicated pneumonia given IMCI pneumonia treatment'
                      ),
        'prob_of_cure_for_pneumonia_with_severe_complication_given_IMCI_severe_pneumonia_treatment':
            Parameter(Types.REAL,
                      'probability of cure for pneumonia with severe complications given IMCI pneumonia treatment'
                      ),
        'prob_seek_follow_up_care_after_treatment_failure':
            Parameter(Types.REAL,
                      'probability of seeking follow-up care after treatment failure'
                      ),
        'oxygen_therapy_effectiveness_ALRI':
            Parameter(Types.REAL,
                      'effectiveness of oxygen therapy on death from Alri with respiratory failure'
                      ),
        'antibiotic_therapy_effectiveness_ALRI':
            Parameter(Types.REAL,
                      'effectiveness of antibiotic therapy on death from Alri with bacterial cause'
                      ),
        '5day_amoxicillin_treatment_failure_by_day6':
            Parameter(Types.REAL,
                      'probability of treatment failure by day 6 of 5-day course amoxicillin for non-severe pneumonia'
                      ),
        '5day_amoxicillin_relapse_by_day14':
            Parameter(Types.REAL,
                      'probability of relapse by day 14 on 5-day amoxicillin for non-severe pneumonia'
                      ),
    }

    PROPERTIES = {
        # ---- Alri status ----
        'ri_current_infection_status':
            Property(Types.BOOL,
                     'Does the person currently have an infection with a pathogen that can cause Alri.'
                     ),

        # ---- The pathogen which is the attributed cause of Alri ----
        'ri_primary_pathogen':
            Property(Types.CATEGORICAL,
                     'If infected, what is the pathogen with which the person is currently infected. (np.nan if not '
                     'infected)',
                     categories=list(all_pathogens)
                     ),
        # ---- The bacterial pathogen which is the attributed co-/secondary infection ----
        'ri_secondary_bacterial_pathogen':
            Property(Types.CATEGORICAL,
                     'If infected, is there a secondary bacterial pathogen (np.nan if none or not applicable)',
                     categories=list(pathogens['bacterial'])
                     ),
        # ---- The underlying Alri condition ----
        'ri_disease_type':
            Property(Types.CATEGORICAL, 'If infected, what disease type is the person currently suffering from.',
                     categories=list(disease_types)
                     ),
        # ---- The peripheral oxygen saturation level ----
        'ri_SpO2_level':
            Property(Types.CATEGORICAL, 'Peripheral oxygen saturation level (Sp02), measure for hypoxaemia',
                     categories=['90%', '90-92%', '>=93%']
                     ),

        # ---- Treatment Status ----
        'ri_on_treatment': Property(Types.BOOL, 'Is this person currently receiving treatment.'),
        'ri_treatment_failure_by_day6': Property(Types.BOOL, 'Treatment failure by day 6'),
        'ri_post_treatment_relapse_by_day14': Property(Types.BOOL, 'Relapse post-treatment between day 7 and 14'),
        'ri_referred_from_hsa_hc': Property(Types.BOOL, 'Has this person been referred from an HSA or Health centre'),

        # < --- other properties of the form 'ri_complication_{complication-name}' are added later -->

        # ---- Internal variables to schedule onset and deaths due to Alri ----
        'ri_start_of_current_episode': Property(Types.DATE,
                                                'date of onset of current Alri event (pd.NaT is not infected)'),
        'ri_scheduled_recovery_date': Property(Types.DATE,
                                               '(scheduled) date of recovery from current Alri event (pd.NaT is not '
                                               'infected or episode is scheduled to end in death)'),
        'ri_scheduled_death_date': Property(Types.DATE,
                                            '(scheduled) date of death caused by current Alri event (pd.NaT is not '
                                            'infected or episode will not cause death)'),
        'ri_end_of_current_episode':
            Property(Types.DATE, 'date on which the last episode of Alri is resolved, (including '
                                 'allowing for the possibility that a cure is scheduled following onset). '
                                 'This is used to determine when a new episode can begin. '
                                 'This stops successive episodes interfering with one another.'),
        'ri_ALRI_tx_start_date': Property(Types.DATE,
                                          'start date of Alri treatment for current episode (pd.NaT is not infected or'
                                          ' treatment has not begun)'),
    }

    def __init__(self, name=None, resourcefilepath=None, log_indivdual=None, do_checks=False):
        super().__init__(name)

        # Store arguments provided
        self.resourcefilepath = resourcefilepath
        self.do_checks = do_checks

        assert (log_indivdual is None or isinstance(log_indivdual, int)) and (not isinstance(log_indivdual, bool))
        self.log_individual = log_indivdual

        # Initialise the pointer to where the models will be stored:
        self.models = None

        # Maximum duration of an episode (beginning with infection and ending with recovery)
        self.max_duration_of_episode = None

        # dict to hold the DALY weights
        self.daly_wts = dict()

        # dict to hold the consumables
        self.consumables_used_in_hsi = dict()

        # Pointer to store the logging event used by this module
        self.logging_event = None

    def read_parameters(self, data_folder):
        """
        * Setup parameters values used by the module
        * Define symptoms
        """
        self.load_parameters_from_dataframe(
            pd.read_excel(Path(self.resourcefilepath) / 'ResourceFile_Alri.xlsx', sheet_name='Parameter_values')
        )

        self.check_params_read_in_ok()

        self.define_symptoms()

    def check_params_read_in_ok(self):
        """Check that every value has been read-in successfully"""
        for param_name, param_type in self.PARAMETERS.items():
            assert param_name in self.parameters, f'Parameter "{param_name}" ' \
                                                  f'is not read in correctly from the resourcefile.'
            assert param_name is not None, f'Parameter "{param_name}" is not read in correctly from the resourcefile.'
            assert isinstance(self.parameters[param_name],
                              param_type.python_type), f'Parameter "{param_name}" ' \
                                                       f'is not read in correctly from the resourcefile.'

    def define_symptoms(self):
        """Define the symptoms that this module will use"""
        all_symptoms = {
            'cough', 'difficult_breathing', 'tachypnoea', 'chest_indrawing', 'danger_signs'
        }

        for symptom_name in all_symptoms:
            if symptom_name not in self.sim.modules['SymptomManager'].generic_symptoms:
                self.sim.modules['SymptomManager'].register_symptom(
                    Symptom(name=symptom_name)
                    # (associates the symptom with the 'average' healthcare seeking)
                )

    def pre_initialise_population(self):
        """Define columns for complications at run-time"""
        for complication in self.complications:
            Alri.PROPERTIES[f"ri_complication_{complication}"] = Property(
                Types.BOOL, f"Whether this person has complication {complication}"
            )

    def initialise_population(self, population):
        """
        Sets that there is no one with Alri at initiation.
        """
        df = population.props  # a shortcut to the data-frame storing data for individuals

        # ---- Key Current Status Classification Properties ----
        df.loc[df.is_alive, 'ri_current_infection_status'] = False
        df.loc[df.is_alive, 'ri_primary_pathogen'] = np.nan
        df.loc[df.is_alive, 'ri_secondary_bacterial_pathogen'] = np.nan
        df.loc[df.is_alive, 'ri_disease_type'] = np.nan
        df.loc[df.is_alive, [
            f"ri_complication_{complication}" for complication in self.complications]
        ] = False
        df.loc[df.is_alive, 'ri_SpO2_level'] = np.nan

        # ---- Internal values ----
        df.loc[df.is_alive, 'ri_start_of_current_episode'] = pd.NaT
        df.loc[df.is_alive, 'ri_scheduled_recovery_date'] = pd.NaT
        df.loc[df.is_alive, 'ri_scheduled_death_date'] = pd.NaT
        df.loc[df.is_alive, 'ri_end_of_current_episode'] = pd.NaT
        df.loc[df.is_alive, 'ri_on_treatment'] = False
        df.loc[df.is_alive, 'ri_ALRI_tx_start_date'] = pd.NaT
        df.loc[df.is_alive, 'ri_treatment_failure_by_day6'] = False
        df.loc[df.is_alive, 'ri_post_treatment_relapse_by_day14'] = False

    def initialise_simulation(self, sim):
        """
        Prepares for simulation:
        * Schedules the main polling event
        * Schedules the main logging event
        * Establishes the linear models and other data structures using the parameters that have been read-in
        """

        # Schedule the main polling event (to first occur immediately)
        sim.schedule_event(AlriPollingEvent(self), sim.date)

        # Schedule the main logging event (to first occur in one year)
        self.logging_event = AlriLoggingEvent(self)
        sim.schedule_event(self.logging_event, sim.date + DateOffset(years=1))

        if self.log_individual is not None:
            # Schedule the individual check logging event (to first occur immediately, and to occur every day)
            sim.schedule_event(AlriIndividualLoggingEvent(self), sim.date)

        if self.do_checks:
            # Schedule the event that does checking every day:
            sim.schedule_event(AlriCheckPropertiesEvent(self), sim.date)

        # Generate the model that determine the Natural History of the disease:
        self.models = Models(self)

        # Get DALY weights
        if 'HealthBurden' in self.sim.modules.keys():
            self.daly_wts['daly_non_severe_ALRI'] = self.sim.modules['HealthBurden'].get_daly_weight(sequlae_code=47)
            self.daly_wts['daly_severe_ALRI'] = self.sim.modules['HealthBurden'].get_daly_weight(sequlae_code=46)

        # Look-up and store the consumables that are required for each HSI
        self.look_up_consumables()

        # Define the max episode duration
        self.max_duration_of_episode = DateOffset(days=(self.parameters['days_between_treatment_and_cure'] + 14))

    def on_birth(self, mother_id, child_id):
        """Initialise properties for a newborn individual.
        This is called by the simulation whenever a new person is born.
        :param mother_id: the mother for this child
        :param child_id: the new child
        """

        df = self.sim.population.props

        # ---- Key Current Status Classification Properties ----
        df.at[child_id, 'ri_current_infection_status'] = False
        df.loc[child_id, ['ri_primary_pathogen',
                          'ri_secondary_bacterial_pathogen',
                          'ri_disease_type']] = np.nan
        df.at[child_id, [f"ri_complication_{complication}" for complication in self.complications]] = False
        df.at[child_id, 'ri_SpO2_level'] = np.nan

        # ---- Internal values ----
        df.loc[child_id, ['ri_start_of_current_episode',
                          'ri_scheduled_recovery_date',
                          'ri_scheduled_death_date',
                          'ri_end_of_current_episode']] = pd.NaT

    def report_daly_values(self):
        """Report DALY incurred in the population in the last month due to ALRI"""
        df = self.sim.population.props

        total_daly_values = pd.Series(data=0.0, index=df.index[df.is_alive])
        total_daly_values.loc[
            self.sim.modules['SymptomManager'].who_has('tachypnoea' or 'chest_indrawing')] = \
            self.daly_wts['daly_non_severe_ALRI']  # TODO: AND no danger_signs
        total_daly_values.loc[
            self.sim.modules['SymptomManager'].who_has('danger_signs')] = self.daly_wts['daly_severe_ALRI']

        # Split out by pathogen that causes the Alri
        dummies_for_pathogen = pd.get_dummies(df.loc[total_daly_values.index, 'ri_primary_pathogen'], dtype='float')
        daly_values_by_pathogen = dummies_for_pathogen.mul(total_daly_values, axis=0)

        # add prefix to label according to the name of the causes of disability declared
        daly_values_by_pathogen = daly_values_by_pathogen.add_prefix('ALRI_')
        return daly_values_by_pathogen

    def look_up_consumables(self):
        """Look up and store the consumables item codes used in each of the HSI."""

        get_item_codes_from_package = self.sim.modules['HealthSystem'].get_item_codes_from_package_name
        get_item_code = self.sim.modules['HealthSystem'].get_item_code_from_item_name

        self.consumables_used_in_hsi['Treatment_Non_severe_Pneumonia'] = \
            get_item_codes_from_package(package='Pneumonia treatment (children)')
        self.consumables_used_in_hsi['Treatment_Severe_Pneumonia'] = \
            get_item_codes_from_package(package='Treatment of severe pneumonia')
        self.consumables_used_in_hsi['First_dose_antibiotic_for_referral'] = \
            [get_item_code(item='Paracetamol, tablet, 100 mg')] + [get_item_code(item='Amoxycillin 250mg_1000_CMST')]

    def pneumonia_classification_iCCM_IMCI(self, person_id, hsi_event, facility_level):
        """Based on symptoms presented, classify WHO-pneumonia severity"""

        df = self.sim.population.props

        symptoms = hsi_event.sim.modules['SymptomManager'].has_what(person_id=person_id)
        # TODO: get other danger signs in iCCM when issue 429 is resolved
        # iccm_danger_signs = symptoms.append() other symptoms child may have that is considered severe in iCCM

        classification = ''

        # ----- For children over 2 months and under 5 years of age -----
        if (df.at[person_id, 'age_exact_years'] >= 1/6) & (df.at[person_id, 'age_exact_years'] < 5):

            if facility_level == '0':
                if 'tachypnoea' in symptoms and ('chest_indrawing' and 'danger_signs' not in symptoms):
                    classification = 'iCCM_non_severe_pneumonia'
                if 'chest_indrawing' in symptoms and ('danger_signs' not in symptoms):
                    classification = 'iCCM_severe_pneumonia'
                if 'danger_signs' in symptoms:
                    classification = 'iCCM_very_severe_pneumonia'

            if facility_level == '1a' or '1b' or '2':
                if ('tachypnoea' or 'chest_indrawing' in symptoms) and ('danger_signs' not in symptoms):
                    classification = 'IMCI_non_severe_pneumonia'
                if 'danger_signs' in symptoms:
                    classification = 'IMCI_severe_pneumonia'

        return classification

    def do_when_presentation_with_cough_or_difficult_breathing_level0(self, person_id, hsi_event):
        """This routine is called when cough or difficulty breathing is a symptom for a child attending
        a Generic HSI Appointment at level 0. It checks for danger signs and schedules HSI Events appropriately."""

        # get the classification
        classification = self.pneumonia_classification_iCCM_IMCI(person_id, hsi_event, facility_level='0')
        if classification == 'iCCM_very_severe_pneumonia' or 'iCCM_severe_pneumonia':
            # refer to health facility, give first dose of antibiotic
            treatment_plan = 'urgent_referral'
        else:
            # treat now
            treatment_plan = 'treat_at_home'

        # schedule the HSI with iCCM depending on treatment plan
        self.sim.modules['HealthSystem'].schedule_hsi_event(HSI_iCCM_Pneumonia_Treatment(
            person_id=person_id,
            treatment_plan=treatment_plan,
            module=self,
            appointment_type='first_appointment'),
            priority=0,
            topen=self.sim.date,
            tclose=None)

    def do_when_presentation_with_cough_or_difficult_breathing_level1(self, person_id, hsi_event):
        """This routine is called when cough or difficulty breathing is a symptom for a child attending
        a Generic HSI Appointment at level 1. It checks for danger signs and schedules HSI Events appropriately."""

        # get the classification
        classification = self.pneumonia_classification_iCCM_IMCI(person_id, hsi_event, facility_level='1a' or '1b')
        if classification == 'IMCI_severe_pneumonia':
            # refer to health facility, give first dose of antibiotic
            treatment_plan = 'treat_severe_pneumonia'
        else:
            # treat now
            treatment_plan = 'treat_non_severe_pneumonia'

        # schedule the HSI with IMCI depending on treatment plan
        self.sim.modules['HealthSystem'].schedule_hsi_event(HSI_IMCI_Pneumonia_Treatment(
            person_id=person_id,
            module=self,
            treatment_plan=treatment_plan,
            appointment_type='first_appointment',
            hsa_hc_referred=False),
            priority=0,
            topen=self.sim.date,
            tclose=None)

    def do_when_presentation_with_cough_or_difficult_breathing_level2(self, person_id, hsi_event):
        """This routine is called when cough or difficulty breathing is a symptom for a child attending
        a Generic HSI Appointment at level 2 """

        # get the classification
        classification = self.pneumonia_classification_iCCM_IMCI(person_id, hsi_event, facility_level='2')
        if classification == 'IMCI_severe_pneumonia':
            # treat severe pneumonia
            treatment_plan = 'treat_severe_pneumonia'
        else:
            # treat pneumonia
            treatment_plan = 'treat_non_severe_pneumonia'

        # schedule the HSI with IMCI depending on treatment plan
        self.sim.modules['HealthSystem'].schedule_hsi_event(HSI_IMCI_Hospital_Pneumonia_Treatment(
            person_id=person_id,
            module=self,
            treatment_plan=treatment_plan,
            appointment_type='first_appointment',
            hsa_hc_referred=False),
            priority=0,
            topen=self.sim.date,
            tclose=None)

    def end_episode(self, person_id):
        """End the episode infection for a person (i.e. reset all properties to show no current infection or
        complications).
        This is called by AlriNaturalRecoveryEvent and AlriCureEvent.
        """
        df = self.sim.population.props

        # Reset properties to show no current infection:
        df.loc[person_id, [
            'ri_current_infection_status',
            'ri_primary_pathogen',
            'ri_secondary_bacterial_pathogen',
            'ri_disease_type',
            'ri_SpO2_level',
            'ri_on_treatment',
            'ri_treatment_failure_by_day6',
            'ri_post_treatment_relapse_by_day14',
            'ri_start_of_current_episode',
            'ri_scheduled_recovery_date',
            'ri_scheduled_death_date',
            'ri_ALRI_tx_start_date']
        ] = [
            False,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            False,
            False,
            False,
            pd.NaT,
            pd.NaT,
            pd.NaT,
            pd.NaT,
        ]
        #  NB. 'ri_end_of_current_episode is not reset: this is used to prevent new infections from occuring whilst
        #  HSI from a previous episode may still be scheduled to occur.

        # Remove all existing complications
        df.loc[person_id, [f"ri_complication_{c}" for c in self.complications]] = False

        # Resolve all the symptoms immediately
        self.sim.modules['SymptomManager'].clear_symptoms(person_id=person_id, disease_module=self)

    def do_treatment(self, person_id, appointment_type, facility_level):
        """Helper function that enacts the effects of a treatment to Alri caused by a pathogen.
        It will only do something if the Alri is caused by a pathogen (this module).
        * Log the treatment
        * Prevent any death event that may be scheduled from occuring (prior to the cure event)
        * Schedules the cure event, at which the episode is ended
        """

        df = self.sim.population.props
        person = df.loc[person_id]
        p = self.parameters

        # Do nothing if the person is not alive
        if not person.is_alive:
            return

        # Do nothing if the person is not infected with a pathogen that can cause ALRI
        if not person['ri_current_infection_status']:
            return

        # Record that the person is now on treatment:
        df.loc[person_id, ['ri_on_treatment', 'ri_ALRI_tx_start_date']] = [True, self.sim.date]

        # First contact with the health system for the current illness
        if appointment_type == 'first_appointment':

            # check if they will have treatment failure by day 6 or relapse by day 14
            if p['5day_amoxicillin_treatment_failure_by_day6'] > self.rng.rand():
                person['ri_treatment_failure_by_day6'] = True
            else:
                person['ri_treatment_failure_by_day6'] = False
                if p['5day_amoxicillin_relapse_by_day14'] > self.rng.rand():
                    person['ri_post_treatment_relapse_by_day14'] = True
                else:
                    person['ri_post_treatment_relapse_by_day14'] = False

            # if no treatment failure or relapse, schedule recovery
            if not (person['ri_treatment_failure_by_day6'] or person['ri_post_treatment_relapse_by_day14']):
                # Cancel the death
                self.cancel_death_date(person_id)
                # Schedule the CureEvent
                cure_date = self.sim.date + DateOffset(days=self.parameters['days_between_treatment_and_cure'])
                self.sim.schedule_event(AlriCureEvent(self, person_id), cure_date)

            # Schedule the Follow-up appointment for all
            self.do_when_follow_up_appointments(person_id, facility_level)

        # -----------------------------------------------
        # Follow-up appointment for the current illness
        if appointment_type == 'follow_up':
            # TODO: add a probability of death or recovery
            # Cancel the death
            self.cancel_death_date(person_id)
            # Schedule the CureEvent
            cure_date = self.sim.date + DateOffset(days=self.parameters['days_between_treatment_and_cure'])
            self.sim.schedule_event(AlriCureEvent(self, person_id), cure_date)

        # -----------------------------------------------
        # Referral appointment for the current illness
        if appointment_type == 'referred':
            # TODO: add a probability of death or recovery
            # Cancel the death
            self.cancel_death_date(person_id)
            # Schedule the CureEvent
            cure_date = self.sim.date + DateOffset(days=self.parameters['days_between_treatment_and_cure'])
            self.sim.schedule_event(AlriCureEvent(self, person_id), cure_date)

    def do_when_follow_up_appointments(self, person_id, facility_level):
        """ Schedules follow-up appointments based on their first appointment facility level"""

        df = self.sim.population.props
        person = df.loc[person_id]

        # schedule follow-up appointment at day 3
        treatment_failure = person['ri_treatment_failure_by_day6']

        # if first appointment at facility level 0, return for follow-up
        if facility_level == '0':
            if treatment_failure:
                treatment_plan = 'urgent_referral'
            else:
                treatment_plan = 'continue_current_treatment'

            # schedule the follow-up appointment
            self.sim.modules['HealthSystem'].schedule_hsi_event(HSI_iCCM_Pneumonia_Follow_up(
                person_id=person_id,
                module=self,
                treatment_plan=treatment_plan,
                appointment_type='follow-up'),
                priority=0,
                topen=self.sim.date + DateOffset(days=3),
                tclose=None)

        # if first appointment at facility level 1a/1b, return for follow-up
        if facility_level == '1a' or '1b':
            if treatment_failure:  # if not feeling better
                treatment_plan = 'urgent_referral'
            else:
                treatment_plan = 'continue_current_treatment'

            # schedule the follow-up appointment
            self.sim.modules['HealthSystem'].schedule_hsi_event(HSI_IMCI_Pneumonia_Follow_up(
                person_id=person_id,
                module=self,
                treatment_plan=treatment_plan,
                appointment_type='follow-up'),
                priority=0,
                topen=self.sim.date + DateOffset(days=3),
                tclose=None)

        # if first appointment at facility level 2, return for follow-up
        # TODO: follow-up at level 2

    def cancel_death_date(self, person_id):
        """Cancels a scheduled date of death due to Alri for a person. This is called within do_treatment_alri function,
        and prior to the scheduling the CureEvent to prevent deaths happening in the time between
        a treatment being given and the cure event occurring.
        :param person_id:
        :return:
        """
        self.sim.population.props.at[person_id, 'ri_scheduled_death_date'] = pd.NaT

    def check_properties(self):
        """This is used in debugging to make sure that the configuration of properties is correct"""

        df = self.sim.population.props

        # identify those who currently have an infection with a pathogen that can cause ALRI:
        curr_inf = df['is_alive'] & df['ri_current_infection_status']
        not_curr_inf = df['is_alive'] & ~df['ri_current_infection_status']

        # For those with no current infection, variables about the current infection should be null
        assert df.loc[not_curr_inf, [
            'ri_primary_pathogen',
            'ri_secondary_bacterial_pathogen',
            'ri_disease_type',
            'ri_start_of_current_episode',
            'ri_scheduled_recovery_date',
            'ri_scheduled_death_date']
        ].isna().all().all()

        # For those with no current infection, 'ri_end_of_current_episode' should be null or in the past or within the
        # period for which the episode can last.
        assert (
            df.loc[not_curr_inf, 'ri_end_of_current_episode'].isna() |
            (df.loc[not_curr_inf, 'ri_end_of_current_episode'] <= self.sim.date) |
            (
                (df.loc[not_curr_inf, 'ri_end_of_current_episode'] - self.sim.date).dt.days
                <= self.max_duration_of_episode.days
            )
                ).all()

        # For those with no current infection, there should be no treatment
        assert not df.loc[not_curr_inf, 'ri_on_treatment'].any()
        assert df.loc[not_curr_inf, 'ri_ALRI_tx_start_date'].isna().all()

        # For those with no current infection, there should be no complications
        assert not df.loc[
            not_curr_inf, [f"ri_complication_{c}" for c in self.complications]
        ].any().any()

        # For those with current infection, variables about the current infection should not be null
        assert not df.loc[curr_inf, [
            'ri_primary_pathogen',
            'ri_disease_type']
        ].isna().any().any()

        # For those with current infection, dates relating to this episode should make sense
        # - start is in the past and end is in the future
        assert (df.loc[curr_inf, 'ri_start_of_current_episode'] <= self.sim.date).all()
        assert (df.loc[curr_inf, 'ri_end_of_current_episode'] >= self.sim.date).all()

        # - a person has exactly one of a recovery_date _or_ a death_date
        assert ((~df.loc[curr_inf, 'ri_scheduled_recovery_date'].isna()) | (
            ~df.loc[curr_inf, 'ri_scheduled_death_date'].isna())).all()
        assert (df.loc[curr_inf, 'ri_scheduled_recovery_date'].isna() != df.loc[
            curr_inf, 'ri_scheduled_death_date'].isna()).all()

        #  If that primary pathogen is bacterial then there should be np.nan for secondary_bacterial_pathogen:
        assert df.loc[
            curr_inf & df['ri_primary_pathogen'].isin(self.pathogens['bacterial']), 'ri_secondary_bacterial_pathogen'
        ].isna().all()

        # If person is on treatment, they should have a treatment start date
        assert (df.loc[curr_inf, 'ri_on_treatment'] != df.loc[curr_inf, 'ri_ALRI_tx_start_date'].isna()).all()

    def impose_symptoms_for_complicated_alri(self, person_id, complication):
        """Impose symptoms for ALRI with any complication."""

        for symptom in self.models.symptoms_for_complicated_alri(person_id, complication):
            self.sim.modules['SymptomManager'].change_symptom(
                person_id=person_id,
                symptom_string=symptom,
                add_or_remove='+',
                disease_module=self,
            )

        df = self.sim.population.props


class Models:
    """Helper-class to store all the models that specify the natural history of the Alri disease"""

    def __init__(self, module):
        self.module = module
        self.p = module.parameters
        self.rng = module.rng

        # dict that will hold the linear models for incidence risk for each pathogen
        self.incidence_equations_by_pathogen = dict()

        # set-up the linear models for the incidence risk for each pathogen
        self.make_model_for_acquisition_risk()

        # set-up the linear model for the death risk
        # self.death_equation = self.make_scaled_linear_model_for_death()
        # self.compute_death_risk(self, person_id)
        self.death_risk = None

    def make_model_for_acquisition_risk(self):
        """"Model for the acquisition of a primary pathogen that can cause ALri"""
        p = self.p
        df = self.module.sim.population.props

        def make_scaled_linear_model_for_incidence(patho):
            """Makes the unscaled linear model with default intercept of 1. Calculates the mean incidents rate for
            0-year-olds and then creates a new linear model with adjusted intercept so incidence in 0-year-olds
            matches the specified value in the model when averaged across the population. This will return an unscaled
            linear model if there are no 0-year-olds in the population.
            """

            def make_naive_linear_model(_patho, intercept=1.0):
                """Make the linear model based exactly on the parameters specified"""

                base_inc_rate = f'base_inc_rate_ALRI_by_{_patho}'
                return LinearModel(
                    LinearModelType.MULTIPLICATIVE,
                    intercept,
                    Predictor(
                        'age_years',
                        conditions_are_mutually_exclusive=True,
                        conditions_are_exhaustive=True,
                    )
                    .when(0, p[base_inc_rate][0])
                    .when(1, p[base_inc_rate][1])
                    .when('.between(2,4)', p[base_inc_rate][2])
                    .when('> 4', 0.0),
                    Predictor('li_wood_burn_stove').when(False, p['rr_ALRI_indoor_air_pollution']),
                    Predictor().when('(va_measles_all_doses == False) & (age_years > 1)',
                                     p['rr_ALRI_incomplete_measles_immunisation']),
                    Predictor('hv_inf').when(True, p['rr_ALRI_HIV/AIDS']),
                    Predictor('un_clinical_acute_malnutrition').when('SAM', p['rr_ALRI_underweight']),
                    Predictor('nb_breastfeeding_status').when('exclusive', 1.0)
                                                        .otherwise(p['rr_ALRI_non_exclusive_breastfeeding'])
                )  # todo: add crowding or wealth?

            zero_year_olds = df.is_alive & (df.age_years == 0)
            unscaled_lm = make_naive_linear_model(patho)

            # If not 0 year-olds then cannot do scaling, return unscaled linear model
            if sum(zero_year_olds) == 0:
                return unscaled_lm

            # If some 0 year-olds then can do scaling:
            target_mean = p[f'base_inc_rate_ALRI_by_{patho}'][0]
            actual_mean = unscaled_lm.predict(df.loc[zero_year_olds]).mean()
            scaled_intercept = 1.0 * (target_mean / actual_mean) \
                if (target_mean != 0 and actual_mean != 0 and ~np.isnan(actual_mean)) else 1.0
            scaled_lm = make_naive_linear_model(patho, intercept=scaled_intercept)

            # check by applying the model to mean incidence of 0-year-olds
            if (df.is_alive & (df.age_years == 0)).sum() > 0:
                assert (target_mean - scaled_lm.predict(df.loc[df.is_alive & (df.age_years == 0)]).mean()) < 1e-10
            return scaled_lm

        for patho in self.module.all_pathogens:
            self.incidence_equations_by_pathogen[patho] = make_scaled_linear_model_for_incidence(patho)

    def compute_risk_of_aquisition(self, pathogen, df):
        """Compute the risk of a pathogen, using the linear model created and the df provided"""
        p = self.p
        lm = self.incidence_equations_by_pathogen[pathogen]

        # run linear model to get baseline risk
        baseline = lm.predict(df)

        # apply the reduced risk of acquisition for those vaccinated
        if pathogen == "Strep_pneumoniae_PCV13":
            baseline.loc[df['va_pneumo_all_doses'] & df['age_years'] < 2] \
                *= p['rr_Strep_pneum_VT_ALRI_with_PCV13_age<2y']
            baseline.loc[df['va_pneumo_all_doses'] & df['age_years'].between(2, 5)] \
                *= p['rr_Strep_pneum_VT_ALRI_with_PCV13_age2to5y']
        elif pathogen == "Hib":
            baseline.loc[df['va_hib_all_doses']] *= p['rr_Hib_ALRI_with_Hib_vaccine']

        return baseline

    def determine_disease_type_and_secondary_bacterial_coinfection(self, pathogen, age,
                                                                   va_hib_all_doses,
                                                                   va_pneumo_all_doses):
        """Determines:
         * the disease that is caused by infection with this pathogen (from among self.disease_types)
         * if there is a bacterial coinfection associated that will cause the dominant disease.
         """
        p = self.p
        bacterial_coinfection = np.nan

        if pathogen in self.module.pathogens['bacterial']:
            if ((age < 1) and p[f'proportion_pneumonia_in_{pathogen}_ALRI'][0] > self.rng.rand()) or \
               ((age >=1 and age < 5) and p[f'proportion_pneumonia_in_{pathogen}_ALRI'][1] > self.rng.rand()):
                disease_type = 'pneumonia'
                # No bacterial co-infection in primary bacterial cause
                bacterial_coinfection = np.nan
            else:
                disease_type = 'bronchiolitis/other_alri'
                bacterial_coinfection = np.nan

        elif pathogen in self.module.pathogens['fungal/other']:
            if ((age < 1) and p[f'proportion_pneumonia_in_{pathogen}_ALRI'][0] > self.rng.rand()) or \
               ((age >=1 and age < 5) and p[f'proportion_pneumonia_in_{pathogen}_ALRI'][1] > self.rng.rand()):
                disease_type = 'pneumonia'
                # No bacterial co-infection in primary fungal or NoS pathogens cause (assumption)
                bacterial_coinfection = np.nan
            else:
                disease_type = 'bronchiolitis/other_alri'
                bacterial_coinfection = np.nan

        elif pathogen in self.module.pathogens['viral']:
            if ((age < 1) and p[f'proportion_pneumonia_in_{pathogen}_ALRI'][0] > self.rng.rand()) or \
               ((age >=1 and age < 5) and p[f'proportion_pneumonia_in_{pathogen}_ALRI'][1] > self.rng.rand()):
                disease_type = 'pneumonia'
                if p['prob_viral_pneumonia_bacterial_coinfection'] > self.rng.rand():
                    bacterial_coinfection = self.secondary_bacterial_infection(va_hib_all_doses, va_pneumo_all_doses)
                else:
                    bacterial_coinfection = np.nan
            else:
                disease_type = 'bronchiolitis/other_alri'
        else:
            raise ValueError('Pathogen is not recognised.')

        assert disease_type in self.module.disease_types
        assert bacterial_coinfection in (self.module.pathogens['bacterial'] + [np.nan])

        return disease_type, bacterial_coinfection

    def secondary_bacterial_infection(self, va_hib_all_doses, va_pneumo_all_doses):
        """Determine which specific bacterial pathogen causes a secondary coinfection, or if there is no secondary
        bacterial infection (due to the effects of the pneumococcal vaccine).
        """
        p = self.p

        # get probability of bacterial coinfection with each pathogen
        list_bacteria_probs = []
        for n in range(len(self.module.pathogens['bacterial'])):
            prob_secondary_patho = 1 / len(self.module.pathogens['bacterial'])  # assume equal distribution
            list_bacteria_probs.append(prob_secondary_patho)
        probs = dict(zip(
            self.module.pathogens['bacterial'], list_bacteria_probs))

        # Edit the probability that the coinfection will be of `Strep_pneumoniae_PCV13` if the person has had
        # the pneumococcal vaccine:
        if va_pneumo_all_doses:
            probs['Strep_pneumoniae_PCV13'] *= p['rr_Strep_pneum_VT_ALRI_with_PCV13_age2to5y']

        # Edit the probability that the coinfection will be of `Hib` if the person has had
        # the hib vaccine:
        if va_hib_all_doses:
            probs['Hib'] *= p['rr_Hib_ALRI_with_Hib_vaccine']

        # Add in the probability that there is none (to ensure that all probabilities sum to 1.0)
        probs['none'] = 1.0 - sum(probs.values())

        # return the random selection of bacterial coinfection (including possibly np.nan for 'none')
        outcome = self.rng.choice(list(probs.keys()), p=list(probs.values()))
        return outcome if not 'none' else np.nan

    def complications(self, person_id):
        """Determine the set of complication for this person"""
        p = self.p
        person = self.module.sim.population.props.loc[person_id]

        primary_path_is_bacterial = person['ri_primary_pathogen'] in self.module.pathogens['bacterial']
        primary_path_is_viral = person['ri_primary_pathogen'] in self.module.pathogens['viral']
        has_secondary_bacterial_inf = pd.notnull(person.ri_secondary_bacterial_pathogen)
        disease_type = person['ri_disease_type']

        probs = defaultdict(float)

        # probabilities for local pulmonary complications
        prob_pulmonary_complications = p['prob_pulmonary_complications_in_pneumonia']
        if disease_type == 'pneumonia' and (prob_pulmonary_complications > self.rng.rand()):
            for c in ['pneumothorax', 'pleural_effusion', 'lung_abscess', 'empyema']:
                probs[c] += p[f'prob_{c}_in_pulmonary_complicated_pneumonia']
                assert any(c)  # TODO: lung abscess, empyema should only apply to (primary or secondary) bacteria ALRIs

        # probabilities for systemic complications
        if disease_type == 'pneumonia' and (primary_path_is_bacterial or has_secondary_bacterial_inf):
            for c in ['sepsis']:
                probs[c] += p['prob_bacteraemia_in_pneumonia'] * \
                            p['prob_progression_to_sepsis_with_bacteraemia']

        if disease_type == 'pneumonia':
            for c in ['hypoxaemia']:
                probs[c] += p['prob_hypoxaemia_in_pneumonia']

        if disease_type == 'bronchiolitis/other_alri':
            for c in ['hypoxaemia']:
                probs[c] += p['prob_hypoxaemia_in_bronchiolitis/other_alri']

        # determine which complications are onset:
        complications = {c for c, p in probs.items() if p > self.rng.rand()}

        return complications

    def hypoxaemia_severity(self, person_id):
        """ Determine the level of severity of hypoxaemia by SpO2 measurement"""
        p = self.p
        df = self.module.sim.population.props

        if p['proportion_hypoxaemia_with_SpO2<90%'] < self.rng():
            df.at[person_id, 'ri_SpO2_level'] = '<90%'
        else:
            df.at[person_id, 'ri_SpO2_level'] = '90-92%'

    def symptoms_for_disease(self, disease_type):
        """Determine set of symptom (before complications) for a given instance of disease"""
        p = self.p

        assert disease_type in self.module.disease_types

        probs = {
            symptom: p[f'prob_{symptom}_in_{disease_type}']
            for symptom in [
                'cough', 'difficult_breathing', 'tachypnoea', 'chest_indrawing']
        }

        # determine which symptoms are onset:
        symptoms = {s for s, p in probs.items() if p > self.rng.rand()}

        return symptoms

    def symptoms_for_complicated_alri(self, person_id, complication):
        """Probability of each symptom for a person given a complication"""
        p = self.p
        df = self.module.sim.population.props

        lung_complications = ['pneumothorax', 'pleural_effusion', 'empyema', 'lung_abscess']

        probs = defaultdict(float)

        if complication == 'sepsis':
            probs = {
                'danger_signs': p['prob_danger_signs_in_sepsis']
            }

        if complication == any(lung_complications):
            probs = {
                'danger_signs': p['prob_danger_signs_in_pulmonary_complications'],
                'chest_indrawing': p['prob_chest_indrawing_in_pulmonary_complications']
            }

        if complication == 'hypoxaemia' and (df.at[person_id, 'ri_SpO2_level'] == '<90%'):
            probs = {
                'danger_signs': p['prob_danger_signs_in_SpO2<90%'],
                'chest_indrawing': p['prob_chest_indrawing_in_SpO2<90%']
            }
        if complication == 'hypoxaemia' and (df.at[person_id, 'ri_SpO2_level'] == '90-92%'):
            probs = {
                'danger_signs': p['prob_danger_signs_in_SpO2_90-92%'],
                'chest_indrawing': p['prob_chest_indrawing_in_SpO2_90-92%']
            }

        # determine which symptoms are onset:
        symptoms = {s for s, p in probs.items() if p > self.rng.rand()}

        return symptoms

    def compute_death_risk(self, person_id):
        """Determine if person will die from Alri. Returns True/False"""
        p = self.p
        df = self.module.sim.population.props

        def set_lm_death():
            """ Linear Model for ALRI death (Logistic regression)"""

            return LinearModel(
                LinearModelType.LOGISTIC,
                p['baseline_odds_alri_death'],
                Predictor('age_exact_years').when(1/6, p['or_death_ALRI_age<2mo']),
                Predictor('ri_primary_pathogen').when('P.jirovecii', p['or_death_ALRI_P.jirovecii']),
                Predictor('un_clinical_acute_malnutrition').when('SAM', p['or_death_ALRI_SAM']),
                Predictor('un_clinical_acute_malnutrition').when('MAM', p['or_death_ALRI_MAM']),
                Predictor('sy_danger_signs').when(2, p['or_death_ALRI_danger_signs']),
                Predictor('sex').when('M', p['or_death_ALRI_male']),
                Predictor('ri_complication_hypoxaemia').when(True, p['or_death_ALRI_SpO2<=92%']),
                Predictor('hv_inf').when(True, p['rr_ALRI_HIV/AIDS']),
                Predictor('referral_from_hsa_hc').when(True, p['or_death_ALRI_hc_referral'])
            )

        self.death_risk = set_lm_death()
        return self.death_risk.predict(df.loc[[person_id]]).values[0] > self.rng.rand()


# ---------------------------------------------------------------------------------------------------------
#   DISEASE MODULE EVENTS
# ---------------------------------------------------------------------------------------------------------

class AlriPollingEvent(RegularEvent, PopulationScopeEventMixin):
    """ This is the main event that runs the acquisition of pathogens that cause Alri.
    It determines who is infected and when and schedules individual IncidentCase events to represent onset.

    A known issue is that Alri events are scheduled based on the risk of current age but occur a short time
    later when the children will be slightly older. This means that when comparing the model output with data, the
    model slightly under-represents incidence among younger age-groups and over-represents incidence among older
    age-groups. This is a small effect when the frequency of the polling event is high."""

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=2))
        self.fraction_of_a_year_until_next_polling_event = self.compute_fraction_of_year_between_polling_event()

    def compute_fraction_of_year_between_polling_event(self):
        """Compute fraction of a year that elapses between polling event. This is used to adjust the risk of
        infection"""
        return (self.sim.date + self.frequency - self.sim.date) / np.timedelta64(1, 'Y')

    def apply(self, population):
        """Determine who will become infected and schedule for them an AlriComplicationOnsetEvent"""

        df = population.props
        m = self.module
        models = m.models

        # Compute the incidence rate for each person getting Alri and then convert into a probability
        # getting all children that do not currently have an Alri episode (never had or last episode resolved)
        mask_could_get_new_alri_event = (
            df.is_alive & (df.age_years < 5) & ~df.ri_current_infection_status &
            ((df.ri_end_of_current_episode < self.sim.date) | pd.isnull(df.ri_end_of_current_episode))
        )

        # Compute the incidence rate for each person acquiring Alri
        inc_of_acquiring_alri = pd.DataFrame(index=df.loc[mask_could_get_new_alri_event].index)
        for pathogen in m.all_pathogens:
            inc_of_acquiring_alri[pathogen] = models.compute_risk_of_aquisition(
                pathogen=pathogen,
                df=df.loc[mask_could_get_new_alri_event]
            )

        probs_of_acquiring_pathogen = 1 - np.exp(
            -inc_of_acquiring_alri * self.fraction_of_a_year_until_next_polling_event
        )

        # Sample to find outcomes:
        outcome = sample_outcome(probs=probs_of_acquiring_pathogen, rng=self.module.rng)

        # For persons that will become infected with a particular pathogen:
        for person_id, pathogen in outcome.items():
            #  Create the event for the onset of infection:
            self.sim.schedule_event(
                event=AlriIncidentCase(
                    module=self.module,
                    person_id=person_id,
                    pathogen=pathogen,
                ),
                date=random_date(self.sim.date, self.sim.date + self.frequency - pd.DateOffset(days=1), m.rng)
            )


class AlriIncidentCase(Event, IndividualScopeEventMixin):
    """
    This Event is for the onset of the infection that causes Alri. It is scheduled by the AlriPollingEvent.
    """

    def __init__(self, module, person_id, pathogen):
        super().__init__(module, person_id=person_id)
        self.pathogen = pathogen

    def apply(self, person_id):
        """
        * Determines the disease and complications associated with this case
        * Updates all the properties so that they pertain to this current episode of Alri
        * Imposes the symptoms
        * Schedules relevant natural history event {(AlriWithPulmonaryComplicationsEvent) and
        (AlriWithSevereComplicationsEvent) (either AlriNaturalRecoveryEvent or AlriDeathEvent)}
        * Updates the counters in the log accordingly.
        """
        df = self.sim.population.props  # shortcut to the dataframe
        person = df.loc[person_id]
        m = self.module
        p = m.parameters
        rng = self.module.rng
        models = m.models

        # The event should not run if the person is not currently alive:
        if not person.is_alive:
            return

        # Add this case to the counter:
        self.module.logging_event.new_case(age=person.age_years, pathogen=self.pathogen)

        # ----------------- Determine the Alri disease type and bacterial coinfection for this case -----------------
        disease_type, bacterial_coinfection = models.determine_disease_type_and_secondary_bacterial_coinfection(
            age=person.age_years, pathogen=self.pathogen, va_hib_all_doses=person.va_hib_all_doses,
            va_pneumo_all_doses=person.va_pneumo_all_doses)

        # ----------------------- Duration of the Alri event -----------------------
        duration_in_days_of_alri = rng.randint(1, 14)  # assumes uniform interval around mean duration with range 7 days

        # Date for outcome (either recovery or death) with uncomplicated Alri
        date_of_outcome = self.module.sim.date + DateOffset(days=duration_in_days_of_alri)

        # Define 'episode end' date. This the date when this episode ends. It is the last possible data that any HSI
        # could affect this episode.
        episode_end = date_of_outcome + DateOffset(days=p['days_between_treatment_and_cure'])

        # Update the properties in the dataframe:
        df.loc[person_id,
               (
                   'ri_current_infection_status',
                   'ri_primary_pathogen',
                   'ri_secondary_bacterial_pathogen',
                   'ri_disease_type',
                   'ri_on_treatment',
                   'ri_start_of_current_episode',
                   'ri_scheduled_recovery_date',
                   'ri_scheduled_death_date',
                   'ri_end_of_current_episode',
                   'ri_ALRI_tx_start_date'
               )] = (
            True,
            self.pathogen,
            bacterial_coinfection,
            disease_type,
            False,
            self.sim.date,
            pd.NaT,
            pd.NaT,
            episode_end,
            pd.NaT
        )

        # ----------------------------------- Clinical Symptoms -----------------------------------
        # impose clinical symptoms for new uncomplicated Alri
        self.impose_symptoms_for_uncomplicated_disease(person_id=person_id, disease_type=disease_type)

        # ----------------------------------- Complications  -----------------------------------
        self.impose_complications(person_id=person_id)

        # ----------------------------------- Outcome  -----------------------------------
        if models.compute_death_risk(person_id=person_id):
            self.sim.schedule_event(AlriDeathEvent(self.module, person_id), date_of_outcome)
            df.loc[person_id, ['ri_scheduled_death_date', 'ri_scheduled_recovery_date']] = [date_of_outcome, pd.NaT]
        else:
            self.sim.schedule_event(AlriNaturalRecoveryEvent(self.module, person_id), date_of_outcome)
            df.loc[person_id, ['ri_scheduled_recovery_date', 'ri_scheduled_death_date']] = [date_of_outcome, pd.NaT]

        # TODO: death should only be applied to those with any complication, currently applies to all ALRIs

    def impose_symptoms_for_uncomplicated_disease(self, person_id, disease_type):
        """
        Imposes the clinical symptoms to uncomplicated Alri. These symptoms are not set to auto-resolve
        """
        m = self.module
        models = m.models

        for symptom in models.symptoms_for_disease(disease_type=disease_type):
            m.sim.modules['SymptomManager'].change_symptom(
                person_id=person_id,
                symptom_string=symptom,
                add_or_remove='+',
                disease_module=m,
            )

    def impose_complications(self, person_id):
        """Choose a set of complications for this person and onset these all instantaneously."""

        df = self.sim.population.props
        m = self.module
        models = m.models

        complications = models.complications(person_id=person_id)
        df.loc[person_id, [f"ri_complication_{complication}" for complication in complications]] = True
        if complications == 'hypoxaemia':
            models.hypoxaemia_severity(person_id=person_id)

        for complication in complications:
            m.impose_symptoms_for_complicated_alri(person_id=person_id, complication=complication)


class AlriNaturalRecoveryEvent(Event, IndividualScopeEventMixin):
    """
    This is the Natural Recovery event. It is scheduled by the AlriIncidentCase Event for someone who will recover
    from the infection even if no care received.
    It calls the 'end_infection' function.
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        df = self.sim.population.props
        person = df.loc[person_id]

        # The event should not run if the person is not currently alive
        if not person.is_alive:
            return

        # Check if person should really recover:
        if (
            person.ri_current_infection_status and
            (person.ri_scheduled_recovery_date == self.sim.date) and
            pd.isnull(person.ri_scheduled_death_date)
        ):
            # Log the recovery
            self.module.logging_event.new_recovered_case(
                age=person.age_years,
                pathogen=person.ri_primary_pathogen
            )

            # Do the episode:
            self.module.end_episode(person_id=person_id)


class AlriCureEvent(Event, IndividualScopeEventMixin):
    """This is the cure event. It is scheduled by an HSI treatment event. It enacts the actual "cure" of the person
    that is caused (after some delay) by the treatment administered."""

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        df = self.sim.population.props
        person = df.loc[person_id]

        # The event should not run if the person is not currently alive
        if not person.is_alive:
            return

        # Check if person should really be cured:
        if person.ri_current_infection_status:
            # Log the cure:
            pathogen = person.ri_primary_pathogen
            self.module.logging_event.new_cured_case(
                age=person.age_years,
                pathogen=pathogen
            )

            # End the episode:
            self.module.end_episode(person_id=person_id)


class AlriDeathEvent(Event, IndividualScopeEventMixin):
    """
    This Event is for the death of someone that is caused by the infection with a pathogen that causes Alri.
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        df = self.sim.population.props  # shortcut to the dataframe
        person = df.loc[person_id]

        # The event should not run if the person is not currently alive
        if not person.is_alive:
            return

        # Check if person should really die of Alri:
        if (
            person.ri_current_infection_status and
            (person.ri_scheduled_death_date == self.sim.date) and
            pd.isnull(person.ri_scheduled_recovery_date)
        ):
            # Do the death:
            pathogen = person.ri_primary_pathogen
            self.module.sim.modules['Demography'].do_death(
                individual_id=person_id,
                cause='ALRI_' + pathogen,
                originating_module=self.module
            )

            # Log the death in the Alri logging system
            self.module.logging_event.new_death(
                age=person.age_years,
                pathogen=pathogen
            )


# ---------------------------------------------------------------------------------------------------------
# ==================================== HEALTH SYSTEM INTERACTION EVENTS ====================================
# ---------------------------------------------------------------------------------------------------------

# class HSI_Alri_GenericTreatment(HSI_Event, IndividualScopeEventMixin):
#     """
#     This is a template for the HSI interaction events. It just shows the checks to use each time.
#     """
#
#     def __init__(self, module, person_id):
#         super().__init__(module, person_id=person_id)
#
#         self.TREATMENT_ID = 'Alri_GenericTreatment'
#         self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'Over5OPD': 1})
#         self.ACCEPTED_FACILITY_LEVEL = '1a'
#         self.ALERT_OTHER_DISEASES = []
#
#     def apply(self, person_id, squeeze_factor):
#         """Do the treatment"""
#
#         df = self.sim.population.props
#         person = df.loc[person_id]
#
#         # Exit if the person is not alive or is not currently infected:
#         if not (person.is_alive and person.ri_current_infection_status):
#             return
#
#         # For example, say that probability of cure = 1.0
#         self.module.do_treatment(person_id=person_id,
#                                  facility_level=self.ACCEPTED_FACILITY_LEVEL)


class HSI_iCCM_Pneumonia_Treatment(HSI_Event, IndividualScopeEventMixin):
    """
    HSI event for treating cough and/ or difficult breathing at the community level
    """

    def __init__(self, module, person_id, treatment_plan, appointment_type):
        super().__init__(module, person_id=person_id)
        self.treatment_plan = treatment_plan
        self.appointment_type = appointment_type

        self.TREATMENT_ID = 'HSI_iCCM_Pneumonia_Treatment'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'ConWithDCSA': 1})
        self.ACCEPTED_FACILITY_LEVEL = '0'
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        """Do the treatment"""

        df = self.sim.population.props
        person = df.loc[person_id]

        # Exit if the person is not alive or is not currently infected:
        if not (person.is_alive and person.ri_current_infection_status):
            return

        if self.treatment_plan == 'urgent_referral':
            # get the first dose of antibiotic
            self.get_consumables(
                item_codes=self.module.consumables_used_in_hsi['First_dose_antibiotic_for_referral'],
                return_individual_results=True)
            # and refer to facility level 1 or 2
            self.sim.modules['HealthSystem'].schedule_hsi_event(
                HSI_IMCI_Pneumonia_Treatment(
                    person_id=person_id,
                    module=self,
                    treatment_plan='severe_pneumonia',
                    appointment_type='referred',
                    hsa_hc_referred=True),
                priority=0,
                topen=self.sim.date,
                tclose=None)
        else:
            if self.get_consumables(
                item_codes=self.module.consumables_used_in_hsi[
                    'Treatment_Non_severe_Pneumonia'], return_individual_results=True):
                self.module.do_treatment(person_id=person_id,
                                         appointment_type=self.appointment_type,
                                         facility_level=self.ACCEPTED_FACILITY_LEVEL)
            else:
                # if no consumables refer to health facility
                self.sim.modules['HealthSystem'].schedule_hsi_event(
                    HSI_IMCI_Pneumonia_Treatment(
                        person_id=person_id,
                        module=self,
                        treatment_plan='treat_non_severe_pneumonia',
                        first_appointment=False,
                        hsa_hc_referred=True),
                    priority=0,
                    topen=self.sim.date,
                    tclose=None)


class HSI_iCCM_Pneumonia_Follow_up(HSI_Event, IndividualScopeEventMixin):
    """
    HSI event for follow-up pneumonia treatment in level 0
    """

    def __init__(self, module, person_id, treatment_plan, appointment_type):
        super().__init__(module, person_id=person_id)
        self.treatment_plan = treatment_plan
        self.appointment_type = appointment_type

        self.TREATMENT_ID = 'HSI_iCCM_Pneumonia_Follow_up'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'ConWithDCSA': 1})
        self.ACCEPTED_FACILITY_LEVEL = '0'
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        """Do the treatment"""

        df = self.sim.population.props
        person = df.loc[person_id]

        # Exit if the person is not alive or is not currently infected:
        if not (person.is_alive and person.ri_current_infection_status):
            return

        if self.treatment_plan == 'urgent_referral':
            # get the first dose of antibiotic
            self.get_consumables(
                item_codes=self.module.consumables_used_in_hsi['First_dose_antibiotic_for_referral'],
                return_individual_results=True)
            # and refer to facility level 1 or 2
            self.sim.modules['HealthSystem'].schedule_hsi_event(
                HSI_IMCI_Pneumonia_Treatment(
                    person_id=person_id,
                    module=self,
                    treatment_plan='treat_severe_pneumonia',
                    appointment_type='follow-up',
                    hsa_hc_referred=True),
                priority=0,
                topen=self.sim.date,
                tclose=None)
        else:
            return


class HSI_IMCI_Pneumonia_Treatment(HSI_Event, IndividualScopeEventMixin):
    """
    HSI event for treating cough and/ or difficult breathing at the primary level (health centres)
    """

    def __init__(self, module, person_id, treatment_plan, appointment_type, hsa_hc_referred):
        super().__init__(module, person_id=person_id)
        self.treatment_plan = treatment_plan
        self.appointment_type = appointment_type
        self.hsa_hc_referred = hsa_hc_referred

        self.TREATMENT_ID = 'HSI_IMCI_Pneumonia_Treatment'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'Under5OPD': 1})
        self.ACCEPTED_FACILITY_LEVEL = '1a'
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        """Do the treatment"""

        df = self.sim.population.props
        person = df.loc[person_id]

        # Exit if the person is not alive or is not currently infected:
        if not (person.is_alive and person.ri_current_infection_status):
            return

        if self.treatment_plan == 'treat_severe_pneumonia':
            # first check if facility level has capabilities to treat severe pneumonia, if not, then refer
            if self.get_consumables(
                item_codes=self.module.consumables_used_in_hsi[
                    'Treatment_Severe_Pneumonia'], return_individual_results=True):
                self.module.do_treatment(person_id=person_id,
                                         appointment_type=self.appointment_type,
                                         facility_level=self.ACCEPTED_FACILITY_LEVEL)

            else:
                # Give first dose of antibiotic and refer to hospital
                self.get_consumables(
                    item_codes=self.module.consumables_used_in_hsi[
                        'First_dose_antibiotic_for_referral'], return_individual_results=True)
                # refer to facility level 1b or 2
                self.sim.modules['HealthSystem'].schedule_hsi_event(
                    HSI_IMCI_Hospital_Pneumonia_Treatment(
                        person_id=person_id,
                        module=self,
                        treatment_plan='treat_severe_pneumonia',
                        appointment_type='referred',
                        hsa_hc_referred=True),
                    priority=0,
                    topen=self.sim.date,
                    tclose=None)

        elif self.treatment_plan == 'treat_non_severe_pneumonia':
            # Treat uncomplicated cases now
            if self.get_consumables(
                item_codes=self.module.consumables_used_in_hsi[
                    'Treatment_Non_severe_Pneumonia'], return_individual_results=True):
                self.module.do_treatment(person_id=person_id,
                                         appointment_type=self.appointment_type,
                                         facility_level=self.ACCEPTED_FACILITY_LEVEL)

        elif self.treatment_plan == 'urgent_referral':
            # refer to facility level 1b or 2
            self.sim.modules['HealthSystem'].schedule_hsi_event(
                HSI_IMCI_Hospital_Pneumonia_Treatment(
                    person_id=person_id,
                    module=self,
                    treatment_plan='treat_severe_pneumonia',
                    appointment_type='referred',
                    hsa_hc_referred=True),
                priority=0,
                topen=self.sim.date,
                tclose=None)

        elif self.treatment_plan == 'continue_current_treatment':
            return


class HSI_IMCI_Pneumonia_Follow_up(HSI_Event, IndividualScopeEventMixin):
    """
    HSI event for follow-up pneumonia treatment in level 1a/1b
    """

    def __init__(self, module, person_id, treatment_plan, appointment_type):
        super().__init__(module, person_id=person_id)
        self.treatment_plan = treatment_plan
        self.appointment_type = appointment_type

        self.TREATMENT_ID = 'HSI_IMCI_Pneumonia_Follow_up'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'Under5OPD': 1})
        self.ACCEPTED_FACILITY_LEVEL = '1a'
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        """Do the treatment"""

        df = self.sim.population.props
        person = df.loc[person_id]

        # Exit if the person is not alive or is not currently infected:
        if not (person.is_alive and person.ri_current_infection_status):
            return

        if self.treatment_plan == 'urgent_referral':
            # and refer to facility level 2
            self.sim.modules['HealthSystem'].schedule_hsi_event(
                HSI_IMCI_Hospital_Pneumonia_Treatment(
                    person_id=person_id,
                    module=self,
                    treatment_plan=self.treatment_plan,
                    appointment_type='follow-up',
                    hsa_hc_referred=True),
                priority=0,
                topen=self.sim.date,
                tclose=None)
        else:
            return


class HSI_IMCI_Hospital_Pneumonia_Treatment(HSI_Event, IndividualScopeEventMixin):
    """
    HSI event for treating cough and/ or difficult breathing at the secondary level (hospital)
    """

    def __init__(self, module, person_id, treatment_plan, appointment_type, hsa_hc_referred):
        super().__init__(module, person_id=person_id)
        self.treatment_plan = treatment_plan
        self.appointment_type = appointment_type
        self.hsa_hc_referred = hsa_hc_referred

        self.TREATMENT_ID = 'HSI_IMCI_Pneumonia_Treatment'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'InpatientDays': 2, 'IPAdmission': 1})
        self.BEDDAYS_FOOTPRINT = self.make_beddays_footprint({'general_bed': 3})
        self.ACCEPTED_FACILITY_LEVEL = '2'
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        """Do the treatment"""

        df = self.sim.population.props
        person = df.loc[person_id]

        # Exit if the person is not alive or is not currently infected:
        if not (person.is_alive and person.ri_current_infection_status):
            return

        # update properties:
        if self.hsa_hc_referred:
            person['ri_referred_from_hsa_hc'] = True

        # Treat now
        print(self.get_consumables(
            item_codes=self.module.consumables_used_in_hsi[
                'Treatment_Severe_Pneumonia'], return_individual_results=True))

        consumables_available = self.get_consumables(
            item_codes=self.module.consumables_used_in_hsi[
                'Treatment_Severe_Pneumonia'], return_individual_results=True)
        x_ray_availability = consumables_available.get(175)
        print(x_ray_availability)

        if self.get_consumables(
            item_codes=self.module.consumables_used_in_hsi[
                'Treatment_Severe_Pneumonia'], return_individual_results=True):
            self.module.do_treatment(person_id=person_id,
                                     appointment_type=self.appointment_type,
                                     facility_level=self.ACCEPTED_FACILITY_LEVEL)


# ---------------------------------------------------------------------------------------------------------
#   LOGGING EVENTS
# ---------------------------------------------------------------------------------------------------------

class AlriLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    """
    This Event logs the number of incident cases that have occurred since the previous logging event.
    Analysis scripts expect that the frequency of this logging event is once per year.
    """

    def __init__(self, module):
        # This event to occur every year
        self.repeat = 12
        super().__init__(module, frequency=DateOffset(months=self.repeat))

        # initialise trakcers of incident cases, new recoveries, new treatments and deaths due to ALRI
        age_grps = {**{0: "0", 1: "1", 2: "2-4", 3: "2-4", 4: "2-4"}, **{x: "5+" for x in range(5, 100)}}

        self.trackers = dict()
        self.trackers['incident_cases'] = Tracker(age_grps=age_grps, pathogens=self.module.all_pathogens)
        self.trackers['recovered_cases'] = Tracker(age_grps=age_grps, pathogens=self.module.all_pathogens)
        self.trackers['cured_cases'] = Tracker(age_grps=age_grps, pathogens=self.module.all_pathogens)
        self.trackers['deaths'] = Tracker(age_grps=age_grps, pathogens=self.module.all_pathogens)

    def new_case(self, age, pathogen):
        self.trackers['incident_cases'].add_one(age=age, pathogen=pathogen)

    def new_recovered_case(self, age, pathogen):
        self.trackers['recovered_cases'].add_one(age=age, pathogen=pathogen)

    def new_cured_case(self, age, pathogen):
        self.trackers['cured_cases'].add_one(age=age, pathogen=pathogen)

    def new_death(self, age, pathogen):
        self.trackers['deaths'].add_one(age=age, pathogen=pathogen)

    def apply(self, population):
        """
        Log:
        1) Number of new cases, by age-group and by pathogen since the last logging event
        2) Total number of cases, recovery, treatments and deaths since the last logging event
        """

        # 1) Number of new cases, by age-group and by pathogen, since the last logging event
        logger.info(
            key='incidence_count_by_age_and_pathogen',
            data=self.trackers['incident_cases'].report_current_counts(),
            description='pathogens incident case counts in the last year'
        )

        # 2) Total number of cases, recovery, treatments and deaths since the last logging event
        logger.info(
            key='event_counts',
            data={k: v.report_current_total() for k, v in self.trackers.items()},
            description='Counts of cases, recovery, treatment and death in the last year'
        )

        # 3) Reset the trackers
        for tracker in self.trackers.values():
            tracker.reset()


class Tracker():
    """Helper class to be a counter for number of events occuring by age-group and by pathogen."""

    def __init__(self, age_grps: dict, pathogens: list):
        """Create and initalise tracker"""

        # Check and store parameters
        self.pathogens = pathogens
        self.age_grps_lookup = age_grps
        self.unique_age_grps = list(set(self.age_grps_lookup.values()))
        self.unique_age_grps.sort()

        # Initialise Tracker
        self.tracker = None
        self.reset()

    def reset(self):
        """Produce a dict of the form: { <Age-Grp>: {<Pathogen>: <Count>} }"""
        self.tracker = {
            age: dict(zip(self.pathogens, [0] * len(self.pathogens))) for age in self.unique_age_grps
        }

    def add_one(self, age, pathogen):
        """Increment counter by one for a specific age and pathogen"""
        assert age in self.age_grps_lookup, 'Age not recognised'
        assert pathogen in self.pathogens, 'Pathogen not recognised'

        # increment by one:
        age_grp = self.age_grps_lookup[age]
        self.tracker[age_grp][pathogen] += 1

    def report_current_counts(self):
        return self.tracker

    def report_current_total(self):
        total = 0
        for _a in self.tracker.keys():
            total += sum(self.tracker[_a].values())
        return total


# ---------------------------------------------------------------------------------------------------------
#   DEBUGGING / TESTING EVENTS
# ---------------------------------------------------------------------------------------------------------

class AlriCheckPropertiesEvent(RegularEvent, PopulationScopeEventMixin):
    """This event runs daily and checks properties are in the right configuration. Only use whilst debugging!
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(days=1))

    def apply(self, population):
        self.module.check_properties()


class AlriIndividualLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    """
    This Event logs the daily occurrence to a single individual child.
    """

    def __init__(self, module):
        # This logging event to occur every day
        super().__init__(module, frequency=DateOffset(days=1))

        self.person_id = self.module.log_individual
        assert self.person_id in module.sim.population.props.index, 'The person identified to be logged does not exist.'

    def apply(self, population):
        """Log all properties for this module"""
        if self.person_id is not None:
            df = self.sim.population.props
            logger.info(
                key='log_individual',
                data=df.loc[self.person_id, self.module.PROPERTIES.keys()].to_dict(),
                description='Properties for one person (the first under-five-year-old in the dataframe), each day.'
            )


class AlriPropertiesOfOtherModules(Module):
    """For the purpose of the testing, this module generates the properties upon which the Alri module relies"""

    INIT_DEPENDENCIES = {'Demography'}

    # Though this module provides some properties from NewbornOutcomes we do not list
    # NewbornOutcomes in the ALTERNATIVE_TO set to allow using in conjunction with
    # SimplifiedBirths which can also be used as an alternative to NewbornOutcomes
    ALTERNATIVE_TO = {'Hiv', 'Epi', 'Wasting'}

    PROPERTIES = {
        'hv_inf': Property(Types.BOOL, 'temporary property'),
        'nb_low_birth_weight_status': Property(Types.CATEGORICAL, 'temporary property',
                                               categories=['extremely_low_birth_weight', 'very_low_birth_weight',
                                                           'low_birth_weight', 'normal_birth_weight']),

        'nb_breastfeeding_status': Property(Types.CATEGORICAL, 'temporary property',
                                            categories=['none', 'non_exclusive', 'exclusive']),
        'va_pneumo_all_doses': Property(Types.BOOL, 'temporary property'),
        'va_measles_all_doses': Property(Types.BOOL, 'temporary property'),
        'va_hib_all_doses': Property(Types.BOOL, 'temporary property'),
        'un_clinical_acute_malnutrition': Property(Types.CATEGORICAL, 'temporary property',
                                                   categories=['MAM', 'SAM', 'well']),
        'referral_from_hsa_hc': Property(Types.BOOL, 'temporary property'),
    }

    def __init__(self, name=None):
        super().__init__(name)

    def read_parameters(self, data_folder):
        pass

    def initialise_population(self, population):
        df = population.props
        df.loc[df.is_alive, 'hv_inf'] = False
        df.loc[df.is_alive, 'nb_low_birth_weight_status'] = 'normal_birth_weight'
        df.loc[df.is_alive, 'nb_breastfeeding_status'] = 'non_exclusive'
        df.loc[df.is_alive, 'va_pneumo_all_doses'] = False
        df.loc[df.is_alive, 'va_hib_all_doses'] = False
        df.loc[df.is_alive, 'va_measles_all_doses'] = False
        df.loc[df.is_alive, 'un_clinical_acute_malnutrition'] = 'well'
        df.loc[df.is_alive, 'referral_from_hsa_hc'] = False

    def initialise_simulation(self, sim):
        pass

    def on_birth(self, mother, child):
        df = self.sim.population.props
        df.at[child, 'hv_inf'] = False
        df.at[child, 'nb_low_birth_weight_status'] = 'normal_birth_weight'
        df.at[child, 'nb_breastfeeding_status'] = 'non_exclusive'
        df.at[child, 'va_pneumo_all_doses'] = False
        df.at[child, 'va_hib_all_doses'] = False
        df.at[child, 'va_hib_all_doses'] = False
        df.at[child, 'va_measles_all_doses'] = False
        df.at[child, 'un_clinical_acute_malnutrition'] = 'well'
        df.at[child, 'referral_from_hsa_hc'] = False

