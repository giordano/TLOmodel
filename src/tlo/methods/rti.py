"""
A skeleton template for disease methods.

"""
from pathlib import Path
import pandas as pd
import numpy as np
from tlo import DateOffset, Module, Parameter, Property, Types, logging
from tlo.events import IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent, Event
from tlo.methods import demography, Metadata
from tlo.methods.healthsystem import HSI_Event
from tlo.lm import LinearModel, LinearModelType, Predictor
from tlo.methods.hsi_generic_first_appts import HSI_GenericEmergencyFirstApptAtFacilityLevel1
from tlo.methods.symptommanager import Symptom

# ---------------------------------------------------------------------------------------------------------
#   MODULE DEFINITIONS
# ---------------------------------------------------------------------------------------------------------

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class RTI(Module):
    """
    RTI module for the TLO model
    """

    # Module parameters
    PARAMETERS = {
        # Transitioning parameters
        'base_rate_injrti': Parameter(
            Types.REAL,
            'Base rate of RTI per year',
        ),
        'rr_injrti_age018': Parameter(
            Types.REAL,
            'risk ratio of RTI in age 0-18 compared to base rate of RTI'
        ),
        'rr_injrti_age1829': Parameter(
            Types.REAL,
            'risk ratio of RTI in age 18-29 compared to base rate of RTI',
        ),
        'rr_injrti_age3039': Parameter(
            Types.REAL,
            'risk ratio of RTI in age 30-39 compared to base rate of RTI',
        ),
        'rr_injrti_age4049': Parameter(
            Types.REAL,
            'risk ratio of RTI in age 40-49 compared to base rate of RTI',
        ),
        'rr_injrti_age5059': Parameter(
            Types.REAL,
            'risk ratio of RTI in age 50-59 compared to base rate of RTI',
        ),
        'rr_injrti_age6069': Parameter(
            Types.REAL,
            'risk ratio of RTI in age 60-69 compared to base rate of RTI',
        ),
        'rr_injrti_age7079': Parameter(
            Types.REAL,
            'risk ratio of RTI in age 70-79 compared to base rate of RTI',
        ),
        'rr_injrti_male': Parameter(
            Types.REAL,
            'risk ratio of RTI when male compared to females',
        ),
        'rr_injrti_excessalcohol': Parameter(
            Types.REAL,
            'risk ratio of RTI in those that consume excess alcohol compared to those who do not'
        ),
        'imm_death_proportion_rti': Parameter(
            Types.REAL,
            'Proportion of those involved in an RTI that die at site of accident or die before seeking medical '
            'intervention'
        ),
        'prob_death_with_med_mild': Parameter(
            Types.REAL,
            'Proportion of people who pass away in the following month after medical treatment for injuries with an ISS'
            'score less than or equal to 15'
        ),
        'prob_death_with_med_severe': Parameter(
            Types.REAL,
            'Proportion of people who pass away in the following month after medical treatment for injuries with an ISS'
            'score of 15 '
        ),
        'prop_death_no_med_ISS_<=_15': Parameter(
            Types.REAL,
            'Proportion of people who pass away in the following month with no treatment for injuries with an ISS'
            'score less than or equal to 15'
        ),
        'prop_death_no_med_ISS_>15': Parameter(
            Types.REAL,
            'Proportion of people who pass away in the following month with no treatment for injuries with an ISS'
            'score of 15 '
        ),
        'prob_perm_disability_with_treatment_severe_TBI': Parameter(
            Types.REAL,
            'probability that someone with a treated severe TBI is permanently disabled'
        ),
        'prob_perm_disability_with_treatment_sci': Parameter(
            Types.REAL,
            'probability that someone with a treated spinal cord injury is permanently disabled'
        ),
        'prob_death_TBI_SCI_no_treatment': Parameter(
            Types.REAL,
            'probability that someone with a spinal cord injury will die without treatment'
        ),
        'prop_death_burns_no_treatment': Parameter(
            Types.REAL,
            'probability that someone with a burn injury will die without treatment'
        ),
        'prob_TBI_require_craniotomy': Parameter(
            Types.REAL,
            'probability that someone with a traumatic brain injury will require a craniotomy surgery'
        ),
        'prob_exploratory_laparotomy': Parameter(
            Types.REAL,
            'probability that someone with an internal organ injury will require a exploratory_laparotomy'
        ),
        'prob_death_fractures_no_treatment': Parameter(
            Types.REAL,
            'probability that someone with a fracture injury will die without treatment'
        ),
        'prob_depressed_skull_fracture': Parameter(
            Types.REAL,
            'Probability that a skull fracture will be depressed and therefore require surgery'
        ),
        'prob_mild_burns': Parameter(
            Types.REAL,
            'Probability that a burn within a region will result in < 10% total body surface area'
        ),
        'prob_dislocation_requires_surgery': Parameter(
            Types.REAL,
            'Probability that a dislocation will require surgery to relocate the joint.'
        ),
        'rr_injrti_mortality_polytrauma': Parameter(
            Types.REAL,
            'Relative risk of mortality for those with polytrauma'
        ),
        'number_of_injured_body_regions_distribution': Parameter(
            Types.LIST,
            'The distribution of number of injured AIS body regions, used to decide how many injuries a person has'
        ),
        'injury_location_distribution': Parameter(
            Types.LIST,
            'The distribution of where injuries are located in the body, based on the AIS body region definition'
        ),
        'head_prob_skin_wound': Parameter(
            Types.REAL,
            'Proportion of head wounds that result in a skin wound'
        ),
        'head_prob_skin_wound_open': Parameter(
            Types.REAL,
            'Proportion of skin wounds in the head that result in an open wound'
        ),
        'head_prob_skin_wound_burn': Parameter(
            Types.REAL,
            'Proportion of skin wounds in the head that result in an open wound'
        ),
        'head_prob_fracture': Parameter(
            Types.REAL,
            'Proportion of head wounds that result in a fractured skull'
        ),
        'head_prob_fracture_unspecified': Parameter(
            Types.REAL,
            'Proportion of skull fractures in an unspecified location in the skull, carrying a lower AIS score'
        ),
        'head_prob_fracture_basilar': Parameter(
            Types.REAL,
            'Proportion of skull fractures in the base of the skull, carrying a higher AIS score'
        ),
        'head_prob_TBI': Parameter(
            Types.REAL,
            'Proportion of head injuries that result in traumatic brain injury'
        ),
        'head_prob_TBI_AIS3': Parameter(
            Types.REAL,
            'Proportion of traumatic brain injuries with an AIS score of 3'
        ),
        'head_prob_TBI_AIS4': Parameter(
            Types.REAL,
            'Proportion of traumatic brain injuries with an AIS score of 4'
        ),
        'head_prob_TBI_AIS5': Parameter(
            Types.REAL,
            'Proportion of traumatic brain injuries with an AIS score of 3'
        ),
        'face_prob_skin_wound': Parameter(
            Types.REAL,
            'Proportion of facial wounds that result in a skin wound'
        ),
        'face_prob_skin_wound_open': Parameter(
            Types.REAL,
            'Proportion of skin wounds in the face that result in an open wound'
        ),
        'face_prob_skin_wound_burn': Parameter(
            Types.REAL,
            'Proportion of skin wounds in the face that result in an open wound'
        ),
        'face_prob_fracture': Parameter(
            Types.REAL,
            'Proportion of facial wounds that result in a fractured skull'
        ),
        'face_prob_fracture_AIS1': Parameter(
            Types.REAL,
            'Proportion of facial fractures with an AIS score of 1'
        ),
        'face_prob_fracture_AIS2': Parameter(
            Types.REAL,
            'Proportion of facial fractures with an AIS score of 2'
        ),
        'face_prob_soft_tissue_injury': Parameter(
            Types.REAL,
            'Proportion of facial injuries that result in soft tissue injury'
        ),
        'face_prob_eye_injury': Parameter(
            Types.REAL,
            'Proportion of facial injuries that result in eye injury'
        ),
        'neck_prob_skin_wound': Parameter(
            Types.REAL,
            'Proportion of neck injuries that result in skin wounds'
        ),
        'neck_prob_skin_wound_open': Parameter(
            Types.REAL,
            'Proportion of skin wounds in the neck that are open wounds'
        ),
        'neck_prob_skin_wound_burn': Parameter(
            Types.REAL,
            'Proportion of skin wounds in the neck that are burns'
        ),
        'neck_prob_soft_tissue_injury': Parameter(
            Types.REAL,
            'Proportion of neck injuries that result in soft tissue injury'
        ),
        'neck_prob_soft_tissue_injury_AIS2': Parameter(
            Types.REAL,
            'Proportion of soft tissue injuries with an AIS score of 2'
        ),
        'neck_prob_soft_tissue_injury_AIS3': Parameter(
            Types.REAL,
            'Proportion of soft tissue injuries with an AIS score of 3'
        ),
        'neck_prob_internal_bleeding': Parameter(
            Types.REAL,
            'Proportion of neck injuries that result in internal bleeding'
        ),
        'neck_prob_internal_bleeding_AIS1': Parameter(
            Types.REAL,
            'Proportion of internal bleeding in the neck with an AIS score of 1'
        ),
        'neck_prob_internal_bleeding_AIS3': Parameter(
            Types.REAL,
            'Proportion of internal bleeding in the neck with an AIS score of 3'
        ),
        'neck_prob_dislocation': Parameter(
            Types.REAL,
            'Proportion of neck injuries that result in a dislocated neck vertebrae'
        ),
        'neck_prob_dislocation_AIS2': Parameter(
            Types.REAL,
            'Proportion dislocated neck vertebrae with an AIS score of 2'
        ),
        'neck_prob_dislocation_AIS3': Parameter(
            Types.REAL,
            'Proportion dislocated neck vertebrae with an AIS score of 3'
        ),
        'thorax_prob_skin_wound': Parameter(
            Types.REAL,
            'Proportion of thorax injuries that result in a skin wound'
        ),
        'thorax_prob_skin_wound_open': Parameter(
            Types.REAL,
            'Proportion of thorax skin wounds that are open wounds'
        ),
        'thorax_prob_skin_wound_burn': Parameter(
            Types.REAL,
            'Proportion of thorax skin wounds that are burns'
        ),
        'thorax_prob_internal_bleeding': Parameter(
            Types.REAL,
            'Proportion of thorax injuries that result in internal bleeding'
        ),
        'thorax_prob_internal_bleeding_AIS1': Parameter(
            Types.REAL,
            'Proportion of internal bleeding in thorax with AIS score of 1'
        ),
        'thorax_prob_internal_bleeding_AIS3': Parameter(
            Types.REAL,
            'Proportion of internal bleeding in thorax with AIS score of 3'
        ),
        'thorax_prob_internal_organ_injury': Parameter(
            Types.REAL,
            'Proportion of thorax injuries that result in internal organ injuries'
        ),
        'thorax_prob_fracture': Parameter(
            Types.REAL,
            'Proportion of thorax injuries that result in rib fractures/ flail chest'
        ),
        'thorax_prob_fracture_ribs': Parameter(
            Types.REAL,
            'Proportion of rib fractures in  thorax fractures'
        ),
        'thorax_prob_fracture_flail_chest': Parameter(
            Types.REAL,
            'Proportion of flail chest in thorax fractures'
        ),
        'thorax_prob_soft_tissue_injury': Parameter(
            Types.REAL,
            'Proportion of thorax injuries resulting in soft tissue injury'
        ),
        'thorax_prob_soft_tissue_injury_AIS1': Parameter(
            Types.REAL,
            'Proportion of soft tissue injuries in the thorax with an AIS score of 1'
        ),
        'thorax_prob_soft_tissue_injury_AIS2': Parameter(
            Types.REAL,
            'Proportion of soft tissue injuries in the thorax with an AIS score of 2'
        ),
        'thorax_prob_soft_tissue_injury_AIS3': Parameter(
            Types.REAL,
            'Proportion of soft tissue injuries in the thorax with an AIS score of 3'
        ),
        'abdomen_prob_skin_wound': Parameter(
            Types.REAL,
            'Proportion of abdomen injuries that are skin wounds'
        ),
        'abdomen_prob_skin_wound_open': Parameter(
            Types.REAL,
            'Proportion skin wounds to the abdomen that are open wounds'
        ),
        'abdomen_prob_skin_wound_burn': Parameter(
            Types.REAL,
            'Proportion skin wounds to the abdomen that are burns'
        ),
        'abdomen_prob_internal_organ_injury': Parameter(
            Types.REAL,
            'Proportion of abdomen injuries that result in internal organ injury'
        ),
        'abdomen_prob_internal_organ_injury_AIS2': Parameter(
            Types.REAL,
            'Proportion of abdomen injuries that result in internal organ injury with an AIS score of 2'
        ),
        'abdomen_prob_internal_organ_injury_AIS3': Parameter(
            Types.REAL,
            'Proportion of abdomen injuries that result in internal organ injury with an AIS score of 2'
        ),
        'abdomen_prob_internal_organ_injury_AIS4': Parameter(
            Types.REAL,
            'Proportion of abdomen injuries that result in internal organ injury with an AIS score of 2'
        ),
        'spine_prob_spinal_cord_lesion': Parameter(
            Types.REAL,
            'Proportion of injuries to spine that result in spinal cord lesions'
        ),
        'spine_prob_spinal_cord_lesion_neck_level': Parameter(
            Types.REAL,
            'Proportion of spinal cord lesions that happen at neck level'
        ),
        'spine_prob_spinal_cord_lesion_neck_level_AIS3': Parameter(
            Types.REAL,
            'Proportion of spinal cord lesions that happen at neck level with an AIS score of 3'
        ),
        'spine_prob_spinal_cord_lesion_neck_level_AIS4': Parameter(
            Types.REAL,
            'Proportion of spinal cord lesions that happen at neck level with an AIS score of 4'
        ),
        'spine_prob_spinal_cord_lesion_neck_level_AIS5': Parameter(
            Types.REAL,
            'Proportion of spinal cord lesions that happen at neck level with an AIS score of 5'
        ),
        'spine_prob_spinal_cord_lesion_neck_level_AIS6': Parameter(
            Types.REAL,
            'Proportion of spinal cord lesions that happen at neck level with an AIS score of 6'
        ),
        'spine_prob_spinal_cord_lesion_below_neck_level': Parameter(
            Types.REAL,
            'Proportion of spinal cord lesions that happen below neck level'
        ),
        'spine_prob_spinal_cord_lesion_below_neck_level_AIS3': Parameter(
            Types.REAL,
            'Proportion of spinal cord lesions that happen below neck level with an AIS score of 3'
        ),
        'spine_prob_spinal_cord_lesion_below_neck_level_AIS4': Parameter(
            Types.REAL,
            'Proportion of spinal cord lesions that happen below neck level with an AIS score of 4'
        ),
        'spine_prob_spinal_cord_lesion_below_neck_level_AIS5': Parameter(
            Types.REAL,
            'Proportion of spinal cord lesions that happen below neck level with an AIS score of 5'
        ),
        'spine_prob_fracture': Parameter(
            Types.REAL,
            'Proportion of spinal injuries that result in vertebrae fractures'
        ),
        'upper_ex_prob_skin_wound': Parameter(
            Types.REAL,
            'Proportion of upper extremity injuries that result in skin wounds'
        ),
        'upper_ex_prob_skin_wound_open': Parameter(
            Types.REAL,
            'Proportion of upper extremity injuries that result in open wounds'
        ),
        'upper_ex_prob_skin_wound_burn': Parameter(
            Types.REAL,
            'Proportion of upper extremity injuries that result in burns'
        ),
        'upper_ex_prob_fracture': Parameter(
            Types.REAL,
            'Proportion of upper extremity injuries that result in fractures'
        ),
        'upper_ex_prob_dislocation': Parameter(
            Types.REAL,
            'Proportion of upper extremity injuries that result in dislocation'
        ),
        'upper_ex_prob_amputation': Parameter(
            Types.REAL,
            'Proportion of upper extremity injuries that result in amputation'
        ),
        'upper_ex_prob_amputation_AIS2': Parameter(
            Types.REAL,
            'Proportion of upper extremity injuries that result in amputation with AIS 2'
        ),
        'upper_ex_prob_amputation_AIS3': Parameter(
            Types.REAL,
            'Proportion of upper extremity injuries that result in amputation with AIS 3'
        ),
        'lower_ex_prob_skin_wound': Parameter(
            Types.REAL,
            'Proportion of lower extremity injuries that result in skin wounds'
        ),
        'lower_ex_prob_skin_wound_open': Parameter(
            Types.REAL,
            'Proportion of lower extremity injuries that result in open wounds'
        ),
        'lower_ex_prob_skin_wound_burn': Parameter(
            Types.REAL,
            'Proportion of lower extremity injuries that result in burns'
        ),
        'lower_ex_prob_fracture': Parameter(
            Types.REAL,
            'Proportion of lower extremity injuries that result in fractures'
        ),
        'lower_ex_prob_fracture_AIS1': Parameter(
            Types.REAL,
            'Proportion of lower extremity injuries that result in fractures with an AIS of 1'
        ),
        'lower_ex_prob_fracture_AIS2': Parameter(
            Types.REAL,
            'Proportion of lower extremity injuries that result in fractures with an AIS of 2'
        ),
        'lower_ex_prob_fracture_AIS3': Parameter(
            Types.REAL,
            'Proportion of lower extremity injuries that result in fractures with an AIS of 3'
        ),
        'lower_ex_prob_dislocation': Parameter(
            Types.REAL,
            'Proportion of lower extremity injuries that result in dislocation'
        ),
        'lower_ex_prob_amputation': Parameter(
            Types.REAL,
            'Proportion of lower extremity injuries that result in amputation'
        ),
        'lower_ex_prob_amputation_AIS2': Parameter(
            Types.REAL,
            'Proportion of lower extremity injuries that result in amputation with AIS 2'
        ),
        'lower_ex_prob_amputation_AIS3': Parameter(
            Types.REAL,
            'Proportion of lower extremity injuries that result in amputation with AIS 3'
        ),
        'lower_ex_prob_amputation_AIS4': Parameter(
            Types.REAL,
            'Proportion of lower extremity injuries that result in amputation with AIS 4'
        ),
        'allowed_interventions': Parameter(
            Types.LIST, 'list of interventions allowed to run, used in analysis'),
        # Length of stay
        'mean_los_ISS_less_than_4': Parameter(
            Types.REAL,
            'Mean length of stay for someone with an ISS score < 4'
        ),
        'sd_los_ISS_less_than_4': Parameter(
            Types.REAL,
            'Standard deviation in length of stay for someone with an ISS score < 4'
        ),
        'mean_los_ISS_4_to_8': Parameter(
            Types.REAL,
            'Mean length of stay for someone with an ISS score between 4 and 8'
        ),
        'sd_los_ISS_4_to_8': Parameter(
            Types.REAL,
            'Standard deviation in length of stay for someone with an ISS score between 4 and 8'
        ),
        'mean_los_ISS_9_to_15': Parameter(
            Types.REAL,
            'Mean length of stay for someone with an ISS score between 9 and 15'
        ),
        'sd_los_ISS_9_to_15': Parameter(
            Types.REAL,
            'Standard deviation in length of stay for someone with an ISS score between 9 and 15'
        ),
        'mean_los_ISS_16_to_24': Parameter(
            Types.REAL,
            'Mean length of stay for someone with an ISS score between 16 and 24'
        ),
        'sd_los_ISS_16_to_24': Parameter(
            Types.REAL,
            'Standard deviation in length of stay for someone with an ISS score between 16 and 24'
        ),
        'mean_los_ISS_more_than_25': Parameter(
            Types.REAL,
            'Mean length of stay for someone with an ISS score between 16 and 24'
        ),
        'sd_los_ISS_more_that_25': Parameter(
            Types.REAL,
            'Standard deviation in length of stay for someone with an ISS score between 16 and 24'
        ),
        # DALY weights
        'daly_wt_unspecified_skull_fracture': Parameter(
            Types.REAL,
            'daly_wt_unspecified_skull_fracture - code 1674'
        ),
        'daly_wt_basilar_skull_fracture': Parameter(
            Types.REAL,
            'daly_wt_basilar_skull_fracture - code 1675'
        ),
        'daly_wt_epidural_hematoma': Parameter(
            Types.REAL,
            'daly_wt_epidural_hematoma - code 1676'
        ),
        'daly_wt_subdural_hematoma': Parameter(
            Types.REAL,
            'daly_wt_subdural_hematoma - code 1677'
        ),
        'daly_wt_subarachnoid_hematoma': Parameter(
            Types.REAL,
            'daly_wt_subarachnoid_hematoma - code 1678'
        ),
        'daly_wt_brain_contusion': Parameter(
            Types.REAL,
            'daly_wt_brain_contusion - code 1679'
        ),
        'daly_wt_intraventricular_haemorrhage': Parameter(
            Types.REAL,
            'daly_wt_intraventricular_haemorrhage - code 1680'
        ),
        'daly_wt_diffuse_axonal_injury': Parameter(
            Types.REAL,
            'daly_wt_diffuse_axonal_injury - code 1681'
        ),
        'daly_wt_subgaleal_hematoma': Parameter(
            Types.REAL,
            'daly_wt_subgaleal_hematoma - code 1682'
        ),
        'daly_wt_midline_shift': Parameter(
            Types.REAL,
            'daly_wt_midline_shift - code 1683'
        ),
        'daly_wt_facial_fracture': Parameter(
            Types.REAL,
            'daly_wt_facial_fracture - code 1684'
        ),
        'daly_wt_facial_soft_tissue_injury': Parameter(
            Types.REAL,
            'daly_wt_facial_soft_tissue_injury - code 1685'
        ),
        'daly_wt_eye_injury': Parameter(
            Types.REAL,
            'daly_wt_eye_injury - code 1686'
        ),
        'daly_wt_neck_soft_tissue_injury': Parameter(
            Types.REAL,
            'daly_wt_neck_soft_tissue_injury - code 1687'
        ),
        'daly_wt_neck_internal_bleeding': Parameter(
            Types.REAL,
            'daly_wt_neck_internal_bleeding - code 1688'
        ),
        'daly_wt_neck_dislocation': Parameter(
            Types.REAL,
            'daly_wt_neck_dislocation - code 1689'
        ),
        'daly_wt_chest_wall_bruises_hematoma': Parameter(
            Types.REAL,
            'daly_wt_chest_wall_bruises_hematoma - code 1690'
        ),
        'daly_wt_hemothorax': Parameter(
            Types.REAL,
            'daly_wt_hemothorax - code 1691'
        ),
        'daly_wt_lung_contusion': Parameter(
            Types.REAL,
            'daly_wt_lung_contusion - code 1692'
        ),
        'daly_wt_diaphragm_rupture': Parameter(
            Types.REAL,
            'daly_wt_diaphragm_rupture - code 1693'
        ),
        'daly_wt_rib_fracture': Parameter(
            Types.REAL,
            'daly_wt_rib_fracture - code 1694'
        ),
        'daly_wt_flail_chest': Parameter(
            Types.REAL,
            'daly_wt_flail_chest - code 1695'
        ),
        'daly_wt_chest_wall_laceration': Parameter(
            Types.REAL,
            'daly_wt_chest_wall_laceration - code 1696'
        ),
        'daly_wt_closed_pneumothorax': Parameter(
            Types.REAL,
            'daly_wt_closed_pneumothorax - code 1697'
        ),
        'daly_wt_open_pneumothorax': Parameter(
            Types.REAL,
            'daly_wt_open_pneumothorax - code 1698'
        ),
        'daly_wt_surgical_emphysema': Parameter(
            Types.REAL,
            'daly_wt_surgical_emphysema aka subcuteal emphysema - code 1699'
        ),
        'daly_wt_abd_internal_organ_injury': Parameter(
            Types.REAL,
            'daly_wt_abd_internal_organ_injury - code 1700'
        ),
        'daly_wt_spinal_cord_lesion_neck_with_treatment': Parameter(
            Types.REAL,
            'daly_wt_spinal_cord_lesion_neck_with_treatment - code 1701'
        ),
        'daly_wt_spinal_cord_lesion_neck_without_treatment': Parameter(
            Types.REAL,
            'daly_wt_spinal_cord_lesion_neck_without_treatment - code 1702'
        ),
        'daly_wt_spinal_cord_lesion_below_neck_with_treatment': Parameter(
            Types.REAL,
            'daly_wt_spinal_cord_lesion_below_neck_with_treatment - code 1703'
        ),
        'daly_wt_spinal_cord_lesion_below_neck_without_treatment': Parameter(
            Types.REAL,
            'daly_wt_spinal_cord_lesion_below_neck_without_treatment - code 1704'
        ),
        'daly_wt_vertebrae_fracture': Parameter(
            Types.REAL,
            'daly_wt_vertebrae_fracture - code 1705'
        ),
        'daly_wt_clavicle_scapula_humerus_fracture': Parameter(
            Types.REAL,
            'daly_wt_clavicle_scapula_humerus_fracture - code 1706'
        ),
        'daly_wt_hand_wrist_fracture_with_treatment': Parameter(
            Types.REAL,
            'daly_wt_hand_wrist_fracture_with_treatment - code 1707'
        ),
        'daly_wt_hand_wrist_fracture_without_treatment': Parameter(
            Types.REAL,
            'daly_wt_hand_wrist_fracture_without_treatment - code 1708'
        ),
        'daly_wt_radius_ulna_fracture_short_term_with_without_treatment': Parameter(
            Types.REAL,
            'daly_wt_radius_ulna_fracture_short_term_with_without_treatment - code 1709'
        ),
        'daly_wt_radius_ulna_fracture_long_term_without_treatment': Parameter(
            Types.REAL,
            'daly_wt_radius_ulna_fracture_long_term_without_treatment - code 1710'
        ),
        'daly_wt_dislocated_shoulder': Parameter(
            Types.REAL,
            'daly_wt_dislocated_shoulder - code 1711'
        ),
        'daly_wt_amputated_finger': Parameter(
            Types.REAL,
            'daly_wt_amputated_finger - code 1712'
        ),
        'daly_wt_amputated_thumb': Parameter(
            Types.REAL,
            'daly_wt_amputated_thumb - code 1713'
        ),
        'daly_wt_unilateral_arm_amputation_with_treatment': Parameter(
            Types.REAL,
            'daly_wt_unilateral_arm_amputation_with_treatment - code 1714'
        ),
        'daly_wt_unilateral_arm_amputation_without_treatment': Parameter(
            Types.REAL,
            'daly_wt_unilateral_arm_amputation_without_treatment - code 1715'
        ),
        'daly_wt_bilateral_arm_amputation_with_treatment': Parameter(
            Types.REAL,
            'daly_wt_bilateral_arm_amputation_with_treatment - code 1716'
        ),
        'daly_wt_bilateral_arm_amputation_without_treatment': Parameter(
            Types.REAL,
            'daly_wt_bilateral_arm_amputation_without_treatment - code 1717'
        ),
        'daly_wt_foot_fracture_short_term_with_without_treatment': Parameter(
            Types.REAL,
            'daly_wt_foot_fracture_short_term_with_without_treatment - code 1718'
        ),
        'daly_wt_foot_fracture_long_term_without_treatment': Parameter(
            Types.REAL,
            'daly_wt_foot_fracture_long_term_without_treatment - code 1719'
        ),
        'daly_wt_patella_tibia_fibula_fracture_with_treatment': Parameter(
            Types.REAL,
            'daly_wt_patella_tibia_fibula_fracture_with_treatment - code 1720'
        ),
        'daly_wt_patella_tibia_fibula_fracture_without_treatment': Parameter(
            Types.REAL,
            'daly_wt_patella_tibia_fibula_fracture_without_treatment - code 1721'
        ),
        'daly_wt_hip_fracture_short_term_with_without_treatment': Parameter(
            Types.REAL,
            'daly_wt_hip_fracture_short_term_with_without_treatment - code 1722'
        ),
        'daly_wt_hip_fracture_long_term_with_treatment': Parameter(
            Types.REAL,
            'daly_wt_hip_fracture_long_term_with_treatment - code 1723'
        ),
        'daly_wt_hip_fracture_long_term_without_treatment': Parameter(
            Types.REAL,
            'daly_wt_hip_fracture_long_term_without_treatment - code 1724'
        ),
        'daly_wt_pelvis_fracture_short_term': Parameter(
            Types.REAL,
            'daly_wt_pelvis_fracture_short_term - code 1725'
        ),
        'daly_wt_pelvis_fracture_long_term': Parameter(
            Types.REAL,
            'daly_wt_pelvis_fracture_long_term - code 1726'
        ),
        'daly_wt_femur_fracture_short_term': Parameter(
            Types.REAL,
            'daly_wt_femur_fracture_short_term - code 1727'
        ),
        'daly_wt_femur_fracture_long_term_without_treatment': Parameter(
            Types.REAL,
            'daly_wt_femur_fracture_long_term_without_treatment - code 1728'
        ),
        'daly_wt_dislocated_hip': Parameter(
            Types.REAL,
            'daly_wt_dislocated_hip - code 1729'
        ),
        'daly_wt_dislocated_knee': Parameter(
            Types.REAL,
            'daly_wt_dislocated_knee - code 1730'
        ),
        'daly_wt_amputated_toes': Parameter(
            Types.REAL,
            'daly_wt_amputated_toes - code 1731'
        ),
        'daly_wt_unilateral_lower_limb_amputation_with_treatment': Parameter(
            Types.REAL,
            'daly_wt_unilateral_lower_limb_amputation_with_treatment - code 1732'
        ),
        'daly_wt_unilateral_lower_limb_amputation_without_treatment': Parameter(
            Types.REAL,
            'daly_wt_unilateral_lower_limb_amputation_without_treatment - code 1733'
        ),
        'daly_wt_bilateral_lower_limb_amputation_with_treatment': Parameter(
            Types.REAL,
            'daly_wt_bilateral_lower_limb_amputation_with_treatment - code 1734'
        ),
        'daly_wt_bilateral_lower_limb_amputation_without_treatment': Parameter(
            Types.REAL,
            'daly_wt_bilateral_lower_limb_amputation_without_treatment - code 1735'
        ),
        'daly_dist_code_133': Parameter(
            Types.LIST,
            'Mapping parameter of injury code 133 to the various injuries associated with the code'
        ),
        'daly_dist_code_134': Parameter(
            Types.LIST,
            'Mapping parameter of injury code 134 to the various injuries associated with the code'
        ),
        'daly_dist_code_453': Parameter(
            Types.LIST,
            'Mapping parameter of injury code 453 to the various injuries associated with the code'
        ),
        'daly_dist_code_673': Parameter(
            Types.LIST,
            'Mapping parameter of injury code 673 to the various injuries associated with the code'
        ),
        'daly_dist_codes_674_675': Parameter(
            Types.LIST,
            'Mapping parameter of injury code 674/675 to the various injuries associated with the codes'
        ),
        'daly_dist_code_712': Parameter(
            Types.LIST,
            'Mapping parameter of injury code 712 to the various injuries associated with the code'
        ),
        'daly_dist_code_782': Parameter(
            Types.LIST,
            'Mapping parameter of injury code 782 to the various injuries associated with the code'
        ),
        'daly_dist_code_813': Parameter(
            Types.LIST,
            'Mapping parameter of injury code 813 to the various injuries associated with the code'
        ),
        'daly_dist_code_822': Parameter(
            Types.LIST,
            'Mapping parameter of injury code 822 to the various injuries associated with the code'
        ),
    }

    PROPERTIES = {
        'rt_road_traffic_inc': Property(Types.BOOL, 'involved in a road traffic injury'),
        'rt_inj_severity': Property(Types.CATEGORICAL,
                                    'Injury status relating to road traffic injury: none, mild, moderate, severe',
                                    categories=['none', 'mild', 'severe'],
                                    ),
        'rt_injury_1': Property(Types.CATEGORICAL, 'Codes for injury 1 from RTI',
                                categories=['none', '112', '113', '133', '133a', '133b', '133c', '133d', '134', '134a',
                                            '134b', '135', '1101', '1114', '211', '212', '241', '2101', '2114', '291',
                                            '342', '343', '361', '363', '322', '323', '3101', '3113', '412', '414',
                                            '461', '463', '453', '453a', '453b', '441', '442', '443', '4101', '4113',
                                            '552', '553', '554', '5101', '5113', '612', '673', '673a', '673b', '674',
                                            '674a', '674b', '675', '675a', '675b', '676', '712', '712a', '712b', '712c',
                                            '722', '782',
                                            '782a', '782b', '782c', '783', '7101', '7113', '811', '812',
                                            '813', '813a', '813b', '813c', '822', '822a', '822b', '882', '883', '884',
                                            '8101', '8113', 'P133', 'P133a', 'P133b', 'P133c', 'P133d', 'P134', 'P134a',
                                            'P134b', 'P135', 'P673', 'P673a', 'P673b', 'P674', 'P674a', 'P674b', 'P675',
                                            'P675a', 'P675b', 'P676', 'P782a', 'P782b', 'P782c', 'P783', 'P882', 'P883',
                                            'P884']),
        'rt_injury_2': Property(Types.CATEGORICAL, 'Codes for injury 2 from RTI',
                                categories=['none', '112', '113', '133', '133a', '133b', '133c', '133d', '134', '134a',
                                            '134b', '135', '1101', '1114', '211', '212', '241', '2101', '2114', '291',
                                            '342', '343', '361', '363', '322', '323', '3101', '3113', '412', '414',
                                            '461', '463', '453', '453a', '453b', '441', '442', '443', '4101', '4113',
                                            '552', '553', '554', '5101', '5113', '612', '673', '673a', '673b', '674',
                                            '674a', '674b', '675', '675a', '675b', '676', '712', '712a', '712b', '712c',
                                            '722', '782',
                                            '782a', '782b', '782c', '783', '7101', '7113', '811', '812',
                                            '813', '813a', '813b', '813c', '822', '822a', '822b', '882', '883', '884',
                                            '8101', '8113', 'P133', 'P133a', 'P133b', 'P133c', 'P133d', 'P134', 'P134a',
                                            'P134b', 'P135', 'P673', 'P673a', 'P673b', 'P674', 'P674a', 'P674b', 'P675',
                                            'P675a', 'P675b', 'P676', 'P782a', 'P782b', 'P782c', 'P783', 'P882', 'P883',
                                            'P884']),
        'rt_injury_3': Property(Types.CATEGORICAL, 'Codes for injury 3 from RTI',
                                categories=['none', '112', '113', '133', '133a', '133b', '133c', '133d', '134', '134a',
                                            '134b', '135', '1101', '1114', '211', '212', '241', '2101', '2114', '291',
                                            '342', '343', '361', '363', '322', '323', '3101', '3113', '412', '414',
                                            '461', '463', '453', '453a', '453b', '441', '442', '443', '4101', '4113',
                                            '552', '553', '554', '5101', '5113', '612', '673', '673a', '673b', '674',
                                            '674a', '674b', '675', '675a', '675b', '676', '712', '712a', '712b', '712c',
                                            '722', '782',
                                            '782a', '782b', '782c', '783', '7101', '7113', '811', '812',
                                            '813', '813a', '813b', '813c', '822', '822a', '822b', '882', '883', '884',
                                            '8101', '8113', 'P133', 'P133a', 'P133b', 'P133c', 'P133d', 'P134', 'P134a',
                                            'P134b', 'P135', 'P673', 'P673a', 'P673b', 'P674', 'P674a', 'P674b', 'P675',
                                            'P675a', 'P675b', 'P676', 'P782a', 'P782b', 'P782c', 'P783', 'P882', 'P883',
                                            'P884']),
        'rt_injury_4': Property(Types.CATEGORICAL, 'Codes for injury 4 from RTI',
                                categories=['none', '112', '113', '133', '133a', '133b', '133c', '133d', '134', '134a',
                                            '134b', '135', '1101', '1114', '211', '212', '241', '2101', '2114', '291',
                                            '342', '343', '361', '363', '322', '323', '3101', '3113', '412', '414',
                                            '461', '463', '453', '453a', '453b', '441', '442', '443', '4101', '4113',
                                            '552', '553', '554', '5101', '5113', '612', '673', '673a', '673b', '674',
                                            '674a', '674b', '675', '675a', '675b', '676', '712', '712a', '712b', '712c',
                                            '722', '782',
                                            '782a', '782b', '782c', '783', '7101', '7113', '811', '812',
                                            '813', '813a', '813b', '813c', '822', '822a', '822b', '882', '883', '884',
                                            '8101', '8113', 'P133', 'P133a', 'P133b', 'P133c', 'P133d', 'P134', 'P134a',
                                            'P134b', 'P135', 'P673', 'P673a', 'P673b', 'P674', 'P674a', 'P674b', 'P675',
                                            'P675a', 'P675b', 'P676', 'P782a', 'P782b', 'P782c', 'P783', 'P882', 'P883',
                                            'P884']),
        'rt_injury_5': Property(Types.CATEGORICAL, 'Codes for injury 5 from RTI',
                                categories=['none', '112', '113', '133', '133a', '133b', '133c', '133d', '134', '134a',
                                            '134b', '135', '1101', '1114', '211', '212', '241', '2101', '2114', '291',
                                            '342', '343', '361', '363', '322', '323', '3101', '3113', '412', '414',
                                            '461', '463', '453', '453a', '453b', '441', '442', '443', '4101', '4113',
                                            '552', '553', '554', '5101', '5113', '612', '673', '673a', '673b', '674',
                                            '674a', '674b', '675', '675a', '675b', '676', '712', '712a', '712b', '712c',
                                            '722', '782',
                                            '782a', '782b', '782c', '783', '7101', '7113', '811', '812',
                                            '813', '813a', '813b', '813c', '822', '822a', '822b', '882', '883', '884',
                                            '8101', '8113', 'P133', 'P133a', 'P133b', 'P133c', 'P133d', 'P134', 'P134a',
                                            'P134b', 'P135', 'P673', 'P673a', 'P673b', 'P674', 'P674a', 'P674b', 'P675',
                                            'P675a', 'P675b', 'P676', 'P782a', 'P782b', 'P782c', 'P783', 'P882', 'P883',
                                            'P884']),
        'rt_injury_6': Property(Types.CATEGORICAL, 'Codes for injury 6 from RTI',
                                categories=['none', '112', '113', '133', '133a', '133b', '133c', '133d', '134', '134a',
                                            '134b', '135', '1101', '1114', '211', '212', '241', '2101', '2114', '291',
                                            '342', '343', '361', '363', '322', '323', '3101', '3113', '412', '414',
                                            '461', '463', '453', '453a', '453b', '441', '442', '443', '4101', '4113',
                                            '552', '553', '554', '5101', '5113', '612', '673', '673a', '673b', '674',
                                            '674a', '674b', '675', '675a', '675b', '676', '712', '712a', '712b', '712c',
                                            '722', '782',
                                            '782a', '782b', '782c', '783', '7101', '7113', '811', '812',
                                            '813', '813a', '813b', '813c', '822', '822a', '822b', '882', '883', '884',
                                            '8101', '8113', 'P133', 'P133a', 'P133b', 'P133c', 'P133d', 'P134', 'P134a',
                                            'P134b', 'P135', 'P673', 'P673a', 'P673b', 'P674', 'P674a', 'P674b', 'P675',
                                            'P675a', 'P675b', 'P676', 'P782a', 'P782b', 'P782c', 'P783', 'P882', 'P883',
                                            'P884']),
        'rt_injury_7': Property(Types.CATEGORICAL, 'Codes for injury 7 from RTI',
                                categories=['none', '112', '113', '133', '133a', '133b', '133c', '133d', '134', '134a',
                                            '134b', '135', '1101', '1114', '211', '212', '241', '2101', '2114', '291',
                                            '342', '343', '361', '363', '322', '323', '3101', '3113', '412', '414',
                                            '461', '463', '453', '453a', '453b', '441', '442', '443', '4101', '4113',
                                            '552', '553', '554', '5101', '5113', '612', '673', '673a', '673b', '674',
                                            '674a', '674b', '675', '675a', '675b', '676', '712', '712a', '712b', '712c',
                                            '722', '782',
                                            '782a', '782b', '782c', '783', '7101', '7113', '811', '812',
                                            '813', '813a', '813b', '813c', '822', '822a', '822b', '882', '883', '884',
                                            '8101', '8113', 'P133', 'P133a', 'P133b', 'P133c', 'P133d', 'P134', 'P134a',
                                            'P134b', 'P135', 'P673', 'P673a', 'P673b', 'P674', 'P674a', 'P674b', 'P675',
                                            'P675a', 'P675b', 'P676', 'P782a', 'P782b', 'P782c', 'P783', 'P882', 'P883',
                                            'P884']),
        'rt_injury_8': Property(Types.CATEGORICAL, 'Codes for injury 8 from RTI',
                                categories=['none', '112', '113', '133', '133a', '133b', '133c', '133d', '134', '134a',
                                            '134b', '135', '1101', '1114', '211', '212', '241', '2101', '2114', '291',
                                            '342', '343', '361', '363', '322', '323', '3101', '3113', '412', '414',
                                            '461', '463', '453', '453a', '453b', '441', '442', '443', '4101', '4113',
                                            '552', '553', '554', '5101', '5113', '612', '673', '673a', '673b', '674',
                                            '674a', '674b', '675', '675a', '675b', '676', '712', '712a', '712b', '712c',
                                            '722', '782',
                                            '782a', '782b', '782c', '783', '7101', '7113', '811', '812',
                                            '813', '813a', '813b', '813c', '822', '822a', '822b', '882', '883', '884',
                                            '8101', '8113', 'P133', 'P133a', 'P133b', 'P133c', 'P133d', 'P134', 'P134a',
                                            'P134b', 'P135', 'P673', 'P673a', 'P673b', 'P674', 'P674a', 'P674b', 'P675',
                                            'P675a', 'P675b', 'P676', 'P782a', 'P782b', 'P782c', 'P783', 'P882', 'P883',
                                            'P884']),
        'rt_ISS_score': Property(Types.INT, 'The ISS score associated with the injuries resulting from a road traffic'
                                            'accident'),
        'rt_perm_disability': Property(Types.BOOL, 'whether the injuries from an RTI result in permanent disability'),
        'rt_polytrauma': Property(Types.BOOL, 'polytrauma from RTI'),
        'rt_imm_death': Property(Types.BOOL, 'death at scene True/False'),
        'rt_diagnosed': Property(Types.BOOL, 'Person has had their injuries diagnosed'),
        'rt_date_to_remove_daly': Property(Types.LIST, 'List of dates to remove the daly weight associated with each '
                                                       'injury'),
        'rt_post_med_death': Property(Types.BOOL, 'death in following month despite medical intervention True/False'),
        'rt_no_med_death': Property(Types.BOOL, 'death in following month without medical intervention True/False'),
        'rt_recovery_no_med': Property(Types.BOOL, 'recovery without medical intervention True/False'),
        'rt_disability': Property(Types.REAL, 'disability weight for current month'),
        'rt_date_inj': Property(Types.DATE, 'date of latest injury'),
        'rt_med_int': Property(Types.BOOL, 'whether this person is currently undergoing medical treatment'),
        'rt_MAIS_military_score': Property(Types.INT, 'the maximum AIS-military score, used as a proxy to calculate the'
                                                      'probability of mortality without medical intervention'),
        'rt_date_death_no_med': Property(Types.DATE, 'the date which the person has is scheduled to die without medical'
                                                     'intervention')
    }

    # Declare Metadata
    METADATA = {
        Metadata.DISEASE_MODULE,  # Disease modules: Any disease module should carry this label.
        Metadata.USES_SYMPTOMMANAGER,  # The 'Symptom Manager' recognises modules with this label.
        Metadata.USES_HEALTHSYSTEM,  # The 'HealthSystem' recognises modules with this label.
        Metadata.USES_HEALTHBURDEN  # The 'HealthBurden' module recognises modules with this label.
    }

    # generic symptom for severely traumatic injuries, mild injuries accounted for in generic symptoms under 'injury'
    # SYMPTOMS = {'em_severe_trauma',
    #             #             'Fracture'
    #             #             'bleeding from wound',
    #             #             'bruising around trauma site',
    #             #             'severe pain at trauma site',
    #             #             'swelling around trauma site',
    #             #             'redness or warmth around trauma site',
    #             #             'visual disturbances',
    #             #             'restlessness',
    #             #             'irritability',
    #             #             'loss of balance',
    #             #             'stiffness',
    #             #             'abnormal pupil behaviour/reflexes',
    #             #             'confusion',
    #             #             'fatigue',
    #             #             'fainting',
    #             #             'excessive salivation',
    #             #             'difficulty swallowing',
    #             #             'nosebleed',
    #             #             'breathing difficulty',
    #             #             'audible signs of injury',
    #             #             'uneven chest rise',
    #             #             'seat belt marks',
    #             #             'visual deformity of body part',
    #             #             'limitation of movement',
    #             #             'inability to walk',
    #             #             # TBI
    #             #             'periorbital ecchymosis',
    #             #             'shock',
    #             #             'hyperbilirubinemia',
    #             #             'abnormal posturing',
    #             #             'nausea',
    #             #             'loss of consciousness',
    #             #             'coma',
    #             #             'seizures',
    #             #             'tinnitus',
    #             #             'sensitive to light',
    #             #             'slurred speech',
    #             #             'personality change',
    #             #             'paralysis',
    #             #             'weakness in one half of body',
    #             #             # Dislocation
    #             #             'numbness in lower back and lower limbs',
    #             #             'muscle spasms',
    #             #             'hypermobile patella'
    #             #             # Soft tissue injury
    #             #             'ataxia',
    #             #             'coughing up blood',
    #             #             'stridor',
    #             #             'subcutaneous air',
    #             #             'blue discoloration of skin or lips',
    #             #             'pressure in chest',
    #             #             'rapid breathing',
    #             #             # Internal organ injury
    #             #             'low blood pressure',
    #             #             'Bluish discoloration of the belly',
    #             #             'Right-sided abdominal pain and right shoulder pain',
    #             #             'Blood in the urine',
    #             #             'Left arm and shoulder pain',
    #             #             'rigid abdomen',
    #             #             'cyanosis',
    #             #             'heart palpitations',
    #             #             'pain in the left shoulder or left side of the chest',
    #             #             'difficulty urinating',
    #             #             'urine leakage',
    #             #             'abdominal distension',
    #             #             'rectal bleeding',
    #             #             # Internal bleeding
    #             #             'sweaty skin',
    #             #             # Spinal cord injury
    #             #             'inability to control bladder',
    #             #             'inability to control bowel',
    #             #             'unnatural positioning of the head',
    #             #             # Amputation - limb's bloody gone
    #             }

    def __init__(self, name=None, resourcefilepath=None):
        # NB. Parameters passed to the module can be inserted in the __init__ definition.
        super().__init__(name)
        self.resourcefilepath = resourcefilepath

    def read_parameters(self, data_folder):
        """ Reads the parameters used in the RTI module"""
        dfd = pd.read_excel(Path(self.resourcefilepath) / 'ResourceFile_RTI.xlsx', sheet_name='parameter_values')
        self.load_parameters_from_dataframe(dfd)
        if "HealthBurden" in self.sim.modules.keys():
            # get the DALY weights of the seq associated with road traffic injuries
            self.parameters["daly_wt_unspecified_skull_fracture"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1674
            )
            self.parameters["daly_wt_basilar_skull_fracture"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1675
            )
            self.parameters["daly_wt_epidural_hematoma"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1676
            )
            self.parameters["daly_wt_subdural_hematoma"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1677
            )
            self.parameters["daly_wt_subarachnoid_hematoma"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1678
            )
            self.parameters["daly_wt_brain_contusion"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1679
            )
            self.parameters["daly_wt_intraventricular_haemorrhage"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1680
            )
            self.parameters["daly_wt_diffuse_axonal_injury"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1681
            )
            self.parameters["daly_wt_subgaleal_hematoma"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1682
            )
            self.parameters["daly_wt_midline_shift"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1683
            )
            self.parameters["daly_wt_facial_fracture"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1684
            )
            self.parameters["daly_wt_facial_soft_tissue_injury"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1685
            )
            self.parameters["daly_wt_eye_injury"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1686
            )
            self.parameters["daly_wt_neck_soft_tissue_injury"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1687
            )
            self.parameters["daly_wt_neck_internal_bleeding"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1688
            )
            self.parameters["daly_wt_neck_dislocation"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1689
            )
            self.parameters["daly_wt_chest_wall_bruises_hematoma"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1690
            )
            self.parameters["daly_wt_hemothorax"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1691
            )
            self.parameters["daly_wt_lung_contusion"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1692
            )
            self.parameters["daly_wt_diaphragm_rupture"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1693
            )
            self.parameters["daly_wt_rib_fracture"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1694
            )
            self.parameters["daly_wt_flail_chest"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1695
            )
            self.parameters["daly_wt_chest_wall_laceration"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1696
            )
            self.parameters["daly_wt_closed_pneumothorax"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1697
            )
            self.parameters["daly_wt_open_pneumothorax"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1698
            )
            self.parameters["daly_wt_surgical_emphysema"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1699
            )
            self.parameters["daly_wt_abd_internal_organ_injury"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1700
            )
            self.parameters["daly_wt_spinal_cord_lesion_neck_with_treatment"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1701
            )
            self.parameters["daly_wt_spinal_cord_lesion_neck_without_treatment"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1702
            )
            self.parameters["daly_wt_spinal_cord_lesion_below_neck_with_treatment"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1703
            )
            self.parameters["daly_wt_spinal_cord_lesion_below_neck_without_treatment"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1704
            )
            self.parameters["daly_wt_vertebrae_fracture"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1705
            )
            self.parameters["daly_wt_clavicle_scapula_humerus_fracture"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1706
            )
            self.parameters["daly_wt_hand_wrist_fracture_with_treatment"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1707
            )
            self.parameters["daly_wt_hand_wrist_fracture_without_treatment"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1708
            )
            self.parameters["daly_wt_radius_ulna_fracture_short_term_with_without_treatment"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1709
            )
            self.parameters["daly_wt_radius_ulna_fracture_long_term_without_treatment"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1710
            )
            self.parameters["daly_wt_dislocated_shoulder"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1711
            )
            self.parameters["daly_wt_amputated_finger"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1712
            )
            self.parameters["daly_wt_amputated_thumb"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1713
            )
            self.parameters["daly_wt_unilateral_arm_amputation_with_treatment"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1714
            )
            self.parameters["daly_wt_unilateral_arm_amputation_without_treatment"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1715
            )
            self.parameters["daly_wt_bilateral_arm_amputation_with_treatment"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1716
            )
            self.parameters["daly_wt_bilateral_arm_amputation_without_treatment"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1717
            )
            self.parameters["daly_wt_foot_fracture_short_term_with_without_treatment"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1718
            )
            self.parameters["daly_wt_foot_fracture_long_term_without_treatment"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1719
            )
            self.parameters["daly_wt_patella_tibia_fibula_fracture_with_treatment"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1720
            )
            self.parameters["daly_wt_patella_tibia_fibula_fracture_without_treatment"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1721
            )
            self.parameters["daly_wt_hip_fracture_short_term_with_without_treatment"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1722
            )
            self.parameters["daly_wt_hip_fracture_long_term_with_treatment"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1723
            )
            self.parameters["daly_wt_hip_fracture_long_term_without_treatment"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1724
            )
            self.parameters["daly_wt_pelvis_fracture_short_term"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1725
            )
            self.parameters["daly_wt_pelvis_fracture_long_term"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1726
            )
            self.parameters["daly_wt_femur_fracture_short_term"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1727
            )
            self.parameters["daly_wt_femur_fracture_long_term_without_treatment"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1728
            )
            self.parameters["daly_wt_dislocated_hip"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1729
            )
            self.parameters["daly_wt_dislocated_knee"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1730
            )
            self.parameters["daly_wt_amputated_toes"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1731
            )
            self.parameters["daly_wt_unilateral_lower_limb_amputation_with_treatment"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1732
            )
            self.parameters["daly_wt_unilateral_lower_limb_amputation_without_treatment"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1733
            )
            self.parameters["daly_wt_bilateral_lower_limb_amputation_with_treatment"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1734
            )
            self.parameters["daly_wt_bilateral_lower_limb_amputation_without_treatment"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1735
            )
            self.parameters["daly_wt_burns_greater_than_20_percent_body_area"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1736
            )
            self.parameters["daly_wt_burns_less_than_20_percent_body_area_with_treatment"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1737
            )
            self.parameters["daly_wt_burns_less_than_20_percent_body_area_without_treatment"] = self.sim.modules[
                "HealthBurden"].get_daly_weight(
                sequlae_code=1738
            )
            # self.sim.modules["HealthSystem"].register_disease_module(self)
        p = self.parameters
        # ================== Test the parameter distributions to see whether they sum to roughly one ===============
        assert 0.9999 < sum(p['number_of_injured_body_regions_distribution'][1]) < 1.0001, \
            "The number of injured body region distribution doesn't sum to one"
        assert 0.9999 < sum(p['injury_location_distribution'][1]) < 1.0001, \
            "The injured body region distribution doesn't sum to one"
        daly_weight_distributions = [val for key, val in p.items() if 'daly_dist_code_' in key]
        for dist in daly_weight_distributions:
            assert 0.9999 < sum(dist) < 1.0001, 'daly weight distribution does not sum to one'
        body_part_strings = ['head_prob_', 'face_prob_', 'neck_prob_', 'thorax_prob_', 'abdomen_prob_',
                             'spine_prob_', 'upper_ex_prob_', 'lower_ex_prob_']
        for body_part in body_part_strings:
            probabilities_to_assign_injuries = [val for key, val in p.items() if body_part in key]
            sum_probabilities = sum(probabilities_to_assign_injuries)
            assert (sum_probabilities % 1 < 0.0001) or (sum_probabilities % 1 > 0.9999), "The probabilities" \
                                                                                         "chosen for assigning" \
                                                                                         "injuries don't" \
                                                                                         "sum to one"
        probabilities = [val for key, val in p.items() if 'prob_' in key]
        for probability in probabilities:
            assert 0 < probability < 1, "Probability is not a feasible value"
        self.sim.modules['SymptomManager'].register_symptom(
            Symptom(
                name='severe_trauma',
                emergency_in_adults=True,
                emergency_in_children=True
            )
        )

    def rti_injury_diagnosis(self, person_id, the_appt_footprint):
        df = self.sim.population.props
        columns = ['rt_injury_1', 'rt_injury_2', 'rt_injury_3', 'rt_injury_4', 'rt_injury_5', 'rt_injury_6',
                   'rt_injury_7', 'rt_injury_8']
        persons_injuries = df.loc[[person_id], columns]
        # ================================ Fractures require x-rays ============================================
        fracture_codes = ['112', '113', '211', '212', '412', '414', '612', '712', '811', '812', '813']
        idx, counts = self.rti_find_and_count_injuries(persons_injuries, fracture_codes)
        if len(idx) > 0:
            the_appt_footprint['DiagRadio'] = 1
        # ========================= Traumatic brain injuries require ct scan ===================================
        codes = ['133', '134', '135']
        idx, counts = self.rti_find_and_count_injuries(persons_injuries, codes)
        if len(idx) > 0:
            the_appt_footprint['Tomography'] = 1  # This appointment requires a ct scan
        # ============================= Abdominal trauma requires ct scan ======================================
        codes = ['552', '553', '554']
        idx, counts = self.rti_find_and_count_injuries(persons_injuries, codes)
        if len(idx) > 0:
            the_appt_footprint['Tomography'] = 1

        # ============================== Spinal cord injury require x ray ======================================
        codes = ['673', '674', '675', '676']
        idx, counts = self.rti_find_and_count_injuries(persons_injuries, codes)
        if len(idx) > 0:
            the_appt_footprint['DiagRadio'] = 1  # This appointment requires an x-ray

        # ============================== Dislocations require x ray ============================================
        codes = ['322', '323', '722', '822']
        idx, counts = self.rti_find_and_count_injuries(persons_injuries, codes)
        if len(idx) > 0:
            the_appt_footprint['DiagRadio'] = 1  # This appointment requires an x-ray

        # --------------------------------- Soft tissue injury in neck -----------------------------------------
        codes = ['342', '343']
        idx, counts = self.rti_find_and_count_injuries(persons_injuries, codes)
        if len(idx) > 0:
            the_appt_footprint['Tomography'] = 1  # This appointment requires a ct scan
            the_appt_footprint['DiagRadio'] = 1  # This appointment requires an x ray

        # --------------------------------- Soft tissue injury in thorax/ lung injury --------------------------
        codes = ['441', '443', '453']
        idx, counts = self.rti_find_and_count_injuries(persons_injuries, codes)
        if len(idx) > 0:
            the_appt_footprint['Tomography'] = 1  # This appointment requires a ct scan
            the_appt_footprint['DiagRadio'] = 1  # This appointment requires an x ray

        # ----------------------------- Internal bleeding ------------------------------------------------------
        codes = ['361', '363', '461', '463']
        idx, counts = self.rti_find_and_count_injuries(persons_injuries, codes)
        if len(idx) > 0:
            the_appt_footprint['Tomography'] = 1  # This appointment requires a ct scan

    def rti_set_default_properties(self, context, idx):
        """
        Function to centralise all calls to set up default property values in a population
        :param context: what context the function is being called from
        :param idx: the index of the population to set up the defaults in
        :return:
        """
        df = self.sim.population.props
        non_list_properties = dict()
        list_properties = dict()
        if context == "initialise_population" or "on_birth":
            non_list_default_properties = [False, 'none', 'none', 'none', 'none', 'none', 'none', 'none', 'none',
                                           'none', False, 0, False, False, False, False, False, False, 0, pd.NaT,
                                           False, 0, pd.NaT]
            list_default_properties = [[pd.NaT] * 8]
            for (key, value) in self.sim.modules['RTI'].PROPERTIES.items():
                if value.python_type != list:
                    non_list_properties[key] = value
                else:
                    list_properties[key] = value
            non_list_module_columns = list(non_list_properties.keys())
            assert len(non_list_default_properties) == len(non_list_module_columns)
            df.loc[idx, non_list_module_columns] = non_list_default_properties
            for prop_number, property in enumerate(list(list_properties.keys())):
                for index, row in df.loc[idx, list(list_properties.keys())].iterrows():
                    df.at[index, property] = list_default_properties[prop_number]


    def initialise_population(self, population):
        """Sets up the default properties used in the RTI module and applies them to the dataframe. The default state
        for the RTI module is that people haven't been involved in a road traffic accident and are therefor alive and
        healthy."""
        df = population.props
        # Set up default properties in the population using centralised function
        # self.sim.modules['RTI'].rti_set_default_properties("initialise_population", df.is_alive.index)
        # drop any properties that are a list
        # Create corresponding list of default properties
        default_properties = []
        # df.loc[df.is_alive, rti_module_columns] = default_properties
        df.loc[df.is_alive, 'rt_road_traffic_inc'] = False
        df.loc[df.is_alive, 'rt_inj_severity'] = "none"  # default: no one has been injured in a RTI
        df.loc[df.is_alive, 'rt_injury_1'] = "none"
        df.loc[df.is_alive, 'rt_injury_2'] = "none"
        df.loc[df.is_alive, 'rt_injury_3'] = "none"
        df.loc[df.is_alive, 'rt_injury_4'] = "none"
        df.loc[df.is_alive, 'rt_injury_5'] = "none"
        df.loc[df.is_alive, 'rt_injury_6'] = "none"
        df.loc[df.is_alive, 'rt_injury_7'] = "none"
        df.loc[df.is_alive, 'rt_injury_8'] = "none"
        df.loc[df.is_alive, 'rt_polytrauma'] = False
        df.loc[df.is_alive, 'rt_ISS_score'] = 0
        df.loc[df.is_alive, 'rt_perm_disability'] = False
        df.loc[df.is_alive, 'rt_imm_death'] = False  # default: no one is dead on scene of crash
        df.loc[df.is_alive, 'rt_diagnosed'] = False
        df.loc[df.is_alive, 'rt_recovery_no_med'] = False  # default: no recovery without medical intervention
        df.loc[df.is_alive, 'rt_post_med_death'] = False  # default: no death after medical intervention
        df.loc[df.is_alive, 'rt_no_med_death'] = False
        df.loc[df.is_alive, 'rt_disability'] = 0  # default: no DALY
        df.loc[df.is_alive, 'rt_date_inj'] = pd.NaT
        df.loc[df.is_alive, 'rt_med_int'] = False
        df.loc[df.is_alive, 'rt_MAIS_military_score'] = 0
        df.loc[df.is_alive, 'rt_date_death_no_med'] = pd.NaT
        for index, row in df.iterrows():
            df.at[index, 'rt_date_to_remove_daly'] = [pd.NaT] * 8  # no one has any injuries to remove dalys for

    def initialise_simulation(self, sim):
        """At the start of the simulation we schedule two RTI events, the first is the main RTI event which takes parts
        of the population and assigns them to be involved in road traffic injuries and providing they survived will
        begin the interaction with the healthcare system. This event runs monthly.

        The second is the begin scheduling the RTI recovery event, which looks at those in the population who have been
        injured in a road traffic accident, checking whether enough time has passed for their injuries to have healed.
        When the injury has healed the associated daly weight is removed. This event runs daily.

        Finally, we also schedule a logging event, which records the relevant information regarding road traffic
        injuries in the last month.
        """
        # Begin modelling road traffic injuries
        event = RTI_Event(self)
        sim.schedule_event(event, sim.date + DateOffset(months=0))
        # Begin checking whether the persons injuries are healed
        event = RTI_Recovery_Event(self)
        sim.schedule_event(event, sim.date + DateOffset(months=0))
        # Begin checking whether those with untreated injuries die
        event = RTI_Check_Death_No_Med(self)
        sim.schedule_event(event, sim.date + DateOffset(months=0))
        # Begin logging the RTI events
        event = RTI_Logging_Event(self)
        sim.schedule_event(event, sim.date + DateOffset(months=0))

    def rti_do_when_diagnosed(self, person_id):
        """
        This function is called by the generic first emergency appointment when an injured person has been diagnosed
        in A&E and needs to progress further in the health system. The injured person will then be scheduled a generic
        'medical intervention' appointment which serves two purposes. The first is to determine what treatments they
        require for their injuries, the second is to contain them in the health care system with inpatient days.

        :param person_id: the person requesting medical care
        :return: n/a
        """
        df = self.sim.population.props
        # Check to see whether they have been sent here from A and E and are alive
        assert df.loc[person_id, 'rt_diagnosed'] & df.loc[person_id, 'is_alive']
        # Get the relevant information about their injuries
        cols = ['rt_injury_1', 'rt_injury_2', 'rt_injury_3', 'rt_injury_4', 'rt_injury_5', 'rt_injury_6',
                'rt_injury_7', 'rt_injury_8']
        person_injuries = df.loc[[person_id], cols]
        # check this person is injured, search they have an injury code that isn't "none"
        idx, counts = RTI.rti_find_and_count_injuries(person_injuries,
                                                      self.PROPERTIES.get('rt_injury_1').categories[1:-1])
        assert counts > 0, 'This person has asked for medical treatment despite not being injured'
        # If they meet the requirements, send them to HSI_RTI_MedicalIntervention for further treatment
        self.sim.modules['HealthSystem'].schedule_hsi_event(
            hsi_event=HSI_RTI_Medical_Intervention(module=self, person_id=person_id),
            priority=0,
            topen=self.sim.date
        )

    def rti_do_for_major_surgeries(self, person_id, count):
        """
        Function called in HSI_RTI_MedicalIntervention to schedule a major surgery if required. In
        HSI_RTI_MedicalIntervention, we determine that they need a surgery. In this function, further to scheduling the
        surgery, we double check that they do meet the conditions for needing a surgery. The conditions for needing a
        surgery is that they are alive, currently seeking medical intervention and have an injury that is treated by
        surgery.
        :param person_id: The person requesting major surgeries
        :param count: The amount of major surgeries required, used when scheduling surgeries to ensure that two major
                      surgeries aren't scheduled on the same day
        :return: n/a
        """
        df = self.sim.population.props
        # Check to see whether they have been sent here from RTI_MedicalIntervention and are alive
        assert df.at[person_id, 'rt_med_int'] & df.at[person_id, 'is_alive']
        # Determine what injuries are able to be treated by surgery by checking the injury codes which are currently
        # treated in this simulation, it seems there is a limited available to treat spinal cord injuries and chest
        # trauma in Malawi, so these are initially left out, but we will test different scenarios to see what happens
        # when we include those treatments
        surgically_treated_codes = ['112', '813a', '813b', '813c', '133a', '133b', '133c', '133d', '134a',
                                    '134b', '135', '552', '553', '554', '342', '343', '414', '361', '363',
                                    '782', '782a', '782b', '782c', '783', '882', '883', '884',
                                    'P133a', 'P133b', 'P133c', 'P133d', 'P134a', 'P134b', 'P135', 'P782a', 'P782b',
                                    'P782c', 'P783', 'P882', 'P883', 'P884'
                                    ]
        # If we allow surgical treatment of spinal cord injuries, extend the surgically treated codes to include spinal
        # cord injury codes
        if 'include_spine_surgery' in self.allowed_interventions:
            additional_codes = ['673a', '673b', '674a', '674b', '675a', '675b', '676', 'P673a', 'P673b', 'P674',
                                'P674a', 'P674b', 'P675', 'P675a', 'P675b', 'P676']
            for code in additional_codes:
                surgically_treated_codes.append(code)
        # If we allow surgical treatment of chest trauma, extend the surgically treated codes to include chest trauma
        # codes.
        if 'include_thoroscopy' in self.allowed_interventions:
            additional_codes = ['441', '443', '453', '453a', '453b', '463']
            for code in additional_codes:
                surgically_treated_codes.append(code)
        person = df.iloc[person_id]
        # isolate the relevant injury information
        cols = ['rt_injury_1', 'rt_injury_2', 'rt_injury_3', 'rt_injury_4', 'rt_injury_5', 'rt_injury_6',
                'rt_injury_7', 'rt_injury_8']
        person_injuries = df.loc[[person_id], cols]
        # Check whether the person sent to surgery has an injury which actually requires surgery
        idx, counts = RTI.rti_find_and_count_injuries(person_injuries, surgically_treated_codes)
        assert counts > 0, 'This person has been sent to major surgery without the right injuries'
        if person.is_alive:
            self.sim.modules['HealthSystem'].schedule_hsi_event(
                hsi_event=HSI_RTI_Major_Surgeries(module=self,
                                                  person_id=person_id),
                priority=0,
                topen=self.sim.date + DateOffset(days=count),
                tclose=self.sim.date + DateOffset(days=15))

    def rti_do_for_minor_surgeries(self, person_id, count):
        """
        Function called in HSI_RTI_MedicalIntervention to schedule a minor surgery if required. In
        HSI_RTI_MedicalIntervention, we determine that they need a surgery. In this function, further to scheduling the
        surgery, we double check that they do meet the conditions for needing a surgery. The conditions for needing a
        surgery is that they are alive, currently seeking medical intervention and have an injury that is treated by
        surgery.
        :param person_id: The person requesting major surgeries
        :param count: The amount of major surgeries required, used when scheduling surgeries to ensure that two minor
                      surgeries aren't scheduled on the same day
        :return:
        """
        df = self.sim.population.props
        # Check to see whether they have been sent here from RTI_MedicalIntervention and they are alive
        assert df.at[person_id, 'rt_med_int'] & df.at[person_id, 'is_alive']
        person = df.iloc[person_id]
        surgically_treated_codes = ['211', '212', '291', '241', '322', '323', '722', '822a', '822b']
        # Isolate the relevant injury information
        cols = ['rt_injury_1', 'rt_injury_2', 'rt_injury_3', 'rt_injury_4', 'rt_injury_5', 'rt_injury_6',
                'rt_injury_7', 'rt_injury_8']
        person_injuries = df.loc[[person_id], cols]
        # Check whether the person requesting minor surgeries has an injury that requires minor surgery
        idx, counts = RTI.rti_find_and_count_injuries(person_injuries, surgically_treated_codes)
        assert counts > 0
        if person.is_alive:
            self.sim.modules['HealthSystem'].schedule_hsi_event(
                hsi_event=HSI_RTI_Minor_Surgeries(module=self,
                                                  person_id=person_id),
                priority=0,
                topen=self.sim.date + DateOffset(days=count),
                tclose=self.sim.date + DateOffset(days=15))

    def rti_acute_pain_management(self, person_id):
        """
        Function called in HSI_RTI_MedicalIntervention to request pain management. This should be called be every alive
        injured person, regardless of what their injuries are. In this function we test whether they meet the
        requirements to ask for pain relief, that is they are alive and currently receiving medical treatment.
        :param person_id: The person requesting pain management
        :return: n/a
        """
        df = self.sim.population.props
        # Check to see whether they have been sent here from RTI_MedicalIntervention and they are alive
        assert df.at[person_id, 'rt_med_int'] & df.at[person_id, 'is_alive']
        # Isolate the relevant injury information
        cols = ['rt_injury_1', 'rt_injury_2', 'rt_injury_3', 'rt_injury_4', 'rt_injury_5', 'rt_injury_6',
                'rt_injury_7', 'rt_injury_8']
        person_injuries = df.loc[[person_id], cols]
        # check this person is injured, search they have an injury code that isn't "none", short hand way to do this is
        # with: self.PROPERTIES.get('rt_injury_1').categories[1:-1], which is a list of the all possible injury codes
        idx, counts = RTI.rti_find_and_count_injuries(person_injuries,
                                                      self.PROPERTIES.get('rt_injury_1').categories[1:-1])
        assert counts > 0, 'This person has asked for pain relief despite not being injured'
        person = df.iloc[person_id]
        if person.is_alive:
            self.sim.modules['HealthSystem'].schedule_hsi_event(
                hsi_event=HSI_RTI_Acute_Pain_Management(module=self,
                                                        person_id=person_id),
                priority=0,
                topen=self.sim.date,
                tclose=self.sim.date + DateOffset(days=15))

    def rti_ask_for_suture_kit(self, person_id):
        """
        Function called by HSI_RTI_MedicalIntervention to centralise all suture kit requests. This function checks
        that the person asking for a suture kit meets the requirements to get one. That is they are alive, currently
        being treated for their injuries and that they have a laceration which needs stitching.
        :param person_id: The person asking for a suture kit
        :return: n/a
        """
        df = self.sim.population.props
        # Check to see whether they have been sent here from RTI_MedicalIntervention and they are alive
        assert df.at[person_id, 'rt_med_int'] & df.at[person_id, 'is_alive']
        # Isolate the relevant injury information
        cols = ['rt_injury_1', 'rt_injury_2', 'rt_injury_3', 'rt_injury_4', 'rt_injury_5', 'rt_injury_6',
                'rt_injury_7', 'rt_injury_8']
        person_injuries = df.loc[[person_id], cols]
        laceration_codes = ['1101', '2101', '3101', '4101', '5101', '6101', '7101', '8101']
        # Check they have a laceration which needs stitches
        idx, counts = RTI.rti_find_and_count_injuries(person_injuries, laceration_codes)
        assert counts > 0, "This person has asked for stiches, but doens't have a laceration"
        person = df.iloc[person_id]
        if person.is_alive:
            self.sim.modules['HealthSystem'].schedule_hsi_event(
                hsi_event=HSI_RTI_Suture(module=self,
                                         person_id=person_id),
                priority=0,
                topen=self.sim.date,
                tclose=self.sim.date + DateOffset(days=15)
            )

    def rti_ask_for_burn_treatment(self, person_id):
        """
        Function called by HSI_RTI_MedicalIntervention to centralise all burn treatment requests. This function
        schedules burn treatments for the person if they meet the requirements, that is they are alive, currently being
        treated, and they have a burn which needs to be treated.
        :param person_id: The person requesting burn treatment
        :return: n/a
        """
        df = self.sim.population.props
        # Check to see whether they have been sent here from RTI_MedicalIntervention and they are alive
        assert df.at[person_id, 'rt_med_int'] & df.at[person_id, 'is_alive']
        # Isolate the relevant injury information
        cols = ['rt_injury_1', 'rt_injury_2', 'rt_injury_3', 'rt_injury_4', 'rt_injury_5', 'rt_injury_6',
                'rt_injury_7', 'rt_injury_8']
        person_injuries = df.loc[[person_id], cols]
        burn_codes = ['1114', '2114', '3113', '4113', '5113', '7113', '8113']
        # Check to see whether they have a burn which needs treatment
        idx, counts = RTI.rti_find_and_count_injuries(person_injuries, burn_codes)
        assert counts > 0, "This person has asked for burn treatment, but doens't have any burns"
        person = df.iloc[person_id]
        if person.is_alive:
            self.sim.modules['HealthSystem'].schedule_hsi_event(
                hsi_event=HSI_RTI_Burn_Management(module=self,
                                                  person_id=person_id),
                priority=0,
                topen=self.sim.date,
                tclose=self.sim.date + DateOffset(days=15)
            )

    def rti_ask_for_fracture_casts(self, person_id):
        """
        Function called by HSI_RTI_MedicalIntervention to centralise all fracture casting. This function schedules the
        fracture cast treatment if they meet the requirements to ask for it. That is they are alive, currently being
        treated and they have a fracture that needs casting (Note that this also handles slings for upper arm/shoulder
        fractures).
        :param person_id: The person asking for fracture cast/sling
        :return: n/a
        """
        df = self.sim.population.props
        # Check to see whether they have been sent here from RTI_MedicalIntervention and they are alive
        assert df.at[person_id, 'rt_med_int'] & df.at[person_id, 'is_alive']
        # Isolate the relevant injury information
        cols = ['rt_injury_1', 'rt_injury_2', 'rt_injury_3', 'rt_injury_4', 'rt_injury_5', 'rt_injury_6',
                'rt_injury_7', 'rt_injury_8']
        person_injuries = df.loc[[person_id], cols]
        fracture_codes = ['712a', '712b', '712c', '811', '812']
        # Check they have an injury treated by HSI_RTI_Fracture_Cast
        idx, counts = RTI.rti_find_and_count_injuries(person_injuries, fracture_codes)
        assert counts > 0, "This person has asked for fracture treatment, but doens't have appropriate fractures"
        person = df.iloc[person_id]
        if person.is_alive:
            self.sim.modules['HealthSystem'].schedule_hsi_event(
                hsi_event=HSI_RTI_Fracture_Cast(module=self,
                                                person_id=person_id),
                priority=0,
                topen=self.sim.date,
                tclose=self.sim.date + DateOffset(days=15)
            )

    def rti_ask_for_tetanus(self, person_id):
        """
        Function called by HSI_RTI_MedicalIntervention to centralise all tetanus requests. This function schedules a
        tetanus event, conditional on whether they are alive, being treated and have an injury that requires a tetanus
        vaccine, i.e. a burn or a laceration.

        :param person_id: the person requesting a tetanus jab
        :return: n/a
        """
        df = self.sim.population.props
        # Check to see whether they have been sent here from RTI_MedicalIntervention and are alive
        assert df.at[person_id, 'rt_med_int'] & df.at[person_id, 'is_alive']
        person = df.iloc[person_id]
        # Isolate the relevant injury information
        codes_for_tetanus = ['1101', '2101', '3101', '4101', '5101', '7101', '8101',
                             '1114', '2114', '3113', '4113', '5113', '7113', '8113']
        cols = ['rt_injury_1', 'rt_injury_2', 'rt_injury_3', 'rt_injury_4', 'rt_injury_5', 'rt_injury_6',
                'rt_injury_7', 'rt_injury_8']
        person_injuries = df.loc[[person_id], cols]
        # Check that they have a burn/laceration
        idx, counts = RTI.rti_find_and_count_injuries(person_injuries, codes_for_tetanus)
        assert counts > 0, "This person has requested a tetanus jab but doesn't require one"
        if person.is_alive:
            self.sim.modules['HealthSystem'].schedule_hsi_event(
                hsi_event=HSI_RTI_Tetanus_Vaccine(module=self,
                                                  person_id=person_id),
                priority=0,
                topen=self.sim.date,
                tclose=self.sim.date + DateOffset(days=15)
            )

    def rti_find_injury_column(self, person_id, codes):
        """
        This function is a tool to find the injury column an injury code occurs in. To call this function you need to
        provide the person who you want to perform the search on and the injury codes which you want to find the
        corresponding injury column for. The function/search will return the injury code which the person has
        from the list of codes you supplied, and which injury column from rt_injury_1 through to rt_injury_8, the code
        appears in.

        :param person_id: The person the search is being performed for
        :param codes: The injury codes being searched for
        :return: which column out of rt_injury_1 to rt_injury_8 the injury code occurs in, and the injury code itself
        """
        df = self.sim.population.props
        # Isolate the relevant injury information
        columns = ['rt_injury_1', 'rt_injury_2', 'rt_injury_3', 'rt_injury_4', 'rt_injury_5', 'rt_injury_6',
                   'rt_injury_7', 'rt_injury_8']
        df = df.loc[[person_id], columns]
        # Set up the number to iterate over
        injury_numbers = range(1, 9)
        # create empty variables to return if the search doesn't find a code/column
        injury_column = ''
        injury_code = ''
        # Iterate over the list of codes to begin the search
        for code in codes:
            # Iterate over the injury columns
            for injury_number in injury_numbers:
                # Create a dataframe where the rows are those who have injury code 'code' within the column
                # 'rt_injury_(injury_number)', if the person doesn't have 'code' in column 'rt_injury_(injury_number)',
                # then the dataframe is empty
                found = df[df[f"rt_injury_{injury_number}"].str.contains(code)]
                # check if the dataframe is non-empty
                if len(found) > 0:
                    # if the dataframe is non-empty, then we have found the injury column corresponding to the injury
                    # code for person 'person_id'. Assign the found column/code to injury_column and injury_code and
                    # break the for loop.
                    injury_column = f"rt_injury_{injury_number}"
                    injury_code = code
                    break
        # Return the found column for the injury code
        return injury_column, injury_code

    def rti_find_all_columns_of_treated_injuries(self, person_id, codes):
        """
        This function searches for treated injuries (supplied by the parameter codes) in person person_id, finding and
        returning all the columns with treated injuries and all the injury codes for the treated injuries.

        :param person_id: The person the search is being performed on
        :param codes: The treated injury codes
        :return: All columns and codes of the successfully treated injuries
        """
        df = self.sim.population.props
        # Isolate the relevant injury information
        columns = ['rt_injury_1', 'rt_injury_2', 'rt_injury_3', 'rt_injury_4', 'rt_injury_5', 'rt_injury_6',
                   'rt_injury_7', 'rt_injury_8']
        df = df.loc[[person_id], columns]
        # create empty variables to return the columns and codes of the treated injuries
        columns_to_return = []
        codes_to_return = []
        injury_numbers = range(1, 9)
        # iterate over the codes in the list codes and also the injury columns
        for code in codes:
            for injury_number in injury_numbers:
                # Search a sub-dataframe that is non-empty if the code is present is in that column and empty if not
                found = len(df[df[f"rt_injury_{injury_number}"] == code])
                if found > 0:
                    # if the code is in the column, store the column and code in columns_to_return and codes_to_return
                    # respectively
                    columns_to_return.append(f"rt_injury_{injury_number}")
                    codes_to_return.append(code)

        return columns_to_return, codes_to_return

    def rti_assign_daly_weights(self, injured_index):
        """
        This function assigns DALY weights associated with each injury when they happen, by default this function
        gives the DALY weight for each condition without treatment, this will then be swapped for the DALY weight
        associated with the injury with treatment when treatment occurs.
        :param injured_index: The people who have been involved in a road traffic accident for the current month and did
                              not die on the scene of the crash
        :return: n/a
        """
        rng = self.rng

        df = self.sim.population.props
        # Check that those sent here have been involved in a road traffic accident
        assert sum(df.loc[injured_index, 'rt_road_traffic_inc']) == len(injured_index)
        # Check everyone here has at least one injury to be given a daly weight to
        assert sum(df.loc[injured_index, 'rt_injury_1'] != "none") == len(injured_index)
        # Check everyone here is alive and hasn't died on scene
        assert (sum(df.loc[injured_index, 'is_alive']) == len(injured_index)) & \
               (sum(df.loc[injured_index, 'rt_imm_death']) == 0)
        columns = ['rt_injury_1', 'rt_injury_2', 'rt_injury_3', 'rt_injury_4', 'rt_injury_5', 'rt_injury_6',
                   'rt_injury_7', 'rt_injury_8']
        p = self.parameters
        selected_for_rti_inj = df.loc[injured_index, columns]
        # =============================== AIS region 1: head ==========================================================
        # ------ Find those with skull fractures and update rt_fracture to match and call for treatment ---------------
        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, ['112'])
        if len(idx) > 0:
            for injuredperson in idx:
                df.loc[injuredperson, 'rt_disability'] += self.daly_wt_unspecified_skull_fracture
        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, ['113'])
        if len(idx) > 0:
            for injuredperson in idx:
                df.loc[injuredperson, 'rt_disability'] += self.daly_wt_basilar_skull_fracture
        # ------ Find those with traumatic brain injury and update rt_tbi to match and call the TBI treatment ---------
        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, ['133'])
        # There is a one-to-many mapping for the daly weights associated with code 133, create the map here and then
        # change the code 133 to reflect the particular daly weight the original code is mapped to
        dalyweightsfor133 = [self.daly_wt_subarachnoid_hematoma, self.daly_wt_brain_contusion,
                             self.daly_wt_intraventricular_haemorrhage, self.daly_wt_subgaleal_hematoma]
        probabilities = p['daly_dist_code_133']
        idx_for_choose = [0, 1, 2, 3]
        if len(idx) > 0:
            for injuredperson in idx:
                choose = rng.choice(idx_for_choose, p=probabilities)
                if choose == 0:
                    # person has a subarachnoid hematoma, assign the correct daly weight to rt_disability
                    df.loc[injuredperson, 'rt_disability'] += dalyweightsfor133[choose]
                    # Change code 133 to 133a so that the correct daly weight will by removed if treatment is successful
                    column, code = RTI.rti_find_injury_column(self, person_id=injuredperson, codes=['133'])
                    df.loc[injuredperson, column] = code + "a"
                elif choose == 1:
                    # person has a brain contusion, assign the correct daly weight to rt_disability
                    df.loc[injuredperson, 'rt_disability'] += dalyweightsfor133[choose]
                    # Change code 133 to 133b so that the correct daly weight will by removed if treatment is successful
                    column, code = RTI.rti_find_injury_column(self, person_id=injuredperson, codes=['133'])
                    df.loc[injuredperson, column] = code + "b"
                elif choose == 2:
                    # person has a intraventricular haemorrhage, assign the correct daly weight to rt_disability
                    df.loc[injuredperson, 'rt_disability'] += dalyweightsfor133[choose]
                    # Change code 133 to 133c so that the correct daly weight will by removed if treatment is successful
                    column, code = RTI.rti_find_injury_column(self, person_id=injuredperson, codes=['133'])
                    df.loc[injuredperson, column] = code + "c"
                else:
                    # person has a subgaleal hematoma, assign the correct daly weight to rt_disability
                    df.loc[injuredperson, 'rt_disability'] += dalyweightsfor133[choose]
                    # Change code 133 to 133d so that the correct daly weight will by removed if treatment is successful
                    column, code = RTI.rti_find_injury_column(self, person_id=injuredperson, codes=['133'])
                    df.loc[injuredperson, column] = code + "d"

        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, ['134'])
        # Code 134 maps on to two daly weights, assign the daly weights and update the codes as required
        dalyweightsfor134 = [self.daly_wt_epidural_hematoma, self.daly_wt_subdural_hematoma]
        probabilities = p['daly_dist_code_134']
        idx_for_choose = [0, 1]
        if len(idx) > 0:
            for injuredperson in idx:
                choose = rng.choice(idx_for_choose, p=probabilities)
                if choose == 0:
                    # This person has an epidural hematoma, assign correct daly weight
                    df.loc[injuredperson, 'rt_disability'] += dalyweightsfor134[choose]
                    column, code = RTI.rti_find_injury_column(self, person_id=injuredperson, codes=['134'])
                    # Update the code
                    df.loc[injuredperson, column] = code + "a"
                else:
                    # This person has a subdural hematoma, assign correct daly weight
                    df.loc[injuredperson, 'rt_disability'] += dalyweightsfor134[choose]
                    column, code = RTI.rti_find_injury_column(self, person_id=injuredperson, codes=['134'])
                    # Update the code
                    df.loc[injuredperson, column] = code + "b"
        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, ['135'])
        if len(idx) > 0:
            df.loc[idx, 'rt_disability'] += self.daly_wt_diffuse_axonal_injury

        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, ['1101'])
        if len(idx) > 0:
            df.loc[idx, 'rt_disability'] += self.daly_wt_facial_soft_tissue_injury
        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, ['1114'])
        if len(idx) > 0:
            df.loc[idx, 'rt_disability'] += self.daly_wt_burns_greater_than_20_percent_body_area

        # =============================== AIS region 2: face ==========================================================
        # ----------------------- Find those with facial fractures and assign DALY weight -----------------------------
        codes = ['211', '212']
        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, codes)
        if counts > 0:
            for injuredperson in idx:
                df.loc[injuredperson, 'rt_disability'] += self.daly_wt_facial_fracture

        # ----------------- Find those with lacerations/soft tissue injuries and assign DALY weight -------------------
        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, ['2101'])
        if len(idx) > 0:
            for injuredperson in idx:
                df.loc[injuredperson, 'rt_disability'] += self.daly_wt_facial_soft_tissue_injury

        # ----------------- Find those with eye injuries and assign DALY weight ---------------------------------------
        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, ['291'])
        if len(idx) > 0:
            df.loc[idx, 'rt_disability'] += self.daly_wt_eye_injury
        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, ['241'])
        if len(idx) > 0:
            df.loc[idx, 'rt_disability'] += self.daly_wt_facial_soft_tissue_injury

        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, ['2114'])
        if len(idx) > 0:
            df.loc[idx, 'rt_disability'] += self.daly_wt_burns_greater_than_20_percent_body_area
        # =============================== AIS region 3: Neck ==========================================================
        # -------------------------- soft tissue injuries and internal bleeding----------------------------------------
        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, ['342', '343', '361', '363'])
        if len(idx) > 0:
            df.loc[idx, 'rt_disability'] += self.daly_wt_neck_internal_bleeding
        # -------------------------------- neck vertebrae dislocation ------------------------------------------------
        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, ['322', '323'])
        if len(idx) > 0:
            df.loc[idx, 'rt_disability'] += self.daly_wt_neck_dislocation
        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, ['3101'])
        if len(idx) > 0:
            df.loc[idx, 'rt_disability'] += self.daly_wt_facial_soft_tissue_injury
        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, ['3113'])
        if len(idx) > 0:
            df.loc[idx, 'rt_disability'] += self.daly_wt_burns_less_than_20_percent_body_area_without_treatment
        # ================================== AIS region 4: Thorax =====================================================
        # --------------------------------- fractures & flail chest ---------------------------------------------------
        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, ['412'])
        if len(idx) > 0:
            df.loc[idx, 'rt_disability'] += self.daly_wt_rib_fracture
        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, ['414'])
        if len(idx) > 0:
            df.loc[idx, 'rt_disability'] += self.daly_wt_flail_chest
        # ------------------------------------ Internal bleeding ------------------------------------------------------
        # chest wall bruises/hematoma
        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, ['461'])
        if len(idx) > 0:
            df.loc[idx, 'rt_disability'] += self.daly_wt_chest_wall_bruises_hematoma
        # hemothorax
        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, ['463'])
        if len(idx) > 0:
            df.loc[idx, 'rt_disability'] += self.daly_wt_hemothorax
        # -------------------------------- Internal organ injury ------------------------------------------------------
        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, ['453'])
        dalyweightsfor453 = [self.daly_wt_diaphragm_rupture, self.daly_wt_lung_contusion]
        probabilities = p['daly_dist_code_453']
        idx_for_choose = [0, 1]
        if len(idx) > 0:
            for injuredperson in idx:
                choose = rng.choice(idx_for_choose, p=probabilities)
                if choose == 0:
                    df.loc[injuredperson, 'rt_disability'] += dalyweightsfor453[choose]
                    column, code = RTI.rti_find_injury_column(self, person_id=injuredperson, codes=['453'])
                    df.loc[injuredperson, column] = code + "a"
                else:
                    df.loc[injuredperson, 'rt_disability'] += dalyweightsfor453[choose]
                    column, code = RTI.rti_find_injury_column(self, person_id=injuredperson, codes=['453'])
                    df.loc[injuredperson, column] = code + "b"
        # ----------------------------------- Soft tissue injury ------------------------------------------------------
        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, ['441'])
        if len(idx) > 0:
            df.loc[idx, 'rt_disability'] += self.daly_wt_chest_wall_laceration
        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, ['442'])
        if len(idx) > 0:
            df.loc[idx, 'rt_disability'] += self.daly_wt_surgical_emphysema
        # ---------------------------------- Pneumothoraxs ------------------------------------------------------------
        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, ['441'])
        if len(idx) > 0:
            df.loc[idx, 'rt_disability'] += self.daly_wt_closed_pneumothorax
        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, ['443'])
        if len(idx) > 0:
            df.loc[idx, 'rt_disability'] += self.daly_wt_open_pneumothorax
        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, ['4101'])
        if len(idx) > 0:
            df.loc[idx, 'rt_disability'] += self.daly_wt_facial_soft_tissue_injury
        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, ['4113'])
        if len(idx) > 0:
            df.loc[idx, 'rt_disability'] += self.daly_wt_burns_less_than_20_percent_body_area_without_treatment
        # ================================== AIS region 5: Abdomen ====================================================
        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, ['552', '553', '554'])
        if len(idx) > 0:
            df.loc[idx, 'rt_disability'] += self.daly_wt_abd_internal_organ_injury
        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, ['5101'])
        if len(idx) > 0:
            df.loc[idx, 'rt_disability'] += self.daly_wt_facial_soft_tissue_injury
        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, ['5113'])
        if len(idx) > 0:
            df.loc[idx, 'rt_disability'] += self.daly_wt_burns_less_than_20_percent_body_area_without_treatment
        # =================================== AIS region 6: spine =====================================================
        # ----------------------------------- vertebrae fracture ------------------------------------------------------
        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, ['612'])
        if len(idx) > 0:
            df.loc[idx, 'rt_disability'] += self.daly_wt_vertebrae_fracture
        # ---------------------------------- Spinal cord injuries -----------------------------------------------------
        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, ['673'])
        dalyweightsfor673 = [self.daly_wt_spinal_cord_lesion_neck_without_treatment,
                             self.daly_wt_spinal_cord_lesion_below_neck_without_treatment]
        probabilities = p['daly_dist_code_673']
        idx_for_choose = [0, 1]
        if len(idx) > 0:
            for injuredperson in idx:
                choose = rng.choice(idx_for_choose, p=probabilities)
                if choose == 0:
                    df.loc[injuredperson, 'rt_disability'] += dalyweightsfor673[choose]
                    column, code = RTI.rti_find_injury_column(self, person_id=injuredperson, codes=['673'])
                    df.loc[injuredperson, column] = code + "a"
                else:
                    df.loc[injuredperson, 'rt_disability'] += dalyweightsfor673[choose]
                    column, code = RTI.rti_find_injury_column(self, person_id=injuredperson, codes=['673'])
                    df.loc[injuredperson, column] = code + "b"

        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, ['674', '675'])
        dalyweightsfor674675 = [self.daly_wt_spinal_cord_lesion_neck_without_treatment,
                                self.daly_wt_spinal_cord_lesion_below_neck_without_treatment]
        probabilities = p['daly_dist_codes_674_675']
        idx_for_choose = [0, 1]
        if len(idx) > 0:
            for injuredperson in idx:
                choose = rng.choice(idx_for_choose, p=probabilities)
                if choose == 0:
                    df.loc[injuredperson, 'rt_disability'] += dalyweightsfor674675[choose]
                    column, code = RTI.rti_find_injury_column(self, person_id=injuredperson, codes=['674', '675'])
                    df.loc[injuredperson, column] = code + "a"
                else:
                    df.loc[injuredperson, 'rt_disability'] += dalyweightsfor674675[choose]
                    column, code = RTI.rti_find_injury_column(self, person_id=injuredperson, codes=['674', '675'])
                    df.loc[injuredperson, column] = code + "b"
        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, ['676'])
        if len(idx) > 0:
            df.loc[idx, 'rt_disability'] += self.daly_wt_spinal_cord_lesion_neck_without_treatment

        # ============================== AIS body region 7: upper extremities ======================================
        # ------------------------------------------ fractures ------------------------------------------------------
        # Fracture to Clavicle, scapula, humerus, Hand/wrist, Radius/ulna
        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, ['712'])
        dalyweightsfor712 = [self.daly_wt_clavicle_scapula_humerus_fracture,
                             self.daly_wt_hand_wrist_fracture_without_treatment,
                             self.daly_wt_radius_ulna_fracture_short_term_with_without_treatment]
        probabilities = p['daly_dist_code_712']
        idx_for_choose = [0, 1, 2]
        if len(idx) > 0:
            for injuredperson in idx:
                choose = rng.choice(idx_for_choose, p=probabilities)
                if choose == 0:
                    df.loc[injuredperson, 'rt_disability'] += dalyweightsfor712[choose]
                    column, code = RTI.rti_find_injury_column(self, person_id=injuredperson, codes=['712'])
                    df.loc[injuredperson, column] = code + "a"
                elif choose == 1:
                    df.loc[injuredperson, 'rt_disability'] += dalyweightsfor712[choose]
                    column, code = RTI.rti_find_injury_column(self, person_id=injuredperson, codes=['712'])
                    df.loc[injuredperson, column] = code + "b"
                else:
                    df.loc[injuredperson, 'rt_disability'] += dalyweightsfor712[choose]
                    column, code = RTI.rti_find_injury_column(self, person_id=injuredperson, codes=['712'])
                    df.loc[injuredperson, column] = code + "c"
        # ------------------------------------ Dislocation of shoulder ---------------------------------------------
        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, ['722'])
        if len(idx) > 0:
            df.loc[idx, 'rt_disability'] += self.daly_wt_dislocated_shoulder
        # ------------------------------------------ Amputations -----------------------------------------------------
        # Amputation of fingers, Unilateral upper limb amputation, Thumb amputation
        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, ['782'])
        dalyweightsfor782 = [self.daly_wt_amputated_finger,
                             self.daly_wt_unilateral_arm_amputation_without_treatment,
                             self.daly_wt_amputated_thumb]
        probabilities = p['daly_dist_code_782']
        idx_for_choose = [0, 1, 2]
        if len(idx) > 0:
            for injuredperson in idx:
                choose = rng.choice(idx_for_choose, p=probabilities)
                if choose == 0:
                    df.loc[injuredperson, 'rt_disability'] += dalyweightsfor782[choose]
                    column, code = RTI.rti_find_injury_column(self, person_id=injuredperson, codes=['782'])
                    df.loc[injuredperson, column] = code + "a"
                if choose == 1:
                    df.loc[injuredperson, 'rt_disability'] += dalyweightsfor782[choose]
                    column, code = RTI.rti_find_injury_column(self, person_id=injuredperson, codes=['782'])
                    df.loc[injuredperson, column] = code + "b"
                else:
                    df.loc[injuredperson, 'rt_disability'] += dalyweightsfor782[choose]
                    column, code = RTI.rti_find_injury_column(self, person_id=injuredperson, codes=['782'])
                    df.loc[injuredperson, column] = code + "c"
        # Bilateral upper limb amputation
        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, ['783'])
        if len(idx) > 0:
            df.loc[idx, 'rt_disability'] += self.daly_wt_bilateral_arm_amputation_without_treatment
        # ----------------------------------- cuts and bruises --------------------------------------------------------
        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, ['7101'])
        if len(idx) > 0:
            df.loc[idx, 'rt_disability'] += self.daly_wt_facial_soft_tissue_injury
        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, ['7113'])
        if len(idx) > 0:
            df.loc[idx, 'rt_disability'] += self.daly_wt_burns_less_than_20_percent_body_area_without_treatment
        # ============================== AIS body region 8: Lower extremities ========================================
        # ------------------------------------------ Fractures -------------------------------------------------------
        # Broken foot
        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, ['811'])
        if len(idx) > 0:
            df.loc[idx, 'rt_disability'] += self.daly_wt_foot_fracture_short_term_with_without_treatment
        # Broken patella, tibia, fibula
        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, ['812'])
        if len(idx) > 0:
            df.loc[idx, 'rt_disability'] += self.daly_wt_patella_tibia_fibula_fracture_without_treatment
        # Broken Hip, Pelvis, Femur other than femoral neck
        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, ['813'])
        dalyweightsfor813 = [self.daly_wt_hip_fracture_short_term_with_without_treatment,
                             self.daly_wt_pelvis_fracture_short_term,
                             self.daly_wt_femur_fracture_short_term]
        probabilities = p['daly_dist_code_813']
        idx_for_choose = [0, 1, 2]
        if len(idx) > 0:
            for injuredperson in idx:
                choose = rng.choice(idx_for_choose, p=probabilities)
                if choose == 0:
                    df.loc[injuredperson, 'rt_disability'] += dalyweightsfor813[choose]
                    column, code = RTI.rti_find_injury_column(self, person_id=injuredperson, codes=['813'])
                    df.loc[injuredperson, column] = code + "a"
                elif choose == 1:
                    df.loc[injuredperson, 'rt_disability'] += dalyweightsfor813[choose]
                    column, code = RTI.rti_find_injury_column(self, person_id=injuredperson, codes=['813'])
                    df.loc[injuredperson, column] = code + "b"
                else:
                    df.loc[injuredperson, 'rt_disability'] += dalyweightsfor813[choose]
                    column, code = RTI.rti_find_injury_column(self, person_id=injuredperson, codes=['813'])
                    df.loc[injuredperson, column] = code + "c"

        # -------------------------------------- Dislocations -------------------------------------------------------
        # Dislocated hip, knee
        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, ['822'])
        dalyweightsfor822 = [self.daly_wt_dislocated_hip,
                             self.daly_wt_dislocated_knee]
        probabilities = p['daly_dist_code_822']
        idx_for_choose = [0, 1]
        if len(idx) > 0:
            for injuredperson in idx:
                choose = rng.choice(idx_for_choose, p=probabilities)
                if choose == 0:
                    df.loc[injuredperson, 'rt_disability'] += dalyweightsfor822[choose]
                    column, code = RTI.rti_find_injury_column(self, person_id=injuredperson, codes=['822'])
                    df.loc[injuredperson, column] = code + "a"
                else:
                    df.loc[injuredperson, 'rt_disability'] += dalyweightsfor822[choose]
                    column, code = RTI.rti_find_injury_column(self, person_id=injuredperson, codes=['822'])
                    df.loc[injuredperson, column] = code + "b"
        # --------------------------------------- Amputations ------------------------------------------------------
        # toes
        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, ['882'])
        if len(idx) > 0:
            df.loc[idx, 'rt_disability'] += self.daly_wt_amputated_toes
        # Unilateral lower limb amputation
        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, ['883'])
        if len(idx) > 0:
            df.loc[idx, 'rt_disability'] += self.daly_wt_unilateral_lower_limb_amputation_without_treatment
        # Bilateral lower limb amputation
        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, ['884'])
        if len(idx) > 0:
            df.loc[idx, 'rt_disability'] += self.daly_wt_bilateral_lower_limb_amputation_without_treatment
        # ------------------------------------ cuts and bruises -----------------------------------------------------
        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, ['8101'])
        if len(idx) > 0:
            df.loc[idx, 'rt_disability'] += self.daly_wt_facial_soft_tissue_injury

        idx, counts = RTI.rti_find_and_count_injuries(selected_for_rti_inj, ['8113'])
        if len(idx) > 0:
            df.loc[idx, 'rt_disability'] += self.daly_wt_burns_less_than_20_percent_body_area_without_treatment
        DALYweightoverlimit = df.index[df['rt_disability'] > 1]
        df.loc[DALYweightoverlimit, 'rt_disability'] = 1
        assert (df.loc[injured_index, 'rt_disability'] > 0).all()

    def rti_alter_daly_post_treatment(self, person_id, codes):
        """
        This function removes the DALY weight associated with each injury code after treament is complete. This
        function is called by RTI_Recovery_event which removes asks to remove the DALY weight when the injury has
        healed
        :param person_id: The person who needs a daly weight removed as their injury has healed
        :param codes: The injury codes for the healed injury/injuries
        :return: n/a
        """

        df = self.sim.population.props
        # Check that people who are sent here have had medical treatment
        assert df.loc[person_id, 'rt_med_int']
        # Check everyone here has at least one injury to be alter the daly weight to
        cols = ['rt_injury_1', 'rt_injury_2', 'rt_injury_3', 'rt_injury_4', 'rt_injury_5', 'rt_injury_6',
                'rt_injury_7', 'rt_injury_8']
        person_injuries = df.loc[[person_id], cols]
        # check this person is injured, search they have an injury code that isn't "none"
        idx, counts = RTI.rti_find_and_count_injuries(person_injuries,
                                                      self.PROPERTIES.get('rt_injury_1').categories[1:-1])
        assert counts > 0, 'This person has asked for medical treatment despite not being injured'
        # Check everyone here is alive and hasn't died on scene
        assert df.loc[person_id, 'is_alive'] & ~df.loc[person_id, 'rt_imm_death']
        # ------------------------------- Remove the daly weights for treated injuries ---------------------------------
        # ==================================== heal with time injuries =================================================
        for code in codes:
            if code == '322' or code == '323':
                df.loc[person_id, 'rt_disability'] -= self.daly_wt_neck_dislocation
            if code == '722':
                df.loc[person_id, 'rt_disability'] -= self.daly_wt_dislocated_shoulder
            if code == '822a':
                df.loc[person_id, 'rt_disability'] -= self.daly_wt_dislocated_hip
            if code == '822b':
                df.loc[person_id, 'rt_disability'] -= self.daly_wt_dislocated_knee
            if code == '112':
                df.loc[person_id, 'rt_disability'] -= self.daly_wt_unspecified_skull_fracture
            if code == '113':
                df.loc[person_id, 'rt_disability'] -= self.daly_wt_basilar_skull_fracture
            if code == '552' or code == '553' or code == '554':
                df.loc[person_id, 'rt_disability'] -= self.daly_wt_abd_internal_organ_injury
            if code == '412':
                df.loc[person_id, 'rt_disability'] -= self.daly_wt_rib_fracture
            if code == '442':
                df.loc[person_id, 'rt_disability'] -= self.daly_wt_surgical_emphysema
            if code == '461':
                df.loc[person_id, 'rt_disability'] -= self.daly_wt_chest_wall_bruises_hematoma
            # ========================== Codes 'treated' with stitches  ==============================================
            laceration_codes = ['1101', '2101', '3101', '4101', '5101', '6101', '7101', '8101']
            if code in laceration_codes:
                logger.debug("This person has had their lacerations treated")
                df.loc[person_id, 'rt_disability'] -= self.daly_wt_facial_soft_tissue_injury
            # ============================== Codes 'treated' with fracture casts ======================================
            if code == '712a':
                df.loc[person_id, 'rt_disability'] -= self.daly_wt_clavicle_scapula_humerus_fracture
            if code == '712b':
                df.loc[person_id, 'rt_disability'] -= self.daly_wt_hand_wrist_fracture_with_treatment
            if code == '712c':
                df.loc[person_id, 'rt_disability'] -= \
                    self.daly_wt_radius_ulna_fracture_short_term_with_without_treatment
            if code == '811':
                df.loc[person_id, 'rt_disability'] -= \
                    self.daly_wt_foot_fracture_short_term_with_without_treatment
            if code == '812':
                df.loc[person_id, 'rt_disability'] -= self.daly_wt_patella_tibia_fibula_fracture_with_treatment
            # ============================== Codes 'treated' with minor surgery =======================================
            if code == '322' or code == '323':
                df.loc[person_id, 'rt_disability'] -= self.daly_wt_neck_dislocation
            if code == '722':
                df.loc[person_id, 'rt_disability'] -= self.daly_wt_dislocated_shoulder
            if code == '822a':
                df.loc[person_id, 'rt_disability'] -= self.daly_wt_dislocated_hip
            if code == '822b':
                df.loc[person_id, 'rt_disability'] -= self.daly_wt_dislocated_knee
            if code == '291':
                df.loc[person_id, 'rt_disability'] -= self.daly_wt_eye_injury
            if code == '241':
                df.loc[person_id, 'rt_disability'] -= self.daly_wt_facial_soft_tissue_injury
            if code == '211' or code == '212':
                df.loc[person_id, 'rt_disability'] -= self.daly_wt_facial_fracture
            # ============================== Codes 'treated' with burn management ======================================
            if code == '1114':
                df.loc[person_id, 'rt_disability'] -= self.daly_wt_burns_greater_than_20_percent_body_area
            if code == '2114':
                df.loc[person_id, 'rt_disability'] -= self.daly_wt_burns_greater_than_20_percent_body_area
            if code == '3113':
                df.loc[person_id, 'rt_disability'] -= self.daly_wt_burns_less_than_20_percent_body_area_with_treatment
            if code == '4113':
                df.loc[person_id, 'rt_disability'] -= self.daly_wt_burns_less_than_20_percent_body_area_with_treatment
            if code == '5113':
                df.loc[person_id, 'rt_disability'] -= self.daly_wt_burns_less_than_20_percent_body_area_with_treatment
            if code == '7113':
                df.loc[person_id, 'rt_disability'] -= self.daly_wt_burns_less_than_20_percent_body_area_with_treatment
            if code == '8113':
                df.loc[person_id, 'rt_disability'] -= self.daly_wt_burns_less_than_20_percent_body_area_with_treatment
            # ============================== Codes 'treated' with major surgery ========================================
            if code == '112':
                df.loc[person_id, 'rt_disability'] -= self.daly_wt_unspecified_skull_fracture
            if code == '113':
                df.loc[person_id, 'rt_disability'] -= self.daly_wt_basilar_skull_fracture
            if code == '813a':
                df.loc[person_id, 'rt_disability'] -= self.daly_wt_hip_fracture_long_term_with_treatment
            if code == '813b':
                df.loc[person_id, 'rt_disability'] -= self.daly_wt_pelvis_fracture_long_term
            if code == '813c':
                df.loc[person_id, 'rt_disability'] -= self.daly_wt_femur_fracture_short_term
            if code == '133a':
                df.loc[person_id, 'rt_disability'] -= self.daly_wt_subarachnoid_hematoma
            if code == '133b':
                df.loc[person_id, 'rt_disability'] -= self.daly_wt_brain_contusion
            if code == '133c':
                df.loc[person_id, 'rt_disability'] -= self.daly_wt_intraventricular_haemorrhage
            if code == '133d':
                df.loc[person_id, 'rt_disability'] -= self.daly_wt_subgaleal_hematoma
            if code == '134a':
                df.loc[person_id, 'rt_disability'] -= self.daly_wt_epidural_hematoma
            if code == '134b':
                df.loc[person_id, 'rt_disability'] -= self.daly_wt_subdural_hematoma
            if code == '135':
                df.loc[person_id, 'rt_disability'] -= self.daly_wt_diffuse_axonal_injury
            if code == '552' or code == '553' or code == '554':
                df.loc[person_id, 'rt_disability'] -= self.daly_wt_abd_internal_organ_injury
            if code == '342' or code == '343' or code == '361' or code == '363':
                df.loc[person_id, 'rt_disability'] -= self.daly_wt_neck_internal_bleeding
            if code == '414':
                df.loc[person_id, 'rt_disability'] -= self.daly_wt_flail_chest
            if code == '441':
                df.loc[person_id, 'rt_disability'] -= self.daly_wt_closed_pneumothorax
            if code == '443':
                df.loc[person_id, 'rt_disability'] -= self.daly_wt_open_pneumothorax
            if code == '453a':
                df.loc[person_id, 'rt_disability'] -= self.daly_wt_diaphragm_rupture
            if code == '453b':
                df.loc[person_id, 'rt_disability'] -= self.daly_wt_lung_contusion
            if code == '463':
                df.loc[person_id, 'rt_disability'] -= self.daly_wt_hemothorax
            if code == 'P782b':
                df.loc[person_id, 'rt_disability'] -= self.daly_wt_unilateral_arm_amputation_with_treatment
            if code == 'P783':
                df.loc[person_id, 'rt_disability'] -= self.daly_wt_bilateral_arm_amputation_with_treatment
            if code == 'P883':
                df.loc[person_id, 'rt_disability'] -= self.daly_wt_unilateral_lower_limb_amputation_with_treatment
            if code == 'P884':
                df.loc[person_id, 'rt_disability'] -= self.daly_wt_bilateral_lower_limb_amputation_with_treatment

        RTI.rti_treated_injuries(self, person_id, codes)
        DALYweightoverlimit = df.index[df['rt_disability'] < 0]
        df.loc[DALYweightoverlimit, 'rt_disability'] = 0

    def rti_swap_injury_daly_upon_treatment(self, person_id, codes):
        """
        This function swaps certain DALY weight codes upon when a person receives treatment(s). Some injuries have a
        different daly weight associated with them for the treated and untreated injuries. If an injury is 'swap-able'
        then this function removes the old daly weight for the untreated injury and gives the daly weight for the
        treated injury.

        :param person_id: The person who has received treatment
        :param codes: the 'swap-able' injury code
        :return: n/a
        """
        df = self.sim.population.props
        # Check the people that are sent here have had medical treatment
        assert df.loc[person_id, 'rt_med_int']
        # Check they are alive
        assert df.loc[person_id, 'is_alive']
        assert ~df.loc[person_id, 'rt_imm_death']
        # Check they have an appropriate injury code to swap

        swapping_codes = ['712b', '812', '3113', '4113', '5113', '7113', '8113', '813a', '813b', 'P673a', 'P673b',
                          'P674a', 'P674b', 'P675a', 'P675b', 'P676', 'P782b', 'P783', 'P883', 'P884']
        relevant_codes = np.intersect1d(codes, swapping_codes)
        cols = ['rt_injury_1', 'rt_injury_2', 'rt_injury_3', 'rt_injury_4', 'rt_injury_5', 'rt_injury_6',
                'rt_injury_7', 'rt_injury_8']
        person_injuries = df.loc[[person_id], cols]
        # check this person is injured, search they have an injury code that is swappable
        idx, counts = RTI.rti_find_and_count_injuries(person_injuries, relevant_codes)
        assert counts > 0, 'This person has asked to swap an injury code, but it is not swap-able'
        # Iterate over the relevant codes
        for code in relevant_codes:
            # swap the relevant code's daly weight, from the daly weight associated with the injury without treatment
            # and the daly weight for the disability with treatment.
            if code == '712b':
                df.loc[person_id, 'rt_disability'] += - self.daly_wt_hand_wrist_fracture_without_treatment + \
                                                      self.daly_wt_hand_wrist_fracture_with_treatment
            if code == '812':
                df.loc[person_id, 'rt_disability'] += - self.daly_wt_patella_tibia_fibula_fracture_without_treatment + \
                                                      self.daly_wt_patella_tibia_fibula_fracture_with_treatment
            if code == '3113':
                df.loc[person_id, 'rt_disability'] += \
                    - self.daly_wt_burns_less_than_20_percent_body_area_without_treatment \
                    + self.daly_wt_burns_less_than_20_percent_body_area_with_treatment
            if code == '4113':
                df.loc[person_id, 'rt_disability'] += \
                    - self.daly_wt_burns_less_than_20_percent_body_area_without_treatment \
                    + self.daly_wt_burns_less_than_20_percent_body_area_with_treatment
            if code == '5113':
                df.loc[person_id, 'rt_disability'] += \
                    - self.daly_wt_burns_less_than_20_percent_body_area_without_treatment \
                    + self.daly_wt_burns_less_than_20_percent_body_area_with_treatment
            if code == '7113':
                df.loc[person_id, 'rt_disability'] += \
                    - self.daly_wt_burns_less_than_20_percent_body_area_without_treatment \
                    + self.daly_wt_burns_less_than_20_percent_body_area_with_treatment
            if code == '8113':
                df.loc[person_id, 'rt_disability'] += \
                    - self.daly_wt_burns_less_than_20_percent_body_area_without_treatment \
                    + self.daly_wt_burns_less_than_20_percent_body_area_with_treatment
            if code == '813a':
                df.loc[person_id, 'rt_disability'] += - self.daly_wt_hip_fracture_short_term_with_without_treatment + \
                                                      self.daly_wt_hip_fracture_long_term_with_treatment
            if code == '813b':
                df.loc[person_id, 'rt_disability'] += - self.daly_wt_pelvis_fracture_short_term + \
                                                      self.daly_wt_pelvis_fracture_long_term
            if code == 'P673a':
                df.loc[person_id, 'rt_disability'] += - self.daly_wt_spinal_cord_lesion_neck_without_treatment + \
                                                      self.daly_wt_spinal_cord_lesion_neck_with_treatment
            if code == 'P673b':
                df.loc[person_id, 'rt_disability'] += - self.daly_wt_spinal_cord_lesion_below_neck_without_treatment + \
                                                      self.daly_wt_spinal_cord_lesion_below_neck_with_treatment
            if code == 'P674a':
                df.loc[person_id, 'rt_disability'] += - self.daly_wt_spinal_cord_lesion_neck_without_treatment + \
                                                      self.daly_wt_spinal_cord_lesion_neck_with_treatment
            if code == 'P674b':
                df.loc[person_id, 'rt_disability'] += - self.daly_wt_spinal_cord_lesion_below_neck_without_treatment + \
                                                      self.daly_wt_spinal_cord_lesion_below_neck_with_treatment
            if code == 'P675a':
                df.loc[person_id, 'rt_disability'] += - self.daly_wt_spinal_cord_lesion_neck_without_treatment + \
                                                      self.daly_wt_spinal_cord_lesion_neck_with_treatment
            if code == 'P675b':
                df.loc[person_id, 'rt_disability'] += - self.daly_wt_spinal_cord_lesion_below_neck_without_treatment + \
                                                      self.daly_wt_spinal_cord_lesion_below_neck_with_treatment
            if code == 'P676':
                df.loc[person_id, 'rt_disability'] += - self.daly_wt_spinal_cord_lesion_neck_without_treatment + \
                                                      self.daly_wt_spinal_cord_lesion_neck_with_treatment
            if code == 'P782b':
                df.loc[person_id, 'rt_disability'] += - self.daly_wt_unilateral_arm_amputation_without_treatment + \
                                                      self.daly_wt_unilateral_arm_amputation_with_treatment
            if code == 'P783':
                df.loc[person_id, 'rt_disability'] += - self.daly_wt_bilateral_arm_amputation_without_treatment + \
                                                      self.daly_wt_bilateral_arm_amputation_with_treatment
            if code == 'P883':
                df.loc[person_id, 'rt_disability'] += - self.daly_wt_unilateral_lower_limb_amputation_without_treatment \
                                                      + self.daly_wt_unilateral_lower_limb_amputation_with_treatment
            if code == 'P884':
                df.loc[person_id, 'rt_disability'] += - self.daly_wt_bilateral_lower_limb_amputation_without_treatment \
                                                      + self.daly_wt_bilateral_lower_limb_amputation_with_treatment
        assert df.loc[person_id, 'rt_disability'] > 0, 'No disability burden where there should be'

    def rti_determine_LOS(self, person_id):
        """
        This function determines the length of stay a person sent to the health care system will require, based on how
        severe their injuries are (determined by the person's ISS score). Currently I use data from China, but once a
        more appropriate source of data is found I can swap this over.
        :param person_id: The person who needs their LOS determined
        :return: the inpatient days required to treat this person (Their LOS)
        """
        p = self.parameters
        df = self.sim.population.props
        # Check that those sent here have been injured and did not die immediately
        assert (df.loc[person_id, 'rt_road_traffic_inc']) & (~df.loc[person_id, 'rt_imm_death']) & \
               (df.loc[person_id, 'is_alive'])
        # Load the parameters needed to determine the length of stay
        mean_los_ISS_less_than_4 = p['mean_los_ISS_less_than_4']
        sd_los_ISS_less_than_4 = p['sd_los_ISS_less_than_4']
        mean_los_ISS_4_to_8 = p['mean_los_ISS_4_to_8']
        sd_los_ISS_4_to_8 = p['sd_los_ISS_4_to_8']
        mean_los_ISS_9_to_15 = p['mean_los_ISS_9_to_15']
        sd_los_ISS_9_to_15 = p['sd_los_ISS_9_to_15']
        mean_los_ISS_16_to_24 = p['mean_los_ISS_16_to_24']
        sd_los_ISS_16_to_24 = p['sd_los_ISS_16_to_24']
        mean_los_ISS_more_than_25 = p['mean_los_ISS_more_than_25']
        sd_los_ISS_more_that_25 = p['sd_los_ISS_more_that_25']
        days_until_treatment_end = 0  # default value to be changed
        # Create the length of stays required for each ISS score boundaries and check that they are >=0
        if df.iloc[person_id]['rt_ISS_score'] < 4:
            inpatient_days_ISS_less_than_4 = int(self.rng.normal(mean_los_ISS_less_than_4,
                                                                 sd_los_ISS_less_than_4, 1))
            days_until_treatment_end = inpatient_days_ISS_less_than_4
        if 4 <= df.iloc[person_id]['rt_ISS_score'] < 9:
            inpatient_days_ISS_4_to_8 = int(self.rng.normal(mean_los_ISS_4_to_8,
                                                            sd_los_ISS_4_to_8, 1))
            days_until_treatment_end = inpatient_days_ISS_4_to_8
        if 9 <= df.iloc[person_id]['rt_ISS_score'] < 16:
            inpatient_days_ISS_9_to_15 = int(self.rng.normal(mean_los_ISS_9_to_15,
                                                             sd_los_ISS_9_to_15, 1))
            days_until_treatment_end = inpatient_days_ISS_9_to_15
        if 16 <= df.iloc[person_id]['rt_ISS_score'] < 25:
            inpatient_days_ISS_16_to_24 = int(self.rng.normal(mean_los_ISS_16_to_24,
                                                              sd_los_ISS_16_to_24, 1))
            days_until_treatment_end = inpatient_days_ISS_16_to_24
        if 25 <= df.iloc[person_id]['rt_ISS_score']:
            inpatient_days_ISS_more_than_25 = int(self.rng.normal(mean_los_ISS_more_than_25,
                                                                  sd_los_ISS_more_that_25, 1))
            days_until_treatment_end = inpatient_days_ISS_more_than_25
        if days_until_treatment_end < 0:
            days_until_treatment_end = 0
        # Return the LOS
        return days_until_treatment_end



    @staticmethod
    def rti_find_and_count_injuries(dataframe, tloinjcodes):
        """
        A function that searches a user given dataframe for a list of injuries (tloinjcodes). If the injury code is
        found in the dataframe, this function returns the index for who has the injury/injuries and the number of
        injuries found. This function works much faster if the dataframe is smaller, hence why the searched dataframe
        is a parameter in the function.

        :param dataframe: The dataframe to search for the tlo injury codes in
        :param tloinjcodes: The injury codes to search for in the data frame
        :return: the df index of who has the injuries and how many injuries in the search were found.
        """
        index = pd.Index([])
        counts = 0
        peoples_injuries = [item for sublist in dataframe.values.tolist() for item in sublist]
        relevant_codes = np.intersect1d(peoples_injuries, tloinjcodes)
        for code in relevant_codes:
            for col in dataframe.columns:
                # Find where a searched for injury code is in the columns, store the matches in counts
                counts += len(dataframe[dataframe[col] == code])
                if len(dataframe[dataframe[col] == code]) > 0:
                    # If you find a matching code, update the index to include the matching person
                    inj = dataframe.apply(lambda row: row.astype(str).str.contains(code).any(0), axis=1)
                    injidx = inj.index[inj]
                    index = index.union(injidx)
        return index, counts

    def rti_treated_injuries(self, person_id, tloinjcodes):
        """
        A function that takes a person with treated injuries and removes the injury code from the properties rt_injury_1
        to rt_injury_8
        :param person_id: The person who needs an injury code removed
        :param tloinjcodes: the injury code(s) to be removed
        :return: n/a
        """
        df = self.sim.population.props
        # Isolate the relevant injury information
        cols = ['rt_injury_1', 'rt_injury_2', 'rt_injury_3', 'rt_injury_4', 'rt_injury_5', 'rt_injury_6',
                'rt_injury_7', 'rt_injury_8']
        permanent_injuries = ['P133', 'P133a', 'P133b', 'P133c', 'P133d', 'P134', 'P134a', 'P134b', 'P135', 'P673',
                              'P673a', 'P673b', 'P674', 'P674a', 'P674b', 'P675', 'P675a', 'P675b', 'P676', 'P782a',
                              'P782b', 'P782c', 'P783', 'P882', 'P883', 'P884']
        person_injuries = df.loc[[person_id], cols]
        # Iterate over the codes
        for code in tloinjcodes:
            if code in permanent_injuries:
                # checks if the injury is permanent, if so the injury code is not removed.
                pass
            else:
                # Find which columns have treated injuries
                injury_cols = person_injuries.columns[(person_injuries.values == code).any(0)].tolist()
                # Reset the treated injury code to "none"
                df.loc[person_id, injury_cols] = "none"

    def on_birth(self, mother_id, child_id):
        """
        When a person is born this function sets up the default properties for the road traffic injuries module
        :param mother_id: The mother
        :param child_id: The newborn
        :return: n/a
        """
        self.sim.modules['RTI'].rti_set_default_properties("on_birth", child_id)
        df = self.sim.population.props
        df.at[child_id, 'rt_road_traffic_inc'] = False
        df.at[child_id, 'rt_inj_severity'] = "none"  # default: no one has been injured in a RTI
        df.at[child_id, 'rt_injury_1'] = "none"
        df.at[child_id, 'rt_injury_2'] = "none"
        df.at[child_id, 'rt_injury_3'] = "none"
        df.at[child_id, 'rt_injury_4'] = "none"
        df.at[child_id, 'rt_injury_5'] = "none"
        df.at[child_id, 'rt_injury_6'] = "none"
        df.at[child_id, 'rt_injury_7'] = "none"
        df.at[child_id, 'rt_injury_8'] = "none"
        df.at[child_id, 'rt_polytrauma'] = False
        df.at[child_id, 'rt_ISS_score'] = 0
        df.at[child_id, 'rt_imm_death'] = False
        df.at[child_id, 'rt_perm_disability'] = False
        df.at[child_id, 'rt_med_int'] = False  # default: no one has a had medical intervention
        df.at[child_id, 'rt_date_to_remove_daly'] = [pd.NaT] * 8
        df.at[child_id, 'rt_diagnosed'] = False
        df.at[child_id, 'rt_recovery_no_med'] = False  # default: no recovery without medical intervention
        df.at[child_id, 'rt_post_med_death'] = False  # default: no death after medical intervention
        df.at[child_id, 'rt_no_med_death'] = False
        df.at[child_id, 'rt_disability'] = 0  # default: no disability due to RTI
        df.at[child_id, 'rt_date_inj'] = pd.NaT
        df.at[child_id, 'rt_MAIS_military_score'] = 0
        df.at[child_id, 'rt_date_death_no_med'] = pd.NaT

    def on_hsi_alert(self, person_id, treatment_id):
        """
        This is called whenever there is an HSI event commissioned by one of the other disease modules.
        """
        logger.debug(
            'This is RTI, being alerted about a health system interaction person %d for: %s',
            person_id,
            treatment_id,
        )

    def report_daly_values(self):
        # This must send back a pd.Series or pd.DataFrame that reports on the average daly-weights that have been
        # experienced by persons in the previous month. Only rows for alive-persons must be returned.
        # The names of the series of columns is taken to be the label of the cause of this disability.
        # It will be recorded by the healthburden module as <ModuleName>_<Cause>.
        logger.debug('This is RTI reporting my daly values')
        df = self.sim.population.props
        disability_series_for_alive_persons = df.loc[df.is_alive, "rt_disability"]
        return disability_series_for_alive_persons

    def assign_injuries(self, number):
        """
        A function that can be called specifying the number of people affected by RTI injuries
         and provides outputs for the number of injuries each person experiences from a RTI event, the location of the
         injury, the TLO injury categories and the severity of the injuries. The severity of the injuries will then be
         used to calculate the injury severity score (ISS), which will then inform mortality and disability

        :param number: The number of people who need injuries assigned to them
        :return: injurydescription - a dataframe for the injury/injuries summarised in the TLO injury code form along
                                     with data on their ISS score, used for calculating mortality and whether they have
                                     polytrauma or not.

        """
        p = self.parameters
        # Import the distribution of injured body regions from the VIBES study
        number_of_injured_body_regions_distribution = p['number_of_injured_body_regions_distribution']
        # Create empty lists to store information on the person's injuries
        predinjlocs = []
        predinjsev = []
        predinjcat = []
        predinjiss = []
        predpolytrauma = []
        injlocstring = []
        injcatstring = []
        injaisstring = []
        # Iterate over the total number of injured people
        for n in range(0, number):

            # Reset the distribution of body regions which can injured for each iteration.
            injlocdist = p['injury_location_distribution']
            injlocdist = np.array(injlocdist)

            # Generate a random number which will decide how many injuries the person will have,

            ninj = self.rng.choice(number_of_injured_body_regions_distribution[0],
                                   p=number_of_injured_body_regions_distribution[1])

            # Create an empty vector which will store the injury locations (numerically coded using the
            # abbreviated injury scale coding system, where 1 corresponds to head, 2 to face, 3 to neck, 4 to
            # thorax, 5 to abdomen, 6 to spine, 7 to upper extremity and 8 to lower extremity
            allinjlocs = []
            # Create an empty vector to store the type of injury
            injcat = []
            # Create an empty vector which will store the severity of the injuries
            injais = []

            for j in range(0, ninj):
                # calculate the sum of the distribution for the remaining uninjured body regions
                upperlim = np.sum(injlocdist[-1:])
                # create a random number between 0 and the sum of the distribution for the remaining uninjured body
                # regions
                locinvector = self.rng.uniform(0, upperlim)
                # create a random variable which will determine the injury category
                cat = self.rng.uniform(0, 1)
                # create a random variable which will determine the severity of the injury
                severity = self.rng.uniform(0, 1)
                # create a base for the cumulative probability
                cprop = 0
                # iterate over all the potential injured body regions
                for k in range(0, len(injlocdist[0])):
                    # for the jth injury we calculate the cumulative frequency of injury location proportion and store
                    # it in cprop, this is essentially sampling from the injury location distribution.
                    injproprow = injlocdist[1, k]
                    cprop += injproprow
                    if cprop > locinvector:
                        injlocs = injlocdist[0, k]
                        # Once we find the region of the cumulative frequency of proportion of injury location
                        # loc falls in, we can determine use this to determine where the injury is located, the jth
                        # injury a person has is stored in injlocs initially and then injlocs is stored the vector
                        # allinjlocs and returned as an output at the end of the function
                        allinjlocs.append(int(injlocs))
                        injlocdist = np.delete(injlocdist, k, 1)
                        # In injury categories I will use the following mapping:
                        # Fracture - 1
                        # Dislocation - 2
                        # Traumatic brain injury - 3
                        # Soft tissue injury - 4
                        # Internal organ injury - 5
                        # Internal bleeding - 6
                        # Spinal cord injury - 7
                        # Amputation - 8
                        # Eye injury - 9
                        # Cuts etc (minor wounds) - 10
                        # Burns - 11

                        if injlocs == 1:
                            # Decide what the injury to the head is going to be:

                            # ask is it a skin wound
                            if cat <= self.head_prob_skin_wound:
                                # is it a laceration else it is a burn
                                if severity <= self.head_prob_skin_wound_open:
                                    # Open wound
                                    injcat.append(int(10))
                                    injais.append(1)
                                else:
                                    # Burn
                                    injcat.append(int(11))
                                    injais.append(4)
                                    logger.debug('gave a burn to head')
                                    logger.debug(severity)
                                    logger.debug(self.head_prob_skin_wound_open)

                            # Ask is it a skull fracture
                            elif self.head_prob_skin_wound < cat <= self.head_prob_skin_wound + self.head_prob_fracture:
                                # Skull fractures
                                injcat.append(int(1))
                                # ask how severe the skull fracture will be
                                if severity <= self.head_prob_fracture_unspecified:
                                    injais.append(2)
                                else:
                                    injais.append(3)
                            # Ask is it a traumatic brain injury
                            elif self.head_prob_skin_wound + self.head_prob_fracture < cat:
                                # Traumatic brain injuries
                                injcat.append(int(3))
                                # Decide how severe the traumatic brain injury will be
                                if severity <= self.head_prob_TBI_AIS3:
                                    # Mild TBI
                                    injais.append(3)
                                elif self.head_prob_TBI_AIS3 < severity <= self.head_prob_TBI_AIS3 + \
                                    self.head_prob_TBI_AIS4:
                                    # Moderate TBI
                                    injais.append(4)
                                elif self.head_prob_TBI_AIS3 + self.head_prob_TBI_AIS4 < severity:
                                    # Severe TBI
                                    injais.append(5)
                        if injlocs == 2:
                            # Decide what the injury to the face will be
                            # Ask will it be a skin wound
                            if cat <= self.face_prob_skin_wound:
                                # decide whether it will be a laceration or a burn
                                if severity <= self.face_prob_skin_wound_open:
                                    # Open wound
                                    injcat.append(int(10))
                                    injais.append(1)
                                else:
                                    # Burn
                                    injcat.append(int(11))
                                    injais.append(4)
                            # Ask if it is a facial fracture
                            elif self.face_prob_skin_wound < cat <= self.face_prob_skin_wound + self.face_prob_fracture:
                                # Facial fracture
                                injcat.append(int(1))
                                # decide how severe the injury will be
                                if severity <= self.face_prob_fracture_AIS1:
                                    # Nasal and unspecified fractures of the face
                                    injais.append(1)
                                else:
                                    # Mandible and Zygomatic fractures
                                    injais.append(2)
                            # Ask if it will be a soft tissue injury
                            elif self.face_prob_skin_wound + self.face_prob_fracture < cat < self.face_prob_skin_wound \
                                + self.face_prob_fracture + self.face_prob_soft_tissue_injury:
                                # soft tissue injury
                                injcat.append(int(4))
                                injais.append(1)
                            # If none of the above the injury is an eye injury
                            elif self.face_prob_skin_wound + self.face_prob_fracture + \
                                self.face_prob_soft_tissue_injury < cat:
                                # eye injury
                                injcat.append(int(9))
                                injais.append(1)
                        # Decide what the injury to the neck will be
                        if injlocs == 3:
                            # ask if the injury is a skin wound
                            if cat <= self.neck_prob_skin_wound:
                                # ask if it is a laceration, else it will be a burn
                                if severity <= self.neck_prob_skin_wound_open:
                                    # Open wound
                                    injcat.append(int(10))
                                    injais.append(1)
                                else:
                                    # Burn
                                    injcat.append(int(11))
                                    injais.append(3)
                            # Ask if the injury is a soft tissue injury
                            elif self.neck_prob_skin_wound < cat <= self.neck_prob_skin_wound + \
                                self.neck_prob_soft_tissue_injury:
                                # Soft tissue injuries of the neck
                                # decide how severe the injury is
                                injcat.append(int(4))
                                if severity <= self.neck_prob_soft_tissue_injury_AIS2:
                                    # Vertebral artery laceration
                                    injais.append(2)
                                else:
                                    # Pharynx contusion
                                    injais.append(3)
                            # Ask if the injury is internal bleeding
                            elif self.neck_prob_skin_wound + self.neck_prob_soft_tissue_injury < cat <= \
                                self.neck_prob_skin_wound + self.neck_prob_soft_tissue_injury + \
                                self.neck_prob_internal_bleeding:
                                # Internal bleeding
                                injcat.append(int(6))
                                # Decide how severe the injury will be
                                if severity <= self.neck_prob_internal_bleeding_AIS1:
                                    # Sternomastoid m. hemorrhage,
                                    # Hemorrhage, supraclavicular triangle
                                    # Hemorrhage, posterior triangle
                                    # Anterior vertebral vessel hemorrhage
                                    # Neck muscle hemorrhage
                                    injais.append(1)
                                else:
                                    # Hematoma in carotid sheath
                                    # Carotid sheath hemorrhage
                                    injais.append(3)
                            # Ask if the injury is a dislocation
                            elif self.neck_prob_skin_wound + self.neck_prob_soft_tissue_injury + \
                                self.neck_prob_internal_bleeding < cat:
                                # Dislocation
                                # Decide how severe the injury will be
                                injcat.append(int(2))
                                if severity <= self.neck_prob_dislocation_AIS3:
                                    # Atlanto-axial subluxation
                                    injais.append(3)
                                else:
                                    # Atlanto-occipital subluxation
                                    injais.append(2)
                        if injlocs == 4:
                            # Decide what the injury to the thorax will be
                            # Ask if the injury is a skin wound
                            if cat <= self.thorax_prob_skin_wound:
                                # Decide if the injury is a laceration or a burn
                                if severity <= self.thorax_prob_skin_wound_open:
                                    # Open wound
                                    injcat.append(int(10))
                                    injais.append(1)
                                else:
                                    # Burn
                                    injcat.append(int(11))
                                    injais.append(3)
                            # Decide if the injury is internal bleeding
                            elif self.thorax_prob_skin_wound < cat <= self.thorax_prob_skin_wound + \
                                self.thorax_prob_internal_bleeding:
                                # Internal Bleeding
                                injcat.append(int(6))
                                # Decide how severe the injury will be
                                if severity <= self.thorax_prob_internal_bleeding_AIS1:
                                    # Chest wall bruises/haematoma
                                    injais.append(1)
                                else:
                                    # Haemothorax
                                    injais.append(3)
                            # Decide if the injury is an internal organ injury
                            elif self.thorax_prob_skin_wound + self.thorax_prob_internal_bleeding < cat <= \
                                self.thorax_prob_skin_wound + self.thorax_prob_internal_bleeding + \
                                self.thorax_prob_internal_organ_injury:
                                # Internal organ injury
                                injcat.append(int(5))
                                # Lung contusion and Diaphragm rupture
                                injais.append(3)
                            # Decide if the injury is a fracture/flail chest
                            elif self.thorax_prob_skin_wound + self.thorax_prob_internal_bleeding + \
                                self.thorax_prob_internal_organ_injury < cat <= self.thorax_prob_skin_wound + \
                                self.thorax_prob_internal_bleeding + self.thorax_prob_internal_organ_injury + \
                                self.thorax_prob_fracture:
                                # Fractures to ribs and flail chest
                                injcat.append(int(1))
                                # Decide how severe the injury is
                                if severity <= self.thorax_prob_fracture_ribs:
                                    # fracture to rib(s)
                                    injais.append(2)
                                else:
                                    # flail chest
                                    injais.append(4)
                            # Decide if the injury is a soft tissue injury
                            elif self.thorax_prob_skin_wound + self.thorax_prob_internal_bleeding + \
                                self.thorax_prob_internal_organ_injury + self.thorax_prob_fracture < cat <= \
                                self.thorax_prob_skin_wound + self.thorax_prob_internal_bleeding + \
                                self.thorax_prob_internal_organ_injury + self.thorax_prob_fracture + \
                                self.thorax_prob_soft_tissue_injury:
                                # Soft tissue injury
                                injcat.append(int(4))
                                # Decide how severe the injury is
                                if severity <= self.thorax_prob_soft_tissue_injury_AIS1:
                                    # Chest wall lacerations/avulsions
                                    injais.append(1)
                                elif self.thorax_prob_soft_tissue_injury_AIS1 < severity <= \
                                    self.thorax_prob_soft_tissue_injury_AIS1 + self.thorax_prob_soft_tissue_injury_AIS2:
                                    # surgical emphysema
                                    injais.append(2)
                                else:
                                    # Open/closed pneumothorax
                                    injais.append(3)
                        # Decide what the injury to the abdomen will be
                        if injlocs == 5:
                            # Decide if it will be a skin wound, otherwise will be an internal organ injury
                            if cat <= self.abdomen_prob_skin_wound:
                                # Decide if the skin wound is a laceration or a burn
                                if severity <= self.abdomen_prob_skin_wound_open:
                                    # Open wound
                                    injcat.append(int(10))
                                    injais.append(1)
                                else:
                                    # Burn
                                    injcat.append(int(11))
                                    injais.append(3)
                            else:
                                # Internal organ injuries
                                injcat.append(int(5))
                                # Decide how severe the injury is
                                if severity <= self.abdomen_prob_internal_organ_injury_AIS2:
                                    # Intestines, Stomach and colon injury
                                    injais.append(2)
                                elif self.abdomen_prob_internal_organ_injury_AIS2 < severity <= \
                                    self.abdomen_prob_internal_organ_injury_AIS2 + \
                                    self.abdomen_prob_internal_organ_injury_AIS3:
                                    # Spleen, bladder, liver, urethra and diaphragm injury
                                    injais.append(3)
                                else:
                                    # Kidney injury
                                    injais.append(4)
                        # Decide what the injury to the spine will be
                        if injlocs == 6:
                            # Ask is it a vertebrae fracture
                            if cat <= self.spine_prob_fracture:
                                # Fracture to vertebrae
                                injcat.append(int(1))
                                injais.append(2)
                            # Ask if it is a spinal cord lesion
                            else:
                                # Spinal cord injury
                                injcat.append(int(7))

                                base1 = self.spine_prob_spinal_cord_lesion_neck_level * \
                                        self.spine_prob_spinal_cord_lesion_neck_level_AIS3 + \
                                        self.spine_prob_spinal_cord_lesion_below_neck_level * \
                                        self.spine_prob_spinal_cord_lesion_below_neck_level_AIS3
                                base2 = self.spine_prob_spinal_cord_lesion_neck_level * \
                                        self.spine_prob_spinal_cord_lesion_neck_level_AIS4 + \
                                        self.spine_prob_spinal_cord_lesion_below_neck_level * \
                                        self.spine_prob_spinal_cord_lesion_below_neck_level_AIS4
                                base3 = self.spine_prob_spinal_cord_lesion_neck_level * \
                                        self.spine_prob_spinal_cord_lesion_neck_level_AIS5 + \
                                        self.spine_prob_spinal_cord_lesion_below_neck_level * \
                                        self.spine_prob_spinal_cord_lesion_below_neck_level_AIS5
                                # Decide how severe the injury is
                                if severity <= base1:
                                    injais.append(3)
                                elif base1 < cat <= base1 + base2:
                                    injais.append(4)
                                elif base1 + base2 < cat <= base1 + base2 + base3:
                                    injais.append(5)
                                else:
                                    injais.append(6)
                        # Decide What the injury to the upper extremities is
                        if injlocs == 7:
                            # Decide if the injury will be a skin wound
                            if cat <= self.upper_ex_prob_skin_wound:
                                # Decide if the injury will be a laceration or a burn
                                if severity <= self.upper_ex_prob_skin_wound_open:
                                    # Open wound
                                    injcat.append(int(10))
                                    injais.append(1)
                                else:
                                    # Burn
                                    injcat.append(int(11))
                                    injais.append(3)
                            # Decide if the injury is a fracture
                            elif self.upper_ex_prob_skin_wound < cat <= self.upper_ex_prob_skin_wound + \
                                self.upper_ex_prob_fracture:
                                # Fracture to arm
                                injcat.append(int(1))
                                injais.append(2)
                            # Decide if the injury is a dislocation
                            elif self.upper_ex_prob_skin_wound + self.upper_ex_prob_fracture < cat <= \
                                self.upper_ex_prob_skin_wound + self.upper_ex_prob_fracture + \
                                self.upper_ex_prob_dislocation:
                                # Dislocation to arm
                                injcat.append(int(2))
                                injais.append(2)
                            # Decide if the injury is an amputation
                            elif self.upper_ex_prob_skin_wound + self.upper_ex_prob_fracture + \
                                self.upper_ex_prob_dislocation < cat:
                                # Amputation in upper limb
                                injcat.append(int(8))
                                # Decide how severe the injury will be
                                if severity <= self.upper_ex_prob_amputation_AIS2:
                                    # Amputation to finger/thumb/unilateral arm
                                    injais.append(2)
                                else:
                                    # Amputation, arm, bilateral
                                    injais.append(3)
                        # Decide what the injury to the lower extremity is
                        if injlocs == 8:
                            # Decide if the injury is a skin wound
                            if cat <= self.lower_ex_prob_skin_wound:
                                # decide if the injury is a laceration or a burn
                                if severity <= self.lower_ex_prob_skin_wound_open:
                                    # Open wound
                                    injcat.append(int(10))
                                    injais.append(1)
                                else:
                                    # Burn
                                    injcat.append(int(11))
                                    injais.append(3)
                            # Decide if the injury is a fracture
                            elif self.lower_ex_prob_skin_wound < cat <= self.lower_ex_prob_skin_wound + \
                                self.lower_ex_prob_fracture:
                                # Fractures
                                injcat.append(int(1))
                                # Decide how severe the injury will be
                                if severity < self.lower_ex_prob_fracture_AIS1:
                                    # Foot fracture
                                    injais.append(1)
                                elif self.lower_ex_prob_fracture_AIS1 < severity <= self.lower_ex_prob_fracture_AIS1 + \
                                    self.lower_ex_prob_fracture_AIS2:
                                    # Lower leg fracture
                                    injais.append(2)
                                else:
                                    # Upper leg fracture
                                    injais.append(3)
                            # Decide if the injury is a dislocation
                            elif self.lower_ex_prob_skin_wound + self.lower_ex_prob_fracture < cat <= \
                                self.lower_ex_prob_skin_wound + self.lower_ex_prob_fracture + \
                                self.lower_ex_prob_dislocation:
                                # dislocation of hip or knee
                                injcat.append(int(2))
                                injais.append(2)
                            # Decide if the injury is an amputation
                            elif self.lower_ex_prob_skin_wound + self.lower_ex_prob_fracture + \
                                self.lower_ex_prob_dislocation < cat:
                                # Amputations
                                injcat.append(int(8))
                                # Decide how severe the injury is
                                if severity <= self.lower_ex_prob_amputation_AIS2:
                                    # Toe/toes amputation
                                    injais.append(2)
                                elif self.lower_ex_prob_amputation_AIS2 < severity <= \
                                    self.lower_ex_prob_amputation_AIS2 + self.lower_ex_prob_amputation_AIS3:
                                    # Unilateral limb amputation
                                    injais.append(3)
                                else:
                                    # Bilateral limb amputation
                                    injais.append(4)

                        break

            # Create a dataframe that stores the injury location and severity for each person, the point of this
            # dataframe is to use some of the pandas tools to manipulate the generated injury data to calculate
            # the ISS score and from this, the probability of mortality resulting from the injuries.
            injlocstring.append(' '.join(map(str, allinjlocs)))
            injcatstring.append(' '.join(map(str, injcat)))
            injaisstring.append(' '.join(map(str, injais)))
            injdata = {'AIS location': allinjlocs, 'AIS severity': injais}
            df = pd.DataFrame(injdata, columns=['AIS location', 'AIS severity'])
            # Find the most severe injury to the person in each body region, creates a new column containing the
            # maximum AIS value of each injured body region
            df['Severity max'] = df.groupby(['AIS location'], sort=False)['AIS severity'].transform(max)
            # column no longer needed and will get in the way of future calculations
            df = df.drop(columns='AIS severity')
            # drops the duplicate values in the location data, preserving the most severe injuries in each body
            # location.
            df = df.drop_duplicates(['AIS location'], keep='first')
            # Finds the AIS score for the most severely injured body regions and stores them in a new dataframe z
            z = df.nlargest(3, 'Severity max', 'first')
            # Find the 3 most severely injured body regions
            z = z.iloc[:3]
            # Need to determine whether the persons injuries qualify as polytrauma as such injuries have a different
            # prognosis, set default as False. Polytrauma is defined via the new Berlin definition, 'when two or more
            # injuries have an AIS severity score of 3 or higher'.
            polytrauma = False
            # Determine where more than one injured body region has occurred
            if len(z) > 1:
                # Find where the injuries have an AIS score of 3 or higher
                cond = np.where(z['Severity max'] > 2)
                if len(z.iloc[cond]) > 1:
                    # if two or more injuries have a AIS score of 3 or higher then this person has polytrauma.
                    polytrauma = True
            # Calculate the squares of the AIS scores for the three most severely injured body regions
            z['sqrsev'] = z['Severity max'] ** 2
            # From the squared AIS scores, calculate the ISS score
            ISSscore = sum(z['sqrsev'])
            # Turn the vectors into a string to store as one entry in a dataframe
            allinjlocs = np.array(allinjlocs)
            allinjlocs = allinjlocs.astype(int)
            allinjlocs = ''.join([str(elem) for elem in allinjlocs])
            predinjlocs.append(allinjlocs)
            predinjsev.append(injais)
            predinjcat.append(injcat)
            predinjiss.append(ISSscore)
            predpolytrauma.append(polytrauma)
        injdf = pd.DataFrame()
        injdf['Injury locations'] = predinjlocs
        injdf['Injury locations string'] = injlocstring
        injdf['Injury AIS'] = predinjsev
        MAIS = []
        for item in injdf['Injury AIS'].tolist():
            MAIS.append(max(item) + 1)
        injdf['Injury AIS string'] = injaisstring
        injdf['Injury category'] = predinjcat
        injdf['Injury category string'] = injcatstring
        injdf['ISS'] = predinjiss
        injdf['Polytrauma'] = predpolytrauma
        injdf['Injury category string'] = injdf['Injury category string'].astype(str)
        injurycategories = injdf['Injury category string'].str.split(expand=True)
        injdf['Injury locations string'] = injdf['Injury locations string'].astype(str)
        injurylocations = injdf['Injury locations string'].str.split(expand=True)
        injdf['Injury AIS string'] = injdf['Injury AIS string'].astype(str)
        injuryais = injdf['Injury AIS string'].str.split(expand=True)
        injurydescription = injurylocations + injurycategories + injuryais
        injurydescription = injurydescription.astype(str)
        for (columnname, columndata) in injurydescription.iteritems():
            injurydescription.rename(
                columns={injurydescription.columns[columnname]: "Injury " + str(columnname + 1)},
                inplace=True)
        injurydescription['ISS'] = predinjiss
        injurydescription['Polytrauma'] = predpolytrauma
        injurydescription['MAIS_M'] = MAIS
        injurydescription = injurydescription.fillna("none")
        return injurydescription


# ---------------------------------------------------------------------------------------------------------
#   DISEASE MODULE EVENTS
#
#   These are the events which drive the simulation of the disease. It may be a regular event that updates
#   the status of all the population of subsections of it at one time. There may also be a set of events
#   that represent disease events for particular persons.
# ---------------------------------------------------------------------------------------------------------

class RTI_Event(RegularEvent, PopulationScopeEventMixin):
    """The regular RTI event which handles all the initial RTI related changes to the dataframe. It can be thought of
     as the actual road traffic accident occuring. Specifically the event  decides who is involved in a road traffic
     accident every month (via the linear model helper class), whether those involved in a road traffic accident die on
     scene or are given injuries (via the assign_injuries function) which they will attempt to interact with the health
     system with for treatment.

     Those who don't die on scene and are injured then attempt to go to an emergency generic first appointment

    This event will change the rt_ properties:
    1) rt_road_traffic_inc - False when not involved in a collision, True when RTI_Event decides they are in a collision

    2) rt_date_inj - Change to current date if the person has been involved in a road traffic accident

    3) rt_imm_death - True if they die on the scene of the crash, false otherwise

    4) rt_injury_1 through to rt_injury_8 - a series of 8 properties which stores the injuries that need treating as a
                                            code

    5) rt_ISS_score - The metric used to calculate the probability of mortality from the person's injuries

    6) rt_disability - after injuries are assigned to a person, RTI_event calls rti_assign_daly_weights to match the
                       person's injury codes in rt_injury_1 through 8 to their corresponding DALY weights

    7) rt_polytrauma - If the person's injuries fit the definition for polytrauma we keep track of this here and use it
                        to calculate the probability for mortality later on.


    """

    def __init__(self, module):
        """Schedule to take place every month
        """
        super().__init__(module, frequency=DateOffset(months=1))
        p = module.parameters
        # Parameters which transition the model between states
        self.base_1m_prob_rti = (p['base_rate_injrti'] / 12)
        if 'reduce_incidence' in p['allowed_interventions']:
            self.base_1m_prob_rti = self.base_1m_prob_rti * 0.335
        self.rr_injrti_age018 = p['rr_injrti_age018']
        self.rr_injrti_age1829 = p['rr_injrti_age1829']
        self.rr_injrti_age3039 = p['rr_injrti_age3039']
        self.rr_injrti_age4049 = p['rr_injrti_age4049']
        self.rr_injrti_age5059 = p['rr_injrti_age5059']
        self.rr_injrti_age6069 = p['rr_injrti_age6069']
        self.rr_injrti_age7079 = p['rr_injrti_age7079']
        self.rr_injrti_male = p['rr_injrti_male']
        self.rr_injrti_excessalcohol = p['rr_injrti_excessalcohol']
        self.imm_death_proportion_rti = p['imm_death_proportion_rti']
        self.prob_perm_disability_with_treatment_severe_TBI = p['prob_perm_disability_with_treatment_severe_TBI']
        self.prob_perm_disability_with_treatment_sci = p['prob_perm_disability_with_treatment_sci']

        # Parameters used to assign injuries in the injrandomizer function
        # Injuries to AIS region 1
        self.head_prob_skin_wound = p['head_prob_skin_wound']
        self.head_prob_skin_wound_open = p['head_prob_skin_wound_open']
        self.head_prob_skin_wound_burn = p['head_prob_skin_wound_burn']
        self.head_prob_fracture = p['head_prob_fracture']
        self.head_prob_fracture_unspecified = p['head_prob_fracture_unspecified']
        self.head_prob_fracture_basilar = p['head_prob_fracture_basilar']
        self.head_prob_TBI = p['head_prob_TBI']
        self.head_prob_TBI_AIS3 = p['head_prob_TBI_AIS3']
        self.head_prob_TBI_AIS4 = p['head_prob_TBI_AIS4']
        self.head_prob_TBI_AIS5 = p['head_prob_TBI_AIS5']
        # Injuries to AIS region 2
        self.face_prob_skin_wound = p['face_prob_skin_wound']
        self.face_prob_skin_wound_open = p['face_prob_skin_wound_open']
        self.face_prob_skin_wound_burn = p['face_prob_skin_wound_burn']
        self.face_prob_fracture = p['face_prob_fracture']
        self.face_prob_fracture_AIS1 = p['face_prob_fracture_AIS1']
        self.face_prob_fracture_AIS2 = p['face_prob_fracture_AIS2']
        self.face_prob_soft_tissue_injury = p['face_prob_soft_tissue_injury']
        self.face_prob_eye_injury = p['face_prob_eye_injury']
        # Injuries to AIS region 3
        self.neck_prob_skin_wound = p['neck_prob_skin_wound']
        self.neck_prob_skin_wound_open = p['neck_prob_skin_wound_open']
        self.neck_prob_skin_wound_burn = p['neck_prob_skin_wound_burn']
        self.neck_prob_soft_tissue_injury = p['neck_prob_soft_tissue_injury']
        self.neck_prob_soft_tissue_injury_AIS2 = p['neck_prob_soft_tissue_injury_AIS2']
        self.neck_prob_soft_tissue_injury_AIS3 = p['neck_prob_soft_tissue_injury_AIS3']
        self.neck_prob_internal_bleeding = p['neck_prob_internal_bleeding']
        self.neck_prob_internal_bleeding_AIS1 = p['neck_prob_internal_bleeding_AIS1']
        self.neck_prob_internal_bleeding_AIS3 = p['neck_prob_internal_bleeding_AIS3']
        self.neck_prob_dislocation = p['neck_prob_dislocation']
        self.neck_prob_dislocation_AIS2 = p['neck_prob_dislocation_AIS2']
        self.neck_prob_dislocation_AIS3 = p['neck_prob_dislocation_AIS3']
        # Injuries to AIS region 4
        self.thorax_prob_skin_wound = p['thorax_prob_skin_wound']
        self.thorax_prob_skin_wound_open = p['thorax_prob_skin_wound_open']
        self.thorax_prob_skin_wound_burn = p['thorax_prob_skin_wound_burn']
        self.thorax_prob_internal_bleeding = p['thorax_prob_internal_bleeding']
        self.thorax_prob_internal_bleeding_AIS1 = p['thorax_prob_internal_bleeding_AIS1']
        self.thorax_prob_internal_bleeding_AIS3 = p['thorax_prob_internal_bleeding_AIS3']
        self.thorax_prob_internal_organ_injury = p['thorax_prob_internal_organ_injury']
        self.thorax_prob_fracture = p['thorax_prob_fracture']
        self.thorax_prob_fracture_ribs = p['thorax_prob_fracture_ribs']
        self.thorax_prob_fracture_flail_chest = p['thorax_prob_fracture_flail_chest']
        self.thorax_prob_soft_tissue_injury = p['thorax_prob_soft_tissue_injury']
        self.thorax_prob_soft_tissue_injury_AIS1 = p['thorax_prob_soft_tissue_injury_AIS1']
        self.thorax_prob_soft_tissue_injury_AIS2 = p['thorax_prob_soft_tissue_injury_AIS2']
        self.thorax_prob_soft_tissue_injury_AIS3 = p['thorax_prob_soft_tissue_injury_AIS3']
        # Injuries to AIS region 5
        self.abdomen_prob_skin_wound = p['abdomen_prob_skin_wound']
        self.abdomen_prob_skin_wound_open = p['abdomen_prob_skin_wound_open']
        self.abdomen_prob_skin_wound_burn = p['abdomen_prob_skin_wound_burn']
        self.abdomen_prob_internal_organ_injury = p['abdomen_prob_internal_organ_injury']
        self.abdomen_prob_internal_organ_injury_AIS2 = p['abdomen_prob_internal_organ_injury_AIS2']
        self.abdomen_prob_internal_organ_injury_AIS3 = p['abdomen_prob_internal_organ_injury_AIS3']
        self.abdomen_prob_internal_organ_injury_AIS4 = p['abdomen_prob_internal_organ_injury_AIS4']
        # Injuries to AIS region 6
        self.spine_prob_spinal_cord_lesion = p['spine_prob_spinal_cord_lesion']
        self.spine_prob_spinal_cord_lesion_neck_level = p['spine_prob_spinal_cord_lesion_neck_level']
        self.spine_prob_spinal_cord_lesion_neck_level_AIS3 = p['spine_prob_spinal_cord_lesion_neck_level_AIS3']
        self.spine_prob_spinal_cord_lesion_neck_level_AIS4 = p['spine_prob_spinal_cord_lesion_neck_level_AIS4']
        self.spine_prob_spinal_cord_lesion_neck_level_AIS5 = p['spine_prob_spinal_cord_lesion_neck_level_AIS5']
        self.spine_prob_spinal_cord_lesion_neck_level_AIS6 = p['spine_prob_spinal_cord_lesion_neck_level_AIS6']
        self.spine_prob_spinal_cord_lesion_below_neck_level = p['spine_prob_spinal_cord_lesion_below_neck_level']
        self.spine_prob_spinal_cord_lesion_below_neck_level_AIS3 = \
            p['spine_prob_spinal_cord_lesion_below_neck_level_AIS3']
        self.spine_prob_spinal_cord_lesion_below_neck_level_AIS4 = \
            p['spine_prob_spinal_cord_lesion_below_neck_level_AIS4']
        self.spine_prob_spinal_cord_lesion_below_neck_level_AIS5 = \
            p['spine_prob_spinal_cord_lesion_below_neck_level_AIS5']
        self.spine_prob_fracture = p['spine_prob_fracture']
        # Injuries to AIS region 7
        self.upper_ex_prob_skin_wound = p['upper_ex_prob_skin_wound']
        self.upper_ex_prob_skin_wound_open = p['upper_ex_prob_skin_wound_open']
        self.upper_ex_prob_skin_wound_burn = p['upper_ex_prob_skin_wound_burn']
        self.upper_ex_prob_fracture = p['upper_ex_prob_fracture']
        self.upper_ex_prob_dislocation = p['upper_ex_prob_dislocation']
        self.upper_ex_prob_amputation = p['upper_ex_prob_amputation']
        self.upper_ex_prob_amputation_AIS2 = p['upper_ex_prob_amputation_AIS2']
        self.upper_ex_prob_amputation_AIS3 = p['upper_ex_prob_amputation_AIS3']
        # Injuries to AIS region 8
        self.lower_ex_prob_skin_wound = p['lower_ex_prob_skin_wound']
        self.lower_ex_prob_skin_wound_open = p['lower_ex_prob_skin_wound_open']
        self.lower_ex_prob_skin_wound_burn = p['lower_ex_prob_skin_wound_burn']
        self.lower_ex_prob_fracture = p['lower_ex_prob_fracture']
        self.lower_ex_prob_fracture_AIS1 = p['lower_ex_prob_fracture_AIS1']
        self.lower_ex_prob_fracture_AIS2 = p['lower_ex_prob_fracture_AIS2']
        self.lower_ex_prob_fracture_AIS3 = p['lower_ex_prob_fracture_AIS3']
        self.lower_ex_prob_dislocation = p['lower_ex_prob_dislocation']
        self.lower_ex_prob_amputation = p['lower_ex_prob_amputation']
        self.lower_ex_prob_amputation_AIS2 = p['lower_ex_prob_amputation_AIS2']
        self.lower_ex_prob_amputation_AIS3 = p['lower_ex_prob_amputation_AIS3']
        self.lower_ex_prob_amputation_AIS4 = p['lower_ex_prob_amputation_AIS3']

        # DALY weights
        self.daly_wt_unspecified_skull_fracture = p['daly_wt_unspecified_skull_fracture']
        self.daly_wt_basilar_skull_fracture = p['daly_wt_basilar_skull_fracture']
        self.daly_wt_epidural_hematoma = p['daly_wt_epidural_hematoma']
        self.daly_wt_subdural_hematoma = p['daly_wt_subdural_hematoma']
        self.daly_wt_subarachnoid_hematoma = p['daly_wt_subarachnoid_hematoma']
        self.daly_wt_brain_contusion = p['daly_wt_brain_contusion']
        self.daly_wt_intraventricular_haemorrhage = p['daly_wt_intraventricular_haemorrhage']
        self.daly_wt_diffuse_axonal_injury = p['daly_wt_diffuse_axonal_injury']
        self.daly_wt_subgaleal_hematoma = p['daly_wt_subgaleal_hematoma']
        self.daly_wt_midline_shift = p['daly_wt_midline_shift']
        self.daly_wt_facial_fracture = p['daly_wt_facial_fracture']
        self.daly_wt_facial_soft_tissue_injury = p['daly_wt_facial_soft_tissue_injury']
        self.daly_wt_eye_injury = p['daly_wt_eye_injury']
        self.daly_wt_neck_soft_tissue_injury = p['daly_wt_neck_soft_tissue_injury']
        self.daly_wt_neck_internal_bleeding = p['daly_wt_neck_internal_bleeding']
        self.daly_wt_neck_dislocation = p['daly_wt_neck_dislocation']
        self.daly_wt_chest_wall_bruises_hematoma = p['daly_wt_chest_wall_bruises_hematoma']
        self.daly_wt_hemothorax = p['daly_wt_hemothorax']
        self.daly_wt_lung_contusion = p['daly_wt_lung_contusion']
        self.daly_wt_diaphragm_rupture = p['daly_wt_diaphragm_rupture']
        self.daly_wt_rib_fracture = p['daly_wt_rib_fracture']
        self.daly_wt_flail_chest = p['daly_wt_flail_chest']
        self.daly_wt_chest_wall_laceration = p['daly_wt_chest_wall_laceration']
        self.daly_wt_closed_pneumothorax = p['daly_wt_closed_pneumothorax']
        self.daly_wt_open_pneumothorax = p['daly_wt_open_pneumothorax']
        self.daly_wt_surgical_emphysema = p['daly_wt_surgical_emphysema']
        self.daly_wt_abd_internal_organ_injury = p['daly_wt_abd_internal_organ_injury']
        self.daly_wt_spinal_cord_lesion_neck_with_treatment = p['daly_wt_spinal_cord_lesion_neck_with_treatment']
        self.daly_wt_spinal_cord_lesion_neck_without_treatment = p['daly_wt_spinal_cord_lesion_neck_without_treatment']
        self.daly_wt_spinal_cord_lesion_below_neck_with_treatment = p[
            'daly_wt_spinal_cord_lesion_below_neck_with_treatment']
        self.daly_wt_spinal_cord_lesion_below_neck_without_treatment = p[
            'daly_wt_spinal_cord_lesion_below_neck_without_treatment']
        self.daly_wt_vertebrae_fracture = p['daly_wt_vertebrae_fracture']
        self.daly_wt_clavicle_scapula_humerus_fracture = p['daly_wt_clavicle_scapula_humerus_fracture']
        self.daly_wt_hand_wrist_fracture_with_treatment = p['daly_wt_hand_wrist_fracture_with_treatment']
        self.daly_wt_hand_wrist_fracture_without_treatment = p['daly_wt_hand_wrist_fracture_without_treatment']
        self.daly_wt_radius_ulna_fracture_short_term_with_without_treatment = p[
            'daly_wt_radius_ulna_fracture_short_term_with_without_treatment']
        self.daly_wt_radius_ulna_fracture_long_term_without_treatment = p[
            'daly_wt_radius_ulna_fracture_long_term_without_treatment']
        self.daly_wt_dislocated_shoulder = p['daly_wt_dislocated_shoulder']
        self.daly_wt_amputated_finger = p['daly_wt_amputated_finger']
        self.daly_wt_amputated_thumb = p['daly_wt_amputated_thumb']
        self.daly_wt_unilateral_arm_amputation_with_treatment = p['daly_wt_unilateral_arm_amputation_with_treatment']
        self.daly_wt_unilateral_arm_amputation_without_treatment = p[
            'daly_wt_unilateral_arm_amputation_without_treatment']
        self.daly_wt_bilateral_arm_amputation_with_treatment = p['daly_wt_bilateral_arm_amputation_with_treatment']
        self.daly_wt_bilateral_arm_amputation_without_treatment = p[
            'daly_wt_bilateral_arm_amputation_without_treatment']
        self.daly_wt_foot_fracture_short_term_with_without_treatment = p[
            'daly_wt_foot_fracture_short_term_with_without_treatment']
        self.daly_wt_foot_fracture_long_term_without_treatment = p['daly_wt_foot_fracture_long_term_without_treatment']
        self.daly_wt_patella_tibia_fibula_fracture_with_treatment = p[
            'daly_wt_patella_tibia_fibula_fracture_with_treatment']
        self.daly_wt_patella_tibia_fibula_fracture_without_treatment = p[
            'daly_wt_patella_tibia_fibula_fracture_without_treatment']
        self.daly_wt_hip_fracture_short_term_with_without_treatment = p[
            'daly_wt_hip_fracture_short_term_with_without_treatment']
        self.daly_wt_hip_fracture_long_term_with_treatment = p['daly_wt_hip_fracture_long_term_with_treatment']
        self.daly_wt_hip_fracture_long_term_without_treatment = p['daly_wt_hip_fracture_long_term_without_treatment']
        self.daly_wt_pelvis_fracture_short_term = p['daly_wt_pelvis_fracture_short_term']
        self.daly_wt_pelvis_fracture_long_term = p['daly_wt_pelvis_fracture_long_term']
        self.daly_wt_femur_fracture_short_term = p['daly_wt_femur_fracture_short_term']
        self.daly_wt_femur_fracture_long_term_without_treatment = p[
            'daly_wt_femur_fracture_long_term_without_treatment']
        self.daly_wt_dislocated_hip = p['daly_wt_dislocated_hip']
        self.daly_wt_dislocated_knee = p['daly_wt_dislocated_knee']
        self.daly_wt_amputated_toes = p['daly_wt_amputated_toes']
        self.daly_wt_unilateral_lower_limb_amputation_with_treatment = p[
            'daly_wt_unilateral_lower_limb_amputation_with_treatment']
        self.daly_wt_unilateral_lower_limb_amputation_without_treatment = p[
            'daly_wt_unilateral_lower_limb_amputation_without_treatment']
        self.daly_wt_bilateral_lower_limb_amputation_with_treatment = p[
            'daly_wt_bilateral_lower_limb_amputation_with_treatment']
        self.daly_wt_bilateral_lower_limb_amputation_without_treatment = p[
            'daly_wt_bilateral_lower_limb_amputation_without_treatment']
        self.daly_wt_burns_greater_than_20_percent_body_area = p['daly_wt_burns_greater_than_20_percent_body_area']
        self.daly_wt_burns_less_than_20_percent_body_area_with_treatment = p[
            'daly_wt_burns_less_than_20_percent_body_area_with_treatment']
        self.daly_wt_burns_less_than_20_percent_body_area_without_treatment = p[
            'daly_wt_burns_less_than_20_percent_body_area_with_treatment']

    def apply(self, population):
        """Apply this event to the population.

        :param population: the current population
        """
        df = population.props
        now = self.sim.date
        # Reset injury properties after death
        immdeathidx = df.index[df.rt_imm_death]
        deathwithmedidx = df.index[df.rt_post_med_death]
        deathwithoutmedidx = df.index[df.rt_no_med_death]
        diedfromrtiidx = immdeathidx.union(deathwithmedidx)
        diedfromrtiidx = diedfromrtiidx.union(deathwithoutmedidx)
        df.loc[diedfromrtiidx, "rt_imm_death"] = False
        df.loc[diedfromrtiidx, "rt_post_med_death"] = False
        df.loc[diedfromrtiidx, "rt_no_med_death"] = False
        df.loc[diedfromrtiidx, "rt_disability"] = 0
        df.loc[diedfromrtiidx, "rt_med_int"] = False
        for index, row in df.loc[diedfromrtiidx].iterrows():
            df.at[index, 'rt_date_to_remove_daly'] = [pd.NaT] * 8
        df.loc[diedfromrtiidx, "rt_diagnosed"] = False
        df.loc[diedfromrtiidx, "rt_polytrauma"] = False
        df.loc[diedfromrtiidx, "rt_inj_severity"] = "none"
        df.loc[diedfromrtiidx, "rt_perm_disability"] = False
        df.loc[diedfromrtiidx, "rt_injury_1"] = "none"
        df.loc[diedfromrtiidx, "rt_injury_2"] = "none"
        df.loc[diedfromrtiidx, "rt_injury_3"] = "none"
        df.loc[diedfromrtiidx, "rt_injury_4"] = "none"
        df.loc[diedfromrtiidx, "rt_injury_5"] = "none"
        df.loc[diedfromrtiidx, "rt_injury_6"] = "none"
        df.loc[diedfromrtiidx, "rt_injury_7"] = "none"
        df.loc[diedfromrtiidx, "rt_injury_8"] = "none"
        df.loc[diedfromrtiidx, 'rt_date_death_no_med'] = pd.NaT
        df.loc[diedfromrtiidx, 'rt_MAIS_military_score'] = 0
        # reset whether they have been selected for an injury this month
        df['rt_road_traffic_inc'] = False
        # reset whether they have sought care this month
        df['rt_diagnosed'] = False
        df.loc[df.is_alive, 'rt_post_med_death'] = False
        df.loc[df.is_alive, 'rt_no_med_death'] = False

        # --------------------------------- UPDATING OF RTI OVER TIME -------------------------------------------------
        # Currently we have the following conditions for being able to be involved in a road traffic injury, they are
        # alive, they aren't currently injured, they didn't die immediately in
        # a road traffic injury in the last month and finally, they aren't currently being treated for a road traffic
        # injury.
        rt_current_non_ind = df.index[df.is_alive & ~df.rt_road_traffic_inc & ~df.rt_imm_death & ~df.rt_med_int &
                                      (df.rt_inj_severity == "none")]

        # ========= Update for people currently not involved in a RTI, make some involved in a RTI event ==============
        # Use linear model helper class
        eq = LinearModel(LinearModelType.MULTIPLICATIVE,
                         self.base_1m_prob_rti,
                         Predictor('sex').when('M', self.rr_injrti_male),
                         Predictor('age_years').when('.between(0,18)', self.rr_injrti_age018),
                         Predictor('age_years').when('.between(18,29)', self.rr_injrti_age1829),
                         Predictor('age_years').when('.between(30,39)', self.rr_injrti_age3039),
                         Predictor('age_years').when('.between(40,49)', self.rr_injrti_age4049),
                         Predictor('age_years').when('.between(50,59)', self.rr_injrti_age5059),
                         Predictor('age_years').when('.between(60,69)', self.rr_injrti_age6069),
                         Predictor('age_years').when('.between(70,79)', self.rr_injrti_age7079),
                         Predictor('li_ex_alc').when(True, self.rr_injrti_excessalcohol)
                         )
        pred = eq.predict(df.iloc[rt_current_non_ind])
        random_draw_in_rti = self.module.rng.random_sample(size=len(rt_current_non_ind))
        selected_for_rti = rt_current_non_ind[pred > random_draw_in_rti]
        # Update to say they have been involved in a rti

        df.loc[selected_for_rti, 'rt_road_traffic_inc'] = True
        # ========================= Take those involved in a RTI and assign some to death ==============================
        for person in selected_for_rti:
            logger.debug('Person %d has been involved in a road traffic accident on date: %s', person, self.sim.date)
        df.loc[selected_for_rti, 'rt_date_inj'] = now
        selected_to_die = selected_for_rti[self.imm_death_proportion_rti >
                                           self.module.rng.random_sample(size=len(selected_for_rti))]
        df.loc[selected_to_die, 'rt_imm_death'] = True

        for individual_id in selected_to_die:
            logger.debug('Person %d has immediately died in the road traffic accident', individual_id)

            self.sim.schedule_event(
                demography.InstantaneousDeath(self.module, individual_id, "RTI_imm_death"),
                self.sim.date
            )

        # ============= Take those remaining people involved in a RTI and assign injuries to them ==================
        selected_for_rti_inj_idx = selected_for_rti.drop(selected_to_die)
        assert len(selected_for_rti_inj_idx.intersection(selected_to_die)) == 0
        selected_for_rti_inj = df.loc[selected_for_rti_inj_idx]
        selected_for_rti_inj = selected_for_rti_inj.loc[df.is_alive & df.rt_road_traffic_inc & ~df.rt_imm_death]
        road_traffic_injuries = self.sim.modules['RTI']
        description = road_traffic_injuries.assign_injuries(len(selected_for_rti_inj))
        description = description.replace('nan', 'none')
        description = description.set_index(selected_for_rti_inj.index)

        selected_for_rti_inj = selected_for_rti_inj.join(description.set_index(selected_for_rti_inj.index))
        for person_id in selected_for_rti_inj.index:
            df.loc[person_id, 'rt_ISS_score'] = description.loc[person_id, 'ISS']
            df.loc[person_id, 'rt_MAIS_military_score'] = description.loc[person_id, 'MAIS_M']
        # ======================== Apply the injuries to the population dataframe ======================================
        # I've left this in its current form, basically in this section I create two dataframes and copy entries from
        # one into the other, this may be faster than the alternative.
        injury_columns = pd.Index(['Injury 1', 'Injury 2', 'Injury 3', 'Injury 4', 'Injury 5', 'Injury 6', 'Injury 7',
                                   'Injury 8'])
        for ninjuries in range(0, len(description.columns.intersection(injury_columns))):
            for person_id in selected_for_rti_inj.index:

                if ninjuries == 0:
                    df.loc[person_id, 'rt_injury_1'] = description.loc[person_id, 'Injury 1']
                if ninjuries == 1:
                    df.loc[person_id, 'rt_injury_2'] = description.loc[person_id, 'Injury 2']
                if ninjuries == 2:
                    df.loc[person_id, 'rt_injury_3'] = description.loc[person_id, 'Injury 3']
                if ninjuries == 3:
                    df.loc[person_id, 'rt_injury_4'] = description.loc[person_id, 'Injury 4']
                if ninjuries == 4:
                    df.loc[person_id, 'rt_injury_5'] = description.loc[person_id, 'Injury 5']
                if ninjuries == 5:
                    df.loc[person_id, 'rt_injury_6'] = description.loc[person_id, 'Injury 6']
                if ninjuries == 6:
                    df.loc[person_id, 'rt_injury_7'] = description.loc[person_id, 'Injury 7']
                if ninjuries == 7:
                    df.loc[person_id, 'rt_injury_8'] = description.loc[person_id, 'Injury 8']
        # All those who are injured in a road traffic accident have this noted in the property 'rt_road_traffic_inc'
        assert sum(df.loc[selected_for_rti, 'rt_road_traffic_inc']) == len(selected_for_rti)
        # All those who are involved in a road traffic accident have these noted in the property 'rt_date_inj'
        assert len(df.loc[selected_for_rti, 'rt_date_inj'] != pd.NaT) == len(selected_for_rti)
        # All those in a car crash either have died immediately or been assigned at least one injury
        # assert len(df.loc[df.rt_imm_death & (df['rt_injury_1'] != 'none')]) == 0, \
        #     'Something has gone wrong when deciding who has what post injury'
        # All those who are injures and do not die immediately have an ISS score > 0
        assert len(df.loc[df.rt_road_traffic_inc & ~df.rt_imm_death, 'rt_ISS_score'] > 0) == \
               len(df.loc[df.rt_road_traffic_inc & ~df.rt_imm_death])
        # ========================== Decide survival time without medical intervention ================================
        # todo: find better time for survival data without med int for ISS scores
        df.loc[selected_for_rti_inj.index, 'rt_date_death_no_med'] = now + DateOffset(days=7)
        # ============================ Injury severity classification =================================================

        # ============================== Non specific injury updates ===============================================
        # Find those with mild injuries and update the rt_roadtrafficinj property so they have a mild injury
        mild_rti_idx = selected_for_rti_inj.index[selected_for_rti_inj.is_alive & selected_for_rti_inj['ISS'] < 15]
        df.loc[mild_rti_idx, 'rt_inj_severity'] = 'mild'
        # Find those with severe injuries and update the rt_roadtrafficinj property so they have a severe injury
        severe_rti_idx = selected_for_rti_inj.index[selected_for_rti_inj['ISS'] >= 15]
        df.loc[severe_rti_idx, 'rt_inj_severity'] = 'severe'
        assert sum(df.loc[df.rt_road_traffic_inc & ~df.rt_imm_death & (df.rt_date_inj == now), 'rt_inj_severity']
                   != 'none') == len(selected_for_rti_inj.index)
        # Find those with polytrauma and update the rt_polytrauma property so they have polytrauma
        polytrauma_idx = selected_for_rti_inj.index[selected_for_rti_inj['Polytrauma'] is True]
        df.loc[polytrauma_idx, 'rt_polytrauma'] = True

        # =+=+=+=+=+=+=+=+=+=+=+=+=+=+ Injury specific updates =+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+
        # =+=+=+=+=+=+=+=+=+=+=+=+=+=+ Assign the DALY weights =+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+
        road_traffic_injuries.rti_assign_daly_weights(selected_for_rti_inj.index)

        # Condition to be sent to the health care system: 1) They must be alive 2) They must have been involved in a
        # road traffic accident 3) they must have not died immediately in the accident 4) they must not have been to an
        # A and E department previously and been diagnosed
        condition_to_be_sent_to_HSI = df.is_alive & df.rt_road_traffic_inc & ~df.rt_diagnosed & ~df.rt_imm_death & \
                                      (df.rt_date_inj == now) & (df.rt_injury_1 != "none")
        assert sum(df.loc[condition_to_be_sent_to_HSI, 'rt_injury_1'] != "none") == \
               len(df.loc[condition_to_be_sent_to_HSI])
        idx = df.index[condition_to_be_sent_to_HSI]
        # ===================================== Assign symptoms ========================================================
        # Currently this is just a generic severe_trauma symptom to get those with injuries into the health system,
        # in the future it will be better to have a more sophisticated symptom system
        for person_id in idx:
            self.sim.modules['SymptomManager'].change_symptom(
                person_id=person_id,
                disease_module=self.module,
                add_or_remove='+',
                symptom_string='severe_trauma',
            )

        # ================================ Generic first appointment ===================================================

        for person_id_to_start_treatment in idx:
            if df.loc[person_id_to_start_treatment, 'rt_diagnosed'] is False:
                event = HSI_GenericEmergencyFirstApptAtFacilityLevel1(module=self.module,
                                                                      person_id=person_id_to_start_treatment)

                logger.debug('Person %d seeks care for an injury from a road traffic '
                             'incident on date: %s', person_id_to_start_treatment, self.sim.date)
                self.sim.modules['HealthSystem'].schedule_hsi_event(
                    event,
                    priority=0,
                    topen=self.sim.date,
                    tclose=self.sim.date + DateOffset(days=5))


class RTI_Check_Death_No_Med(RegularEvent, PopulationScopeEventMixin):
    """
    A regular event which organises whether a person who has not received medical treatment should die as a result of
    their injuries. This even makes use of the maximum AIS-military score, a trauma scoring system developed for
    injuries in a military environment, assumed here to be a good indicator of the probability of mortality without
    access to a medical system.
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(days=1))
        assert isinstance(module, RTI)
        self.prob_death_MAIS3 = 0.05
        self.prob_death_MAIS4 = 0.31
        self.prob_death_MAIS5 = 0.59
        self.prob_death_MAIS6 = 0.83

    def apply(self, population):
        df = population.props
        now = self.sim.date
        if len(df.loc[df['rt_date_death_no_med'] == now]) > 0:
            due_to_die_today_without_med_int = df.loc[df['rt_date_death_no_med'] == now].index
            rand_for_death = self.sim.rng.random_sample(1)
            for person_id in due_to_die_today_without_med_int:
                died = False
                if (df.loc[person_id, 'rt_MAIS_military_score'] == 3) & (rand_for_death < self.prob_death_MAIS3):
                    died = True
                elif (df.loc[person_id, 'rt_MAIS_military_score'] == 4) & (rand_for_death < self.prob_death_MAIS4):
                    died = True
                elif (df.loc[person_id, 'rt_MAIS_military_score'] == 5) & (rand_for_death < self.prob_death_MAIS5):
                    died = True
                elif (df.loc[person_id, 'rt_MAIS_military_score'] == 6) & (rand_for_death < self.prob_death_MAIS6):
                    died = True
                if died:
                    df.loc[person_id, 'rt_no_med_death'] = True
                    self.sim.schedule_event(demography.InstantaneousDeath(self.module, person_id,
                                                                          cause='RTI_death_without_med'), self.sim.date)


class RTI_Recovery_Event(RegularEvent, PopulationScopeEventMixin):
    """
    A regular event which checks the recovery date determined by each injury in columns rt_injury_1 through
    rt_injury_8, which is being stored in rt_date_to_remove_daly, a list property with 8 entries. This event
    checks the dates stored in rt_date_to_remove_daly property, when the date matches one of the entries,
    the daly weight is removed and the injury is fully healed.
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(days=1))
        assert isinstance(module, RTI)

    def apply(self, population):
        road_traffic_injuries = self.module
        df = population.props
        now = self.sim.date
        # Isolate the relevant population
        treated_persons = df.loc[df.is_alive & df.rt_med_int]
        # Isolate the relevant information
        recovery_dates = treated_persons['rt_date_to_remove_daly']
        default_recovery = [pd.NaT, pd.NaT, pd.NaT, pd.NaT, pd.NaT, pd.NaT, pd.NaT, pd.NaT]
        # Iterate over all the injured people who are having medical treatment
        for person in recovery_dates.index:
            # Iterate over all the dates in 'rt_date_to_remove_daly'
            for date in treated_persons.loc[person, 'rt_date_to_remove_daly']:
                if date == now:
                    # find the index for the injury which the person has recovered from
                    dateindex = treated_persons.loc[person, 'rt_date_to_remove_daly'].index(date)
                    # find the injury code associated with the healed injury
                    code_to_remove = [df.loc[person, f'rt_injury_{dateindex + 1}']]
                    # Set the healed injury recovery data back to the default state
                    df.loc[person, 'rt_date_to_remove_daly'][dateindex] = pd.NaT
                    # Remove the daly weight associated with the healed injury code
                    road_traffic_injuries.rti_alter_daly_post_treatment(person, code_to_remove)
                    # Check whether all their injuries are healed so the injury properties can be reset
                    if df.loc[person, 'rt_date_to_remove_daly'] == default_recovery:
                        df.loc[person, 'rt_med_int'] = False
                        df.loc[person, 'rt_inj_severity'] = "none"


# ---------------------------------------------------------------------------------------------------------
#   RTI SPECIFIC HEALTH SYSTEM INTERACTION EVENTS
#
#   Here are all the different Health System Interactions Events that this module will use.
# ---------------------------------------------------------------------------------------------------------

class HSI_RTI_Medical_Intervention(HSI_Event, IndividualScopeEventMixin):
    """This is a Health System Interaction Event.
    An appointment of a person who has experienced a road traffic injury, had their injuries diagnosed through A&E
    and now needs treatment.

    This appointment is designed to organise the treatments needed. In the __init__ section the appointment footprint
    is altered to fit the requirements of the person's treatment need. In this section we count the number of
    minor/major surgeries required and determine how long they will be in the health system for. For some injuries,
    the treatment plan is not entirely set into stone and may vary, for example, some skull fractures will need surgery
    whilst some will not. The treatment plan in its entirety is designed here.

    In the apply section, we send those who need surgeries to either HSI_RTI_Major_Surgery or HSI_RTI_Minor_Surgery,
    those who need stitches to HSI_RTI_Suture, those who need burn treatment to HSI_RTI_Burn_Management and those who
    need fracture casts to HSI_RTI_Casting.

    Pain medication is also requested here with HSI_RTI_Acute_Pain_Management.

    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        road_traffic_injuries = self.sim.modules['RTI']

        df = self.sim.population.props
        p = module.parameters
        # Load the parameters used in this event
        self.prob_depressed_skull_fracture = p['prob_depressed_skull_fracture']  # proportion of depressed skull
        # fractures in https://doi.org/10.1016/j.wneu.2017.09.084
        self.prob_mild_burns = p['prob_mild_burns']  # proportion of burns accross SSA with TBSA < 10
        # https://doi.org/10.1016/j.burns.2015.04.006
        self.prob_TBI_require_craniotomy = p['prob_TBI_require_craniotomy']
        self.prob_exploratory_laparotomy = p['prob_exploratory_laparotomy']
        self.prob_death_with_med_mild = p['prob_death_with_med_mild']
        self.prob_death_with_med_severe = p['prob_death_with_med_severe']
        self.prob_dislocation_requires_surgery = p['prob_dislocation_requires_surgery']
        self.allowed_interventions = p['allowed_interventions']
        self.prob_perm_disability_with_treatment_severe_TBI = p['prob_perm_disability_with_treatment_severe_TBI']
        # Create an empty list for injuries that are potentially healed without further medical intervention
        self.heal_with_time_injuries = []
        # Define the call on resources of this treatment event: Time of Officers (Appointments)
        #   - get an 'empty' foot
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_accepted_facility_level = 1
        # Place holder appointment footprints to ensure there is at least one
        is_child = self.sim.population.props.at[person_id, 'age_years'] < 5.0

        if is_child:
            the_appt_footprint['Under5OPD'] = 1.0  # Child out-patient appointment
        else:
            the_appt_footprint['Over5OPD'] = 1.0  # Adult out-patient appointment

        # ======================= Design treatment plan, appointment type =============================================
        """ Here, RTI_MedInt designs the treatment plan of the person's injuries, the following determines what the
        major and minor surgery requirements will be

        """
        # Create variables to count how many major or minor surgeries will be required to treat this person
        self.major_surgery_counts = 0
        self.minor_surgery_counts = 0
        # Isolate the relevant injury information
        cols = ['rt_injury_1', 'rt_injury_2', 'rt_injury_3', 'rt_injury_4', 'rt_injury_5', 'rt_injury_6',
                'rt_injury_7', 'rt_injury_8']
        person_injuries = df.loc[[person_id], cols]
        # ------------------------------- Skull fractures -------------------------------------------------------------
        # Check if the person has a skull fracture and whether the skull fracture is a depressed skull fracture. If the
        # fracture is depressed, schedule a surgery.
        codes = ['112']
        idx, counts = road_traffic_injuries.rti_find_and_count_injuries(person_injuries, codes)
        require_surgery = self.module.rng.random_sample(size=1)
        # Find probability that skull fractures will require surgery
        if counts > 0:
            if require_surgery < self.prob_depressed_skull_fracture:
                logger.debug('Person %d has a depressed skull fracture which will require surgery', person_id)
                self.major_surgery_counts += 1
            else:
                actual_injury = np.intersect1d(codes, person_injuries.values)
                self.heal_with_time_injuries = np.append(self.heal_with_time_injuries, actual_injury)
        codes = ['113']
        idx, counts = road_traffic_injuries.rti_find_and_count_injuries(person_injuries, codes)
        if counts > 0:
            self.heal_with_time_injuries = np.append(self.heal_with_time_injuries, '113')
        # -------------------------------- Facial fractures -----------------------------------------------------------
        # Check whether the person has facial fractures, then if they do schedule a surgery to treat it
        codes = ['211', '212']
        idx, counts = road_traffic_injuries.rti_find_and_count_injuries(person_injuries, codes)
        if counts > 0:
            self.minor_surgery_counts += 1
        # consumables required: closed reduction. In some cases surgery
        # --------------------------------- Thorax Fractures -----------------------------------------------------------
        # Check whether the person has a broken rib (and therefor needs no further medical care apart from pain
        # management) or if they have flail chest, a life threatening condition which will require surgery.
        codes = ['412']
        idx, counts = road_traffic_injuries.rti_find_and_count_injuries(person_injuries, codes)
        if counts > 0:
            actual_injury = np.intersect1d(codes, person_injuries.values)
            self.heal_with_time_injuries = np.append(self.heal_with_time_injuries, actual_injury)
        codes = ['414']
        idx, counts = road_traffic_injuries.rti_find_and_count_injuries(person_injuries, codes)
        if counts > 0:
            self.major_surgery_counts += 1
        # --------------------------------- Lower extremity fractures --------------------------------------------------
        # Check if the person has a broken femur/hip which will require a surgery.
        codes = ['813', '813a', '813b', '813c']
        idx1, counts = road_traffic_injuries.rti_find_and_count_injuries(person_injuries, codes)
        # Major surgery required for broken femur/hip/pelvis
        if counts > 0:
            self.major_surgery_counts += 1
        # ------------------------------ Traumatic brain injury requirements ------------------------------------------
        # Check whether the person has a severe traumatic brain injury, which in some cases will require a major surgery
        # to treat
        codes = ['133', '133a', '133b', '133c', '133d' '134', '134a', '134b', '135']
        idx, counts = road_traffic_injuries.rti_find_and_count_injuries(person_injuries, codes)
        require_surgery = self.module.rng.random_sample(size=1)
        if counts > 0:
            if require_surgery < self.prob_TBI_require_craniotomy:
                self.major_surgery_counts += 1
            else:
                actual_injury = np.intersect1d(codes, person_injuries.values)
                self.heal_with_time_injuries = np.append(self.heal_with_time_injuries, actual_injury)
        # ------------------------------ Abdominal organ injury requirements ------------------------------------------
        # Check if the person has any abodominal organ injuries, if they do, determine whether they require a surgery or
        # not
        codes = ['552', '553', '554']
        idx, counts = road_traffic_injuries.rti_find_and_count_injuries(person_injuries, codes)
        require_surgery = self.module.rng.random_sample(size=1)
        if counts > 0:
            if require_surgery < self.prob_exploratory_laparotomy:
                self.major_surgery_counts += 1
            else:
                actual_injury = np.intersect1d(codes, person_injuries.values)
                self.heal_with_time_injuries = np.append(self.heal_with_time_injuries, actual_injury)

        # -------------------------------- Spinal cord injury requirements --------------------------------------------
        # Check whether they have a spinal cord injury, if we allow spinal cord surgery capacilities here, ask for a
        # surgery
        codes = ['673', '673a', '673b', '674', '674a', '674b', '675', '675a', '675b', '676']
        idx, counts = road_traffic_injuries.rti_find_and_count_injuries(person_injuries, codes)
        if (counts > 0) & ('include_spine_surgery' in self.allowed_interventions):
            self.major_surgery_counts += 1
        elif counts > 0:
            df.at[person_id, 'rt_perm_disability'] = True
            # Find the column and code where the permanent injury is stored
            column, code = road_traffic_injuries.rti_find_injury_column(person_id=person_id, codes=codes)
            logger.debug('@@@@@@@@@@ Person %d had intervention for SCI on %s but still disabled!!!!!!',
                         person_id, self.sim.date)
            df.loc[person_id, column] = "P" + code
            code = df.loc[person_id, column]
            columns, codes = road_traffic_injuries.rti_find_all_columns_of_treated_injuries(person_id, [code])
            for col in columns:
                # schedule the recovery date for the permanent injury for beyond the end of the simulation (making
                # it permanent)
                df.loc[person_id, 'rt_date_to_remove_daly'][int(col[-1]) - 1] = self.sim.end_date + \
                                                                                DateOffset(days=1)

        # --------------------------------- Dislocations --------------------------------------------------------------
        # Check if they have a dislocation, will require surgery but otherwise they can be taken care of in the RTI med
        # app
        codes = ['322', '323', '722', '822', '822a', '822b']
        idx, counts = road_traffic_injuries.rti_find_and_count_injuries(person_injuries, codes)
        require_surgery = self.module.rng.random_sample(size=1)
        if counts > 0:
            if require_surgery < self.prob_dislocation_requires_surgery:
                self.minor_surgery_counts += 1
            else:
                actual_injury = np.intersect1d(codes, person_injuries.values)
                self.heal_with_time_injuries = np.append(self.heal_with_time_injuries, actual_injury)

        # --------------------------------- Soft tissue injury in neck -------------------------------------------------
        # check whether they have a soft tissue/internal bleeding injury in the neck. If so schedule a surgery.
        codes = ['342', '343']
        idx, counts = road_traffic_injuries.rti_find_and_count_injuries(person_injuries, codes)
        if counts > 0:
            self.major_surgery_counts += 1

        # --------------------------------- Soft tissue injury in thorax/ lung injury ----------------------------------
        # Check whether they have any soft tissue injuries in the thorax, if so schedule surgery if required else make
        # the injuries heal over time without further medical care
        codes = ['441', '443', '453', '453a', '453b']
        idx, counts = road_traffic_injuries.rti_find_and_count_injuries(person_injuries, codes)
        if (counts > 0) & ('include_thoroscopy' in self.allowed_interventions):
            self.major_surgery_counts += 1
        codes = ['442']
        idx, counts = road_traffic_injuries.rti_find_and_count_injuries(person_injuries, codes)
        if counts > 0:
            actual_injury = np.intersect1d(codes, person_injuries.values)
            self.heal_with_time_injuries = np.append(self.heal_with_time_injuries, actual_injury)
        # -------------------------------- Internal bleeding -----------------------------------------------------------
        # Check if they have any internal bleeding in the neck, if so schedule a major surgery
        codes = ['361', '363']
        idx, counts = road_traffic_injuries.rti_find_and_count_injuries(person_injuries, codes)
        if counts > 0:
            self.major_surgery_counts += 1
        # check if they have internal bleeding in the thorax, and if the surgery is available, schedule a major surgery
        codes = ['463']
        idx, counts = road_traffic_injuries.rti_find_and_count_injuries(person_injuries, codes)
        if (counts > 0) & ('include_thoroscopy' in self.allowed_interventions):
            self.major_surgery_counts += 1
        codes = ['461']
        idx, counts = road_traffic_injuries.rti_find_and_count_injuries(person_injuries, codes)
        if counts > 0:
            actual_injury = np.intersect1d(codes, person_injuries.values)
            self.heal_with_time_injuries = np.append(self.heal_with_time_injuries, actual_injury)
        # ------------------------------------- Amputations ------------------------------------------------------------
        # Check if they have had an amputation and schedule a major surgery if so.
        codes = ['782', '782a', '782b', '782c', '783', '882', '883', '884']
        idx, counts = road_traffic_injuries.rti_find_and_count_injuries(person_injuries, codes)
        if counts > 0:
            self.major_surgery_counts += 1
        # --------------------------------------- Eye injury -----------------------------------------------------------
        # check if they have an eye injury and schedule a minor surgery if so
        codes = ['291']
        idx, counts = road_traffic_injuries.rti_find_and_count_injuries(person_injuries, codes)
        if counts > 0:
            self.minor_surgery_counts += 1
        # ------------------------------ Soft tissue injury in face ----------------------------------------------------
        # check if they have any facial soft tissue damage and schedule a minor surgery if so.
        codes = ['241']
        idx, counts = road_traffic_injuries.rti_find_and_count_injuries(person_injuries, codes)
        if counts > 0:
            self.minor_surgery_counts += 1

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'RTI_MedicalIntervention'  # This must begin with the module name
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = the_accepted_facility_level
        self.ALERT_OTHER_DISEASES = []

        # ================ Determine how long the person will be in hospital based on their ISS score ==================
        self.inpatient_days = road_traffic_injuries.rti_determine_LOS(person_id)
        self.EXPECTED_APPT_FOOTPRINT.update({'InpatientDays': self.inpatient_days})

    def apply(self, person_id, squeeze_factor):
        road_traffic_injuries = self.sim.modules['RTI']
        df = self.sim.population.props
        # Remove the scheduled death without medical intervention
        df.loc[person_id, 'rt_date_death_no_med'] = pd.NaT
        # Isolate relevant injury information
        cols = ['rt_injury_1', 'rt_injury_2', 'rt_injury_3', 'rt_injury_4', 'rt_injury_5', 'rt_injury_6',
                'rt_injury_7', 'rt_injury_8']
        person_injuries = df.loc[[person_id], cols]
        non_empty_injuries = person_injuries[person_injuries != "none"]
        injury_columns = non_empty_injuries.columns
        # Check that those who arrive here are alive and have been through the first generic appointment, and didn't die
        # at the scene of the crash
        assert df.loc[person_id, 'is_alive'] & df.loc[person_id, 'rt_diagnosed'] & ~df.loc[person_id, 'rt_imm_death']
        # Check that those who arrive here have at least one injury
        idx, counts = RTI.rti_find_and_count_injuries(person_injuries,
                                                      self.module.PROPERTIES.get('rt_injury_1').categories[1:-1])
        assert counts > 0, 'This person has asked for medical treatment despite not being injured'
        df.at[person_id, 'rt_med_int'] = True
        # =============================== Make 'healed with time' injuries disappear ===================================
        if len(self.heal_with_time_injuries) > 0:
            # check whether the heal with time injuries include dislocations, which may have been sent to surgery
            dislocations = ['322', '323', '722', '822', '822a', '822b']
            dislocations_injury = np.intersect1d(dislocations, self.heal_with_time_injuries)
            if len(dislocations_injury) > 0:
                for code in dislocations_injury:
                    # if the heal with time injury is a dislocation, shedule a recovery date
                    if code == '322' or '323':
                        columns = injury_columns.get_loc(road_traffic_injuries.rti_find_injury_column(person_id,
                                                                                                      [code])[0])
                        # using estimated 7 weeks to recover from dislocated neck
                        df.loc[person_id, 'rt_date_to_remove_daly'][columns] = self.sim.date + DateOffset(weeks=6)
                    elif code == '722':
                        columns = injury_columns.get_loc(road_traffic_injuries.rti_find_injury_column(person_id,
                                                                                                      [code])[0])
                        # using estimated 12 weeks to recover from dislocated shoulder
                        df.loc[person_id, 'rt_date_to_remove_daly'][columns] = self.sim.date + DateOffset(weeks=12)
                    elif code == '822a':
                        columns = injury_columns.get_loc(road_traffic_injuries.rti_find_injury_column(person_id,
                                                                                                      [code])[0])
                        # using estimated 2 months to recover from dislocated hip
                        df.loc[person_id, 'rt_date_to_remove_daly'][columns] = self.sim.date + DateOffset(months=2)
                    else:
                        columns = injury_columns.get_loc(road_traffic_injuries.rti_find_injury_column(person_id,
                                                                                                      [code])[0])
                        # using estimated 6 months to recover from dislocated knee
                        df.loc[person_id, 'rt_date_to_remove_daly'][columns] = self.sim.date + DateOffset(weeks=6)
            # Check whether the heal with time injury is a skull fracture, which may have been sent to surgery
            fractures = ['112', '113']
            fractures_injury = np.intersect1d(fractures, self.heal_with_time_injuries)
            if len(fractures_injury) > 0:
                for code in fractures_injury:
                    if code == '112' or '113':
                        # schedule a recovery date for the skull fracture
                        columns = injury_columns.get_loc(road_traffic_injuries.rti_find_injury_column(person_id,
                                                                                                      [code])[0]
                                                         )
                        # using estimated 7 weeks PLACEHOLDER FOR SKULL FRACTURE
                        df.loc[person_id, 'rt_date_to_remove_daly'][columns] = self.sim.date + DateOffset(weeks=7)
            abdominal = ['552', '553', '554']
            abdominal_injury = np.intersect1d(abdominal, self.heal_with_time_injuries)
            # check whether the heal with time injury is an abdominal injury
            if len(abdominal_injury) > 0:
                for code in abdominal_injury:
                    if code == '552' or '553' or '554':
                        # Schedule the recovery date for the injury
                        columns = injury_columns.get_loc(road_traffic_injuries.rti_find_injury_column(person_id,
                                                                                                      [code])[0]
                                                         )
                        # using estimated 3 months PLACEHOLDER FOR ABDOMINAL TRAUMA
                        df.loc[person_id, 'rt_date_to_remove_daly'][columns] = self.sim.date + DateOffset(months=3)
            tbi = ['133', '133a', '133b', '133c', '133d' '134', '134a', '134b', '135']
            tbi_injury = np.intersect1d(tbi, self.heal_with_time_injuries)
            if len(tbi_injury) > 0:
                columns = injury_columns.get_loc(road_traffic_injuries.rti_find_injury_column(person_id, tbi_injury)
                                                 [0])
                # ask if this injury will be permanent
                perm_injury = self.module.rng.random_sample(size=1)
                # todo: find a parameter estimate for the probability of perm TBI when a craniotomy isn't required
                if perm_injury < self.prob_perm_disability_with_treatment_severe_TBI:
                    column, code = road_traffic_injuries.rti_find_injury_column(person_id=person_id, codes=tbi_injury)
                    df.loc[person_id, column] = "P" + code

                # using estimated 6 months PLACEHOLDER FOR TRAUMATIC BRAIN INJURY
                df.loc[person_id, 'rt_date_to_remove_daly'][columns] = self.sim.date + DateOffset(months=6)
        # ======================================= Schedule surgeries ==================================================
        # Schedule the surgeries by calling the functions rti_do_for_major/minor_surgeries which in turn schedules the
        # surgeries
        if self.major_surgery_counts > 0:
            for count in range(0, self.major_surgery_counts):
                road_traffic_injuries.rti_do_for_major_surgeries(person_id=person_id, count=count)
        if self.minor_surgery_counts > 0:
            for count in range(0, self.minor_surgery_counts):
                road_traffic_injuries.rti_do_for_minor_surgeries(person_id=person_id, count=count)

        # --------------------------- Lacerations will get stitches here -----------------------------------------------
        # Schedule the laceration sutures by calling the functions rti_ask_for_stitches which in turn schedules the
        # treatment
        codes = ['1101', '2101', '3101', '4101', '5101', '7101', '8101']
        idx, lacerationcounts = road_traffic_injuries.rti_find_and_count_injuries(person_injuries, codes)
        if lacerationcounts > 0:
            road_traffic_injuries.rti_ask_for_suture_kit(person_id=person_id)

        # =================================== Burns consumables =======================================================
        # Schedule the burn treatments  by calling the functions rti_ask_for_burn_treatment which in turn schedules the
        # treatment
        codes = ['1114', '2114', '3113', '4113', '5113', '7113', '8113']
        idx, burncounts = road_traffic_injuries.rti_find_and_count_injuries(person_injuries, codes)

        if burncounts > 0:
            road_traffic_injuries.rti_ask_for_burn_treatment(person_id=person_id)

        # ==================================== Fractures ==============================================================
        # ------------------------------ Cast-able fractures ----------------------------------------------------------
        # Schedule the fracture treatments by calling the functions rti_ask_for_fracture_casts which in turn schedules
        # the treatment
        codes = ['712', '712a', '712b', '712c', '811', '812']
        idx, fracturecounts = road_traffic_injuries.rti_find_and_count_injuries(person_injuries, codes)
        if fracturecounts > 0:
            road_traffic_injuries.rti_ask_for_fracture_casts(person_id=person_id)

        # ============================== Generic injury management =====================================================

        # ================================= Pain management ============================================================
        # Most injuries will require some level of pain relief, we need to determine:
        # 1) What drug the person will require
        # 2) What to do if the drug they are after isn't available
        # 3) Whether to run the event even if the drugs aren't available
        # Determine whether this person dies with medical treatment or not with the RTIMediaclInterventionDeathEvent

        road_traffic_injuries.rti_acute_pain_management(person_id=person_id)
        # ==================================== Tetanus management ======================================================
        # Check if they have had a laceration or a burn, if so request a tetanus jab
        codes_for_tetanus = ['1101', '2101', '3101', '4101', '5101', '7101', '8101',
                             '1114', '2114', '3113', '4113', '5113', '7113', '8113']

        idx, counts = road_traffic_injuries.rti_find_and_count_injuries(person_injuries, codes_for_tetanus)
        if counts > 0:
            road_traffic_injuries.rti_ask_for_tetanus(person_id=person_id)
        # ============================== Ask if they die even with treatment ===========================================
        self.sim.schedule_event(RTI_Medical_Intervention_Death_Event(self.module, person_id), self.sim.date +
                                DateOffset(days=self.inpatient_days))
        logger.debug('This is RTIMedicalInterventionEvent scheduling a potential death on date %s (end of treatment)'
                     ' for person %d', self.sim.date + DateOffset(days=self.inpatient_days), person_id)

    def did_not_run(self):
        person_id = self.target
        df = self.sim.population.props
        logger.debug('RTI_MedicalInterventionEvent: did not run for person  %d on date %s',
                     person_id, self.sim.date)
        person = df.loc[person_id]
        injurycodes = {'First injury': df.loc[person_id, 'rt_injury_1'],
                       'Second injury': df.loc[person_id, 'rt_injury_2'],
                       'Third injury': df.loc[person_id, 'rt_injury_3'],
                       'Fourth injury': df.loc[person_id, 'rt_injury_4'],
                       'Fifth injury': df.loc[person_id, 'rt_injury_5'],
                       'Sixth injury': df.loc[person_id, 'rt_injury_6'],
                       'Seventh injury': df.loc[person_id, 'rt_injury_7'],
                       'Eight injury': df.loc[person_id, 'rt_injury_8']}
        logger.debug(f'Injury profile of person %d, {injurycodes}', person_id)


class HSI_RTI_Fracture_Cast(HSI_Event, IndividualScopeEventMixin):
    """
    This HSI event handles fracture casts/giving slings for those who need it. The HSI event tests whether the injured
    person has an appropriate injury code, determines how many fractures the person and then requests fracture
    treatment as required.

    The injury codes dealt with in this HSI event are:
    '712a' - broken clavicle, scapula, humerus
    '712b' - broken hand/wrist
    '712c' - broken radius/ulna
    '811' - Fractured foot
    '812' - broken tibia/fibula
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, RTI)
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['Over5OPD'] = 1  # Placeholder requirement
        the_accepted_facility_level = 1
        self.TREATMENT_ID = 'RTI_Fracture_Cast'  # This must begin with the module name
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = the_accepted_facility_level
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        road_traffic_injuries = self.sim.modules['RTI']
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        # isolate the relevant injury information
        cols = ['rt_injury_1', 'rt_injury_2', 'rt_injury_3', 'rt_injury_4', 'rt_injury_5', 'rt_injury_6',
                'rt_injury_7', 'rt_injury_8']
        person_injuries = df.loc[[person_id], cols]
        # check if they have a fracture that requires a cast
        codes = ['712b', '712c', '811', '812']
        idx, fracturecastcounts = road_traffic_injuries.rti_find_and_count_injuries(person_injuries, codes)
        # check if they have a fracture that requires a sling
        codes = ['712a']
        idx, slingcounts = road_traffic_injuries.rti_find_and_count_injuries(person_injuries, codes)
        consumables_fractures = {'Intervention_Package_Code': dict(), 'Item_Code': dict()}
        # Check the person sent here is alive, been through the generic first appointment,
        # been through the RTI med intervention
        assert df.loc[person_id, 'is_alive'] & df.loc[person_id, 'rt_diagnosed'] & df.loc[person_id, 'rt_med_int']
        # Check that the person sent here has an injury treated by this module
        assert fracturecastcounts + slingcounts > 0
        # If they have a fracture that needs a cast, ask for plaster of paris
        if fracturecastcounts > 0:
            plaster_of_paris_code = pd.unique(
                consumables.loc[consumables['Items'] ==
                                'Plaster of Paris (POP) 10cm x 7.5cm slab_12_CMST', 'Item_Code'])[0]
            consumables_fractures['Item_Code'].update({plaster_of_paris_code: fracturecastcounts})
        # If they have a fracture that needs a sling, ask for bandage.

        if slingcounts > 0:
            sling_code = pd.unique(
                consumables.loc[consumables['Items'] ==
                                'Bandage, crepe 7.5cm x 1.4m long , when stretched', 'Item_Code'])[0]
            consumables_fractures['Item_Code'].update({sling_code: slingcounts})
        # Check that there are enough consumables to treat this person's fractures
        is_cons_available = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=self,
            cons_req_as_footprint=consumables_fractures,
            to_log=False)
        if is_cons_available:
            logger.debug(f"Fracture casts available for person %d's {fracturecastcounts + slingcounts} fractures",
                         person_id)
            df.at[person_id, 'rt_med_int'] = True
            # Find the persons injuries
            non_empty_injuries = person_injuries[person_injuries != "none"]
            non_empty_injuries = non_empty_injuries.dropna(axis=1)
            # Find the injury codes treated by fracture casts/slings
            codes = ['712a', '712b', '712c', '811', '812']
            relevant_codes = np.intersect1d(non_empty_injuries.values, codes)
            # Some TLO codes have daly weights associated with treated and non-treated injuries, swap-able codes are
            # listed below
            swapping_codes = ['712b', '812', '3113', '4113', '5113', '7113', '8113', '813a', '813b', 'P673a', 'P673b',
                              'P674a', 'P674b', 'P675a', 'P675b', 'P676', 'P782b', 'P783', 'P883', 'P884']
            # iterate over the treated codes
            for code in relevant_codes:
                # check if the treated code is one where we have to swap the daly weights for
                if code in swapping_codes:
                    road_traffic_injuries.rti_swap_injury_daly_upon_treatment(person_id, relevant_codes)
                    break
            # Find the injuries that have been treated and then schedule a recovery date
            columns, codes = road_traffic_injuries.rti_find_all_columns_of_treated_injuries(person_id, codes)
            for col in columns:
                df.loc[person_id, 'rt_date_to_remove_daly'][int(col[-1]) - 1] = self.sim.date + \
                                                                                DateOffset(weeks=7)
            else:
                logger.debug(f"Person %d's has {fracturecastcounts + slingcounts} fractures without treatment",
                             person_id)

    def did_not_run(self, person_id):
        logger.debug('Fracture casts unavailable for person %d', person_id)


class HSI_RTI_Suture(HSI_Event, IndividualScopeEventMixin):
    """
    This HSI event handles lacerations giving suture kits for those who need it. The HSI event tests whether the injured
    person has an appropriate injury code, determines how many lacerations the person and then requests suture kits
     as required.


    The codes dealt with are:
    '1101' - Laceration to the head
    '2101' - Laceration to the face
    '3101' - Laceration to the neck
    '4101' - Laceration to the thorax
    '5101' - Laceration to the abdomen
    '7101' - Laceration to the upper extremity
    '8101' - Laceration to the lower extremity
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, RTI)
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['Over5OPD'] = 1  # Placeholder requirement
        the_accepted_facility_level = 1
        self.TREATMENT_ID = 'RTI_Suture'  # This must begin with the module name
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = the_accepted_facility_level
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        df = self.sim.population.props
        road_traffic_injuries = self.sim.modules['RTI']

        cols = ['rt_injury_1', 'rt_injury_2', 'rt_injury_3', 'rt_injury_4', 'rt_injury_5', 'rt_injury_6',
                'rt_injury_7', 'rt_injury_8']
        person_injuries = df.loc[[person_id], cols]
        codes = ['1101', '2101', '3101', '4101', '5101', '7101', '8101']
        idx, lacerationcounts = road_traffic_injuries.rti_find_and_count_injuries(person_injuries, codes)
        # Check the person sent here is alive, has been through A&E, through Med int
        assert df.loc[person_id, 'is_alive'] & df.loc[person_id, 'rt_diagnosed'] & df.loc[person_id, 'rt_med_int']
        # Check that the person sent here has an injury that is treated by this HSI event
        assert lacerationcounts > 0
        if lacerationcounts > 0:
            # check the number of suture kits required and request them
            item_code_suture_kit = pd.unique(
                consumables.loc[consumables['Items'] == 'Suture pack', 'Item_Code'])[0]
            item_code_cetrimide_chlorhexidine = pd.unique(
                consumables.loc[consumables['Items'] ==
                                'Cetrimide 15% + chlorhexidine 1.5% solution.for dilution _5_CMST', 'Item_Code'])[0]
            consumables_open_wound_1 = {
                'Intervention_Package_Code': dict(),
                'Item_Code': {item_code_suture_kit: lacerationcounts,
                              item_code_cetrimide_chlorhexidine: lacerationcounts}
            }

            is_cons_available_1 = self.sim.modules['HealthSystem'].request_consumables(
                hsi_event=self,
                cons_req_as_footprint=consumables_open_wound_1,
                to_log=False)['Item_Code']

            cond = is_cons_available_1

            # Availability of consumables determines if the intervention is delivered...
            if cond[item_code_suture_kit]:
                logger.debug('This facility has open wound treatment available which has been used for person %d.',
                             person_id)
                logger.debug(f'This facility treated their {lacerationcounts} open wounds')
                if cond[item_code_cetrimide_chlorhexidine]:
                    logger.debug('This laceration was cleaned before stitching')
                    df.at[person_id, 'rt_med_int'] = True
                    columns, codes = road_traffic_injuries.rti_find_all_columns_of_treated_injuries(person_id, codes)
                    for col in columns:
                        # heal time for lacerations is roughly two weeks according to:
                        # https://www.facs.org/~/media/files/education/patient%20ed/wound_lacerations.ashx#:~:text=of%20
                        # wound%20and%20your%20general,have%20a%20weakened%20immune%20system.
                        df.loc[person_id, 'rt_date_to_remove_daly'][int(col[-1]) - 1] = self.sim.date + \
                                                                                        DateOffset(days=14)
                else:
                    logger.debug("This laceration wasn't cleaned before stitching, person %d is at risk of infection",
                                 person_id)
                    df.at[person_id, 'rt_med_int'] = True
                    columns, codes = road_traffic_injuries.rti_find_all_columns_of_treated_injuries(person_id, codes)
                    for col in columns:
                        df.loc[person_id, 'rt_date_to_remove_daly'][int(col[-1]) - 1] = self.sim.date + \
                                                                                        DateOffset(days=14)

            else:
                logger.debug('This facility has no treatment for open wounds available.')

        else:
            logger.debug("Did event run????")
            logger.debug(person_id)

            pass

    def did_not_run(self, person_id):
        logger.debug('Suture kits unavailable for person %d', person_id)


class HSI_RTI_Burn_Management(HSI_Event, IndividualScopeEventMixin):
    """
    This HSI event handles burns giving treatment for those who need it. The HSI event tests whether the injured
    person has an appropriate injury code, determines how many burns the person and then requests appropriate treatment
     as required.



    The codes dealt with in this HSI event are:
    '1114' - Burns to the head
    '2114' - Burns to the face
    '3113' - Burns to the neck
    '4113' - Burns to the thorax
    '5113' - Burns to the abdomen
    '7113' - Burns to the upper extremities
    '8113' - Burns to the lower extremities
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, RTI)
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['MinorSurg'] = 1
        the_accepted_facility_level = 1
        self.TREATMENT_ID = 'RTI_Burn_Management'  # This must begin with the module name
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = the_accepted_facility_level
        self.ALERT_OTHER_DISEASES = []
        p = self.module.parameters
        self.prob_mild_burns = p['prob_mild_burns']

    def apply(self, person_id, squeeze_factor):
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        df = self.sim.population.props
        road_traffic_injuries = self.sim.modules['RTI']

        cols = ['rt_injury_1', 'rt_injury_2', 'rt_injury_3', 'rt_injury_4', 'rt_injury_5', 'rt_injury_6',
                'rt_injury_7', 'rt_injury_8']
        person_injuries = df.loc[[person_id], cols]
        codes = ['1114', '2114', '3113', '4113', '5113', '7113', '8113']
        idx, burncounts = road_traffic_injuries.rti_find_and_count_injuries(person_injuries, codes)
        # check the person sent here is alive, has been through A and E and had RTI_med_int
        assert df.loc[person_id, 'is_alive'] & df.loc[person_id, 'rt_diagnosed'] & df.loc[person_id, 'rt_med_int']
        # check the person sent here has an injury treated by this module
        assert burncounts > 0
        if burncounts > 0:
            # Request materials for burn treatment
            possible_large_TBSA_burn_codes = ['7113', '8113', '4113', '5113']
            idx2, bigburncounts = \
                road_traffic_injuries.rti_find_and_count_injuries(person_injuries, possible_large_TBSA_burn_codes)
            item_code_cetrimide_chlorhexidine = pd.unique(
                consumables.loc[consumables['Items'] ==
                                'Cetrimide 15% + chlorhexidine 1.5% solution.for dilution _5_CMST', 'Item_Code'])[0]
            item_code_gauze = pd.unique(
                consumables.loc[
                    consumables['Items'] == "Dressing, paraffin gauze 9.5cm x 9.5cm (square)_packof 36_CMST",
                    'Item_Code'])[0]

            random_for_severe_burn = self.module.rng.random_sample(size=1)
            # ======================== If burns severe enough then give IV fluid replacement ===========================
            if (burncounts > 1) or ((len(idx2) > 0) & (random_for_severe_burn > self.prob_mild_burns)):
                # check if they have multiple burns, which implies a higher burned total body surface area (TBSA) which
                # will alter the treatment plan

                item_code_fluid_replacement = pd.unique(
                    consumables.loc[consumables['Items'] ==
                                    "ringer's lactate (Hartmann's solution), 500 ml_20_IDA", 'Item_Code'])[0]
                consumables_burns = {
                    'Intervention_Package_Code': dict(),
                    'Item_Code': {item_code_cetrimide_chlorhexidine: burncounts,
                                  item_code_fluid_replacement: 1, item_code_gauze: burncounts}}

            else:
                consumables_burns = {
                    'Intervention_Package_Code': dict(),
                    'Item_Code': {item_code_cetrimide_chlorhexidine: burncounts,
                                  item_code_gauze: burncounts}}
            is_cons_available = self.sim.modules['HealthSystem'].request_consumables(
                hsi_event=self,
                cons_req_as_footprint=consumables_burns,
                to_log=False)
            logger.debug(is_cons_available)
            cond = is_cons_available
            if all(value == 1 for value in cond.values()):
                logger.debug('This facility has burn treatment available which has been used for person %d.',
                             person_id)
                logger.debug(f'This facility treated their {burncounts} burns')
                df.at[person_id, 'rt_med_int'] = True
                non_empty_injuries = person_injuries[person_injuries != "none"]
                injury_columns = non_empty_injuries.columns
                columns = injury_columns.get_loc(road_traffic_injuries.rti_find_injury_column(person_id, codes)[0])
                # estimate burns take 4 weeks to heal
                df.loc[person_id, 'rt_date_to_remove_daly'][columns] = self.sim.date + DateOffset(weeks=4)

                columns = ['rt_injury_1', 'rt_injury_2', 'rt_injury_3', 'rt_injury_4', 'rt_injury_5', 'rt_injury_6',
                           'rt_injury_7', 'rt_injury_8']
                persons_injuries = df.loc[[person_id], columns]
                non_empty_injuries = persons_injuries[persons_injuries != "none"]
                non_empty_injuries = non_empty_injuries.dropna(axis=1)
                relevant_codes = np.intersect1d(non_empty_injuries.values, codes)
                swapping_codes = ['712b', '812', '3113', '4113', '5113', '7113', '8113', '813a', '813b', 'P673a',
                                  'P673b',
                                  'P674a', 'P674b', 'P675a', 'P675b', 'P676', 'P782b', 'P783', 'P883', 'P884']
                for code in relevant_codes:
                    if code in swapping_codes:
                        road_traffic_injuries.rti_swap_injury_daly_upon_treatment(person_id, relevant_codes)
                        break
            else:
                logger.debug('This facility has no treatment for burns available.')

        else:
            logger.debug("Did event run????")
            logger.debug(person_id)
            pass

    def did_not_run(self, person_id):
        logger.debug('Burn treatment unavailable for person %d', person_id)


class HSI_RTI_Tetanus_Vaccine(HSI_Event, IndividualScopeEventMixin):
    """
    This HSI event handles tetanus vaccine requests, the idea being that by separating these from the burn and
    laceration and burn treatments, those treatments can go ahead without the availability of tetanus stopping the event
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, RTI)
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['Over5OPD'] = 1  # Placeholder requirement
        the_accepted_facility_level = 1
        self.TREATMENT_ID = 'RTI_Tetanus_Vaccine'  # This must begin with the module name
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = the_accepted_facility_level
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        cols = ['rt_injury_1', 'rt_injury_2', 'rt_injury_3', 'rt_injury_4', 'rt_injury_5', 'rt_injury_6',
                'rt_injury_7', 'rt_injury_8']
        person_injuries = df.loc[[person_id], cols]
        # check the person sent here is alive, has been through A and E and had RTI_med_int
        assert df.loc[person_id, 'is_alive'] & df.loc[person_id, 'rt_diagnosed'] & df.loc[person_id, 'rt_med_int']
        # check the person sent here has an injury treated by this module
        codes_for_tetanus = ['1101', '2101', '3101', '4101', '5101', '7101', '8101',
                             '1114', '2114', '3113', '4113', '5113', '7113', '8113']

        idx, counts = RTI.rti_find_and_count_injuries(person_injuries, codes_for_tetanus)
        assert counts > 0
        # If they have a laceration/burn ask request the tetanus vaccine
        if counts > 0:
            consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
            item_code_tetanus = pd.unique(
                consumables.loc[consumables['Items'] == 'Tetanus toxin vaccine (TTV)', 'Item_Code'])[0]
            consumables_tetanus = {
                'Intervention_Package_Code': dict(),
                'Item_Code': {item_code_tetanus: 1}
            }
            is_tetanus_available = self.sim.modules['HealthSystem'].request_consumables(
                hsi_event=self,
                cons_req_as_footprint=consumables_tetanus,
                to_log=True)
            if is_tetanus_available:
                logger.debug("Tetanus vaccine requested for person %d and given", person_id)

    def did_not_run(self, person_id):
        logger.debug('Tetanus vaccine unavailable for person %d', person_id)


class HSI_RTI_Acute_Pain_Management(HSI_Event, IndividualScopeEventMixin):
    """ This HSI event handles all requests for pain management here, all injuries will pass through here and the pain
    medicine required will be set to manage the level of pain they are experiencing, with mild pain being managed with
    paracetamol/NSAIDS, moderate pain being managed with tramadol and severe pain being managed with morphine.

     "There is a mismatch between the burden of musculoskeletal pain conditions and appropriate health policy response
     and planning internationally that can be addressed with an integrated research and policy agenda."
     SEE https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6301413/
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, RTI)

        # Define the call on resources of this treatment event: Time of Officers (Appointments)
        #   - get an 'empty' footprint:
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        #   - update to reflect the appointments that are required
        the_appt_footprint['Over5OPD'] = 1  # This requires one out patient

        # Define the facilities at which this event can occur (only one is allowed)
        # Choose from: list(pd.unique(self.sim.modules['HealthSystem'].parameters['Facilities_For_Each_District']
        #                            ['Facility_Level']))
        the_accepted_facility_level = 1

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'RTI_Acute_Pain_Management'  # This must begin with the module name
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = the_accepted_facility_level
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        # Check that the person sent here is alive, has been through A&E and RTI_Med_int
        assert df.loc[person_id, 'is_alive'] & df.loc[person_id, 'rt_diagnosed'] & df.loc[person_id, 'rt_med_int']
        cols = ['rt_injury_1', 'rt_injury_2', 'rt_injury_3', 'rt_injury_4', 'rt_injury_5', 'rt_injury_6',
                'rt_injury_7', 'rt_injury_8']
        person_injuries = df.loc[[person_id], cols]
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        road_traffic_injuries = self.sim.modules['RTI']
        pain_level = "none"
        # Injuries causing mild pain include: Lacerations, mild soft tissue injuries, TBI (for now), eye injury
        Mild_Pain_Codes = ['1101', '2101', '3101', '4101', '5101', '7101', '8101',  # lacerations
                           '241',  # Minor soft tissue injuries
                           '133', '133a', '133b', '133c', '133d', '134', '134a', '134b', '135',  # TBI
                           'P133', 'P133a', 'P133b', 'P133c', 'P133d', 'P134', 'P134a', 'P134b', 'P135',  # Perm TBI
                           '291',  # Eye injury
                           '442'
                           ]
        mild_idx, mild_counts = road_traffic_injuries.rti_find_and_count_injuries(person_injuries, Mild_Pain_Codes)
        # Injuries causing moderate pain include: Fractures, dislocations, soft tissue and neck trauma
        Moderate_Pain_Codes = ['112', '113', '211', '212', '412', '414', '612', '712', '712a', '712b', '712c',
                               '811', '812', '813', '813a', '813b', '813c',  # fractures
                               '322', '323', '722', '822', '822a', '822b',  # dislocations
                               '342', '343', '361', '363',  # neck trauma
                               '461',  # chest wall bruising
                               ]
        moderate_idx, moderate_counts = road_traffic_injuries.rti_find_and_count_injuries(person_injuries,
                                                                                          Moderate_Pain_Codes)
        # Injuries causing severe pain include: All burns, amputations, spinal cord injuries, abdominal trauma see
        # (https://bestbets.org/bets/bet.php?id=1247), severe chest trauma
        Severe_Pain_Codes = ['1114', '2114', '3113', '4113', '5113', '7113', '8113',  # burns
                             'P782', 'P782a', 'P782b', 'P782c', 'P783', 'P882', 'P883', 'P884',  # amputations
                             '673', '673a', '673b', '674', '674a', '674b', '675', '675a', '675b', '676',
                             'P673', 'P673a', 'P673b', 'P674', 'P674a', 'P674b', 'P675', 'P675a', 'P675b', 'P676',
                             # SCI
                             '552', '553', '554',  # abdominal trauma
                             '463', '453', '453a', '453b', '441', '443'  # severe chest trauma
                             ]
        severe_idx, severe_counts = road_traffic_injuries.rti_find_and_count_injuries(person_injuries,
                                                                                      Severe_Pain_Codes)
        # check that the people here have at least one injury
        assert mild_counts + moderate_counts + severe_counts > 0
        if len(severe_idx) > 0:
            pain_level = "severe"
        elif len(moderate_idx) > 0:
            pain_level = "moderate"
        elif len(mild_idx) > 0:
            pain_level = "mild"

        if pain_level is "mild":
            # Multiple options, some are conditional
            # Give paracetamol
            # Give NSAIDS such as aspirin (unless they are under 16) for soft tissue pain, but not if they are pregnant
            dict_to_output = {'person': person_id,
                              'pain level': pain_level}
            logger.info(key='Requested_Pain_Management',
                        data=dict_to_output,
                        description='Summary of the pain medicine requested by each person')
            item_code_paracetamol = pd.unique(
                consumables.loc[consumables['Items'] == "Paracetamol 500mg_1000_CMST",
                                'Item_Code'])[0]
            item_code_diclofenac = pd.unique(
                consumables.loc[consumables['Items'] == "diclofenac sodium 25 mg, enteric coated_1000_IDA",
                                'Item_Code'])[0]

            pain_management_strategy_paracetamol = {
                'Intervention_Package_Code': dict(),
                'Item_Code': {item_code_paracetamol: 1}}
            pain_management_strategy_diclofenac = {
                'Intervention_Package_Code': dict(),
                'Item_Code': {item_code_diclofenac: 1}}

            if df.iloc[person_id]['age_years'] < 16:
                # If they are under 16 or pregnant (still to do) only give them paracetamol
                logger.debug(pain_management_strategy_paracetamol)
                is_paracetamol_available = self.sim.modules['HealthSystem'].request_consumables(
                    hsi_event=self,
                    cons_req_as_footprint=pain_management_strategy_paracetamol,
                    to_log=True)['Item_Code'][item_code_paracetamol]
                cond = is_paracetamol_available
                logger.debug('Person %d requested paracetamol for their pain relief', person_id)
            else:
                # Multiple options, give them what's available or random pick between them (for now)
                is_diclofenac_available = self.sim.modules['HealthSystem'].request_consumables(
                    hsi_event=self,
                    cons_req_as_footprint=pain_management_strategy_diclofenac,
                    to_log=True)['Item_Code'][item_code_diclofenac]

                is_paracetamol_available = self.sim.modules['HealthSystem'].request_consumables(
                    hsi_event=self,
                    cons_req_as_footprint=pain_management_strategy_paracetamol,
                    to_log=True)['Item_Code'][item_code_paracetamol]

                cond1 = is_paracetamol_available
                cond2 = is_diclofenac_available
                if (cond1 is True) & (cond2 is True):
                    which = self.module.rng.random_sample(size=1)
                    if which <= 0.5:
                        cond = cond1
                        logger.debug('Person %d requested paracetamol for their pain relief', person_id)
                    else:
                        cond = cond2
                        logger.debug('Person %d requested diclofenac for their pain relief', person_id)
                elif (cond1 is True) & (cond2 is False):
                    cond = cond1
                    logger.debug('Person %d requested paracetamol for their pain relief', person_id)
                elif (cond1 is False) & (cond2 is True):
                    cond = cond2
                    logger.debug('Person %d requested diclofenac for their pain relief', person_id)
                else:
                    which = self.module.rng.random_sample(size=1)
                    if which <= 0.5:
                        cond = cond1
                        logger.debug('Person %d requested paracetamol for their pain relief', person_id)
                    else:
                        cond = cond2
                        logger.debug('Person %d requested diclofenac for their pain relief', person_id)
            # Availability of consumables determines if the intervention is delivered...
            if cond:
                logger.debug('This facility has pain management available for mild pain which has been used for '
                             'person %d.', person_id)
                dict_to_output = {'person': person_id,
                                  'pain level': pain_level}
                logger.info(key='Successful_Pain_Management',
                            data=dict_to_output,
                            description='Pain medicine successfully provided to the person')
            else:
                logger.debug('This facility has no pain management available for their mild pain, person %d.',
                             person_id)

        if pain_level is "moderate":
            dict_to_output = {'person': person_id,
                              'pain level': pain_level}
            logger.info(key='Requested_Pain_Management',
                        data=dict_to_output,
                        description='Summary of the pain medicine requested by each person')
            item_code_tramadol = pd.unique(
                consumables.loc[consumables['Items'] == "tramadol HCl 100 mg/2 ml, for injection_100_IDA",
                                'Item_Code'])[0]

            pain_management_strategy_tramadol = {
                'Intervention_Package_Code': dict(),
                'Item_Code': {item_code_tramadol: 1}}

            is_cons_available = self.sim.modules['HealthSystem'].request_consumables(
                hsi_event=self,
                cons_req_as_footprint=pain_management_strategy_tramadol,
                to_log=False)['Item_Code'][item_code_tramadol]
            cond = is_cons_available
            logger.debug('Person %d has requested tramadol for moderate pain relief', person_id)

            if cond:
                logger.debug('This facility has pain management available for moderate pain which has been used for '
                             'person %d.', person_id)
                dict_to_output = {'person': person_id,
                                  'pain level': pain_level}
                logger.info(key='Successful_Pain_Management',
                            data=dict_to_output,
                            description='Pain medicine successfully provided to the person')
            else:
                logger.debug('This facility has no pain management available for moderate pain for person %d.',
                             person_id)

        if pain_level is "severe":
            dict_to_output = {'person': person_id,
                              'pain level': pain_level}
            logger.info(key='Requested_Pain_Management',
                        data=dict_to_output,
                        description='Summary of the pain medicine requested by each person')
            # give morphine
            item_code_morphine = pd.unique(
                consumables.loc[consumables['Items'] == "morphine sulphate 10 mg/ml, 1 ml, injection (nt)_10_IDA",
                                'Item_Code'])[0]

            pain_management_strategy = {
                'Intervention_Package_Code': dict(),
                'Item_Code': {item_code_morphine: 1}}

            is_cons_available = self.sim.modules['HealthSystem'].request_consumables(
                hsi_event=self,
                cons_req_as_footprint=pain_management_strategy,
                to_log=False)
            cond = is_cons_available
            logger.debug('Person %d has requested morphine for severe pain relief', person_id)

            if cond:
                logger.debug('This facility has pain management available for severe pain which has been used for '
                             'person %d.', person_id)
                dict_to_output = {'person': person_id,
                                  'pain level': pain_level}
                logger.info(key='Successful_Pain_Management',
                            data=dict_to_output,
                            description='Pain medicine successfully provided to the person')
            else:
                logger.debug('This facility has no pain management available for severe pain for person %d.', person_id)

    def did_not_run(self, person_id):
        df = self.sim.population.props
        logger.debug('Pain relief unavailable for person %d', person_id)
        injurycodes = {'First injury': df.loc[person_id, 'rt_injury_1'],
                       'Second injury': df.loc[person_id, 'rt_injury_2'],
                       'Third injury': df.loc[person_id, 'rt_injury_3'],
                       'Fourth injury': df.loc[person_id, 'rt_injury_4'],
                       'Fifth injury': df.loc[person_id, 'rt_injury_5'],
                       'Sixth injury': df.loc[person_id, 'rt_injury_6'],
                       'Seventh injury': df.loc[person_id, 'rt_injury_7'],
                       'Eight injury': df.loc[person_id, 'rt_injury_8']}
        logger.debug(f'Injury profile of person %d, {injurycodes}', person_id)


class HSI_RTI_Major_Surgeries(HSI_Event, IndividualScopeEventMixin):
    """This is a Health System Interaction Event.
        An appointment of a person who has experienced a road traffic injury, had their injuries diagnosed through
        A and E and requires major surgery.

        Major surgeries are defined here as surgeries that include extensive work such as entering a body cavity,
        removing an organ or altering the body’s anatomy

        The injuries treated in this module are as follows:

        FRACTURES:
        While district hospitals can provide some
        emergency trauma care and surgeries, only central hospitals
        are equipped to provide advanced orthopaedic surgery. - Lavy et al. 2007

        '112' - Depressed skull fracture - reported use of surgery in Eaton et al. 2017
        '813a' - Fractured hip - reported use of surgery and Lavy et al. 2007
        '813b' - Fractured pelvis - reported use of surgery and Lavy et al. 2007
        '813c' - Fractured femur - reported use of surgery and Lavy et al. 2007
        '414' - Flail chest - https://www.sciencedirect.com/science/article/pii/S0020138303002900

        SOFT TISSUE INJURIES:
        '342' - Soft tissue injury of the neck
        '343' - Soft tissue injury of the neck

        Thoroscopy treated injuries:
        https://www.unthsc.edu/texas-college-of-osteopathic-medicine/wp-content/uploads/sites/9/
        Pediatric_Handbook_for_Malawi.pdf
        '441' - Closed pneumothorax
        '443' - Open pneumothorax
        '463' - Haemothorax
        '453a' - Diaphragm rupture
        '453b' - Lung contusion

        INTERNAL BLEEDING:
        '361' - Internal bleeding in neck
        '363' - Internal bleeding in neck


        TRAUMATIC BRAIN INJURIES THAT REQUIRE A CRANIOTOMOY - reported use of surgery in Eaton et al 2017 and Lavy et
        al. 2007

        '133a' - Subarachnoid hematoma
        '133b' - Brain contusion
        '133c' - Intraventricular haemorrhage
        '133d' - Subgaleal hematoma
        '134a' - Epidural hematoma
        '134b' - Subdural hematoma
        '135' - diffuse axonal injury

        Laparotomy - Recorded in Lavy et al. 2007 and here: https://www.ajol.info/index.php/mmj/article/view/174378

        '552' - Injury to Intestine, stomach and colon
        '553' - Injury to Spleen, Urinary bladder, Liver, Urethra, Diaphragm
        '554' - Injury to kidney


        SPINAL CORD LESIONS, REQUIRING LAMINOTOMY/FORAMINOTOMY/INTERSPINOUS PROCESS SPACER
        Quote from Eaton et al. 2019:
        "No patients received thoracolumbar braces or underwent spinal surgery."
        https://journals.sagepub.com/doi/pdf/10.1177/0049475518808969
        So those with spinal cord injuries are not likely to be treated here in RTI_Major_Surgeries..

        '673a' - Spinal cord lesion at neck level
        '673b' - Spinal cord lesion below neck level
        '674a' - Spinal cord lesion at neck level
        '674b' - Spinal cord lesion below neck level
        '675a' - Spinal cord lesion at neck level
        '675b' - Spinal cord lesion below neck level
        '676' - Spinal cord lesion at neck level

        AMPUTATIONS - Reported in Crudziak et al. 2019
        '782a' - Amputated finger
        '782b' - Unilateral arm amputation
        '782c' - Amputated thumb
        '783' - Bilateral arm amputation
        '882' - Amputated toe
        '883' - Unilateral lower limb amputation
        '884' - Bilateral lower limb amputation
        """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, RTI)
        self.TREATMENT_ID = 'RTI_Major_Surgeries'
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['MajorSurg'] = 1  # This requires major surgery

        the_accepted_facility_level = 1
        p = self.module.parameters

        # Define the necessary information for an HSI
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = the_accepted_facility_level
        self.ALERT_OTHER_DISEASES = []
        self.prob_perm_disability_with_treatment_severe_TBI = p['prob_perm_disability_with_treatment_severe_TBI']
        self.prob_perm_disability_with_treatment_sci = p['prob_perm_disability_with_treatment_sci']
        self.allowed_interventions = p['allowed_interventions']
        self.treated_code = 'none'

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        rng = self.module.rng
        road_traffic_injuries = self.sim.modules['RTI']
        # check the people sent here are alive, have had their injuries diagnosed and been through RTI_Med
        assert df.loc[person_id, 'is_alive'] & df.loc[person_id, 'rt_diagnosed'] & df.loc[person_id, 'rt_med_int']
        # Isolate the relevant injury information
        columns = ['rt_injury_1', 'rt_injury_2', 'rt_injury_3', 'rt_injury_4', 'rt_injury_5', 'rt_injury_6',
                   'rt_injury_7', 'rt_injury_8']
        surgically_treated_codes = ['112', '813a', '813b', '813c', '133a', '133b', '133c', '133d', '134a',
                                    '134b', '135', '552', '553', '554', '342', '343', '414', '361', '363',
                                    '782', '782a', '782b', '782c', '783', '882', '883', '884',
                                    'P133a', 'P133b', 'P133c', 'P133d', 'P134a', 'P134b', 'P135', 'P782a', 'P782b',
                                    'P782c', 'P783', 'P882', 'P883', 'P884']
        # If we have allowed spinal cord surgeries to be treated in this simulation, include the associated injury codes
        # here
        if 'include_spine_surgery' in self.allowed_interventions:
            additional_codes = ['673a', '673b', '674a', '674b', '675a', '675b', '676', 'P673a', 'P673b', 'P674',
                                'P674a', 'P674b', 'P675', 'P675a', 'P675b', 'P676']
            for code in additional_codes:
                surgically_treated_codes.append(code)
        # If we have allowed greater access to thoroscopy, include the codes treated by thoroscopy here
        if 'include_thoroscopy' in self.allowed_interventions:
            additional_codes = ['441', '443', '453', '453a', '453b', '463']
            for code in additional_codes:
                surgically_treated_codes.append(code)
        persons_injuries = df.loc[[person_id], columns]
        # check the people sent here have at least one injury treated by this HSI event
        idx, counts = road_traffic_injuries.rti_find_and_count_injuries(persons_injuries, surgically_treated_codes)
        assert counts > 0
        # People can be sent here for multiple surgeries, but only one injury can be treated at a time. Decide which
        # injury is being treated in this surgery
        idx_for_untreated_injuries = []
        for index, time in enumerate(df.loc[person_id, 'rt_date_to_remove_daly']):
            if pd.isnull(time):
                idx_for_untreated_injuries.append(index)

        relevant_codes = np.intersect1d(persons_injuries.values[0][idx_for_untreated_injuries],
                                        surgically_treated_codes)
        code = rng.choice(relevant_codes)
        self.treated_code = code
        # ------------------------ Track permanent disabilities with treatment ----------------------------------------
        # --------------------------------- Perm disability from TBI --------------------------------------------------
        codes = ['133', '133a', '133b', '133c', '133d', '134', '134a', '134b', '135']

        """ Of patients that survived, 80.1% (n 148) had a good recovery with no appreciable clinical neurologic
        deficits, 13.1% (n 24) had a moderate disability with deficits that still allowed the patient to live
        independently, 4.9% (n 9) had severe disability which will require assistance with activities of daily life,
        and 1.1% (n 2) were in a vegetative state
        """
        # Check whether the person having treatment for their tbi will be left permanently disabled
        if (code in codes) & ('include_spine_surgery' in self.allowed_interventions):
            prob_perm_disability = self.module.rng.random_sample(size=1)
            if prob_perm_disability < self.prob_perm_disability_with_treatment_severe_TBI:
                # Track whether they are permanently disabled
                df.at[person_id, 'rt_perm_disability'] = True
                # Find the column and code where the permanent injury is stored
                column, code = road_traffic_injuries.rti_find_injury_column(person_id=person_id, codes=codes)
                logger.debug('@@@@@@@@@@ Person %d had intervention for TBI on %s but still disabled!!!!!!',
                             person_id, self.sim.date)
                # Update the code to make the injury permanent, so it will not have the associated daly weight removed
                # later on
                df.loc[person_id, column] = "P" + code
                code = df.loc[person_id, column]
                columns, codes = road_traffic_injuries.rti_find_all_columns_of_treated_injuries(person_id,
                                                                                                [code])
                for col in columns:
                    # schedule the recovery date for the permanent injury for beyond the end of the simulation (making
                    # it permanent)
                    df.loc[person_id, 'rt_date_to_remove_daly'][int(col[-1]) - 1] = self.sim.end_date + \
                                                                                    DateOffset(days=1)
        # ------------------------------------- Perm disability from SCI ----------------------------------------------
        if 'include_spine_surgery' in self.allowed_interventions:
            codes = ['673', '673a', '673b', '674', '674a', '674b', '675', '675a', '675b', '676']
            if code in codes:
                # Track whether they are permanently disabled
                df.at[person_id, 'rt_perm_disability'] = True
                # Find the column and code where the permanent injury is stored
                column, code = road_traffic_injuries.rti_find_injury_column(person_id=person_id, codes=codes)
                logger.debug('@@@@@@@@@@ Person %d had intervention for SCI on %s but still disabled!!!!!!',
                             person_id, self.sim.date)
                df.loc[person_id, column] = "P" + code
                code = df.loc[person_id, column]
                columns, codes = road_traffic_injuries.rti_find_all_columns_of_treated_injuries(person_id, [code])
                for col in columns:
                    # schedule the recovery date for the permanent injury for beyond the end of the simulation (making
                    # it permanent)
                    df.loc[person_id, 'rt_date_to_remove_daly'][int(col[-1]) - 1] = self.sim.end_date + \
                                                                                    DateOffset(days=1)
        # ------------------------------------- Perm disability from amputation ----------------------------------------
        codes = ['782', '782a', '782b', '782c', '783', '882', '883', '884']
        if code in codes:
            # Track whether they are permanently disabled
            df.at[person_id, 'rt_perm_disability'] = True
            # Find the column and code where the permanent injury is stored
            column, code = road_traffic_injuries.rti_find_injury_column(person_id=person_id, codes=codes)
            logger.debug('@@@@@@@@@@ Person %d had intervention for an amputation on %s but still disabled!!!!!!',
                         person_id, self.sim.date)
            # Update the code to make the injury permanent, so it will not have the associated daly weight removed
            # later on
            df.loc[person_id, column] = "P" + code
            code = df.loc[person_id, column]
            columns, codes = road_traffic_injuries.rti_find_all_columns_of_treated_injuries(person_id,
                                                                                            [code])
            # Schedule recovery for the end of the simulation, thereby making the injury permanent
            for col in columns:
                df.loc[person_id, 'rt_date_to_remove_daly'][int(col[-1]) - 1] = self.sim.end_date + \
                                                                                DateOffset(days=1)
        # ============================== Schedule the recovery dates for the non-permanent injuries ==================
        non_empty_injuries = persons_injuries[persons_injuries != "none"]
        non_empty_injuries = non_empty_injuries.dropna(axis=1)
        injury_columns = non_empty_injuries.columns
        if code == '112':
            columns = injury_columns.get_loc(road_traffic_injuries.rti_find_injury_column(person_id,
                                                                                          [code])[0])
            # using estimated 6 weeks to recover from brain/head injury surgery

            # performing check to see whether an injury is deemed to heal over time, if it is, then we change the code
            # this scheduled surgery treats
            if pd.isnull(df.loc[person_id, 'rt_date_to_remove_daly'][columns]):
                df.loc[person_id, 'rt_date_to_remove_daly'][columns] = self.sim.date + DateOffset(weeks=6)
            else:
                non_empty_injuries.drop(non_empty_injuries.columns[columns], axis=1, inplace=True)
                relevant_codes = np.intersect1d(non_empty_injuries.values, surgically_treated_codes)
                code = rng.choice(relevant_codes)
        if code == '552' or code == '553' or code == '554':
            columns = injury_columns.get_loc(road_traffic_injuries.rti_find_injury_column(person_id,
                                                                                          [code])[0])
            # using estimated 3 months to recover from laparotomy
            if pd.isnull(df.loc[person_id, 'rt_date_to_remove_daly'][columns]):
                df.loc[person_id, 'rt_date_to_remove_daly'][columns] = self.sim.date + DateOffset(months=3)
            else:
                non_empty_injuries.drop(non_empty_injuries.columns[columns], axis=1, inplace=True)
                relevant_codes = np.intersect1d(non_empty_injuries.values, surgically_treated_codes)
                assert len(relevant_codes) > 0
                code = rng.choice(relevant_codes)
        if code == '813a':
            columns = injury_columns.get_loc(road_traffic_injuries.rti_find_injury_column(person_id,
                                                                                          [code])[0])
            # using estimated 6 - 12 months to recover from a hip fracture
            df.loc[person_id, 'rt_date_to_remove_daly'][columns] = self.sim.date + DateOffset(months=9)
        if code == '813b':
            columns = injury_columns.get_loc(road_traffic_injuries.rti_find_injury_column(person_id,
                                                                                          [code])[0])
            # using estimated 8 - 12 weeks to recover from a pelvis fracture
            df.loc[person_id, 'rt_date_to_remove_daly'][columns] = self.sim.date + DateOffset(weeks=10)
        if code == '813c':
            columns = injury_columns.get_loc(road_traffic_injuries.rti_find_injury_column(person_id,
                                                                                          [code])[0])
            # using estimated 3 - 6 months to recover from a femur fracture
            df.loc[person_id, 'rt_date_to_remove_daly'][columns] = self.sim.date + DateOffset(months=4)
        tbi_codes = ['133a', '133b', '133c', '133d', '134a', '134b', '135']
        if code in tbi_codes:
            columns = injury_columns.get_loc(road_traffic_injuries.rti_find_injury_column(person_id, [code])[0])
            # using estimated 6 weeks to recover from brain/head injury surgery
            df.loc[person_id, 'rt_date_to_remove_daly'][columns] = self.sim.date + DateOffset(weeks=6)
        if code == '342':
            columns = injury_columns.get_loc(road_traffic_injuries.rti_find_injury_column(person_id,
                                                                                          [code])[0])
            # using estimated 6 weeks PLACEHOLDER FOR VERTEBRAL ARTERY LACERATION
            df.loc[person_id, 'rt_date_to_remove_daly'][columns] = self.sim.date + DateOffset(weeks=6)
        if code == '343':
            columns = injury_columns.get_loc(road_traffic_injuries.rti_find_injury_column(person_id,
                                                                                          [code])[0])
            # using estimated 6 weeks PLACEHOLDER FOR PHARYNX CONTUSION
            df.loc[person_id, 'rt_date_to_remove_daly'][columns] = self.sim.date + DateOffset(weeks=6)
        if code == '414':
            columns = injury_columns.get_loc(road_traffic_injuries.rti_find_injury_column(person_id,
                                                                                          [code])[0])
            # using estimated 1 year recovery for flail chest
            df.loc[person_id, 'rt_date_to_remove_daly'][columns] = self.sim.date + DateOffset(years=1)
        if code == '441' or code == '443':
            columns = injury_columns.get_loc(road_traffic_injuries.rti_find_injury_column(person_id,
                                                                                          [code])[0])
            # using estimated 1 - 2 week recovery time for pneumothorax
            df.loc[person_id, 'rt_date_to_remove_daly'][columns] = self.sim.date + DateOffset(weeks=2)
        if code == '453a':
            columns = injury_columns.get_loc(road_traffic_injuries.rti_find_injury_column(person_id,
                                                                                          [code])[0])
            # using estimated 6 weeks PLACEHOLDER FOR DIAPHRAGM RUPTURE
            df.loc[person_id, 'rt_date_to_remove_daly'][columns] = self.sim.date + DateOffset(weeks=6)
        if code == '361' or code == '363' or code == '463':
            columns = injury_columns.get_loc(road_traffic_injuries.rti_find_injury_column(person_id,
                                                                                          [code])[0])
            # using estimated 1 weeks PLACEHOLDER FOR INTERNAL BLEEDING
            df.loc[person_id, 'rt_date_to_remove_daly'][columns] = self.sim.date + DateOffset(weeks=1)
        swapping_codes = ['712b', '812', '3113', '4113', '5113', '7113', '8113', '813a', '813b', 'P673a', 'P673b',
                          'P674a', 'P674b', 'P675a', 'P675b', 'P676', 'P782b', 'P783', 'P883', 'P884']
        if code in swapping_codes:
            road_traffic_injuries.rti_swap_injury_daly_upon_treatment(person_id, [code])
        logger.debug('This is RTI_Major_Surgeries supplying surgery for person %d on date %s!!!!!!, removing code %s',
                     person_id, self.sim.date)

    def did_not_run(self, person_id):
        df = self.sim.population.props
        logger.debug('Major surgery not scheduled for person %d', person_id)
        injurycodes = {'First injury': df.loc[person_id, 'rt_injury_1'],
                       'Second injury': df.loc[person_id, 'rt_injury_2'],
                       'Third injury': df.loc[person_id, 'rt_injury_3'],
                       'Fourth injury': df.loc[person_id, 'rt_injury_4'],
                       'Fifth injury': df.loc[person_id, 'rt_injury_5'],
                       'Sixth injury': df.loc[person_id, 'rt_injury_6'],
                       'Seventh injury': df.loc[person_id, 'rt_injury_7'],
                       'Eight injury': df.loc[person_id, 'rt_injury_8']}
        logger.debug(f'Injury profile of person %d, {injurycodes}', person_id)

        # If the surgery was a life-saving surgery, then send them to RTI_No_Medical_Intervention_Death_Event
        life_threatening_injuries = ['133a', '133b', '133c', '133d', '134a', '134b', '135',  # TBI
                                     '112',  # Depressed skull fracture
                                     'P133a', 'P133b', 'P133c', 'P133d', 'P134a', 'P134b', 'P135',  # Perm TBI
                                     '342', '343', '361', '363',  # Injuries to neck
                                     '414', '441', '443', '463', '453a', '453b',  # Severe chest trauma
                                     '782b',  # Unilateral arm amputation
                                     '783',  # Bilateral arm amputation
                                     '883',  # Unilateral lower limb amputation
                                     '884',  # Bilateral lower limb amputation
                                     '552', '553', '554'  # Internal organ injuries
                                     ]
        if (self.treated_code in life_threatening_injuries) & df.loc[person_id, 'is_alive']:
            self.sim.schedule_event(RTI_No_Lifesaving_Medical_Intervention_Death_Event(self.module, person_id),
                                    self.sim.date)


class HSI_RTI_Minor_Surgeries(HSI_Event, IndividualScopeEventMixin):
    """This is a Health System Interaction Event.
        An appointment of a person who has experienced a road traffic injury, had their injuries diagnosed through
        A and E, treatment plan organised by RTI_MedInt and requires minor surgery.

        Minor surgeries are defined here as surgeries are generally superficial and do not require penetration of a
        body cavity. They do not involve assisted breathing or anesthesia and are usually performed by a single doctor.

        The injuries treated in this module are as follows:

        Evidence for all from Mkandawire et al. 2008:
        https://link.springer.com/content/pdf/10.1007/s11999-008-0366-5.pdf
        '211' - Facial fractures
        '212' - Facial fractures
        '291' - Injury to the eye
        '241' - Soft tissue injury of the face

        '322' - Dislocation in the neck
        '323' - Dislocation in the neck

        '722' - Dislocated shoulder
        '822a' - Dislocated hip
        '822b' - Dislocated knee
        """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, RTI)
        self.TREATMENT_ID = 'RTI_Minor_Surgeries'
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['MinorSurg'] = 1  # This requires major surgery

        the_accepted_facility_level = 1

        # Define the necessary information for an HSI
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = the_accepted_facility_level
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        rng = self.module.rng
        road_traffic_injuries = self.sim.modules['RTI']
        columns = ['rt_injury_1', 'rt_injury_2', 'rt_injury_3', 'rt_injury_4', 'rt_injury_5', 'rt_injury_6',
                   'rt_injury_7', 'rt_injury_8']
        surgically_treated_codes = ['322', '211', '212', '323', '722', '822a', '822b', '291', '241']
        persons_injuries = df.loc[[person_id], columns]
        # =========================================== Tests ============================================================
        # check alive, diagnosed by a and e, housed by rt_med_int
        assert df.loc[person_id, 'is_alive'] & df.loc[person_id, 'rt_diagnosed'] & df.loc[person_id, 'rt_med_int']
        # check they have at least one injury treated by minor surgery
        idx, counts = road_traffic_injuries.rti_find_and_count_injuries(persons_injuries, surgically_treated_codes)
        assert counts > 0
        non_empty_injuries = persons_injuries[persons_injuries != "none"]
        non_empty_injuries = non_empty_injuries.dropna(axis=1)
        relevant_codes = np.intersect1d(non_empty_injuries.values, surgically_treated_codes)
        # Check that a code has been selected to be treated
        assert len(relevant_codes) > 0
        code = rng.choice(relevant_codes)
        injury_columns = non_empty_injuries.columns
        if code == '322' or code == '323':
            columns = injury_columns.get_loc(road_traffic_injuries.rti_find_injury_column(person_id,
                                                                                          [code])[0])

            # using estimated 6 months to recover from neck surgery
            if pd.isnull(df.loc[person_id, 'rt_date_to_remove_daly'][columns]):
                df.loc[person_id, 'rt_date_to_remove_daly'][columns] = self.sim.date + DateOffset(months=6)
            else:
                non_empty_injuries.drop(non_empty_injuries.columns[columns], axis=1, inplace=True)
                relevant_codes = np.intersect1d(non_empty_injuries.values, surgically_treated_codes)
                assert len(relevant_codes) > 0
                code = rng.choice(relevant_codes)

        if code == '722':
            columns = injury_columns.get_loc(road_traffic_injuries.rti_find_injury_column(person_id,
                                                                                          [code])[0])
            # using estimated 7 weeks to recover from dislocated shoulder surgery
            if pd.isnull(df.loc[person_id, 'rt_date_to_remove_daly'][columns]):
                df.loc[person_id, 'rt_date_to_remove_daly'][columns] = self.sim.date + DateOffset(weeks=7)
            else:
                non_empty_injuries.drop(non_empty_injuries.columns[columns], axis=1, inplace=True)
                relevant_codes = np.intersect1d(non_empty_injuries.values, surgically_treated_codes)
                code = rng.choice(relevant_codes)
        if code == '822a':
            columns = injury_columns.get_loc(road_traffic_injuries.rti_find_injury_column(person_id,
                                                                                          [code])[0])
            # using estimated 7 weeks to recover from dislocated hip surgery
            if pd.isnull(df.loc[person_id, 'rt_date_to_remove_daly'][columns]):
                df.loc[person_id, 'rt_date_to_remove_daly'][columns] = self.sim.date + DateOffset(weeks=7)
            else:
                non_empty_injuries.drop(non_empty_injuries.columns[columns], axis=1, inplace=True)
                relevant_codes = np.intersect1d(non_empty_injuries.values, surgically_treated_codes)
                code = rng.choice(relevant_codes)
        if code == '822b':
            columns = injury_columns.get_loc(road_traffic_injuries.rti_find_injury_column(person_id,
                                                                                          [code])[0])
            # using estimated 7 weeks to recover from dislocated knee surgery
            if pd.isnull(df.loc[person_id, 'rt_date_to_remove_daly'][columns]):
                df.loc[person_id, 'rt_date_to_remove_daly'][columns] = self.sim.date + DateOffset(weeks=7)
            else:
                non_empty_injuries.drop(non_empty_injuries.columns[columns], axis=1, inplace=True)
                relevant_codes = np.intersect1d(non_empty_injuries.values, surgically_treated_codes)
                code = rng.choice(relevant_codes)

        if code == '211' or code == '212':
            columns = injury_columns.get_loc(road_traffic_injuries.rti_find_injury_column(person_id,
                                                                                          [code])[0])
            # using estimated 7 weeks to recover from facial fracture surgery
            df.loc[person_id, 'rt_date_to_remove_daly'][columns] = self.sim.date + DateOffset(weeks=7)
        if code == '291':
            columns = injury_columns.get_loc(road_traffic_injuries.rti_find_injury_column(person_id,
                                                                                          [code])[0])
            # using estimated 1 week to recover from eye surgery
            df.loc[person_id, 'rt_date_to_remove_daly'][columns] = self.sim.date + DateOffset(weeks=1)
        if code == '241':
            columns = injury_columns.get_loc(road_traffic_injuries.rti_find_injury_column(person_id,
                                                                                          [code])[0])
            # using estimated 1 week to recover from eye surgery
            df.loc[person_id, 'rt_date_to_remove_daly'][columns] = self.sim.date + DateOffset(month=1)
        swapping_codes = ['712b', '812', '3113', '4113', '5113', '7113', '8113', '813a', '813b', 'P673a', 'P673b',
                          'P674a', 'P674b', 'P675a', 'P675b', 'P676', 'P782b', 'P783', 'P883', 'P884']
        if code in swapping_codes:
            road_traffic_injuries.rti_swap_injury_daly_upon_treatment(person_id, [code])
        logger.debug('This is RTI_Minor_Surgeries supplying minor surgeries for person %d on date %s!!!!!!',
                     person_id, self.sim.date)
        df.at[person_id, 'rt_med_int'] = True

    def did_not_run(self, person_id):
        df = self.sim.population.props
        logger.debug('Minor surgery not scheduled for person %d', person_id)
        injurycodes = {'First injury': df.loc[person_id, 'rt_injury_1'],
                       'Second injury': df.loc[person_id, 'rt_injury_2'],
                       'Third injury': df.loc[person_id, 'rt_injury_3'],
                       'Fourth injury': df.loc[person_id, 'rt_injury_4'],
                       'Fifth injury': df.loc[person_id, 'rt_injury_5'],
                       'Sixth injury': df.loc[person_id, 'rt_injury_6'],
                       'Seventh injury': df.loc[person_id, 'rt_injury_7'],
                       'Eight injury': df.loc[person_id, 'rt_injury_8']}
        logger.debug(f'Injury profile of person %d, {injurycodes}', person_id)


class RTI_Medical_Intervention_Death_Event(Event, IndividualScopeEventMixin):
    """This is the MedicalInterventionDeathEvent. It is scheduled by the MedicalInterventionEvent to occur at the end of
     the person's determined length of stay. The risk of mortality for the person wil medical intervention is determined
     by the persons ISS score and whether they have polytrauma.
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        p = self.module.parameters
        self.prob_death_with_med_mild = p['prob_death_with_med_mild']
        self.prob_death_with_med_severe = p['prob_death_with_med_severe']
        self.rr_injrti_mortality_polytrauma = p['rr_injrti_mortality_polytrauma']

    def apply(self, person_id):
        df = self.sim.population.props
        randfordeath = self.module.rng.random_sample(size=1)
        # ======================================== Tests ==============================================================
        # Check the person sent here is alive
        assert df.loc[person_id, 'is_alive'] or (df.loc[person_id, 'cause_of_death'] == 'Other')
        # Schedule death for those who died from their injuries despite medical intervention
        if df.loc[person_id, 'cause_of_death'] == 'Other':
            pass
        if df.loc[person_id, 'rt_inj_severity'] == 'mild':
            # Check if the people with mild injuries have the polytrauma property, if so, judge mortality accordingly
            if df.loc[person_id, 'rt_polytrauma'] is True:
                if randfordeath < self.prob_death_with_med_mild * self.rr_injrti_mortality_polytrauma:
                    df.loc[person_id, 'rt_post_med_death'] = True
                    dict_to_output = {'person': person_id,
                                      'First injury': df.loc[person_id, 'rt_injury_1'],
                                      'Second injury': df.loc[person_id, 'rt_injury_2'],
                                      'Third injury': df.loc[person_id, 'rt_injury_3'],
                                      'Fourth injury': df.loc[person_id, 'rt_injury_4'],
                                      'Fifth injury': df.loc[person_id, 'rt_injury_5'],
                                      'Sixth injury': df.loc[person_id, 'rt_injury_6'],
                                      'Seventh injury': df.loc[person_id, 'rt_injury_7'],
                                      'Eight injury': df.loc[person_id, 'rt_injury_8']}
                    logger.info(key='RTI_Death_Injury_Profile',
                                data=dict_to_output,
                                description='The injury profile of those who have died due to rtis despite medical care'
                                )
                    # Schedule the death
                    self.sim.schedule_event(demography.InstantaneousDeath(self.module, person_id,
                                                                          cause='RTI_death_with_med'), self.sim.date)
                    # Log the death
                    logger.debug('This is RTIMedicalInterventionDeathEvent scheduling a death for person %d who was '
                                 'treated for their injuries but still died on date %s',
                                 person_id, self.sim.date)
            elif randfordeath < self.prob_death_with_med_mild:
                # No polytrauma
                df.loc[person_id, 'rt_post_med_death'] = True
                dict_to_output = {'person': person_id,
                                  'First injury': df.loc[person_id, 'rt_injury_1'],
                                  'Second injury': df.loc[person_id, 'rt_injury_2'],
                                  'Third injury': df.loc[person_id, 'rt_injury_3'],
                                  'Fourth injury': df.loc[person_id, 'rt_injury_4'],
                                  'Fifth injury': df.loc[person_id, 'rt_injury_5'],
                                  'Sixth injury': df.loc[person_id, 'rt_injury_6'],
                                  'Seventh injury': df.loc[person_id, 'rt_injury_7'],
                                  'Eight injury': df.loc[person_id, 'rt_injury_8']}
                logger.info(key='RTI_Death_Injury_Profile',
                            data=dict_to_output,
                            description='The injury profile of those who have died due to rtis despite medical care'
                            )
                # Schedule the death
                self.sim.schedule_event(demography.InstantaneousDeath(self.module, person_id,
                                                                      cause='RTI_death_with_med'), self.sim.date)
                # Log the death
                logger.debug('This is RTIMedicalInterventionDeathEvent scheduling a death for person %d who was'
                             'treated for their injuries but still died on date %s',
                             person_id, self.sim.date)
            else:
                logger.debug('RTIMedicalInterventionDeathEvent determining that person %d was treated for injuries and '
                             'survived on date %s',
                             person_id, self.sim.date)
        if df.loc[person_id, 'rt_inj_severity'] == 'severe':
            if df.loc[person_id, 'rt_polytrauma'] is True:
                # Predict death if they have polytrauma for severe injuries
                if randfordeath < self.prob_death_with_med_severe * self.rr_injrti_mortality_polytrauma:
                    df.loc[person_id, 'rt_post_med_death'] = True
                    dict_to_output = {'person': person_id,
                                      'First injury': df.loc[person_id, 'rt_injury_1'],
                                      'Second injury': df.loc[person_id, 'rt_injury_2'],
                                      'Third injury': df.loc[person_id, 'rt_injury_3'],
                                      'Fourth injury': df.loc[person_id, 'rt_injury_4'],
                                      'Fifth injury': df.loc[person_id, 'rt_injury_5'],
                                      'Sixth injury': df.loc[person_id, 'rt_injury_6'],
                                      'Seventh injury': df.loc[person_id, 'rt_injury_7'],
                                      'Eight injury': df.loc[person_id, 'rt_injury_8']}
                    logger.info(key='RTI_Death_Injury_Profile',
                                data=dict_to_output,
                                description='The injury profile of those who have died due to rtis despite medical care'
                                )
                    # Schedule the death
                    self.sim.schedule_event(demography.InstantaneousDeath(self.module, person_id,
                                                                          cause='RTI_death_with_med'), self.sim.date)
                    # Log the death
                    logger.debug('This is RTIMedicalInterventionDeathEvent scheduling a death for person %d who was '
                                 'treated for their injuries but still died on date %s',
                                 person_id, self.sim.date)
                elif randfordeath < self.prob_death_with_med_severe:
                    # Predict death without polytrauma for severe injuries
                    df.loc[person_id, 'rt_post_med_death'] = True
                    dict_to_output = {'person': person_id,
                                      'First injury': df.loc[person_id, 'rt_injury_1'],
                                      'Second injury': df.loc[person_id, 'rt_injury_2'],
                                      'Third injury': df.loc[person_id, 'rt_injury_3'],
                                      'Fourth injury': df.loc[person_id, 'rt_injury_4'],
                                      'Fifth injury': df.loc[person_id, 'rt_injury_5'],
                                      'Sixth injury': df.loc[person_id, 'rt_injury_6'],
                                      'Seventh injury': df.loc[person_id, 'rt_injury_7'],
                                      'Eight injury': df.loc[person_id, 'rt_injury_8']}
                    logger.info(key='RTI_Death_Injury_Profile',
                                data=dict_to_output,
                                description='The injury profile of those who have died due to rtis despite medical care'
                                )
                    self.sim.schedule_event(demography.InstantaneousDeath(self.module, person_id,
                                                                          cause='RTI_death_with_med'), self.sim.date)
                    # Log the death
                    logger.debug('This is RTIMedicalInterventionDeathEvent scheduling a death for person %d who was'
                                 'treated for their injuries but still died on date %s',
                                 person_id, self.sim.date)
            else:
                logger.debug('RTIMedicalInterventionDeathEvent has determined person %d was '
                             'treated for injuries and survived on date %s',
                             person_id, self.sim.date)


class RTI_No_Lifesaving_Medical_Intervention_Death_Event(Event, IndividualScopeEventMixin):
    """This is the NoMedicalInterventionDeathEvent. It is scheduled by the MedicalInterventionEvent which determines the
    resources required to treat that person and if they aren't present, the person is sent here. This function is also
    called by the did not run function for rti_major_surgeries for certain injuries, implying that if life saving
    surgery is not available for the person, then we have to ask the probability of them dying without having this life
    saving surgery.

    some information on time to craniotomy here:
    https://thejns.org/focus/view/journals/neurosurg-focus/45/6/article-pE2.xml?body=pdf-10653
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        p = self.module.parameters
        # load the parameteres used for this event
        self.prob_death_TBI_SCI_no_treatment = p['prob_death_TBI_SCI_no_treatment']
        self.prob_death_fractures_no_treatment = p['prob_death_fractures_no_treatment']
        self.prop_death_burns_no_treatment = p['prop_death_burns_no_treatment']
        # self.scheduled_death = 0

    def apply(self, person_id):
        # self.scheduled_death = 0
        df = self.sim.population.props
        # =========================================== Tests ============================================================
        # Check that the people sent here are alive
        assert df.loc[person_id, 'is_alive'], f'person died before being sent here {person_id}'
        columns = ['rt_injury_1', 'rt_injury_2', 'rt_injury_3', 'rt_injury_4', 'rt_injury_5', 'rt_injury_6',
                   'rt_injury_7', 'rt_injury_8']
        non_lethal_injuries = ['241', '291', '322', '323', '461', '442', '1101', '2101', '3101', '4101', '5101', '7101',
                               '8101', '722', '822a', '822b']
        severeinjurycodes = ['133', '133a', '133b', '133c', '133d', '134', '134a', '134b', '135',
                             '342', '343', '361', '363', '414', '441', '443', '453a', '453b', '463', '552',
                             '553', '554', '673', '673a', '673b', '674', '674a', '674b', '675', '675a', '675b',
                             '676', '782a', '782b', '782c', '783', '882', '883', '884', 'P133', 'P133a', 'P133b',
                             'P133c', 'P133d', 'P134', 'P134a', 'P134b', 'P135', 'P673', 'P673a', 'P673b', 'P674',
                             'P674a', 'P674b', 'P675', 'P675a', 'P675b', 'P676', 'P782a', 'P782b', 'P782c', 'P783',
                             'P882', 'P883', 'P884']
        fractureinjurycodes = ['112', '113', '211', '212', '412', '612', '712', '712a', '712b', '712c', '811',
                               '812', '813', '813a', '813b', '813c']
        burninjurycodes = ['1114', '2114', '3113', '4113', '5113', '7113', '8113']
        persons_injuries = df.loc[[person_id], columns]
        non_empty_injuries = persons_injuries[persons_injuries != "none"]
        non_empty_injuries = non_empty_injuries.dropna(axis=1)
        untreated_injuries = []
        prob_death = 0
        # Find which injuries are left untreated by finding injuries which haven't been set a recovery time
        for col in non_empty_injuries:
            if pd.isnull(df.loc[person_id, 'rt_date_to_remove_daly'][int(col[-1]) - 1]):
                untreated_injuries.append(df.at[person_id, col])
        for injury in untreated_injuries:
            if injury in severeinjurycodes:
                if prob_death < self.prob_death_TBI_SCI_no_treatment:
                    prob_death = self.prob_death_TBI_SCI_no_treatment
            elif injury in fractureinjurycodes:
                if prob_death < self.prob_death_fractures_no_treatment:
                    prob_death = self.prob_death_fractures_no_treatment
            elif injury in burninjurycodes:
                if prob_death < self.prop_death_burns_no_treatment:
                    prob_death = self.prop_death_burns_no_treatment
            elif injury in non_lethal_injuries:
                pass
        randfordeath = self.module.rng.random_sample(size=1)
        if randfordeath < prob_death:
            df.loc[person_id, 'rt_no_med_death'] = True
            self.sim.schedule_event(demography.InstantaneousDeath(self.module, person_id,
                                                                  cause='RTI_unavailable_med'), self.sim.date)
            # Log the death
            logger.debug(
                'This is RTINoMedicalInterventionDeathEvent scheduling a death for person %d on date %s',
                person_id, self.sim.date)


# ---------------------------------------------------------------------------------------------------------
#   LOGGING EVENTS
#
#   Put the logging events here. There should be a regular logger outputting current states of the
#   population. There may also be a loggig event that is driven by particular events.
# ---------------------------------------------------------------------------------------------------------


class RTI_Logging_Event(RegularEvent, PopulationScopeEventMixin):

    def __init__(self, module):
        """Produce a summary of the numbers of people with respect to the action of this module.
        This is a regular event that can output current states of people or cumulative events since last logging event.
        """

        # run this event every month
        self.repeat = 1
        super().__init__(module, frequency=DateOffset(months=self.repeat))
        assert isinstance(module, RTI)
        # Create variables used to store simulation data in
        # Number of injured body region data
        self.tot1inj = 0
        self.tot2inj = 0
        self.tot3inj = 0
        self.tot4inj = 0
        self.tot5inj = 0
        self.tot6inj = 0
        self.tot7inj = 0
        self.tot8inj = 0
        # Injury category data
        self.totfracnumber = 0
        self.totdisnumber = 0
        self.tottbi = 0
        self.totsoft = 0
        self.totintorg = 0
        self.totintbled = 0
        self.totsci = 0
        self.totamp = 0
        self.toteye = 0
        self.totextlac = 0
        self.totburns = 0
        # Injury location data
        self.totAIS1 = 0
        self.totAIS2 = 0
        self.totAIS3 = 0
        self.totAIS4 = 0
        self.totAIS5 = 0
        self.totAIS6 = 0
        self.totAIS7 = 0
        self.totAIS8 = 0
        # Injury severity data
        self.totmild = 0
        self.totsevere = 0
        # More general RTI data
        self.totinjured = 0
        self.deathonscene = 0
        self.soughtmedcare = 0
        self.deathaftermed = 0
        self.deathwithoutmed = 0
        self.permdis = 0
        self.ISSscore = []
        self.numerator = 0
        self.denominator = 0
        self.death_inc_numerator = 0
        self.death_in_denominator = 0
        self.fracdenominator = 0
        self.fracdist = [0, 0, 0, 0, 0, 0, 0, 0]
        self.openwounddist = [0, 0, 0, 0, 0, 0, 0, 0]
        self.burndist = [0, 0, 0, 0, 0, 0, 0, 0]
        self.severe_pain = 0
        self.moderate_pain = 0
        self.mild_pain = 0
        self.rti_demographics = pd.DataFrame()

    def apply(self, population):
        # Make some summary statitics
        df = population.props
        road_traffic_injuries = self.sim.modules['RTI']
        columns = ['rt_injury_1', 'rt_injury_2', 'rt_injury_3', 'rt_injury_4', 'rt_injury_5', 'rt_injury_6',
                   'rt_injury_7', 'rt_injury_8']
        thoseininjuries = df.loc[df.rt_road_traffic_inc]
        df_injuries = thoseininjuries.loc[:, columns]
        # ==================================== Number of injuries =====================================================
        oneinjury = len(df_injuries.loc[df_injuries['rt_injury_2'] == 'none'])
        self.tot1inj += oneinjury
        twoinjury = len(df_injuries.loc[(df_injuries['rt_injury_2'] != 'none') &
                                        (df_injuries['rt_injury_3'] == 'none')])
        self.tot2inj += twoinjury
        threeinjury = len(df_injuries.loc[(df_injuries['rt_injury_3'] != 'none') &
                                          (df_injuries['rt_injury_4'] == 'none')])
        self.tot3inj += threeinjury
        fourinjury = len(df_injuries.loc[(df_injuries['rt_injury_4'] != 'none') &
                                         (df_injuries['rt_injury_5'] == 'none')])
        self.tot4inj += fourinjury
        fiveinjury = len(df_injuries.loc[(df_injuries['rt_injury_5'] != 'none') &
                                         (df_injuries['rt_injury_6'] == 'none')])
        self.tot5inj += fiveinjury
        sixinjury = len(df_injuries.loc[(df_injuries['rt_injury_6'] != 'none') &
                                        (df_injuries['rt_injury_7'] == 'none')])
        self.tot6inj += sixinjury
        seveninjury = len(df_injuries.loc[(df_injuries['rt_injury_7'] != 'none') &
                                          (df_injuries['rt_injury_8'] == 'none')])
        self.tot7inj += seveninjury
        eightinjury = len(df_injuries.loc[df_injuries['rt_injury_8'] != 'none'])
        self.tot8inj += eightinjury
        dict_to_output = {
            'total_one_injured_body_region': self.tot1inj,
            'total_two_injured_body_region': self.tot2inj,
            'total_three_injured_body_region': self.tot3inj,
            'total_four_injured_body_region': self.tot4inj,
            'total_five_injured_body_region': self.tot5inj,
            'total_six_injured_body_region': self.tot6inj,
            'total_seven_injured_body_region': self.tot7inj,
            'total_eight_injured_body_region': self.tot8inj,
        }
        logger.info(key='number_of_injuries',
                    data=dict_to_output,
                    description='The total number of injured body regions in the simulation')
        # ====================================== AIS body regions =====================================================
        AIS1codes = ['112', '113', '133', '133a', '133b', '133c', '133d', '134', '134a', '134b', '135', '1101', '1114']
        AIS2codes = ['211', '212', '2101', '291', '241', '2114']
        AIS3codes = ['342', '343', '361', '363', '322', '323', '3101', '3113']
        AIS4codes = ['412', '414', '461', '463', '453', '453a', '453b', '441', '442', '443', '4101', '4114']
        AIS5codes = ['552', '553', '554', '5101', '5114']
        AIS6codes = ['612', '673', '673a', '673b', '674', '674a', '674b', '675', '675a', '675b', '676']
        AIS7codes = ['712', '712a', '712b', '712c', '722', '782', '782a', '782b', '782c', '783', '7101', '7114']
        AIS8codes = ['811', '812', '813', '813a', '813b', '813c', '822', '822a', '822b', '882', '883', '884', '8101',
                     '8114']
        idx, AIS1counts = road_traffic_injuries.rti_find_and_count_injuries(df_injuries, AIS1codes)
        self.totAIS1 += AIS1counts
        idx, AIS2counts = road_traffic_injuries.rti_find_and_count_injuries(df_injuries, AIS2codes)
        self.totAIS2 += AIS2counts
        idx, AIS3counts = road_traffic_injuries.rti_find_and_count_injuries(df_injuries, AIS3codes)
        self.totAIS3 += AIS3counts
        idx, AIS4counts = road_traffic_injuries.rti_find_and_count_injuries(df_injuries, AIS4codes)
        self.totAIS4 += AIS4counts
        idx, AIS5counts = road_traffic_injuries.rti_find_and_count_injuries(df_injuries, AIS5codes)
        self.totAIS5 += AIS5counts
        idx, AIS6counts = road_traffic_injuries.rti_find_and_count_injuries(df_injuries, AIS6codes)
        self.totAIS6 += AIS6counts
        idx, AIS7counts = road_traffic_injuries.rti_find_and_count_injuries(df_injuries, AIS7codes)
        self.totAIS7 += AIS7counts
        idx, AIS8counts = road_traffic_injuries.rti_find_and_count_injuries(df_injuries, AIS8codes)
        self.totAIS8 += AIS8counts
        dict_to_output = {'total_number_of_head_injuries': self.totAIS1,
                          'total_number_of_facial_injuries': self.totAIS2,
                          'total_number_of_neck_injuries': self.totAIS3,
                          'total_number_of_thorax_injuries': self.totAIS4,
                          'total_number_of_abdomen_injuries': self.totAIS5,
                          'total_number_of_spinal_injuries': self.totAIS6,
                          'total_number_of_upper_ex_injuries': self.totAIS7,
                          'total_number_of_lower_ex_injuries': self.totAIS8}
        logger.info(key='injury_location_data',
                    data=dict_to_output,
                    description='Data on distribution of the injury location on the body')
        skullfracs = ['112', '113']
        idx, skullfraccounts = road_traffic_injuries.rti_find_and_count_injuries(df_injuries, skullfracs)
        self.fracdist[0] += skullfraccounts
        facefracs = ['211', '212']
        idx, facefraccounts = road_traffic_injuries.rti_find_and_count_injuries(df_injuries, facefracs)
        self.fracdist[1] += facefraccounts
        thoraxfracs = ['412', '414']
        idx, thorfraccounts = road_traffic_injuries.rti_find_and_count_injuries(df_injuries, thoraxfracs)
        self.fracdist[3] += thorfraccounts
        spinefracs = ['612']
        idx, spinfraccounts = road_traffic_injuries.rti_find_and_count_injuries(df_injuries, spinefracs)
        self.fracdist[5] += spinfraccounts
        upperexfracs = ['712', '712a', '712b', '712c']
        idx, upperexfraccounts = road_traffic_injuries.rti_find_and_count_injuries(df_injuries, upperexfracs)
        self.fracdist[6] += upperexfraccounts
        lowerexfracs = ['811', '812', '813', '813a', '813b', '813c']
        idx, lowerexfraccounts = road_traffic_injuries.rti_find_and_count_injuries(df_injuries, lowerexfracs)
        self.fracdist[7] += lowerexfraccounts
        dict_to_output = {
            'total_head_fractures': self.fracdist[0],
            'total_facial_fractures': self.fracdist[1],
            'total_thorax_fractures': self.fracdist[3],
            'total_spinal_fractures': self.fracdist[5],
            'total_upper_ex_fractures': self.fracdist[6],
            'total_lower_ex_fractures': self.fracdist[7]
        }
        logger.info(key='fracture_location_data',
                    data=dict_to_output,
                    description='data on where the fractures occurred on the body')
        skullopen = ['1101']
        idx, skullopencounts = road_traffic_injuries.rti_find_and_count_injuries(df_injuries, skullopen)
        self.openwounddist[0] += skullopencounts
        faceopen = ['2101']
        idx, faceopencounts = road_traffic_injuries.rti_find_and_count_injuries(df_injuries, faceopen)
        self.openwounddist[1] += faceopencounts
        neckopen = ['3101']
        idx, neckopencounts = road_traffic_injuries.rti_find_and_count_injuries(df_injuries, neckopen)
        self.openwounddist[2] += neckopencounts
        thoraxopen = ['4101']
        idx, thoropencounts = road_traffic_injuries.rti_find_and_count_injuries(df_injuries, thoraxopen)
        self.openwounddist[3] += thoropencounts
        abdopen = ['5101']
        idx, abdopencounts = road_traffic_injuries.rti_find_and_count_injuries(df_injuries, abdopen)
        self.openwounddist[4] += abdopencounts
        upperexopen = ['7101']
        idx, upperexopencounts = road_traffic_injuries.rti_find_and_count_injuries(df_injuries, upperexopen)
        self.openwounddist[6] += upperexopencounts
        lowerexopen = ['8101']
        idx, lowerexopencounts = road_traffic_injuries.rti_find_and_count_injuries(df_injuries, lowerexopen)
        self.openwounddist[7] += lowerexopencounts
        dict_to_output = {
            'total_head_laceration': self.openwounddist[0],
            'total_facial_laceration': self.openwounddist[1],
            'total_neck_laceration': self.openwounddist[2],
            'total_thorax_laceration': self.openwounddist[3],
            'total_abdomen_laceration': self.openwounddist[4],
            'total_upper_ex_laceration': self.openwounddist[6],
            'total_lower_ex_laceration': self.openwounddist[7]
        }
        logger.info(key='laceration_location_data',
                    data=dict_to_output,
                    description='data on where the lacerations occurred on the body')

        burncodes = ['1114', '2114', '3113', '4113', '5113', '7113', '8113']
        idx, skullburncounts = road_traffic_injuries.rti_find_and_count_injuries(df_injuries, burncodes[0])
        self.burndist[0] = skullburncounts
        idx, faceburncounts = road_traffic_injuries.rti_find_and_count_injuries(df_injuries, burncodes[1])
        self.burndist[1] = faceburncounts
        idx, neckburncounts = road_traffic_injuries.rti_find_and_count_injuries(df_injuries, burncodes[2])
        self.burndist[2] = neckburncounts
        idx, thorburncounts = road_traffic_injuries.rti_find_and_count_injuries(df_injuries, burncodes[3])
        self.burndist[3] = thorburncounts
        idx, abdburncounts = road_traffic_injuries.rti_find_and_count_injuries(df_injuries, burncodes[4])
        self.burndist[4] = abdburncounts
        idx, upperexburncounts = road_traffic_injuries.rti_find_and_count_injuries(df_injuries, burncodes[5])
        self.burndist[6] = upperexburncounts
        idx, lowerexburncounts = road_traffic_injuries.rti_find_and_count_injuries(df_injuries, burncodes[6])
        self.burndist[7] = lowerexburncounts
        dict_to_output = {
            'total_head_laceration': self.burndist[0],
            'total_facial_laceration': self.burndist[1],
            'total_neck_laceration': self.burndist[2],
            'total_thorax_laceration': self.burndist[3],
            'total_abdomen_laceration': self.burndist[4],
            'total_upper_ex_laceration': self.burndist[6],
            'total_lower_ex_laceration': self.burndist[7]
        }
        logger.info(key='burn_location_data',
                    data=dict_to_output,
                    description='data on where the burns occurred on the body')
        # ===================================== Pain severity =========================================================
        # Injuries causing mild pain include: Lacerations, mild soft tissue injuries, TBI (for now), eye injury
        Mild_Pain_Codes = ['1101', '2101', '3101', '4101', '5101', '7101', '8101',  # lacerations
                           '241',  # Minor soft tissue injuries
                           '133', '133a', '133b', '133c', '133d', '134', '134a', '134b', '135',  # TBI
                           '291',  # Eye injury
                           ]
        mild_idx, mild_counts = road_traffic_injuries.rti_find_and_count_injuries(df_injuries, Mild_Pain_Codes)
        # Injuries causing moderate pain include: Fractures, dislocations, soft tissue and neck trauma
        Moderate_Pain_Codes = ['112', '113', '211', '212', '412', '414', '612', '712', '712a', '712b', '712c',
                               '811', '812', '813', '813a', '813b', '813c',  # fractures
                               '322', '323', '722', '822', '822a', '822b',  # dislocations
                               '342', '343', '361', '363'  # neck trauma
                               ]
        moderate_idx, moderate_counts = road_traffic_injuries.rti_find_and_count_injuries(df_injuries,
                                                                                          Moderate_Pain_Codes)
        # Injuries causing severe pain include: All burns, amputations, spinal cord injuries, abdominal trauma,
        # severe chest trauma
        Severe_Pain_Codes = ['1114', '2114', '3113', '4113', '5113', '7113', '8113',  # burns
                             '782', '782a', '782b', '782c', '783', '882', '883', '884',  # amputations
                             '673', '673a', '673b', '674', '674a', '674b', '675', '675a', '675b', '676',  # sci
                             '552', '553', '554',  # abdominal trauma
                             '461', '463', '453', '453a', '453b', '441', '442', '443'  # severe chest trauma
                             ]
        severe_idx, severe_counts = road_traffic_injuries.rti_find_and_count_injuries(df_injuries, Severe_Pain_Codes)
        in_severe_pain = severe_idx
        self.severe_pain += len(in_severe_pain)
        in_moderate_pain = moderate_idx.difference(moderate_idx.intersection(severe_idx))
        self.moderate_pain += len(in_moderate_pain)
        in_mild_pain = mild_idx.difference(moderate_idx.union(severe_idx))
        self.mild_pain += len(in_mild_pain)
        dict_to_output = {'mild_pain': mild_counts,
                          'moderate_pain': moderate_counts,
                          'severe_pain': severe_counts}
        logger.info(key='pain_information',
                    data=dict_to_output,
                    description='data on the pain level from injuries in the simulation'
                    )
        # ================================== Injury characteristics ===================================================
        allfraccodes = ['112', '113', '211', '212', '412', '414', '612', '712', '712a', '712b', '712c',
                        '811', '812', '813', '813a', '813b', '813c']
        idx, fraccounts = road_traffic_injuries.rti_find_and_count_injuries(df_injuries, allfraccodes)
        self.totfracnumber += fraccounts
        n_alive = df.is_alive.sum()

        self.fracdenominator += (n_alive - fraccounts) / 12
        dislocationcodes = ['322', '323', '722', '822', '822a', '822b']
        idx, dislocationcounts = road_traffic_injuries.rti_find_and_count_injuries(df_injuries, dislocationcodes)
        self.totdisnumber += dislocationcounts
        allheadinjcodes = ['133', '133a', '133b', '133c', '133d', '134', '134a', '134b', '135']
        idx, tbicounts = road_traffic_injuries.rti_find_and_count_injuries(df_injuries, allheadinjcodes)
        self.tottbi += tbicounts
        softtissueinjcodes = ['241', '342', '343', '441', '442', '443']
        idx, softtissueinjcounts = road_traffic_injuries.rti_find_and_count_injuries(df_injuries, softtissueinjcodes)
        self.totsoft += softtissueinjcounts
        organinjurycodes = ['453', '453a', '453b', '552', '553', '554']
        idx, organinjurycounts = road_traffic_injuries.rti_find_and_count_injuries(df_injuries, organinjurycodes)
        self.totintorg += organinjurycounts
        internalbleedingcodes = ['361', '363', '461', '463']
        idx, internalbleedingcounts = road_traffic_injuries.rti_find_and_count_injuries(df_injuries,
                                                                                        internalbleedingcodes)
        self.totintbled += internalbleedingcounts
        spinalcordinjurycodes = ['673', '673a', '673b', '674', '674a', '674b', '675', '675a', '675b', '676']
        idx, spinalcordinjurycounts = road_traffic_injuries.rti_find_and_count_injuries(df_injuries,
                                                                                        spinalcordinjurycodes)
        self.totsci += spinalcordinjurycounts
        amputationcodes = ['782', '782a', '782b', '783', '882', '883', '884']
        idx, amputationcounts = road_traffic_injuries.rti_find_and_count_injuries(df_injuries, amputationcodes)
        self.totamp += amputationcounts
        eyecodes = ['291']
        idx, eyecounts = road_traffic_injuries.rti_find_and_count_injuries(df_injuries, eyecodes)
        self.toteye += eyecounts
        externallacerationcodes = ['1101', '2101', '3101', '4101', '5101', '7101', '8101']
        idx, externallacerationcounts = road_traffic_injuries.rti_find_and_count_injuries(df_injuries,
                                                                                          externallacerationcodes)
        self.totextlac += externallacerationcounts
        burncodes = ['1114', '2114', '3113', '4113', '5113', '7113', '8113']
        idx, burncounts = road_traffic_injuries.rti_find_and_count_injuries(df_injuries, burncodes)
        self.totburns += burncounts
        totalinj = fraccounts + dislocationcounts + tbicounts + softtissueinjcounts + organinjurycounts + \
                   internalbleedingcounts + spinalcordinjurycounts + amputationcounts + externallacerationcounts + \
                   burncounts

        dict_to_output = {
            'total_fractures': self.totfracnumber,
            'total_dislocations': self.totdisnumber,
            'total_traumatic_brain_injuries': self.tottbi,
            'total_soft_tissue_injuries': self.totsoft,
            'total_internal_organ_injuries': self.totintorg,
            'total_internal_bleeding': self.totintbled,
            'total_spinal_cord_injuries': self.totsci,
            'total_amputations': self.totamp,
            'total_eye_injuries': self.toteye,
            'total_lacerations': self.totextlac,
            'total_burns': self.totburns
        }
        logger.info(key='injury_characteristics',
                    data=dict_to_output,
                    description='the injury categories produced in the simulation')
        # ================================= Injury severity ===========================================================
        sev = df.loc[df.rt_road_traffic_inc]
        sev = sev['rt_inj_severity']
        ISSlist = thoseininjuries['rt_ISS_score'].tolist()
        ISSlist = list(filter(lambda num: num != 0, ISSlist))
        self.ISSscore += ISSlist
        severity, severitycount = np.unique(sev, return_counts=True)
        if 'mild' in severity:
            idx = np.where(severity == 'mild')
            self.totmild += len(idx)
        if 'severe' in severity:
            idx = np.where(severity == 'severe')
            self.totsevere += len(idx)
        dict_to_output = {
            'total_mild_injuries': self.totmild,
            'total_severe_injuries': self.totsevere,
            'ISS_score': self.ISSscore
        }
        logger.info(key='injury_severity',
                    data=dict_to_output,
                    description='severity of injuries in simulation')
        # ==================================== Incidence ==============================================================
        # How many were involved in a RTI
        n_in_RTI = df.rt_road_traffic_inc.sum()
        self.numerator += n_in_RTI
        self.totinjured += n_in_RTI
        # How many were disabled
        n_perm_disabled = (df.is_alive & df.rt_perm_disability).sum()
        # self.permdis += n_perm_disabled
        n_alive = df.is_alive.sum()
        self.denominator += (n_alive - n_in_RTI) * (1 / 12)
        n_immediate_death = (df.rt_road_traffic_inc & df.rt_imm_death).sum()
        self.deathonscene += n_immediate_death
        immdeathidx = df.index[df.rt_imm_death]
        deathwithmedidx = df.index[df.rt_post_med_death]
        deathwithoutmedidx = df.index[df.rt_no_med_death]
        diedfromrtiidx = immdeathidx.union(deathwithmedidx)
        diedfromrtiidx = diedfromrtiidx.union(deathwithoutmedidx)
        n_sought_care = (df.rt_road_traffic_inc & df.rt_med_int).sum()
        self.soughtmedcare += n_sought_care
        n_death_post_med = (df.is_alive & df.rt_post_med_death).sum()
        self.deathaftermed += n_death_post_med
        self.deathwithoutmed += len(df.loc[df.rt_no_med_death])
        self.death_inc_numerator += n_immediate_death + n_death_post_med + len(df.loc[df.rt_no_med_death])
        self.death_in_denominator += (n_alive - (n_immediate_death + n_death_post_med + len(df.loc[df.rt_no_med_death])
                                                 )) * \
                                     (1 / 12)
        percent_accidents_result_in_death = (self.deathonscene + self.deathaftermed + self.deathwithoutmed) / \
                                            self.numerator
        maleinrti = len(df.loc[df.rt_road_traffic_inc & (df['sex'] == 'M')])
        femaleinrti = len(df.loc[df.rt_road_traffic_inc & (df['sex'] == 'F')])

        divider = min(maleinrti, femaleinrti)
        if divider > 0:
            maleinrti = maleinrti / divider
            femaleinrti = femaleinrti / divider
        else:
            maleinrti = 1
            femaleinrti = 0
        mfratio = [maleinrti, femaleinrti]
        incidence_of_injuries = (totalinj / (n_alive - n_in_RTI)) * 12 * 100000
        dict_to_output = {
            'number involved in a rti': n_in_RTI,
            'incidence of rti per 100,000': (n_in_RTI / ((n_alive - n_in_RTI) * (1 / 12))) * 100000,
            'incidence of rti death per 100,000': (len(diedfromrtiidx) /
                                                   ((n_alive - len(diedfromrtiidx)) * (1 / 12))) * 100000,
            'incidence of fractures per 100,000': (self.totfracnumber / self.fracdenominator) * 100000,
            'injury incidence per 100,000': incidence_of_injuries,
            'number alive': n_alive,
            'number immediate deaths': n_immediate_death,
            'number permanently disabled': n_perm_disabled,
            'percent of crashes that are fatal': percent_accidents_result_in_death,
            'total injuries': totalinj,
            'male:female ratio': mfratio,
        }
        logger.info(key='summary_1m',
                    data=dict_to_output,
                    description='Summary of the rti injuries in the last month')
        # =========================== Get population demographics of those with RTIs ==================================
        columnsOfInterest = ['sex', 'age_years', 'li_ex_alc']
        injuredDemographics = df.loc[df.rt_road_traffic_inc]

        injuredDemographics = injuredDemographics.loc[:, columnsOfInterest]
        injured_demography_summary = {
            'males_in_rti': injuredDemographics['sex'].value_counts()['M'],
            'females_in_rti': injuredDemographics['sex'].value_counts()['F'],
            'age': injuredDemographics['age_years'].values.tolist()
        }
        logger.info(key='rti_demography',
                    data=injured_demography_summary,
                    description='Demographics of those in rti')

        # =================================== Flows through the model ==================================================
        dict_to_output = {'total_injured': self.totinjured,
                          'total_died_on_scene': self.deathonscene,
                          'total_sought_medical_care': self.soughtmedcare,
                          'total_died_after_medical_intervention': self.deathaftermed,
                          'total_permanently_disabled': n_perm_disabled}
        logger.info(key='model_progression',
                    data=dict_to_output,
                    description='Flows through the rti module')

        # ============================ Injury category incidence ======================================================
        in_rti_this_month = df.loc[df.rt_road_traffic_inc]
        in_rti_this_month = in_rti_this_month.loc[:, columns]
        idx, amputationcounts = road_traffic_injuries.rti_find_and_count_injuries(in_rti_this_month, amputationcodes)
        inc_amputations = amputationcounts / ((n_alive - amputationcounts) * 1 / 12) * 100000
        idx, burncounts = road_traffic_injuries.rti_find_and_count_injuries(in_rti_this_month, burncodes)
        inc_burns = burncounts / ((n_alive - burncounts) * 1 / 12) * 100000
        idx, fraccounts = road_traffic_injuries.rti_find_and_count_injuries(in_rti_this_month, allfraccodes)
        inc_fractures = fraccounts / ((n_alive - fraccounts) * 1 / 12) * 100000
        idx, tbicounts = road_traffic_injuries.rti_find_and_count_injuries(in_rti_this_month, allheadinjcodes)
        inc_tbi = tbicounts / ((n_alive - tbicounts) * 1 / 12) * 100000
        idx, spinalcordinjurycounts = road_traffic_injuries.rti_find_and_count_injuries(in_rti_this_month,
                                                                                        spinalcordinjurycodes)
        inc_sci = spinalcordinjurycounts / ((n_alive - spinalcordinjurycounts) * 1 / 12) * 100000
        minor_injury_codes = ['1101', '2101', '3101', '4101', '5101', '6101', '7101', '8101']
        idx, minorinjurycounts = road_traffic_injuries.rti_find_and_count_injuries(in_rti_this_month,
                                                                                   minor_injury_codes)
        inc_minor = minorinjurycounts / ((n_alive - minorinjurycounts) * 1 / 12) * 100000
        other_codes = ['322', '323', '722', '822', '822a', '822b', '291', '361', '363', '461', '463', '412', '414',
                       '453', '453a', '453b', '552', '553', '554']
        idx, other_counts = road_traffic_injuries.rti_find_and_count_injuries(in_rti_this_month,
                                                                              other_codes)
        inc_other = other_counts / ((n_alive - other_counts) * 1 / 12) * 100000
        tot_inc_all_inj = inc_amputations + inc_burns + inc_fractures + inc_tbi + inc_sci + inc_minor + inc_other
        dict_to_output = {'inc_amputations': inc_amputations,
                          'inc_burns': inc_burns,
                          'inc_fractures': inc_fractures,
                          'inc_tbi': inc_tbi,
                          'inc_sci': inc_sci,
                          'inc_minor': inc_minor,
                          'inc_other': inc_other,
                          'tot_inc_injuries': tot_inc_all_inj}

        logger.info(key='Inj_category_incidence',
                    data=dict_to_output,
                    description='Incidence of each injury grouped as per the GBD definition')
