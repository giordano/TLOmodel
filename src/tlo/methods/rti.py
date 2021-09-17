"""
Road traffic injury module.

"""
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

# ---------------------------------------------------------------------------------------------------------
#   MODULE DEFINITIONS
# ---------------------------------------------------------------------------------------------------------

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class RTI(Module):
    """
    The road traffic injuries module for the TLO model, handling all injuries related to road traffic accidents.
    """

    def __init__(self, name=None, resourcefilepath=None):
        # NB. Parameters passed to the module can be inserted in the __init__ definition.
        super().__init__(name)
        self.resourcefilepath = resourcefilepath

    INIT_DEPENDENCIES = {"SymptomManager",
                         "HealthBurden"}
    ADDITIONAL_DEPENDENCIES = {
        'Demography',
        'Lifestyle',
        'HealthSystem',
    }

    INJURY_COLUMNS = ['rt_injury_1', 'rt_injury_2', 'rt_injury_3', 'rt_injury_4', 'rt_injury_5', 'rt_injury_6',
                      'rt_injury_7', 'rt_injury_8']

    INJURY_CODES = ['none', '112', '113', '133', '133a', '133b', '133c', '133d', '134', '134a',
                    '134b', '135', '1101', '1114', '211', '212', '241', '2101', '2114', '291',
                    '342', '343', '361', '363', '322', '323', '3101', '3113', '412', '414',
                    '461', '463', '453', '453a', '453b', '441', '442', '443', '4101', '4113',
                    '552', '553', '554', '5101', '5113', '612', '673', '673a', '673b', '674',
                    '674a', '674b', '675', '675a', '675b', '676', '712', '712a', '712b', '712c',
                    '722', '782', '782a', '782b', '782c', '783', '7101', '7113', '811', '813do',
                    '812', '813eo', '813', '813a', '813b', '813bo', '813c', '813co', '822',
                    '822a', '822b', '882', '883', '884', '8101', '8113', 'P133', 'P133a',
                    'P133b', 'P133c', 'P133d', 'P134', 'P134a', 'P134b', 'P135', 'P673',
                    'P673a', 'P673b', 'P674', 'P674a', 'P674b', 'P675', 'P675a', 'P675b',
                    'P676', 'P782a', 'P782b', 'P782c', 'P783', 'P882', 'P883', 'P884']

    SWAPPING_CODES = ['712b', '812', '3113', '4113', '5113', '7113', '8113', '813a', '813b', 'P673a', 'P673b', 'P674a',
                      'P674b', 'P675a', 'P675b', 'P676', 'P782b', 'P783', 'P883', 'P884', '813bo', '813co', '813do',
                      '813eo']

    # Module parameters
    PARAMETERS = {

        'base_rate_injrti': Parameter(
            Types.REAL,
            'Base rate of RTI per year',
        ),
        'rr_injrti_age04': Parameter(
            Types.REAL,
            'risk ratio of RTI in age 0-4 compared to base rate of RTI'
        ),
        'rr_injrti_age59': Parameter(
            Types.REAL,
            'risk ratio of RTI in age 5-9 compared to base rate of RTI'
        ),
        'rr_injrti_age1017': Parameter(
            Types.REAL,
            'risk ratio of RTI in age 10-17 compared to base rate of RTI'
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
        'prob_death_iss_less_than_9': Parameter(
            Types.REAL,
            'Proportion of people who pass away in the following month after medical treatment for injuries with an ISS'
            'score less than or equal to 9'
        ),
        'prob_death_iss_10_15': Parameter(
            Types.REAL,
            'Proportion of people who pass away in the following month after medical treatment for injuries with an ISS'
            'score from 10 to 15'
        ),
        'prob_death_iss_16_24': Parameter(
            Types.REAL,
            'Proportion of people who pass away in the following month after medical treatment for injuries with an ISS'
            'score from 16 to 24'
        ),
        'prob_death_iss_25_35': Parameter(
            Types.REAL,
            'Proportion of people who pass away in the following month after medical treatment for injuries with an ISS'
            'score from 25 to 34'
        ),
        'prob_death_iss_35_plus': Parameter(
            Types.REAL,
            'Proportion of people who pass away in the following month after medical treatment for injuries with an ISS'
            'score 35 and above'
        ),
        'prob_perm_disability_with_treatment_severe_TBI': Parameter(
            Types.REAL,
            'probability that someone with a treated severe TBI is permanently disabled'
        ),
        'prob_death_TBI_SCI_no_treatment': Parameter(
            Types.REAL,
            'probability that someone with a spinal cord injury will die without treatment'
        ),
        'prop_death_burns_no_treatment': Parameter(
            Types.REAL,
            'probability that someone with a burn injury will die without treatment'
        ),
        'prob_death_fractures_no_treatment': Parameter(
            Types.REAL,
            'probability that someone with a fracture injury will die without treatment'
        ),
        'prob_TBI_require_craniotomy': Parameter(
            Types.REAL,
            'probability that someone with a traumatic brain injury will require a craniotomy surgery'
        ),
        'prob_exploratory_laparotomy': Parameter(
            Types.REAL,
            'probability that someone with an internal organ injury will require a exploratory_laparotomy'
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
        'rt_emergency_care_ISS_score_cut_off': Parameter(
            Types.INT,
            'A parameter to determine which level of injury severity corresponds to the emergency health care seeking '
            'symptom and which to the non-emergency generic injury symptom'
        ),
        'prob_death_MAIS3': Parameter(
            Types.REAL,
            'A parameter to determine the probability of death without medical intervention with a military AIS'
            'score of 3'
        ),
        'prob_death_MAIS4': Parameter(
            Types.REAL,
            'A parameter to determine the probability of death without medical intervention with a military AIS'
            'score of 4'
        ),
        'prob_death_MAIS5': Parameter(
            Types.REAL,
            'A parameter to determine the probability of death without medical intervention with a military AIS'
            'score of 5'
        ),
        'prob_death_MAIS6': Parameter(
            Types.REAL,
            'A parameter to determine the probability of death without medical intervention with a military AIS'
            'score of 6'
        ),
        'femur_fracture_skeletal_traction_mean_los': Parameter(
            Types.INT,
            'The mean length of stay for a person with a femur fracture being treated with skeletal traction'
        ),
        'other_skeletal_traction_los': Parameter(
            Types.INT,
            'The mean length of stay for a person with a non-femur fracture being treated with skeletal traction'
        ),
        'prob_foot_frac_require_cast': Parameter(
            Types.REAL,
            'The probability that a person with a foot fracture will be treated with a plaster cast'
        ),
        'prob_foot_frac_require_maj_surg': Parameter(
            Types.REAL,
            'The probability that a person with a foot fracture will be treated with a major surgery'
        ),
        'prob_foot_frac_require_min_surg': Parameter(
            Types.REAL,
            'The probability that a person with a foot fracture will be treated with a major surgery'
        ),
        'prob_foot_frac_require_amp': Parameter(
            Types.REAL,
            'The probability that a person with a foot fracture will be treated with amputation via a major surgery'
        ),
        'prob_tib_fib_frac_require_cast': Parameter(
            Types.REAL,
            'The probability that a person with a tibia/fibula fracture will be treated with a plaster cast'
        ),
        'prob_tib_fib_frac_require_maj_surg': Parameter(
            Types.REAL,
            'The probability that a person with a tibia/fibula fracture will be treated with a major surgery'
        ),
        'prob_tib_fib_frac_require_min_surg': Parameter(
            Types.REAL,
            'The probability that a person with a tibia/fibula fracture will be treated with a minor surgery'
        ),
        'prob_tib_fib_frac_require_amp': Parameter(
            Types.REAL,
            'The probability that a person with a tibia/fibula fracture will be treated with an amputation via major '
            'surgery'
        ),
        'prob_tib_fib_frac_require_traction': Parameter(
            Types.REAL,
            'The probability that a person with a tibia/fibula fracture will be treated with skeletal traction'
        ),
        'prob_femural_fracture_require_major_surgery': Parameter(
            Types.REAL,
            'The probability that a person with a femur fracture will be treated with major surgery'
        ),
        'prob_femural_fracture_require_minor_surgery': Parameter(
            Types.REAL,
            'The probability that a person with a femur fracture will be treated with minor surgery'
        ),
        'prob_femural_fracture_require_cast': Parameter(
            Types.REAL,
            'The probability that a person with a femur fracture will be treated with a plaster cast'
        ),
        'prob_femural_fracture_require_amputation': Parameter(
            Types.REAL,
            'The probability that a person with a femur fracture will be treated with amputation via major surgery'
        ),
        'prob_femural_fracture_require_traction': Parameter(
            Types.REAL,
            'The probability that a person with a femur fracture will be treated with skeletal traction'
        ),
        'prob_pelvis_fracture_traction': Parameter(
            Types.REAL,
            'The probability that a person with a pelvis fracture will be treated with skeletal traction'
        ),
        'prob_pelvis_frac_major_surgery': Parameter(
            Types.REAL,
            'The probability that a person with a pelvis fracture will be treated with major surgery'
        ),
        'prob_pelvis_frac_minor_surgery': Parameter(
            Types.REAL,
            'The probability that a person with a pelvis fracture will be treated with minor surgery'
        ),
        'prob_pelvis_frac_cast': Parameter(
            Types.REAL,
            'The probability that a person with a pelvis fracture will be treated with a cast'
        ),
        'prob_dis_hip_require_maj_surg': Parameter(
            Types.REAL,
            'The probability that a person with a dislocated hip will be treated with a major surgery'
        ),
        'prob_dis_hip_require_cast': Parameter(
            Types.REAL,
            'The probability that a person with a dislocated hip will be treated with a plaster cast'
        ),
        'prob_hip_dis_require_traction': Parameter(
            Types.REAL,
            'The probability that a person with a dislocated hip will be treated with skeletal traction'
        ),
        'hdu_cut_off_iss_score': Parameter(
            Types.INT,
            'The ISS score used as a criteria to admit patients to the HDU/ICU units'
        ),
        'mean_icu_days': Parameter(
            Types.REAL,
            'The mean length of stay in the ICUfor those without TBI'
        ),
        'sd_icu_days': Parameter(
            Types.REAL,
            'The standard deviation in length of stay in the ICU for those without TBI'
        ),
        'mean_tbi_icu_days': Parameter(
            Types.REAL,
            'The mean length of stay in the ICU for those with TBI'
        ),
        'sd_tbi_icu_days': Parameter(
            Types.REAL,
            'The standard deviation in length of stay in the ICU for those with TBI'
        ),
        'prob_foot_fracture_open': Parameter(
            Types.REAL,
            'The probability that a foot fracture will be open'
        ),
        'prob_patella_tibia_fibula_ankle_fracture_open': Parameter(
            Types.REAL,
            'The probability that a patella/tibia/fibula/ankle fracture will be open'
        ),
        'prob_pelvis_fracture_open': Parameter(
            Types.REAL,
            'The probability that a pelvis fracture will be open'
        ),
        'prob_femur_fracture_open': Parameter(
            Types.REAL,
            'The probability that a femur fracture will be open'
        ),
        'prob_open_fracture_contaminated': Parameter(
            Types.REAL,
            'The probability that an open fracture will be contaminated'
        ),
        'allowed_interventions': Parameter(
            Types.LIST,
            'List of additional interventions that can be included when performing model analysis'
        )
    }

    # Define the module's parameters
    PROPERTIES = {
        'rt_road_traffic_inc': Property(Types.BOOL, 'involved in a road traffic injury'),
        'rt_inj_severity': Property(Types.CATEGORICAL,
                                    'Injury status relating to road traffic injury: none, mild, severe',
                                    categories=['none', 'mild', 'severe'],
                                    ),
        'rt_injury_1': Property(Types.CATEGORICAL, 'Codes for injury 1 from RTI', categories=INJURY_CODES),
        'rt_injury_2': Property(Types.CATEGORICAL, 'Codes for injury 2 from RTI', categories=INJURY_CODES),
        'rt_injury_3': Property(Types.CATEGORICAL, 'Codes for injury 3 from RTI', categories=INJURY_CODES),
        'rt_injury_4': Property(Types.CATEGORICAL, 'Codes for injury 4 from RTI', categories=INJURY_CODES),
        'rt_injury_5': Property(Types.CATEGORICAL, 'Codes for injury 5 from RTI', categories=INJURY_CODES),
        'rt_injury_6': Property(Types.CATEGORICAL, 'Codes for injury 6 from RTI', categories=INJURY_CODES),
        'rt_injury_7': Property(Types.CATEGORICAL, 'Codes for injury 7 from RTI', categories=INJURY_CODES),
        'rt_injury_8': Property(Types.CATEGORICAL, 'Codes for injury 8 from RTI', categories=INJURY_CODES),
        'rt_in_shock': Property(Types.BOOL, 'A property determining if this person is in shock'),
        'rt_death_from_shock': Property(Types.BOOL, 'whether this person died from shock'),
        'rt_injuries_to_cast': Property(Types.LIST, 'A list of injuries that are to be treated with casts'),
        'rt_injuries_for_minor_surgery': Property(Types.LIST, 'A list of injuries that are to be treated with a minor'
                                                              'surgery'),
        'rt_injuries_for_major_surgery': Property(Types.LIST, 'A list of injuries that are to be treated with a minor'
                                                              'surgery'),
        'rt_injuries_to_heal_with_time': Property(Types.LIST, 'A list of injuries that heal without further treatment'),
        'rt_injuries_for_open_fracture_treatment': Property(Types.LIST, 'A list of injuries that with open fracture '
                                                                        'treatment'),
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
        'rt_unavailable_med_death': Property(Types.BOOL, 'death in the following month without medical intervention '
                                                         'being able to be provided'),
        'rt_recovery_no_med': Property(Types.BOOL, 'recovery without medical intervention True/False'),
        'rt_disability': Property(Types.REAL, 'disability weight for current month'),
        'rt_date_inj': Property(Types.DATE, 'date of latest injury'),
        'rt_med_int': Property(Types.BOOL, 'whether this person is currently undergoing medical treatment'),
        'rt_in_icu_or_hdu': Property(Types.BOOL, 'whether this person is currently in ICU for RTI'),
        'rt_MAIS_military_score': Property(Types.INT, 'the maximum AIS-military score, used as a proxy to calculate the'
                                                      'probability of mortality without medical intervention'),
        'rt_date_death_no_med': Property(Types.DATE, 'the date which the person has is scheduled to die without medical'
                                                     'intervention'),
        'rt_debugging_DALY_wt': Property(Types.REAL, 'The true value of the DALY weight burden')
    }

    # Declare Metadata
    METADATA = {
        Metadata.DISEASE_MODULE,  # Disease modules: Any disease module should carry this label.
        Metadata.USES_SYMPTOMMANAGER,  # The 'Symptom Manager' recognises modules with this label.
        Metadata.USES_HEALTHSYSTEM,  # The 'HealthSystem' recognises modules with this label.
        Metadata.USES_HEALTHBURDEN  # The 'HealthBurden' module recognises modules with this label.
    }

    # Declare Causes of Death
    CAUSES_OF_DEATH = {
        'RTI_death_without_med': Cause(gbd_causes='Road injuries', label='Transport Injuries'),
        'RTI_death_with_med': Cause(gbd_causes='Road injuries', label='Transport Injuries'),
        'RTI_unavailable_med': Cause(gbd_causes='Road injuries', label='Transport Injuries'),
        'RTI_imm_death': Cause(gbd_causes='Road injuries', label='Transport Injuries'),
        'RTI_death_shock': Cause(gbd_causes='Road injuries', label='Transport Injuries'),
    }

    # Declare Causes of Death and Disability
    CAUSES_OF_DISABILITY = {
        'RTI': Cause(gbd_causes='Road injuries', label='Transport Injuries')
    }

    def read_parameters(self, data_folder):
        """ Reads the parameters used in the RTI module"""
        p = self.parameters

        dfd = pd.read_excel(Path(self.resourcefilepath) / 'ResourceFile_RTI.xlsx', sheet_name='parameter_values')
        self.load_parameters_from_dataframe(dfd)
        if "HealthBurden" in self.sim.modules:
            # get the DALY weights of the seq associated with road traffic injuries
            daly_sequlae_codes = {
                'daly_wt_unspecified_skull_fracture': 1674,
                'daly_wt_basilar_skull_fracture': 1675,
                'daly_wt_epidural_hematoma': 1676,
                'daly_wt_subdural_hematoma': 1677,
                'daly_wt_subarachnoid_hematoma': 1678,
                'daly_wt_brain_contusion': 1679,
                'daly_wt_intraventricular_haemorrhage': 1680,
                'daly_wt_diffuse_axonal_injury': 1681,
                'daly_wt_subgaleal_hematoma': 1682,
                'daly_wt_midline_shift': 1683,
                'daly_wt_facial_fracture': 1684,
                'daly_wt_facial_soft_tissue_injury': 1685,
                'daly_wt_eye_injury': 1686,
                'daly_wt_neck_soft_tissue_injury': 1687,
                'daly_wt_neck_internal_bleeding': 1688,
                'daly_wt_neck_dislocation': 1689,
                'daly_wt_chest_wall_bruises_hematoma': 1690,
                'daly_wt_hemothorax': 1691,
                'daly_wt_lung_contusion': 1692,
                'daly_wt_diaphragm_rupture': 1693,
                'daly_wt_rib_fracture': 1694,
                'daly_wt_flail_chest': 1695,
                'daly_wt_chest_wall_laceration': 1696,
                'daly_wt_closed_pneumothorax': 1697,
                'daly_wt_open_pneumothorax': 1698,
                'daly_wt_surgical_emphysema': 1699,
                'daly_wt_abd_internal_organ_injury': 1700,
                'daly_wt_spinal_cord_lesion_neck_with_treatment': 1701,
                'daly_wt_spinal_cord_lesion_neck_without_treatment': 1702,
                'daly_wt_spinal_cord_lesion_below_neck_with_treatment': 1703,
                'daly_wt_spinal_cord_lesion_below_neck_without_treatment': 1704,
                'daly_wt_vertebrae_fracture': 1705,
                'daly_wt_clavicle_scapula_humerus_fracture': 1706,
                'daly_wt_hand_wrist_fracture_with_treatment': 1707,
                'daly_wt_hand_wrist_fracture_without_treatment': 1708,
                'daly_wt_radius_ulna_fracture_short_term_with_without_treatment': 1709,
                'daly_wt_radius_ulna_fracture_long_term_without_treatment': 1710,
                'daly_wt_dislocated_shoulder': 1711,
                'daly_wt_amputated_finger': 1712,
                'daly_wt_amputated_thumb': 1713,
                'daly_wt_unilateral_arm_amputation_with_treatment': 1714,
                'daly_wt_unilateral_arm_amputation_without_treatment': 1715,
                'daly_wt_bilateral_arm_amputation_with_treatment': 1716,
                'daly_wt_bilateral_arm_amputation_without_treatment': 1717,
                'daly_wt_foot_fracture_short_term_with_without_treatment': 1718,
                'daly_wt_foot_fracture_long_term_without_treatment': 1719,
                'daly_wt_patella_tibia_fibula_fracture_with_treatment': 1720,
                'daly_wt_patella_tibia_fibula_fracture_without_treatment': 1721,
                'daly_wt_hip_fracture_short_term_with_without_treatment': 1722,
                'daly_wt_hip_fracture_long_term_with_treatment': 1723,
                'daly_wt_hip_fracture_long_term_without_treatment': 1724,
                'daly_wt_pelvis_fracture_short_term': 1725,
                'daly_wt_pelvis_fracture_long_term': 1726,
                'daly_wt_femur_fracture_short_term': 1727,
                'daly_wt_femur_fracture_long_term_without_treatment': 1728,
                'daly_wt_dislocated_hip': 1729,
                'daly_wt_dislocated_knee': 1730,
                'daly_wt_amputated_toes': 1731,
                'daly_wt_unilateral_lower_limb_amputation_with_treatment': 1732,
                'daly_wt_unilateral_lower_limb_amputation_without_treatment': 1733,
                'daly_wt_bilateral_lower_limb_amputation_with_treatment': 1734,
                'daly_wt_bilateral_lower_limb_amputation_without_treatment': 1735,
                'daly_wt_burns_greater_than_20_percent_body_area': 1736,
                'daly_wt_burns_less_than_20_percent_body_area_with_treatment': 1737,
                'daly_wt_burns_less_than_20_percent_body_area_without_treatment': 1738,
            }

            hb = self.sim.modules["HealthBurden"]
            for key, value in daly_sequlae_codes.items():
                p[key] = hb.get_daly_weight(sequlae_code=value)

        # ================== Test the parameter distributions to see whether they sum to roughly one ===============
        # test the distribution of the number of injured body regions
        assert 0.9999 < sum(p['number_of_injured_body_regions_distribution'][1]) < 1.0001, \
            "The number of injured body region distribution doesn't sum to one"
        # test the injury location distribution
        assert 0.9999 < sum(p['injury_location_distribution'][1]) < 1.0001, \
            "The injured body region distribution doesn't sum to one"
        # test the distributions used to assign daly weights for certain injury codes
        daly_weight_distributions = [val for key, val in p.items() if 'daly_dist_code_' in key]
        for dist in daly_weight_distributions:
            assert 0.9999 < sum(dist) < 1.0001, 'daly weight distribution does not sum to one'
        # test the distributions to assign injuries to certain body regions
        # get the first characters of the parameter names
        body_part_strings = ['head_prob_', 'face_prob_', 'neck_prob_', 'thorax_prob_', 'abdomen_prob_',
                             'spine_prob_', 'upper_ex_prob_', 'lower_ex_prob_']
        # iterate over each body part, check the probabilities add to one
        for body_part in body_part_strings:
            probabilities_to_assign_injuries = [val for key, val in p.items() if body_part in key]
            sum_probabilities = sum(probabilities_to_assign_injuries)
            assert (sum_probabilities % 1 < 0.0001) or (sum_probabilities % 1 > 0.9999), "The probabilities" \
                                                                                         "chosen for assigning" \
                                                                                         "injuries don't" \
                                                                                         "sum to one"
        # Check all other probabilities are between 0 and 1
        probabilities = [val for key, val in p.items() if 'prob_' in key]
        for probability in probabilities:
            assert 0 <= probability <= 1, "Probability is not a feasible value"
        # create a generic severe trauma symptom, which forces people into the health system
        self.sim.modules['SymptomManager'].register_symptom(
            Symptom(
                name='severe_trauma',
                emergency_in_adults=True,
                emergency_in_children=True
            )
        )

    def rti_injury_diagnosis(self, person_id, the_appt_footprint):
        """
        A function used to alter the appointment footprint of the generic first appointments, based on the needs of
        the patient to be properly diagnosed. Specifically, this function will assign x-rays/ct-scans for injuries
        that require those diagnosis tools.
        :param person_id: the person in a generic appointment with an injury
        :param the_appt_footprint: the current appointment footprint to be altered
        :return: the altered appointment footprint
        """
        df = self.sim.population.props
        # Filter the dataframe by the columns the injuries are stored in
        persons_injuries = df.loc[[person_id], RTI.INJURY_COLUMNS]

        # Injuries that require x rays are: fractures, spinal cord injuries, dislocations, soft tissue injuries in neck
        # and soft tissue injury in thorax/ lung injury
        codes_requiring_xrays = ['112', '113', '211', '212', '412', '414', '612', '712a', '712b', '712c', '811', '812',
                                 '813a', '813b', '813c', '822a', '822b', '813bo', '813co', '813do', '813eo', '673',
                                 '674', '675', '676', '322', '323', '722', '342', '343', '441', '443', '453']
        # Injuries that require a ct scan are TBIs, abdominal trauma, soft tissue injury in neck, soft tissue injury in
        # thorax/ lung injury and abdominal trauma
        codes_requiring_ct_scan = ['133', '134', '135', '552', '553', '554', '342', '343', '441', '443', '453', '361',
                                   '363', '461', '463']

        def adjust_appt_footprint(_codes, _requirement):
            _, counts = self.rti_find_and_count_injuries(persons_injuries, _codes)
            if counts > 0:
                the_appt_footprint[_requirement] = 1

        adjust_appt_footprint(codes_requiring_xrays, 'DiagRadio')
        adjust_appt_footprint(codes_requiring_ct_scan, 'Tomography')

    def initialise_population(self, population):
        """Sets up the default properties used in the RTI module and applies them to the dataframe. The default state
        for the RTI module is that people haven't been involved in a road traffic accident and are therefor alive and
        healthy."""
        df = population.props
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
        df.loc[df.is_alive, 'rt_in_shock'] = False
        df.loc[df.is_alive, 'rt_death_from_shock'] = False
        df.loc[df.is_alive, 'rt_polytrauma'] = False
        df.loc[df.is_alive, 'rt_ISS_score'] = 0
        df.loc[df.is_alive, 'rt_perm_disability'] = False
        df.loc[df.is_alive, 'rt_imm_death'] = False  # default: no one is dead on scene of crash
        df.loc[df.is_alive, 'rt_diagnosed'] = False
        df.loc[df.is_alive, 'rt_recovery_no_med'] = False  # default: no recovery without medical intervention
        df.loc[df.is_alive, 'rt_post_med_death'] = False  # default: no death after medical intervention
        df.loc[df.is_alive, 'rt_no_med_death'] = False
        df.loc[df.is_alive, 'rt_unavailable_med_death'] = False
        df.loc[df.is_alive, 'rt_disability'] = 0  # default: no DALY
        df.loc[df.is_alive, 'rt_date_inj'] = pd.NaT
        df.loc[df.is_alive, 'rt_med_int'] = False
        df.loc[df.is_alive, 'rt_in_icu_or_hdu'] = False
        df.loc[df.is_alive, 'rt_MAIS_military_score'] = 0
        df.loc[df.is_alive, 'rt_date_death_no_med'] = pd.NaT
        df.loc[df.is_alive, 'rt_debugging_DALY_wt'] = 0
        alive_count = sum(df.is_alive)
        df.loc[df.is_alive, 'rt_date_to_remove_daly'] = pd.Series([[pd.NaT] * 8 for _ in range(alive_count)])
        df.loc[df.is_alive, 'rt_injuries_to_cast'] = pd.Series([[] for _ in range(alive_count)])
        df.loc[df.is_alive, 'rt_injuries_for_minor_surgery'] = pd.Series([[] for _ in range(alive_count)])
        df.loc[df.is_alive, 'rt_injuries_for_major_surgery'] = pd.Series([[] for _ in range(alive_count)])
        df.loc[df.is_alive, 'rt_injuries_to_heal_with_time'] = pd.Series([[] for _ in range(alive_count)])
        df.loc[df.is_alive, 'rt_injuries_for_open_fracture_treatment'] = pd.Series([[] for _ in range(alive_count)])

    def initialise_simulation(self, sim):
        """At the start of the simulation we schedule a logging event, which records the relevant information
        regarding road traffic injuries in the last month.

        Afterwards, we schedule three RTI events, the first is the main RTI event which takes parts
        of the population and assigns them to be involved in road traffic injuries and providing they survived will
        begin the interaction with the healthcare system. This event runs monthly.

        The second is the begin scheduling the RTI recovery event, which looks at those in the population who have been
        injured in a road traffic accident, checking every day whether enough time has passed for their injuries to have
        healed. When the injury has healed the associated daly weight is removed.

        The final event is one which checks if this person has not sought sought care or been given care, if they
        haven't then it asks whether they should die away from their injuries
        """
        # Begin modelling road traffic injuries
        sim.schedule_event(RTIPollingEvent(self), sim.date + DateOffset(months=0))
        # Begin checking whether the persons injuries are healed
        sim.schedule_event(RTI_Recovery_Event(self), sim.date + DateOffset(months=0))
        # Begin checking whether those with untreated injuries die
        sim.schedule_event(RTI_Check_Death_No_Med(self), sim.date + DateOffset(months=0))
        # Begin logging the RTI events
        sim.schedule_event(RTI_Logging_Event(self), sim.date + DateOffset(months=1))

    def rti_do_when_diagnosed(self, person_id):
        """
        This function is called by the generic first appointments when an injured person has been diagnosed
        in A&E and needs to progress further in the health system. The injured person will then be scheduled a generic
        'medical intervention' appointment which serves three purposes. The first is to determine what treatments they
        require for their injuries and shedule those, the second is to contain them in the health care system with
        inpatient days and finally, the appointment treats injuries that heal over time without further need for
        resources in the health system.

        :param person_id: the person requesting medical care
        :return: n/a
        """
        df = self.sim.population.props
        # Check to see whether they have been sent here from A and E
        assert df.loc[person_id, 'rt_diagnosed']
        # Get the relevant information about their injuries
        person_injuries = df.loc[[person_id], RTI.INJURY_COLUMNS]
        # check this person is injured, search they have an injury code that isn't "none"
        _, counts = RTI.rti_find_and_count_injuries(person_injuries, RTI.INJURY_CODES[1:])
        # also test whether the regular injury symptom has been given to the person via spurious symptoms
        assert (counts > 0) or self.sim.modules['SymptomManager'].spurious_symptoms, \
            'This person has asked for medical treatment despite not being injured'

        # If they meet the requirements, send them to HSI_RTI_MedicalIntervention for further treatment
        # Using counts condition to stop spurious symptoms progressing people through the model
        if counts > 0:
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
        person = df.loc[person_id]
        # Check to see whether they have been sent here from RTI_MedicalIntervention and they haven't died due to rti
        assert person.rt_med_int, 'person sent here not been through RTI_MedInt'
        # Determine what injuries are able to be treated by surgery by checking the injury codes which are currently
        # treated in this simulation, it seems there is a limited available to treat spinal cord injuries and chest
        # trauma in Malawi, so these are initially left out, but we will test different scenarios to see what happens
        # when we include those treatments
        surgically_treated_codes = ['112', '811', '812', '813a', '813b', '813c', '133a', '133b', '133c', '133d', '134a',
                                    '134b', '135', '552', '553', '554', '342', '343', '414', '361', '363',
                                    '782', '782a', '782b', '782c', '783', '822a', '882', '883', '884',
                                    'P133a', 'P133b', 'P133c', 'P133d', 'P134a', 'P134b', 'P135', 'P782a', 'P782b',
                                    'P782c', 'P783', 'P882', 'P883', 'P884'
                                    ]

        # If we allow surgical treatment of spinal cord injuries, extend the surgically treated codes to include spinal
        # cord injury codes
        if 'include_spine_surgery' in self.allowed_interventions:
            additional_codes = ['673a', '673b', '674a', '674b', '675a', '675b', '676', 'P673a', 'P673b', 'P674',
                                'P674a', 'P674b', 'P675', 'P675a', 'P675b', 'P676']
            surgically_treated_codes.extend(additional_codes)
        # If we allow surgical treatment of chest trauma, extend the surgically treated codes to include chest trauma
        # codes.
        if 'include_thoroscopy' in self.allowed_interventions:
            additional_codes = ['441', '443', '453', '453a', '453b', '463']
            surgically_treated_codes.extend(additional_codes)
        # check this person has an injury which should be treated here
        assert len(set(person.rt_injuries_for_major_surgery) & set(surgically_treated_codes)) > 0, \
            'This person has asked for surgery but does not have an appropriate injury'
        # isolate the relevant injury information
        person_injuries = df.loc[[person_id], RTI.INJURY_COLUMNS]
        # Check whether the person sent to surgery has an injury which actually requires surgery
        _, counts = RTI.rti_find_and_count_injuries(person_injuries, surgically_treated_codes)
        assert counts > 0, 'This person has been sent to major surgery without the right injuries'
        assert len(person.rt_injuries_for_major_surgery) > 0
        # for each injury which has been assigned to be treated by major surgery make sure that the injury hasn't
        # already been treated
        for code in person.rt_injuries_for_major_surgery:
            column, found_code = self.rti_find_injury_column(person_id, [code])
            index_in_rt_recovery_dates = int(column[-1]) - 1
            if not pd.isnull(person.rt_date_to_remove_daly[index_in_rt_recovery_dates]):
                df.loc[person_id, 'rt_date_to_remove_daly'][index_in_rt_recovery_dates] = pd.NaT
        # If this person is alive schedule major surgeries
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
        # Check to see whether they have been sent here from RTI_MedicalIntervention and they haven't been killed by the
        # RTI module
        assert df.at[person_id, 'rt_med_int'], 'Person sent for treatment did not go through rti med int'
        # Isolate the person
        person = df.loc[person_id]
        # state the codes treated by minor surgery
        surgically_treated_codes = ['211', '212', '291', '241', '322', '323', '722', '811', '812', '813a',
                                    '813b', '813c']
        # check that the person requesting surgery has an injury in their minor surgery treatment plan
        assert len(df.loc[person_id, 'rt_injuries_for_minor_surgery']) > 0, \
            'this person has asked for a minor surgery but does not need it'
        # check that for each injury due to be treated with a minor surgery, the injury hasn't previously been treated
        for code in df.loc[person_id, 'rt_injuries_for_minor_surgery']:
            column, found_code = self.rti_find_injury_column(person_id, [code])
            index_in_rt_recovery_dates = int(column[-1]) - 1
            assert pd.isnull(df.loc[person_id, 'rt_date_to_remove_daly'][index_in_rt_recovery_dates])
        # check that this person's injuries that were decided to be treated with a minor surgery and the injuries
        # actually treated by minor surgeries coincide
        assert len(set(df.loc[person_id, 'rt_injuries_for_minor_surgery']) & set(surgically_treated_codes)) > 0, \
            'This person has asked for a minor surgery but does not need it'
        # Isolate the relevant injury information
        person_injuries = df.loc[[person_id], RTI.INJURY_COLUMNS]
        # Check whether the person requesting minor surgeries has an injury that requires minor surgery
        _, counts = RTI.rti_find_and_count_injuries(person_injuries, surgically_treated_codes)
        assert counts > 0
        # if this person is alive schedule the minor surgery
        if person.is_alive:
            self.sim.modules['HealthSystem'].schedule_hsi_event(
                hsi_event=HSI_RTI_Minor_Surgeries(module=self,
                                                  person_id=person_id),
                priority=0,
                topen=self.sim.date + DateOffset(days=count),
                tclose=self.sim.date + DateOffset(days=15))

    def rti_acute_pain_management(self, person_id):
        """
        Function called in HSI_RTI_MedicalIntervention to request pain management. This should be called for every alive
        injured person, regardless of what their injuries are. In this function we test whether they meet the
        requirements to recieve for pain relief, that is they are alive and currently receiving medical treatment.
        :param person_id: The person requesting pain management
        :return: n/a
        """
        df = self.sim.population.props
        # Check to see whether they have been sent here from RTI_MedicalIntervention and they haven't died due to rti
        assert df.at[person_id, 'rt_med_int'], 'person sent here not been through rti med int'
        # Isolate the relevant injury information
        person_injuries = df.loc[[person_id], RTI.INJURY_COLUMNS]
        # check this person is injured, search they have an injury code that isn't "none".
        idx, counts = RTI.rti_find_and_count_injuries(person_injuries,
                                                      self.PROPERTIES.get('rt_injury_1').categories[1:])
        assert counts > 0, 'This person has asked for pain relief despite not being injured'
        person = df.loc[person_id]
        # if the person is alive schedule pain management
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
        person = df.loc[person_id]
        if not person.is_alive:
            return
        # Check to see whether they have been sent here from RTI_MedicalIntervention and they haven't died due to rti
        assert df.at[person_id, 'rt_med_int'], 'person sent here not been through rti med int'
        # Isolate the relevant injury information
        person_injuries = df.loc[[person_id], RTI.INJURY_COLUMNS]
        laceration_codes = ['1101', '2101', '3101', '4101', '5101', '6101', '7101', '8101']
        # Check they have a laceration which needs stitches
        _, counts = RTI.rti_find_and_count_injuries(person_injuries, laceration_codes)
        assert counts > 0, "This person has asked for stiches, but doens't have a laceration"
        # if the person is alive request the hsi event
        self.sim.modules['HealthSystem'].schedule_hsi_event(
            hsi_event=HSI_RTI_Suture(module=self,
                                     person_id=person_id),
            priority=0,
            topen=self.sim.date,
            tclose=self.sim.date + DateOffset(days=15)
        )

    def rti_ask_for_shock_treatment(self, person_id):
        """
        A function called by the generic emergency appointment to treat the onset of hypovolemic shock
        :param person_id:
        :return:
        """
        df = self.sim.population.props
        person = df.loc[person_id]
        if not person.is_alive:
            return
        assert person.rt_in_shock, 'person requesting shock treatment is not in shock'

        self.sim.modules['HealthSystem'].schedule_hsi_event(
            hsi_event=HSI_RTI_Shock_Treatment(module=self,
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
        person = df.loc[person_id]

        if not person.is_alive:
            return

        # Check to see whether they have been sent here from RTI_MedicalIntervention and they haven't died due to rti
        assert person.rt_med_int, 'person not been through rti med int'
        # Isolate the relevant injury information
        person_injuries = df.loc[[person_id], RTI.INJURY_COLUMNS]
        burn_codes = ['1114', '2114', '3113', '4113', '5113', '7113', '8113']
        # Check to see whether they have a burn which needs treatment
        _, counts = RTI.rti_find_and_count_injuries(person_injuries, burn_codes)
        assert counts > 0, "This person has asked for burn treatment, but doens't have any burns"

        # if this person is alive ask for the hsi event
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
        person = df.loc[person_id]
        if not person.is_alive:
            return
        # Check to see whether they have been sent here from RTI_MedicalIntervention and they haven't died due to rti
        assert df.at[person_id, 'rt_med_int'], 'person sent here not been through rti med int'
        # Isolate the relevant injury information
        person_injuries = df.loc[[person_id], RTI.INJURY_COLUMNS]
        fracture_codes = ['712a', '712b', '712c', '811', '812', '813a', '813b', '813c', '822a', '822b']
        # check that the codes assigned for treatment by rt_injuries_to_cast and the codes treated by rti_fracture_cast
        # coincide
        assert len(set(df.loc[person_id, 'rt_injuries_to_cast']) & set(fracture_codes)) > 0, \
            'This person has asked for a fracture cast'
        # Check they have an injury treated by HSI_RTI_Fracture_Cast
        _, counts = RTI.rti_find_and_count_injuries(person_injuries, fracture_codes)
        assert counts > 0, "This person has asked for fracture treatment, but doens't have appropriate fractures"
        # if this person is alive request the hsi
        self.sim.modules['HealthSystem'].schedule_hsi_event(
            hsi_event=HSI_RTI_Fracture_Cast(module=self,
                                            person_id=person_id),
            priority=0,
            topen=self.sim.date,
            tclose=self.sim.date + DateOffset(days=15)
        )

    def rti_ask_for_open_fracture_treatment(self, person_id, counts):
        """Function called by HSI_RTI_MedicalIntervention to centralise open fracture treatment requests. This function
        schedules an open fracture event, conditional on whether they are alive, being treated and have an appropriate
        injury.

        :param person_id: the person requesting a tetanus jab
        :param counts: the number of open fractures that requires a treatment
        :return: n/a
        """
        df = self.sim.population.props
        person = df.loc[person_id]
        if not person.is_alive:
            return
        # Check to see whether they have been sent here from RTI_MedicalIntervention and are haven't died due to rti
        assert df.at[person_id, 'rt_med_int'], 'person sent here not been through rti med int'
        # Isolate the relevant injury information

        open_fracture_codes = ['813bo', '813co', '813do', '813eo']
        person_injuries = df.loc[[person_id], RTI.INJURY_COLUMNS]
        # Check that they have an open fracture
        _, counts = RTI.rti_find_and_count_injuries(person_injuries, open_fracture_codes)
        assert counts > 0, "This person has requested open fracture treatment but doesn't require one"
        # if the person is alive request the hsi
        for i in range(0, counts):
            # shedule the treatments, say the treatments occur a day appart for now
            self.sim.modules['HealthSystem'].schedule_hsi_event(
                hsi_event=HSI_RTI_Open_Fracture_Treatment(module=self, person_id=person_id),
                priority=0,
                topen=self.sim.date + DateOffset(days=0 + i),
                tclose=self.sim.date + DateOffset(days=15 + i)
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
        person = df.loc[person_id]
        if not person.is_alive:
            return
        # Check to see whether they have been sent here from RTI_MedicalIntervention and are haven't died due to rti
        assert person.rt_med_int, 'person sent here not been through rti med int'
        # Isolate the relevant injury information
        codes_for_tetanus = ['1101', '2101', '3101', '4101', '5101', '7101', '8101',
                             '1114', '2114', '3113', '4113', '5113', '7113', '8113']
        person_injuries = df.loc[[person_id], RTI.INJURY_COLUMNS]
        # Check that they have a burn/laceration
        _, counts = RTI.rti_find_and_count_injuries(person_injuries, codes_for_tetanus)
        assert counts > 0, "This person has requested a tetanus jab but doesn't require one"
        # if this person is alive, ask for the hsi
        self.sim.modules['HealthSystem'].schedule_hsi_event(
            hsi_event=HSI_RTI_Tetanus_Vaccine(module=self,
                                              person_id=person_id),
            priority=0,
            topen=self.sim.date,
            tclose=self.sim.date + DateOffset(days=15)
        )

    def rti_find_injury_column(self, person_id, codes):
        """
        This function is a tool to find the injury column an injury code occurs in, when calling this funtion
        you will need to guarentee that the person has at least one of the code you are searching for, else this
        function will raise an assertion error.
        To call this function you need to provide the person who you want to perform the search on and the injury
        codes which you want to find the corresponding injury column for. The function/search will return the injury
        code which the person has from the list of codes you supplied, and which injury column from rt_injury_1 through
        to rt_injury_8, the code appears in.

        :param person_id: The person the search is being performed for
        :param codes: The injury codes being searched for
        :return: which column out of rt_injury_1 to rt_injury_8 the injury code occurs in, and the injury code itself
        """
        df = self.sim.population.props.loc[[person_id], RTI.INJURY_COLUMNS]
        # iterate over the codes to search the dataframe for
        injury_column = ''
        injury_code = ''
        for code in codes:
            # iterate over the columns where the code can be found
            for col in RTI.INJURY_COLUMNS:
                # if the code appears in the series, store
                if df[col].str.contains(code).any():
                    injury_column = col
                    injury_code = code
                    break
        # Check that the search found the injury column
        assert injury_column != '', df
        # Return the found column for the injury code
        return injury_column, injury_code

    def rti_find_all_columns_of_treated_injuries(self, person_id, codes):
        """
        This function searches for treated injuries (supplied by the parameter codes) for a specific person, finding and
        returning all the columns with treated injuries and all the injury codes for the treated injuries.

        :param person_id: The person the search is being performed on
        :param codes: The treated injury codes
        :return: All columns and codes of the successfully treated injuries
        """
        df = self.sim.population.props.loc[[person_id], RTI.INJURY_COLUMNS]
        # create empty variables to return the columns and codes of the treated injuries
        columns_to_return = []
        codes_to_return = []
        # iterate over the codes in the list codes and also the injury columns
        for code in codes:
            for col in df.columns:
                # Search a sub-dataframe that is non-empty if the code is present is in that column and empty if not
                if df[col].str.contains(code).any():
                    columns_to_return.append(col)
                    codes_to_return.append(code)

        return columns_to_return, codes_to_return

    def rti_assign_daly_weights(self, injured_index):
        """
        This function assigns DALY weights associated with each injury when they happen.

        By default this function gives the DALY weight for each condition without treatment, this will then be swapped
        for the DALY weight associated with the injury with treatment when treatment occurs.

        The properties that this function alters are rt_disability, which is the property used to report the
        disability burden that this person has and rt_debugging_DALY_wt, which stores the true value of the
        the disability.

        :param injured_index: The people who have been involved in a road traffic accident for the current month and did
                              not die on the scene of the crash
        :return: n/a
        """
        df = self.sim.population.props
        p = self.parameters
        # ============================= DALY weights ===================================================================
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
        # ==============================================================================================================
        # Check that those sent here have been involved in a road traffic accident
        assert sum(df.loc[injured_index, 'rt_road_traffic_inc']) == len(injured_index)
        # Check everyone here has at least one injury to be given a daly weight to
        assert sum(df.loc[injured_index, 'rt_injury_1'] != "none") == len(injured_index)
        # Check everyone here is alive and hasn't died due to rti
        rti_deaths = ['RTI_death_without_med', 'RTI_death_with_med', 'RTI_unavailable_med', 'RTI_imm_death',
                      'RTI_death_shock']
        assert (sum(~df.loc[injured_index, 'cause_of_death'].isin(rti_deaths)) == len(injured_index)) & \
               (sum(df.loc[injured_index, 'rt_imm_death']) == 0)
        selected_for_rti_inj = df.loc[injured_index, RTI.INJURY_COLUMNS]

        daly_lookup = {
            # =============================== AIS region 1: head =======================================================
            # ------ Find those with skull fractures and update rt_fracture to match and call for treatment ------------
            '112': self.daly_wt_unspecified_skull_fracture,
            '113': self.daly_wt_basilar_skull_fracture,
            # ------ Find those with traumatic brain injury and update rt_tbi to match and call the TBI treatment-------
            '133a': self.daly_wt_subarachnoid_hematoma,
            '133b': self.daly_wt_brain_contusion,
            '133c': self.daly_wt_intraventricular_haemorrhage,
            '133d': self.daly_wt_subgaleal_hematoma,
            '134a': self.daly_wt_epidural_hematoma,
            '134b': self.daly_wt_subdural_hematoma,
            '135': self.daly_wt_diffuse_axonal_injury,
            '1101': self.daly_wt_facial_soft_tissue_injury,
            '1114': self.daly_wt_burns_greater_than_20_percent_body_area,
            # =============================== AIS region 2: face =======================================================
            # ----------------------- Find those with facial fractures and assign DALY weight --------------------------
            '211': self.daly_wt_facial_fracture,
            '212': self.daly_wt_facial_fracture,
            # ----------------- Find those with lacerations/soft tissue injuries and assign DALY weight ----------------
            '2101': self.daly_wt_facial_soft_tissue_injury,
            # ----------------- Find those with eye injuries and assign DALY weight ------------------------------------
            '291': self.daly_wt_eye_injury,
            '241': self.daly_wt_facial_soft_tissue_injury,
            '2114': self.daly_wt_burns_greater_than_20_percent_body_area,
            # =============================== AIS region 3: Neck =======================================================
            # -------------------------- soft tissue injuries and internal bleeding-------------------------------------
            '342': self.daly_wt_neck_internal_bleeding,
            '343': self.daly_wt_neck_internal_bleeding,
            '361': self.daly_wt_neck_internal_bleeding,
            '363': self.daly_wt_neck_internal_bleeding,
            # -------------------------------- neck vertebrae dislocation ----------------------------------------------
            '322': self.daly_wt_neck_dislocation,
            '323': self.daly_wt_neck_dislocation,
            '3101': self.daly_wt_facial_soft_tissue_injury,
            '3113': self.daly_wt_burns_less_than_20_percent_body_area_without_treatment,
            # ================================== AIS region 4: Thorax ==================================================
            # --------------------------------- fractures & flail chest ------------------------------------------------
            '412': self.daly_wt_rib_fracture,
            '414': self.daly_wt_flail_chest,
            # ------------------------------------ Internal bleeding ---------------------------------------------------
            # chest wall bruises/hematoma
            '461': self.daly_wt_chest_wall_bruises_hematoma,
            '463': self.daly_wt_hemothorax,
            # -------------------------------- Internal organ injury ---------------------------------------------------
            '453a': self.daly_wt_diaphragm_rupture,
            '453b': self.daly_wt_lung_contusion,
            # ----------------------------------- Soft tissue injury ---------------------------------------------------
            '442': self.daly_wt_surgical_emphysema,
            # ---------------------------------- Pneumothoraxs ---------------------------------------------------------
            '441': self.daly_wt_closed_pneumothorax,
            '443': self.daly_wt_open_pneumothorax,
            '4101': self.daly_wt_facial_soft_tissue_injury,
            '4113': self.daly_wt_burns_less_than_20_percent_body_area_without_treatment,
            # ================================== AIS region 5: Abdomen =================================================
            '552': self.daly_wt_abd_internal_organ_injury,
            '553': self.daly_wt_abd_internal_organ_injury,
            '554': self.daly_wt_abd_internal_organ_injury,
            '5101': self.daly_wt_facial_soft_tissue_injury,
            '5113': self.daly_wt_burns_less_than_20_percent_body_area_without_treatment,
            # =================================== AIS region 6: spine ==================================================
            # ----------------------------------- vertebrae fracture ---------------------------------------------------
            '612': self.daly_wt_vertebrae_fracture,
            # ---------------------------------- Spinal cord injuries --------------------------------------------------
            '673a': self.daly_wt_spinal_cord_lesion_neck_without_treatment,
            '673b': self.daly_wt_spinal_cord_lesion_below_neck_without_treatment,
            '674a': self.daly_wt_spinal_cord_lesion_neck_without_treatment,
            '674b': self.daly_wt_spinal_cord_lesion_below_neck_without_treatment,
            '675a': self.daly_wt_spinal_cord_lesion_neck_without_treatment,
            '675b': self.daly_wt_spinal_cord_lesion_below_neck_without_treatment,
            '676': self.daly_wt_spinal_cord_lesion_neck_without_treatment,
            # ============================== AIS body region 7: upper extremities ======================================
            # ------------------------------------------ fractures -----------------------------------------------------
            # Fracture to Clavicle, scapula, humerus, Hand/wrist, Radius/ulna
            '712a': self.daly_wt_clavicle_scapula_humerus_fracture,
            '712b': self.daly_wt_hand_wrist_fracture_without_treatment,
            '712c': self.daly_wt_radius_ulna_fracture_short_term_with_without_treatment,
            # ------------------------------------ Dislocation of shoulder ---------------------------------------------
            '722': self.daly_wt_dislocated_shoulder,
            # ------------------------------------------ Amputations ---------------------------------------------------
            # Amputation of fingers, Unilateral upper limb amputation, Thumb amputation
            '782a': self.daly_wt_amputated_finger,
            '782b': self.daly_wt_unilateral_arm_amputation_without_treatment,
            '782c': self.daly_wt_amputated_thumb,
            '783': self.daly_wt_bilateral_arm_amputation_without_treatment,
            # ----------------------------------- cuts and bruises -----------------------------------------------------
            '7101': self.daly_wt_facial_soft_tissue_injury,
            '7113': self.daly_wt_burns_less_than_20_percent_body_area_without_treatment,
            # ============================== AIS body region 8: Lower extremities ======================================
            # ------------------------------------------ Fractures -----------------------------------------------------
            # Broken foot
            '811': self.daly_wt_foot_fracture_short_term_with_without_treatment,
            # Broken foot (open), currently combining the daly weight used for open wounds and the fracture
            '813do': (self.daly_wt_foot_fracture_short_term_with_without_treatment +
                      self.daly_wt_facial_soft_tissue_injury),
            # Broken patella, tibia, fibula
            '812': self.daly_wt_patella_tibia_fibula_fracture_without_treatment,
            # Broken foot (open), currently combining the daly weight used for open wounds and the fracture
            '813eo': (self.daly_wt_patella_tibia_fibula_fracture_without_treatment +
                      self.daly_wt_facial_soft_tissue_injury),
            # Broken Hip, Pelvis, Femur other than femoral neck
            '813a': self.daly_wt_hip_fracture_short_term_with_without_treatment,
            '813b': self.daly_wt_pelvis_fracture_short_term,
            # broken pelvis (open)
            '813bo': self.daly_wt_pelvis_fracture_short_term + self.daly_wt_facial_soft_tissue_injury,
            '813c': self.daly_wt_femur_fracture_short_term,
            # broken femur (open)
            '813co': self.daly_wt_femur_fracture_short_term + self.daly_wt_facial_soft_tissue_injury,
            # -------------------------------------- Dislocations ------------------------------------------------------
            # Dislocated hip, knee
            '822a': self.daly_wt_dislocated_hip,
            '822b': self.daly_wt_dislocated_knee,
            # --------------------------------------- Amputations ------------------------------------------------------
            # toes
            '882': self.daly_wt_amputated_toes,
            # Unilateral lower limb amputation
            '883': self.daly_wt_unilateral_lower_limb_amputation_without_treatment,
            # Bilateral lower limb amputation
            '884': self.daly_wt_bilateral_lower_limb_amputation_without_treatment,
            # ------------------------------------ cuts and bruises ----------------------------------------------------
            '8101': self.daly_wt_facial_soft_tissue_injury,
            '8113': self.daly_wt_burns_less_than_20_percent_body_area_without_treatment
        }

        daly_change = selected_for_rti_inj.applymap(lambda code: daly_lookup.get(code, 0)).sum(axis=1)
        df.loc[injured_index, 'rt_disability'] += daly_change

        # Store the true sum of DALY weights in the df
        df.loc[injured_index, 'rt_debugging_DALY_wt'] = df.loc[injured_index, 'rt_disability']
        # Find who's disability burden is greater than one
        DALYweightoverlimit = df.index[df['rt_disability'] > 1]
        # Set the total daly weights to one in this case
        df.loc[DALYweightoverlimit, 'rt_disability'] = 1
        # Find who's disability burden is less than one
        DALYweightunderlimit = df.index[df.rt_road_traffic_inc & ~ df.rt_imm_death & (df['rt_disability'] <= 0)]
        # Check that no one has a disability burden less than or equal to zero
        assert len(DALYweightunderlimit) == 0, ('Someone has not been given an injury burden',
                                                selected_for_rti_inj.loc[DALYweightunderlimit])
        df.loc[DALYweightunderlimit, 'rt_disability'] = 0
        assert (df.loc[injured_index, 'rt_disability'] > 0).all()

    def rti_alter_daly_post_treatment(self, person_id, codes):
        """
        This function removes the DALY weight associated with each injury code after treatment is complete. This
        function is called by RTI_Recovery_event which removes asks to remove the DALY weight when the injury has
        healed

        The properties that this function alters are rt_disability, which is the property used to report the
        disability burden that this person has and rt_debugging_DALY_wt, which stores the true value of the
        the disability.

        :param person_id: The person who needs a daly weight removed as their injury has healed
        :param codes: The injury codes for the healed injury/injuries
        :return: n/a
        """

        df = self.sim.population.props
        # Check everyone here has at least one injury to be alter the daly weight to
        person_injuries = df.loc[[person_id], RTI.INJURY_COLUMNS]
        # check this person is injured, search they have an injury code that isn't "none"
        idx, counts = RTI.rti_find_and_count_injuries(person_injuries,
                                                      self.PROPERTIES.get('rt_injury_1').categories[1:])
        assert counts > 0, 'This person has asked for medical treatment despite not being injured'
        # Check everyone here is alive and hasn't died on scene
        assert ~df.loc[person_id, 'rt_imm_death']

        # ------------------------------- Remove the daly weights for treated injuries ---------------------------------
        # ==================================== heal with time injuries =================================================
        # store open fracture daly weight codes in one variable
        daly_wt_813bo = self.daly_wt_pelvis_fracture_long_term + self.daly_wt_facial_soft_tissue_injury
        daly_wt_813co = self.daly_wt_femur_fracture_short_term + self.daly_wt_facial_soft_tissue_injury
        daly_wt_813do = \
            self.daly_wt_foot_fracture_short_term_with_without_treatment + self.daly_wt_facial_soft_tissue_injury
        daly_wt_813eo = \
            self.daly_wt_patella_tibia_fibula_fracture_without_treatment + self.daly_wt_facial_soft_tissue_injury
        daly_weight_removal_lookup = {
            # heal with time injuries
            '322': self.daly_wt_neck_dislocation,
            '323': self.daly_wt_neck_dislocation,
            '822a': self.daly_wt_dislocated_hip,
            '822b': self.daly_wt_dislocated_knee,
            '112': self.daly_wt_unspecified_skull_fracture,
            '113': self.daly_wt_basilar_skull_fracture,
            '552': self.daly_wt_abd_internal_organ_injury,
            '553': self.daly_wt_abd_internal_organ_injury,
            '554': self.daly_wt_abd_internal_organ_injury,
            '412': self.daly_wt_rib_fracture,
            '442': self.daly_wt_surgical_emphysema,
            '461': self.daly_wt_chest_wall_bruises_hematoma,
            '612': self.daly_wt_vertebrae_fracture,
            # injuries treated with suture
            '1101': self.daly_wt_facial_soft_tissue_injury,
            '2101': self.daly_wt_facial_soft_tissue_injury,
            '3101': self.daly_wt_facial_soft_tissue_injury,
            '4101': self.daly_wt_facial_soft_tissue_injury,
            '5101': self.daly_wt_facial_soft_tissue_injury,
            '7101': self.daly_wt_facial_soft_tissue_injury,
            '8101': self.daly_wt_facial_soft_tissue_injury,
            # injuries treated with a cast
            '712a': self.daly_wt_clavicle_scapula_humerus_fracture,
            '712b': self.daly_wt_hand_wrist_fracture_with_treatment,
            '712c': self.daly_wt_radius_ulna_fracture_short_term_with_without_treatment,
            '811': self.daly_wt_foot_fracture_short_term_with_without_treatment,
            '812': self.daly_wt_patella_tibia_fibula_fracture_with_treatment,
            # injuries treated with minor surgery
            '722': self.daly_wt_dislocated_shoulder,
            '291': self.daly_wt_eye_injury,
            '241': self.daly_wt_facial_soft_tissue_injury,
            '211': self.daly_wt_facial_fracture,
            '212': self.daly_wt_facial_fracture,
            # injuries treated with burn management
            '1114': self.daly_wt_burns_greater_than_20_percent_body_area,
            '2114': self.daly_wt_burns_greater_than_20_percent_body_area,
            '3113': self.daly_wt_burns_less_than_20_percent_body_area_with_treatment,
            '4113': self.daly_wt_burns_less_than_20_percent_body_area_with_treatment,
            '5113': self.daly_wt_burns_less_than_20_percent_body_area_with_treatment,
            '7113': self.daly_wt_burns_less_than_20_percent_body_area_with_treatment,
            '8113': self.daly_wt_burns_less_than_20_percent_body_area_with_treatment,
            # injuries treated with major surgery
            '813a': self.daly_wt_hip_fracture_long_term_with_treatment,
            '813b': self.daly_wt_pelvis_fracture_long_term,
            '813c': self.daly_wt_femur_fracture_short_term,
            '133a': self.daly_wt_subarachnoid_hematoma,
            '133b': self.daly_wt_brain_contusion,
            '133c': self.daly_wt_intraventricular_haemorrhage,
            '133d': self.daly_wt_subgaleal_hematoma,
            '134a': self.daly_wt_epidural_hematoma,
            '134b': self.daly_wt_subdural_hematoma,
            '135': self.daly_wt_diffuse_axonal_injury,
            '342': self.daly_wt_neck_internal_bleeding,
            '343': self.daly_wt_neck_internal_bleeding,
            '361': self.daly_wt_neck_internal_bleeding,
            '363': self.daly_wt_neck_internal_bleeding,
            '414': self.daly_wt_flail_chest,
            '441': self.daly_wt_closed_pneumothorax,
            '443': self.daly_wt_open_pneumothorax,
            '453a': self.daly_wt_diaphragm_rupture,
            '453b': self.daly_wt_lung_contusion,
            '463': self.daly_wt_hemothorax,
            # injuries treated with open fracture treatment
            '813bo': daly_wt_813bo,
            '813co': daly_wt_813co,
            '813do': daly_wt_813do,
            '813eo': daly_wt_813eo,

        }
        # update the total values of the daly weights
        df.loc[person_id, 'rt_debugging_DALY_wt'] -= sum([daly_weight_removal_lookup[code] for code in codes])
        # round off any potential floating point errors
        df.loc[person_id, 'rt_debugging_DALY_wt'] = np.round(df.loc[person_id, 'rt_debugging_DALY_wt'], 4)
        # if the person's true total for daly weights is greater than one, report rt_disability as one, if not
        # report the true disability burden.
        if df.loc[person_id, 'rt_debugging_DALY_wt'] > 1:
            df.loc[person_id, 'rt_disability'] = 1
        else:
            df.loc[person_id, 'rt_disability'] = df.loc[person_id, 'rt_debugging_DALY_wt']
        # if the reported daly weight is below zero add make the model report the true (and always positive) daly weight
        if df.loc[person_id, 'rt_disability'] < 0:
            df.loc[person_id, 'rt_disability'] = df.loc[person_id, 'rt_debugging_DALY_wt']
        # Make sure the true disability burden is greater or equal to zero
        assert df.loc[person_id, 'rt_debugging_DALY_wt'] >= 0, (person_injuries.values,
                                                                df.loc[person_id, 'rt_debugging_DALY_wt'])
        # the reported disability should satisfy 0<=disability<=1, check that they do
        assert df.loc[person_id, 'rt_disability'] >= 0, 'Negative disability burden'
        assert df.loc[person_id, 'rt_disability'] <= 1, 'Too large disability burden'
        # remover the treated injury code from the person using rti_treated_injuries
        RTI.rti_treated_injuries(self, person_id, codes)

    def rti_swap_injury_daly_upon_treatment(self, person_id, codes):
        """
        This function swaps certain DALY weight codes upon when a person receives treatment(s). Some injuries have a
        different daly weight associated with them for the treated and untreated injuries. If an injury is 'swap-able'
        then this function removes the old daly weight for the untreated injury and gives the daly weight for the
        treated injury.

        The properties that this function alters are rt_disability, which is the property used to report the
        disability burden that this person has and rt_debugging_DALY_wt, which stores the true value of the
        the disability.


        :param person_id: The person who has received treatment
        :param codes: the 'swap-able' injury code
        :return: n/a

        """
        df = self.sim.population.props
        # Check the people that are sent here have had medical treatment
        assert df.loc[person_id, 'rt_med_int']
        # Check they have an appropriate injury code to swap

        swapping_codes = RTI.SWAPPING_CODES[:]
        relevant_codes = np.intersect1d(codes, swapping_codes)
        person_injuries = df.loc[[person_id], RTI.INJURY_COLUMNS]
        # check this person is injured, search they have an injury code that is swappable
        idx, counts = RTI.rti_find_and_count_injuries(person_injuries, list(relevant_codes))
        assert counts > 0, 'This person has asked to swap an injury code, but it is not swap-able'

        daly_weight_change_lookup = {
            '712b': (- self.daly_wt_hand_wrist_fracture_without_treatment +
                     self.daly_wt_hand_wrist_fracture_with_treatment),
            '812': (- self.daly_wt_patella_tibia_fibula_fracture_without_treatment +
                    self.daly_wt_patella_tibia_fibula_fracture_with_treatment),
            '3113': (- self.daly_wt_burns_less_than_20_percent_body_area_without_treatment +
                     self.daly_wt_burns_less_than_20_percent_body_area_with_treatment),
            '4113': (- self.daly_wt_burns_less_than_20_percent_body_area_without_treatment +
                     self.daly_wt_burns_less_than_20_percent_body_area_with_treatment),
            '5113': (- self.daly_wt_burns_less_than_20_percent_body_area_without_treatment +
                     self.daly_wt_burns_less_than_20_percent_body_area_with_treatment),
            '7113': (- self.daly_wt_burns_less_than_20_percent_body_area_without_treatment +
                     self.daly_wt_burns_less_than_20_percent_body_area_with_treatment),
            '8113': (- self.daly_wt_burns_less_than_20_percent_body_area_without_treatment +
                     self.daly_wt_burns_less_than_20_percent_body_area_with_treatment),
            '813a': (- self.daly_wt_hip_fracture_short_term_with_without_treatment +
                     self.daly_wt_hip_fracture_long_term_with_treatment),
            '813b': - self.daly_wt_pelvis_fracture_short_term + self.daly_wt_pelvis_fracture_long_term,
            '813bo': - self.daly_wt_pelvis_fracture_short_term + self.daly_wt_pelvis_fracture_long_term,
            'P673a': (- self.daly_wt_spinal_cord_lesion_neck_without_treatment +
                      self.daly_wt_spinal_cord_lesion_neck_with_treatment),
            'P673b': (- self.daly_wt_spinal_cord_lesion_below_neck_without_treatment +
                      self.daly_wt_spinal_cord_lesion_below_neck_with_treatment),
            'P674a': (- self.daly_wt_spinal_cord_lesion_neck_without_treatment +
                      self.daly_wt_spinal_cord_lesion_neck_with_treatment),
            'P674b': (- self.daly_wt_spinal_cord_lesion_below_neck_without_treatment +
                      self.daly_wt_spinal_cord_lesion_below_neck_with_treatment),
            'P675a': (- self.daly_wt_spinal_cord_lesion_neck_without_treatment +
                      self.daly_wt_spinal_cord_lesion_neck_with_treatment),
            'P675b': (- self.daly_wt_spinal_cord_lesion_below_neck_without_treatment +
                      self.daly_wt_spinal_cord_lesion_below_neck_with_treatment),
            'P676': (- self.daly_wt_spinal_cord_lesion_neck_without_treatment +
                     self.daly_wt_spinal_cord_lesion_neck_with_treatment),
            'P782b': (- self.daly_wt_unilateral_arm_amputation_without_treatment +
                      self.daly_wt_unilateral_arm_amputation_with_treatment),
            'P783': (- self.daly_wt_bilateral_arm_amputation_without_treatment +
                     self.daly_wt_bilateral_arm_amputation_with_treatment),
            'P883': (- self.daly_wt_unilateral_lower_limb_amputation_without_treatment +
                     self.daly_wt_unilateral_lower_limb_amputation_with_treatment),
            'P884': (- self.daly_wt_bilateral_lower_limb_amputation_without_treatment +
                     self.daly_wt_bilateral_lower_limb_amputation_with_treatment)
        }

        # swap the relevant code's daly weight, from the daly weight associated with the injury without treatment
        # and the daly weight for the disability with treatment.
        # keep track of the changes to the daly weights
        # update the disability burdens
        df.loc[person_id, 'rt_debugging_DALY_wt'] += sum([daly_weight_change_lookup[code] for code in relevant_codes])
        df.loc[person_id, 'rt_debugging_DALY_wt'] = np.round(df.loc[person_id, 'rt_debugging_DALY_wt'], 4)
        # Check that the person's true disability burden is positive
        assert df.loc[person_id, 'rt_debugging_DALY_wt'] >= 0, (person_injuries.values,
                                                                df.loc[person_id, 'rt_debugging_DALY_wt'])
        # catch rounding point errors where the disability weights should be zero but aren't
        if df.loc[person_id, 'rt_disability'] < 0:
            df.loc[person_id, 'rt_disability'] = df.loc[person_id, 'rt_debugging_DALY_wt']
        # Catch cases where the disability burden is greater than one in reality but needs to be
        # capped at one, if not report the true disability burden
        if df.loc[person_id, 'rt_debugging_DALY_wt'] > 1:
            df.loc[person_id, 'rt_disability'] = 1
        else:
            df.loc[person_id, 'rt_disability'] = df.loc[person_id, 'rt_debugging_DALY_wt']
        # Check the daly weights fall within the accepted bounds
        assert df.loc[person_id, 'rt_disability'] >= 0, 'Negative disability burden'
        assert df.loc[person_id, 'rt_disability'] <= 1, 'Too large disability burden'

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

        def draw_days(_mean, _sd):
            return int(self.rng.normal(_mean, _sd, 1))

        # Create the length of stays required for each ISS score boundaries and check that they are >=0
        rt_iss_score = df.loc[person_id, 'rt_ISS_score']

        if rt_iss_score < 4:
            days_until_treatment_end = draw_days(p["mean_los_ISS_less_than_4"], p["sd_los_ISS_less_than_4"])
        elif 4 <= rt_iss_score < 9:
            days_until_treatment_end = draw_days(p["mean_los_ISS_4_to_8"], p["sd_los_ISS_4_to_8"])
        elif 9 <= rt_iss_score < 16:
            days_until_treatment_end = draw_days(p["mean_los_ISS_9_to_15"], p["sd_los_ISS_9_to_15"])
        elif 16 <= rt_iss_score < 25:
            days_until_treatment_end = draw_days(p["mean_los_ISS_16_to_24"], p["sd_los_ISS_16_to_24"])
        elif 25 <= rt_iss_score:
            days_until_treatment_end = draw_days(p["mean_los_ISS_more_than_25"], p["sd_los_ISS_more_that_25"])
        else:
            days_until_treatment_end = 0

        if days_until_treatment_end < 0:
            days_until_treatment_end = 0

        # Return the LOS
        return days_until_treatment_end

    @staticmethod
    def rti_find_and_count_injuries(persons_injury_properties: pd.DataFrame, injury_codes: list):
        """
        A function that searches a user given dataframe for a list of injuries (injury_codes). If the injury code is
        found in the dataframe, this function returns the index for who has the injury/injuries and the number of
        injuries found. This function works much faster if the dataframe is smaller, hence why the searched dataframe
        is a parameter in the function.

        :param persons_injury_properties: The dataframe to search for the tlo injury codes in
        :param injury_codes: The injury codes to search for in the data frame
        :return: the df index of who has the injuries and how many injuries in the search were found.
        """
        assert isinstance(persons_injury_properties, pd.DataFrame)
        assert isinstance(injury_codes, list)
        injury_counts = persons_injury_properties.isin(injury_codes).sum(axis=1)
        people_with_given_injuries = injury_counts[injury_counts > 0]
        return people_with_given_injuries.index, people_with_given_injuries.sum()

    def rti_treated_injuries(self, person_id, tloinjcodes):
        """
        A function that takes a person with treated injuries and removes the injury code from the properties rt_injury_1
        to rt_injury_8

        The properties that this function alters are rt_injury_1 through rt_injury_8 and the symptoms properties

        :param person_id: The person who needs an injury code removed
        :param tloinjcodes: the injury code(s) to be removed
        :return: n/a
        """
        df = self.sim.population.props
        # Isolate the relevant injury information
        permanent_injuries = ['P133', 'P133a', 'P133b', 'P133c', 'P133d', 'P134', 'P134a', 'P134b', 'P135', 'P673',
                              'P673a', 'P673b', 'P674', 'P674a', 'P674b', 'P675', 'P675a', 'P675b', 'P676', 'P782a',
                              'P782b', 'P782c', 'P783', 'P882', 'P883', 'P884']
        person_injuries = df.loc[[person_id], RTI.INJURY_COLUMNS]
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
                # Reset symptoms so that after being treated for an injury the person won't interact with the
                # healthsystem again.
                if df.loc[person_id, 'sy_injury'] != 0:
                    self.sim.modules['SymptomManager'].change_symptom(
                        person_id=person_id,
                        disease_module=self.sim.modules['RTI'],
                        add_or_remove='-',
                        symptom_string='injury',
                    )
                if df.loc[person_id, 'sy_severe_trauma'] != 0:
                    self.sim.modules['SymptomManager'].change_symptom(
                        person_id=person_id,
                        disease_module=self.sim.modules['RTI'],
                        add_or_remove='-',
                        symptom_string='severe_trauma',
                    )

    def on_birth(self, mother_id, child_id):
        """
        When a person is born this function sets up the default properties for the road traffic injuries module
        :param mother_id: The mother
        :param child_id: The newborn
        :return: n/a
        """
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
        df.at[child_id, 'rt_in_shock'] = False
        df.at[child_id, 'rt_death_from_shock'] = False
        df.at[child_id, 'rt_injuries_to_cast'] = []
        df.at[child_id, 'rt_injuries_for_minor_surgery'] = []
        df.at[child_id, 'rt_injuries_for_major_surgery'] = []
        df.at[child_id, 'rt_injuries_to_heal_with_time'] = []
        df.at[child_id, 'rt_injuries_for_open_fracture_treatment'] = []
        df.at[child_id, 'rt_polytrauma'] = False
        df.at[child_id, 'rt_ISS_score'] = 0
        df.at[child_id, 'rt_imm_death'] = False
        df.at[child_id, 'rt_perm_disability'] = False
        df.at[child_id, 'rt_med_int'] = False  # default: no one has a had medical intervention
        df.at[child_id, 'rt_in_icu_or_hdu'] = False
        df.at[child_id, 'rt_date_to_remove_daly'] = [pd.NaT] * 8
        df.at[child_id, 'rt_diagnosed'] = False
        df.at[child_id, 'rt_recovery_no_med'] = False  # default: no recovery without medical intervention
        df.at[child_id, 'rt_post_med_death'] = False  # default: no death after medical intervention
        df.at[child_id, 'rt_no_med_death'] = False
        df.at[child_id, 'rt_unavailable_med_death'] = False
        df.at[child_id, 'rt_disability'] = 0  # default: no disability due to RTI
        df.at[child_id, 'rt_date_inj'] = pd.NaT
        df.at[child_id, 'rt_MAIS_military_score'] = 0
        df.at[child_id, 'rt_date_death_no_med'] = pd.NaT
        df.at[child_id, 'rt_debugging_DALY_wt'] = 0

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

    def rti_assign_injuries(self, number):
        """
        A function that can be called specifying the number of people affected by RTI injuries
         and provides outputs for the number of injuries each person experiences from a RTI event, the location of the
         injury, the TLO injury categories and the severity of the injuries. The severity of the injuries will then be
         used to calculate the injury severity score (ISS), which will then inform mortality and disability from road
         traffic injuries with treatment and the military abreviated injury score (MAIS) which will be used to predict
         mortality without medical intervention.


        :param number: The number of people who need injuries assigned to them
        :return: injurydescription - a dataframe for the injury/injuries summarised in the TLO injury code form along
                                     with data on their ISS score, used for calculating mortality and whether they have
                                     polytrauma or not.
        Todo: see if we can include the following factors for injury severity (taken from a preprint sent to me after
            MOH meeting):
            - The setting of the patient (rural/urban) as rural location was a predictor for severe injury AOR 2.41
            (1.49-3.90)
            - Seatbelt use (none compared to using AOR 4.49 (1.47-13.76))
            - Role of person in crash (Different risk for different type, see paper)
            - Alcohol use (AOR 1.74 (1.11-2.74) compared to none)
        """
        p = self.parameters

        # Parameters used to assign injuries
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

        # Import the distribution of injured body regions from the VIBES study
        number_of_injured_body_regions_distribution = p['number_of_injured_body_regions_distribution']
        # Create empty lists to store information on the person's injuries
        # predicted injury
        predinj = []
        # predicted injury location
        predinjlocs = []
        # predicted injury severity
        predinjsev = []
        # predicted injury category
        predinjcat = []
        # predicted injury ISS score
        predinjiss = []
        # whether the prediction injuries are classed as polytrauma
        predpolytrauma = []
        # whether this predicted injury requires a greater level of detail that can't be determined by location
        # category and severity alone
        # Create empty lists which will be used to combine the injury location, category, severity and detail
        # information
        injlocstring = []
        injcatstring = []
        injaisstring = []
        # create empty lists to store the qualitative description of injury severity and the number of injuries
        # each person has
        severity_category = []
        number_of_injuries = []
        # define all injuries that can be assigned in a dictionary, information will be stored in the following order:
        # location, the subdict containing the probability of injury occuring, the ais injury location, category and
        # ais score
        # creat shorthand variable names for spinal chord injuries
        prob_673a = self.spine_prob_spinal_cord_lesion * p['daly_dist_code_673'][0] * (
            self.spine_prob_spinal_cord_lesion_neck_level * self.spine_prob_spinal_cord_lesion_neck_level_AIS3 +
            self.spine_prob_spinal_cord_lesion_below_neck_level *
            self.spine_prob_spinal_cord_lesion_below_neck_level_AIS3
        )
        prob_673b = self.spine_prob_spinal_cord_lesion * p['daly_dist_code_673'][1] * (
            self.spine_prob_spinal_cord_lesion_neck_level * self.spine_prob_spinal_cord_lesion_neck_level_AIS3 +
            self.spine_prob_spinal_cord_lesion_below_neck_level *
            self.spine_prob_spinal_cord_lesion_below_neck_level_AIS3
        )
        prob_674a = self.spine_prob_spinal_cord_lesion * p['daly_dist_codes_674_675'][0] * (
            self.spine_prob_spinal_cord_lesion_neck_level * self.spine_prob_spinal_cord_lesion_neck_level_AIS4 +
            self.spine_prob_spinal_cord_lesion_below_neck_level *
            self.spine_prob_spinal_cord_lesion_below_neck_level_AIS4
        )
        prob_674b = self.spine_prob_spinal_cord_lesion * p['daly_dist_codes_674_675'][1] * (
            self.spine_prob_spinal_cord_lesion_neck_level * self.spine_prob_spinal_cord_lesion_neck_level_AIS4 +
            self.spine_prob_spinal_cord_lesion_below_neck_level *
            self.spine_prob_spinal_cord_lesion_below_neck_level_AIS4
        )
        prob_675a = self.spine_prob_spinal_cord_lesion * p['daly_dist_codes_674_675'][0] * (
            self.spine_prob_spinal_cord_lesion_neck_level * self.spine_prob_spinal_cord_lesion_neck_level_AIS5 +
            self.spine_prob_spinal_cord_lesion_below_neck_level *
            self.spine_prob_spinal_cord_lesion_below_neck_level_AIS5
        )
        prob_675b = self.spine_prob_spinal_cord_lesion * p['daly_dist_codes_674_675'][1] * (
            self.spine_prob_spinal_cord_lesion_neck_level * self.spine_prob_spinal_cord_lesion_neck_level_AIS5 +
            self.spine_prob_spinal_cord_lesion_below_neck_level *
            self.spine_prob_spinal_cord_lesion_below_neck_level_AIS5
        )
        prob_676 = \
            self.spine_prob_spinal_cord_lesion * self.spine_prob_spinal_cord_lesion_neck_level * \
            self.spine_prob_spinal_cord_lesion_neck_level_AIS6

        injuries = {
            # injuries to the head
            'head': {'112': [self.head_prob_fracture * self.head_prob_fracture_unspecified, 1, 1, 2],
                     '113': [self.head_prob_fracture * self.head_prob_fracture_basilar, 1, 1, 3],
                     '133a': [self.head_prob_TBI * self.head_prob_TBI_AIS3 * p['daly_dist_code_133'][0], 1, 3, 3],
                     '133b': [self.head_prob_TBI * self.head_prob_TBI_AIS3 * p['daly_dist_code_133'][1], 1, 3, 3],
                     '133c': [self.head_prob_TBI * self.head_prob_TBI_AIS3 * p['daly_dist_code_133'][2], 1, 3, 3],
                     '133d': [self.head_prob_TBI * self.head_prob_TBI_AIS3 * p['daly_dist_code_133'][3], 1, 3, 3],
                     '134a': [self.head_prob_TBI * self.head_prob_TBI_AIS4 * p['daly_dist_code_134'][0], 1, 3, 4],
                     '134b': [self.head_prob_TBI * self.head_prob_TBI_AIS4 * p['daly_dist_code_134'][1], 1, 3, 4],
                     '135': [self.head_prob_TBI * self.head_prob_TBI_AIS5, 1, 3, 5],
                     '1101': [self.head_prob_skin_wound * self.head_prob_skin_wound_open, 1, 10, 1],
                     '1114': [self.head_prob_skin_wound * self.head_prob_skin_wound_burn, 1, 11, 4]},
            # injuries to the face
            'face': {'211': [self.face_prob_fracture * self.face_prob_fracture_AIS1, 2, 1, 1],
                     '212': [self.face_prob_fracture * self.face_prob_fracture_AIS2, 2, 1, 2],
                     '241': [self.face_prob_soft_tissue_injury, 2, 4, 1],
                     '2101': [self.face_prob_skin_wound * self.face_prob_skin_wound_open, 2, 10, 1],
                     '2114': [self.face_prob_skin_wound * self.face_prob_skin_wound_burn, 2, 11, 4],
                     '291': [self.face_prob_eye_injury, 2, 9, 1]},
            # injuries to the neck
            'neck': {'3101': [self.neck_prob_skin_wound * self.neck_prob_skin_wound_open, 3, 10, 1],
                     '3113': [self.neck_prob_skin_wound * self.neck_prob_skin_wound_burn, 3, 11, 3],
                     '342': [self.neck_prob_soft_tissue_injury * self.neck_prob_soft_tissue_injury_AIS2, 3, 4, 2],
                     '343': [self.neck_prob_soft_tissue_injury * self.neck_prob_soft_tissue_injury_AIS3, 3, 4, 3],
                     '361': [self.neck_prob_internal_bleeding * self.neck_prob_internal_bleeding_AIS1, 3, 6, 1],
                     '363': [self.neck_prob_internal_bleeding * self.neck_prob_internal_bleeding_AIS3, 3, 6, 3],
                     '322': [self.neck_prob_dislocation * self.neck_prob_dislocation_AIS2, 3, 2, 2],
                     '323': [self.neck_prob_dislocation * self.neck_prob_dislocation_AIS3, 3, 2, 3],
                     },
            # injuries to the chest
            'chest': {'4101': [self.thorax_prob_skin_wound * self.thorax_prob_skin_wound_open, 4, 10, 1],
                      '4113': [self.thorax_prob_skin_wound * self.thorax_prob_skin_wound_burn, 4, 11, 3],
                      '461': [self.thorax_prob_internal_bleeding * self.thorax_prob_internal_bleeding_AIS1, 4, 6, 1],
                      '463': [self.thorax_prob_internal_bleeding * self.thorax_prob_internal_bleeding_AIS3, 4, 6, 3],
                      '453a': [self.thorax_prob_internal_organ_injury * p['daly_dist_code_453'][0], 4, 5, 3],
                      '453b': [self.thorax_prob_internal_organ_injury * p['daly_dist_code_453'][1], 4, 5, 3],
                      '412': [self.thorax_prob_fracture * self.thorax_prob_fracture_ribs, 4, 1, 2],
                      '414': [self.thorax_prob_fracture * self.thorax_prob_fracture_flail_chest, 4, 1, 4],
                      '441': [self.thorax_prob_soft_tissue_injury * self.thorax_prob_soft_tissue_injury_AIS1, 4, 4, 1],
                      '442': [self.thorax_prob_soft_tissue_injury * self.thorax_prob_soft_tissue_injury_AIS2, 4, 4, 2],
                      '443': [self.thorax_prob_soft_tissue_injury * self.thorax_prob_soft_tissue_injury_AIS3, 4, 4, 3],
                      },
            # injuries to the abdomen
            'abdomen': {'5101': [self.abdomen_prob_skin_wound * self.abdomen_prob_skin_wound_open, 5, 10, 1],
                        '5113': [self.abdomen_prob_skin_wound * self.abdomen_prob_skin_wound_burn, 5, 11, 3],
                        '552': [self.abdomen_prob_internal_organ_injury * self.abdomen_prob_internal_organ_injury_AIS2,
                                5, 5, 2],
                        '553': [self.abdomen_prob_internal_organ_injury * self.abdomen_prob_internal_organ_injury_AIS3,
                                5, 5, 3],
                        '554': [self.abdomen_prob_internal_organ_injury * self.abdomen_prob_internal_organ_injury_AIS4,
                                5, 5, 4]},
            # injuries to the spine
            'spine': {'612': [self.spine_prob_fracture, 6, 1, 2],
                      '673a': [prob_673a, 6, 7, 3],
                      '673b': [prob_673b, 6, 7, 3],
                      '674a': [prob_674a, 6, 7, 4],
                      '674b': [prob_674b, 6, 7, 4],
                      '675a': [prob_675a, 6, 7, 5],
                      '675b': [prob_675b, 6, 7, 5],
                      '676': [prob_676, 6, 7, 6]
                      },
            # injuries to the upper extremities
            'upper_ex': {'7101': [self.upper_ex_prob_skin_wound * self.upper_ex_prob_skin_wound_open, 7, 10, 1],
                         '7113': [self.upper_ex_prob_skin_wound * self.upper_ex_prob_skin_wound_burn, 7, 11, 3],
                         '712a': [self.upper_ex_prob_fracture * p['daly_dist_code_712'][0], 7, 1, 2],
                         '712b': [self.upper_ex_prob_fracture * p['daly_dist_code_712'][1], 7, 1, 2],
                         '712c': [self.upper_ex_prob_fracture * p['daly_dist_code_712'][2], 7, 1, 2],
                         '722': [self.upper_ex_prob_dislocation, 7, 2, 2],
                         '782a': [(self.upper_ex_prob_amputation * self.upper_ex_prob_amputation_AIS2 *
                                  p['daly_dist_code_782'][0]), 7, 8, 2],
                         '782b': [(self.upper_ex_prob_amputation * self.upper_ex_prob_amputation_AIS2 *
                                  p['daly_dist_code_782'][1]), 7, 8, 2],
                         '782c': [(self.upper_ex_prob_amputation * self.upper_ex_prob_amputation_AIS2 *
                                  p['daly_dist_code_782'][2]), 7, 8, 2],
                         '783': [self.upper_ex_prob_amputation * self.upper_ex_prob_amputation_AIS3, 7, 8, 3]
                         },
            # injuries to the lower extremities
            'lower_ex': {'8101': [self.lower_ex_prob_skin_wound * self.lower_ex_prob_skin_wound_open, 8, 10, 1],
                         '8113': [self.lower_ex_prob_skin_wound * self.lower_ex_prob_skin_wound_burn, 8, 11, 3],
                         # foot fracture, can be open or not, open is more severe
                         '811': [self.lower_ex_prob_fracture * self.lower_ex_prob_fracture_AIS1 *
                                 (1 - p['prob_foot_fracture_open']), 8, 1, 1],
                         '813do': [self.lower_ex_prob_fracture * self.lower_ex_prob_fracture_AIS1 *
                                   p['prob_foot_fracture_open'], 8, 1, 3],
                         # lower leg fracture can be open or not
                         '812':  [self.lower_ex_prob_fracture * self.lower_ex_prob_fracture_AIS2 *
                                  (1 - p['prob_patella_tibia_fibula_ankle_fracture_open']), 8, 1, 2],
                         '813eo': [self.lower_ex_prob_fracture * self.lower_ex_prob_fracture_AIS2 *
                                   p['prob_patella_tibia_fibula_ankle_fracture_open'], 8, 1, 3],
                         '813a': [self.lower_ex_prob_fracture * self.lower_ex_prob_fracture_AIS3 *
                                  p['daly_dist_code_813'][0], 8, 1, 3],
                         # pelvis fracture can be open or closed
                         '813b': [self.lower_ex_prob_fracture * self.lower_ex_prob_fracture_AIS3 *
                                  p['daly_dist_code_813'][1] * (1 - p['prob_pelvis_fracture_open']), 8, 1, 3],
                         '813bo': [self.lower_ex_prob_fracture * self.lower_ex_prob_fracture_AIS3 *
                                   p['daly_dist_code_813'][1] * p['prob_pelvis_fracture_open'], 8, 1, 3],
                         # femur fracture can be open or closed
                         '813c': [self.lower_ex_prob_fracture * self.lower_ex_prob_fracture_AIS3 *
                                  p['daly_dist_code_813'][2] * (1 - p['prob_femur_fracture_open']), 8, 1, 3],
                         '813co': [self.lower_ex_prob_fracture * self.lower_ex_prob_fracture_AIS3 *
                                   p['daly_dist_code_813'][2] * p['prob_femur_fracture_open'], 8, 1, 3],
                         '822a': [self.lower_ex_prob_dislocation * p['daly_dist_code_822'][0], 8, 2, 2],
                         '822b': [self.lower_ex_prob_dislocation * p['daly_dist_code_822'][1], 8, 2, 2],
                         '882': [self.lower_ex_prob_amputation * self.lower_ex_prob_amputation_AIS2, 8, 8, 2],
                         '883': [self.lower_ex_prob_amputation * self.lower_ex_prob_amputation_AIS3, 8, 8, 3],
                         '884': [self.lower_ex_prob_amputation * self.lower_ex_prob_amputation_AIS4, 8, 8, 4]
                         }
        }
        # ============================= Begin assigning injuries to people =====================================
        # Iterate over the total number of injured people
        for n in range(0, number):
            # Get the distribution of body regions which can be injured for each iteration.
            injlocdist = p['injury_location_distribution']
            # Convert the parameter to a numpy array
            injlocdist = [list(injuries.keys()), p['injury_location_distribution'][1]]
            # Generate a random number which will decide how many injuries the person will have,
            ninj = self.rng.choice(number_of_injured_body_regions_distribution[0],
                                   p=number_of_injured_body_regions_distribution[1])
            # store the number of injuries this person recieves
            number_of_injuries.append(ninj)
            # create an empty list which stores the injury chosen
            injuries_chosen = []
            # Create an empty vector which will store the injury locations (numerically coded using the
            # abbreviated injury scale coding system, where 1 corresponds to head, 2 to face, 3 to neck, 4 to
            # thorax, 5 to abdomen, 6 to spine, 7 to upper extremity and 8 to lower extremity
            allinjlocs = []
            # Create an empty vector to store the type of injury
            injcat = []
            # Create an empty vector which will store the severity of the injuries
            injais = []
            # generate the locations of the injuries for this person
            injurylocation = self.rng.choice(injlocdist[0], ninj, p=injlocdist[1], replace=False)
            # iterate over the chosen injury locations to determine the exact injuries that this person will have
            for injlocs in injurylocation:
                # find the probability of each injury
                prob_of_each_injury_at_location = [val[0] for val in injuries[injlocs].values()]
                # make sure there are no rounding errors
                prob_of_each_injury_at_location = np.divide(prob_of_each_injury_at_location,
                                                            sum(prob_of_each_injury_at_location))
                # chose an injury to occur at this location
                injury_chosen = self.rng.choice(list(injuries[injlocs].keys()), p=prob_of_each_injury_at_location)
                # store this persons chosen injury at this location
                injuries_chosen.append(injury_chosen)
                # Store this person's injury location
                allinjlocs.append(injuries[injlocs][injury_chosen][1])
                # store the injury category chosen
                injcat.append(injuries[injlocs][injury_chosen][2])
                # store the severity of the injury chosen
                injais.append(injuries[injlocs][injury_chosen][3])

            # Check that all the relevant injury information has been decided by checking there is a injury category,
            # AIS score, injury location for all injuries
            assert len(injcat) == ninj
            assert len(injais) == ninj
            assert len(allinjlocs) == ninj
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
            # (variable name arbitraty, but only used in the next few lines)
            z = df.nlargest(3, 'Severity max', 'first')
            # Find the 3 most severely injured body regions
            z = z.iloc[:3]
            # Need to determine whether the persons injuries qualify as polytrauma as such injuries have a different
            # prognosis, set default as False. Polytrauma is defined via the new Berlin definition, 'when two or more
            # injuries have an AIS severity score of 3 or higher'.
            # set polytrauma as False by default
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
            ISSscore = int(sum(z['sqrsev']))
            if ISSscore < 15:
                severity_category.append('mild')
            else:
                severity_category.append('severe')
            # Turn the vectors into a string to store as one entry in a dataframe
            predinj.append(injuries_chosen)
            predinjlocs.append(allinjlocs)
            predinjsev.append(injais)
            predinjcat.append(injcat)
            predinjiss.append(ISSscore)
            predpolytrauma.append(polytrauma)
        # create a new data frame
        injdf = pd.DataFrame()
        # store the predicted injury codes
        injdf['Injury codes'] = predinj
        # expand injdf['Injury codes'] into its own dataframe
        if len(predinj) > 0:
            # if injuries are assigned split the injury codes
            injdf = injdf['Injury codes'].apply(pd.Series)
            # rename each variable in injdf if people have actually been injured
            injdf = injdf.rename(columns=lambda x: 'rt_injury_' + str(x + 1))

        # store the predicted injury severity scores
        injdf['Injury AIS'] = predinjsev
        injdf['ISS'] = predinjiss
        # Store the predicted occurence of polytrauma
        injdf['Polytrauma'] = predpolytrauma
        # create empty list to store the Military AIS scores used to predict morality without medical care
        MAIS = []
        # iterate of the injur AIS scores and calculate the associated MAIS score
        if number > 0:
            for item in injdf['Injury AIS'].tolist():
                MAIS.append(int(max(item) + 1))
        # Store the predicted Military AIS scores
        injdf['MAIS_M'] = MAIS
        # store the number of injuries this person received
        injdf['ninj'] = number_of_injuries
        # Fill dataframe entries where a person has not had an injury assigned with 'none'
        injdf = injdf.fillna("none")
        # Get injury information in an easily interpreted form to be logged.
        # create a list of the predicted injury locations
        flattened_injury_locations = [str(item) for sublist in predinjlocs for item in sublist]
        # create a list of the predicted injury categories
        flattened_injury_category = [str(item) for sublist in predinjcat for item in sublist]
        # create a list of the predicted injury severity scores
        flattened_injury_ais = [str(item) for sublist in predinjsev for item in sublist]

        # ============================ Injury category incidence ======================================================
        df = self.sim.population.props
        # log the incidence of each injury category
        n_alive = len(df.is_alive)
        amputationcounts = sum(1 for i in flattened_injury_category if i == '8')
        burncounts = sum(1 for i in flattened_injury_category if i == '11')
        fraccounts = sum(1 for i in flattened_injury_category if i == '1')
        tbicounts = sum(1 for i in flattened_injury_category if i == '3')
        minorinjurycounts = sum(1 for i in flattened_injury_category if i == '10')
        spinalcordinjurycounts = sum(1 for i in flattened_injury_category if i == '7')
        other_counts = sum(1 for i in flattened_injury_category if i in ['2', '4', '5', '6', '9'])
        inc_amputations = amputationcounts / ((n_alive - amputationcounts) * 1 / 12) * 100000
        inc_burns = burncounts / ((n_alive - burncounts) * 1 / 12) * 100000
        inc_fractures = fraccounts / ((n_alive - fraccounts) * 1 / 12) * 100000
        inc_tbi = tbicounts / ((n_alive - tbicounts) * 1 / 12) * 100000
        inc_sci = spinalcordinjurycounts / ((n_alive - spinalcordinjurycounts) * 1 / 12) * 100000
        inc_minor = minorinjurycounts / ((n_alive - minorinjurycounts) * 1 / 12) * 100000
        inc_other = other_counts / ((n_alive - other_counts) * 1 / 12) * 100000
        tot_inc_all_inj = inc_amputations + inc_burns + inc_fractures + inc_tbi + inc_sci + inc_minor + inc_other
        if number > 0:
            number_of_injuries = injdf['ninj'].tolist()
        else:
            number_of_injuries = 0
        dict_to_output = {'inc_amputations': inc_amputations,
                          'inc_burns': inc_burns,
                          'inc_fractures': inc_fractures,
                          'inc_tbi': inc_tbi,
                          'inc_sci': inc_sci,
                          'inc_minor': inc_minor,
                          'inc_other': inc_other,
                          'tot_inc_injuries': tot_inc_all_inj,
                          'number_of_injuries': number_of_injuries}

        logger.info(key='Inj_category_incidence',
                    data=dict_to_output,
                    description='Incidence of each injury grouped as per the GBD definition')
        # Log injury information
        injury_info = {'Number_of_injuries': number_of_injuries,
                       'Location_of_injuries': flattened_injury_locations,
                       'Injury_category': flattened_injury_category,
                       'Per_injury_severity': flattened_injury_ais,
                       'Per_person_injury_severity': predinjiss,
                       'Per_person_MAIS_score': MAIS,
                       'Per_person_severity_category': severity_category
                       }
        logger.info(key='Injury_information',
                    data=injury_info,
                    description='Relevant information on the injuries from road traffic accidents when they are '
                                'assigned')
        # log the fraction of lower extremity fractions that are open
        flattened_injuries = [str(item) for sublist in predinj for item in sublist]
        lx_frac_codes = ['811', '813do', '812', '813eo', '813', '813a', '813b', '813bo', '813c', '813co']
        lx_open_frac_codes = ['813do', '813eo', '813bo', '813co']
        n_lx_fracs = len([inj for inj in flattened_injuries if inj in lx_frac_codes])
        n_open_lx_fracs = len([inj for inj in flattened_injuries if inj in lx_open_frac_codes])
        if n_lx_fracs > 0:
            proportion_lx_fracture_open = n_open_lx_fracs / n_lx_fracs
        else:
            proportion_lx_fracture_open = 'no_lx_fractures'
        injury_info = {'Proportion_lx_fracture_open': proportion_lx_fracture_open}
        logger.info(key='Open_fracture_information',
                    data=injury_info,
                    description='The proportion of fractures that are open in specific body regions')
        # Finally return the injury description information
        return injdf


# ---------------------------------------------------------------------------------------------------------
#   DISEASE MODULE EVENTS
#
#   These are the events which drive the simulation of the disease. It may be a regular event that updates
#   the status of all the population of subsections of it at one time. There may also be a set of events
#   that represent disease events for particular persons.
# ---------------------------------------------------------------------------------------------------------

class RTIPollingEvent(RegularEvent, PopulationScopeEventMixin):
    """The regular RTI event which handles all the initial RTI related changes to the dataframe. It can be thought of
     as the actual road traffic accident occurring. Specifically the event decides who is involved in a road traffic
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

    6) rt_MAIS_military_score - The metric used to calculate the probability of mortality without medical intervention

    7) rt_disability - after injuries are assigned to a person, RTI_event calls rti_assign_daly_weights to match the
                       person's injury codes in rt_injury_1 through 8 to their corresponding DALY weights

    8) rt_polytrauma - If the person's injuries fit the definition for polytrauma we keep track of this here and use it
                        to calculate the probability for mortality later on.
    9) rt_date_death_no_med - the projected date to determine mortality for those who haven't sought medical care

    10) rt_inj_severity - The qualitative description of the severity of this person's injuries

    11) the symptoms this person has
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
        self.rr_injrti_age04 = p['rr_injrti_age04']
        self.rr_injrti_age59 = p['rr_injrti_age59']
        self.rr_injrti_age1017 = p['rr_injrti_age1017']
        self.rr_injrti_age1829 = p['rr_injrti_age1829']
        self.rr_injrti_age3039 = p['rr_injrti_age3039']
        self.rr_injrti_age4049 = p['rr_injrti_age4049']
        self.rr_injrti_age5059 = p['rr_injrti_age5059']
        self.rr_injrti_age6069 = p['rr_injrti_age6069']
        self.rr_injrti_age7079 = p['rr_injrti_age7079']
        self.rr_injrti_male = p['rr_injrti_male']
        self.rr_injrti_excessalcohol = p['rr_injrti_excessalcohol']
        self.imm_death_proportion_rti = p['imm_death_proportion_rti']
        self.rt_emergency_care_ISS_score_cut_off = p['rt_emergency_care_ISS_score_cut_off']

    def apply(self, population):
        """Apply this event to the population.

        :param population: the current population
        """
        df = population.props
        now = self.sim.date
        # Reset injury properties after death, get an index of people who have died due to RTI, all causes
        diedfromrtiidx = df.index[df.rt_imm_death | df.rt_post_med_death | df.rt_no_med_death | df.rt_death_from_shock |
                                  df.rt_unavailable_med_death]
        df.loc[diedfromrtiidx, "rt_imm_death"] = False
        df.loc[diedfromrtiidx, "rt_post_med_death"] = False
        df.loc[diedfromrtiidx, "rt_no_med_death"] = False
        df.loc[diedfromrtiidx, "rt_unavailable_med_death"] = False
        df.loc[diedfromrtiidx, "rt_disability"] = 0
        df.loc[diedfromrtiidx, "rt_med_int"] = False
        df.loc[diedfromrtiidx, 'rt_in_icu_or_hdu'] = False
        for index, row in df.loc[diedfromrtiidx].iterrows():
            df.at[index, 'rt_date_to_remove_daly'] = [pd.NaT] * 8
            df.at[index, 'rt_injuries_to_cast'] = []
            df.at[index, 'rt_injuries_for_minor_surgery'] = []
            df.at[index, 'rt_injuries_for_major_surgery'] = []
            df.at[index, 'rt_injuries_to_heal_with_time'] = []
            df.at[index, 'rt_injuries_for_open_fracture_treatment'] = []
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
        df.loc[diedfromrtiidx, 'rt_debugging_DALY_wt'] = 0
        # reset whether they have been selected for an injury this month
        df['rt_road_traffic_inc'] = False

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
                         Predictor('age_years').when('.between(0,4)', self.rr_injrti_age04),
                         Predictor('age_years').when('.between(5,9)', self.rr_injrti_age59),
                         Predictor('age_years').when('.between(10,17)', self.rr_injrti_age1017),
                         Predictor('age_years').when('.between(18,29)', self.rr_injrti_age1829),
                         Predictor('age_years').when('.between(30,39)', self.rr_injrti_age3039),
                         Predictor('age_years').when('.between(40,49)', self.rr_injrti_age4049),
                         Predictor('age_years').when('.between(50,59)', self.rr_injrti_age5059),
                         Predictor('age_years').when('.between(60,69)', self.rr_injrti_age6069),
                         Predictor('age_years').when('.between(70,79)', self.rr_injrti_age7079),
                         Predictor('li_ex_alc').when(True, self.rr_injrti_excessalcohol)
                         )
        pred = eq.predict(df.loc[rt_current_non_ind])
        random_draw_in_rti = self.module.rng.random_sample(size=len(rt_current_non_ind))
        selected_for_rti = rt_current_non_ind[pred > random_draw_in_rti]
        # Update to say they have been involved in a rti
        df.loc[selected_for_rti, 'rt_road_traffic_inc'] = True
        # Set the date that people were injured to now
        df.loc[selected_for_rti, 'rt_date_inj'] = now
        # ========================= Take those involved in a RTI and assign some to death ==============================
        # This section accounts for pre-hospital mortality, where a person is so severy injured that they die before
        # being able to seek medical care
        selected_to_die = selected_for_rti[self.imm_death_proportion_rti >
                                           self.module.rng.random_sample(size=len(selected_for_rti))]
        # Keep track of who experience pre-hospital mortality with the property rt_imm_death
        df.loc[selected_to_die, 'rt_imm_death'] = True
        # For each person selected to experience pre-hospital mortality, schedule an InstantaneosDeath event
        for individual_id in selected_to_die:
            self.sim.modules['Demography'].do_death(individual_id=individual_id, cause="RTI_imm_death",
                                                    originating_module=self.module)
        # ============= Take those remaining people involved in a RTI and assign injuries to them ==================
        # Drop those who have died immediately
        selected_for_rti_inj_idx = selected_for_rti.drop(selected_to_die)
        # Check that none remain
        assert len(selected_for_rti_inj_idx.intersection(selected_to_die)) == 0
        # take a copy dataframe, used to get the index of the population affected by RTI
        selected_for_rti_inj = df.loc[selected_for_rti_inj_idx]
        # Again make sure that those who have injuries assigned to them are alive, involved in a crash and didn't die on
        # scene
        selected_for_rti_inj = selected_for_rti_inj.loc[df.is_alive & df.rt_road_traffic_inc & ~df.rt_imm_death]
        # To stop people who have died from causes outside of the RTI module progressing through the model, remove
        # any person with the condition 'cause_of_death' is not null
        died_elsewhere_index = selected_for_rti_inj[~ selected_for_rti_inj['cause_of_death'].isnull()].index
        # drop the died_elsewhere_index from selected_for_rti_inj
        selected_for_rti_inj.drop(died_elsewhere_index, inplace=True)
        # Create shorthand link to RTI module
        road_traffic_injuries = self.sim.modules['RTI']

        # if people have been chosen to be injured, assign the injuries using the assign injuries function
        description = road_traffic_injuries.rti_assign_injuries(len(selected_for_rti_inj))
        # replace the nan values with 'none', this is so that the injuries can be copied over from this temporarily used
        # pandas dataframe will fit in with the categories in the columns rt_injury_1 through rt_injury_8
        description = description.replace('nan', 'none')
        # set the index of the description dataframe, so that we can join it to the selected_for_rti_inj dataframe
        description = description.set_index(selected_for_rti_inj.index)
        # copy over values from the assign injury dataframe to self.sim.population.props

        df.loc[selected_for_rti_inj.index, 'rt_ISS_score'] = \
            description.loc[selected_for_rti_inj.index, 'ISS'].astype(int)
        df.loc[selected_for_rti_inj.index, 'rt_MAIS_military_score'] = \
            description.loc[selected_for_rti_inj.index, 'MAIS_M'].astype(int)
        # ======================== Apply the injuries to the population dataframe ======================================
        # Find the corresponding column names
        injury_columns = pd.Index(RTI.INJURY_COLUMNS)
        matching_columns = description.columns.intersection(injury_columns)
        for col in matching_columns:
            df.loc[selected_for_rti_inj.index, col] = description.loc[selected_for_rti_inj.index, col]
        # Run assert statements to make sure the model is behaving as it should
        # All those who are injured in a road traffic accident have this noted in the property 'rt_road_traffic_inc'
        assert sum(df.loc[selected_for_rti, 'rt_road_traffic_inc']) == len(selected_for_rti)
        # All those who are involved in a road traffic accident have these noted in the property 'rt_date_inj'
        assert len(df.loc[selected_for_rti, 'rt_date_inj'] != pd.NaT) == len(selected_for_rti)
        # All those who are injures and do not die immediately have an ISS score > 0
        assert len(df.loc[df.rt_road_traffic_inc & ~df.rt_imm_death, 'rt_ISS_score'] > 0) == \
               len(df.loc[df.rt_road_traffic_inc & ~df.rt_imm_death])
        # ========================== Determine who will experience shock from blood loss ==============================
        # todo: improve this section, currently using a blanket assumption that those with internal bleeding or open
        # fractures will have shock, this is a temporary fix.
        internal_bleeding_codes = ['361', '363', '461', '463', '813bo', '813co', '813do', '813eo']
        df = self.sim.population.props

        shock_index, _ = \
            road_traffic_injuries.rti_find_and_count_injuries(df.loc[df.rt_road_traffic_inc, RTI.INJURY_COLUMNS],
                                                              internal_bleeding_codes)
        df.loc[shock_index, 'rt_in_shock'] = True
        # ========================== Decide survival time without medical intervention ================================
        # todo: find better time for survival data without med int for ISS scores
        # Assign a date in the future for which when the simulation reaches that date, the person's mortality will be
        # checked if they haven't sought care
        df.loc[selected_for_rti_inj.index, 'rt_date_death_no_med'] = now + DateOffset(days=7)
        # ============================ Injury severity classification =================================================
        # Find those with mild injuries and update the rt_inj_severity property so they have a mild injury
        injured_this_month = df.loc[selected_for_rti_inj.index]
        mild_rti_idx = injured_this_month.index[injured_this_month.is_alive & injured_this_month['rt_ISS_score'] < 15]
        df.loc[mild_rti_idx, 'rt_inj_severity'] = 'mild'
        # Find those with severe injuries and update the rt_inj_severity property so they have a severe injury
        severe_rti_idx = injured_this_month.index[injured_this_month['rt_ISS_score'] >= 15]
        df.loc[severe_rti_idx, 'rt_inj_severity'] = 'severe'
        # check that everyone who has been assigned an injury this month has an associated injury severity
        assert sum(df.loc[df.rt_road_traffic_inc & ~df.rt_imm_death & (df.rt_date_inj == now), 'rt_inj_severity']
                   != 'none') == len(selected_for_rti_inj.index)
        # Find those with polytrauma and update the rt_polytrauma property so they have polytrauma
        polytrauma_idx = description.loc[description.Polytrauma].index
        df.loc[polytrauma_idx, 'rt_polytrauma'] = True
        # Assign daly weights for each person's injuries with the function rti_assign_daly_weights
        road_traffic_injuries.rti_assign_daly_weights(selected_for_rti_inj.index)

        # =============================== Health seeking behaviour set up =======================================
        # Set up health seeking behaviour. Two symptoms are used in the RTI module, the generic injury symptom and an
        # emergency symptom 'severe_trauma'.

        # The condition to be sent to the health care system: 1) They must be alive 2) They must have been involved in a
        # road traffic accident 3) they must have not died immediately in the accident 4) they must not have been to an
        # A and E department previously and been diagnosed

        # The symptom they are assigned depends injury severity, those with mild injuries will be assigned the generic
        # symptom, those with severe injuries will have the emergency injury symptom

        # Create the logical conditions for each symptom
        condition_to_be_sent_to_em = \
            df.is_alive & df.rt_road_traffic_inc & ~df.rt_diagnosed & ~df.rt_imm_death & (df.rt_date_inj == now) & \
            (df.rt_injury_1 != "none") & (df.rt_ISS_score >= self.rt_emergency_care_ISS_score_cut_off)
        condition_to_be_sent_to_begin_non_emergency = \
            df.is_alive & df.rt_road_traffic_inc & ~df.rt_diagnosed & ~df.rt_imm_death & (df.rt_date_inj == now) & \
            (df.rt_injury_1 != "none") & (df.rt_ISS_score < self.rt_emergency_care_ISS_score_cut_off)
        # check that all those who meet the conditions to try and seek healthcare have at least one injury
        assert sum(df.loc[condition_to_be_sent_to_em, 'rt_injury_1'] != "none") == \
               len(df.loc[condition_to_be_sent_to_em])
        assert sum(df.loc[condition_to_be_sent_to_begin_non_emergency, 'rt_injury_1'] != "none") == \
               len(df.loc[condition_to_be_sent_to_begin_non_emergency])
        # create indexes of people to be assigned each rti symptom
        em_idx = df.index[condition_to_be_sent_to_em]
        non_em_idx = df.index[condition_to_be_sent_to_begin_non_emergency]
        # Assign the symptoms
        self.sim.modules['SymptomManager'].change_symptom(
            person_id=em_idx.tolist(),
            disease_module=self.module,
            add_or_remove='+',
            symptom_string='severe_trauma',
        )
        self.sim.modules['SymptomManager'].change_symptom(
            person_id=non_em_idx.tolist(),
            disease_module=self.module,
            add_or_remove='+',
            symptom_string='injury',
        )


class RTI_Check_Death_No_Med(RegularEvent, PopulationScopeEventMixin):
    """
    A regular event which organises whether a person who has not received medical treatment should die as a result of
    their injuries. This even makes use of the maximum AIS-military score, a trauma scoring system developed for
    injuries in a military environment, assumed here to be an indicator of the probability of mortality without
    access to a medical system.

    The properties this function changes are:
    1) rt_no_med_death - the boolean property tracking who dies from road traffic injuries without medical intervention

    2) rt_date_death_no_med - resetting the date to check the person's mortality without medical intervention if
                              they survive
    3) rt_disability - if the person survives a non-fatal injury then this injury may heal and therefore the disability
                       burden is changed
    4) rt_debugging_DALY_wt - if the person survives a non-fatal injury then this injury may heal and therefore the
                              disability burden is changed, this property keeping track of the true disability burden
    5) rt_date_to_remove_daly - In the event of recovering from a non-fatal injury without medical intervention
                                a recovery date will scheduled

    If the person is sent here and they don't die, we need to correctly model the level of disability they experience
    from their untreated injuries, some injuries that are left untreated will have an associated daly weight for long
    term disability without treatment, others don't.

    # todo: consult with a doctor about the likelihood of survival without medical treatment

    Currently I am assuming that anyone with an injury severity score of 9 or higher will seek care and have an
    emergency symptom, that means that I have to decide what to do with the following injuries:

    Lacerations - [1101, 2101, 3101, 4101, 5101, 7101, 8101]
    What would a laceration do without stitching? Take longer to heal most likely
    Fractures - ['112', '211', '212, '412', '612', '712', '712a', '712b', '712c', '811', '812']

    Some fractures have an associated level of disability to them, others do not. So things like fractured radius/ulna
    have a code to swap, but others don't. Some are of the no treatment type, such as fractured skulls, fractured ribs
    or fractured vertebrae, so we can just add the same recovery time for these injuries. So '112', '412' and '612' will
    need to have recovery events checked and recovered.
    Dislocations will presumably be popped back into place, the injury will feasably be able to heal but most likely
    with more pain and probably with more time
    Amputations - ['782','782a', '782b', '782c', '882']
    Amputations will presumably trigger emergency health seeking behaviour so they shouldn't turn up here really
    soft tissue injuries - ['241', '342', '441', '442']
    Presumably soft tissue injuries that turn up here will heal over time but with more pain
    Internal organ injury - ['552']
    Injury to the gastrointestinal organs can result in complications later on, but
    Internal bleedings - ['361', '461']
    Surviving internal bleeding is concievably possible, these are comparitively minor bleeds


    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(days=1))
        assert isinstance(module, RTI)
        p = module.parameters
        # Load parameters used by this event
        self.prob_death_MAIS3 = p['prob_death_MAIS3']
        self.prob_death_MAIS4 = p['prob_death_MAIS4']
        self.prob_death_MAIS5 = p['prob_death_MAIS5']
        self.prob_death_MAIS6 = p['prob_death_MAIS6']
        self.daly_wt_radius_ulna_fracture_short_term_with_without_treatment = \
            p['daly_wt_radius_ulna_fracture_short_term_with_without_treatment']
        self.daly_wt_radius_ulna_fracture_long_term_without_treatment = \
            p['daly_wt_radius_ulna_fracture_long_term_without_treatment']
        self.daly_wt_foot_fracture_short_term_with_without_treatment = \
            p['daly_wt_foot_fracture_short_term_with_without_treatment']
        self.daly_wt_foot_fracture_long_term_without_treatment = \
            p['daly_wt_foot_fracture_long_term_without_treatment']
        self.daly_wt_hip_fracture_short_term_with_without_treatment = \
            p['daly_wt_hip_fracture_short_term_with_without_treatment']
        self.daly_wt_hip_fracture_long_term_without_treatment = \
            p['daly_wt_hip_fracture_long_term_without_treatment']
        self.daly_wt_pelvis_fracture_short_term = p['daly_wt_pelvis_fracture_short_term']
        self.daly_wt_pelvis_fracture_long_term = \
            p['daly_wt_pelvis_fracture_long_term']
        self.daly_wt_femur_fracture_short_term = p['daly_wt_femur_fracture_short_term']
        self.daly_wt_femur_fracture_long_term_without_treatment = \
            p['daly_wt_femur_fracture_long_term_without_treatment']

    def apply(self, population):
        df = population.props
        now = self.sim.date
        probabilities_of_death = {
            '1': 0,
            '2': 0,
            '3': self.prob_death_MAIS3,
            '4': self.prob_death_MAIS4,
            '5': self.prob_death_MAIS5,
            '6': self.prob_death_MAIS6
        }
        # check if anyone is due to have their mortality without medical intervention determined today
        if len(df.loc[df['rt_date_death_no_med'] == now]) > 0:
            # Get an index of those scheduled to have their mortality checked
            due_to_die_today_without_med_int = df.loc[df['rt_date_death_no_med'] == now].index
            # iterate over those scheduled to die
            for person in due_to_die_today_without_med_int:
                # Create a random number to determine mortality
                rand_for_death = self.module.rng.random_sample(1)
                # create a variable to show if a person has died due to their untreated injuries
                # for each rt_MAIS_military_score, determine mortality
                prob_death = probabilities_of_death[str(df.loc[person, 'rt_MAIS_military_score'])]
                if rand_for_death < prob_death:
                    # If determined to die, schedule a death without med
                    df.loc[person, 'rt_no_med_death'] = True
                    self.sim.modules['Demography'].do_death(individual_id=person, cause="RTI_death_without_med",
                                                            originating_module=self.module)
                else:
                    # If the people do not die from their injuries despite not getting care, we have to decide when and
                    # to what degree their injuries will heal.
                    df.loc[[person], 'rt_recovery_no_med'] = True
                    # Reset the date to check if they die
                    df.loc[[person], 'rt_date_death_no_med'] = pd.NaT
                    swapping_codes = ['712c', '811', '813a', '813b', '813c']
                    # create a dictionary to reference changes to daly weights done here
                    swapping_daly_weights_lookup = {
                        '712c': (- self.daly_wt_radius_ulna_fracture_short_term_with_without_treatment +
                                 self.daly_wt_radius_ulna_fracture_long_term_without_treatment),
                        '811': (- self.daly_wt_foot_fracture_short_term_with_without_treatment +
                                self.daly_wt_foot_fracture_long_term_without_treatment),
                        '813a': (- self.daly_wt_hip_fracture_short_term_with_without_treatment +
                                 self.daly_wt_hip_fracture_long_term_without_treatment),
                        '813b': - self.daly_wt_pelvis_fracture_short_term + self.daly_wt_pelvis_fracture_long_term,
                        '813c': (- self.daly_wt_femur_fracture_short_term +
                                 self.daly_wt_femur_fracture_long_term_without_treatment),
                        'none': 0
                    }
                    road_traffic_injuries = self.sim.modules['RTI']
                    # If those who haven't sought health care have an injury for which we have a daly code
                    # associated with that injury long term without treatment, swap it
                    # Iterate over the person's injuries
                    injuries = df.loc[[person], RTI.INJURY_COLUMNS].values.tolist()
                    # Cannot iterate correctly over list like [[1,2,3]], so need to flatten
                    flattened_injuries = [item for sublist in injuries for item in sublist if item != 'none']
                    persons_injuries = df.loc[[person], RTI.INJURY_COLUMNS]
                    for code in flattened_injuries:
                        swapable_code = np.intersect1d(code, swapping_codes)
                        if len(swapable_code) > 0:
                            swapable_code = swapable_code[0]
                        else:
                            swapable_code = 'none'
                        # check that the person has the injury code
                        _, counts = road_traffic_injuries.rti_find_and_count_injuries(persons_injuries, [code])
                        assert counts > 0
                        df.loc[person, 'rt_debugging_DALY_wt'] += swapping_daly_weights_lookup[swapable_code]
                        if df.loc[person, 'rt_debugging_DALY_wt'] > 1:
                            df.loc[person, 'rt_disability'] = 1
                        else:
                            df.loc[person, 'rt_disability'] = df.loc[person, 'rt_debugging_DALY_wt']
                        # if the code is swappable, swap it
                        if df.loc[person, 'rt_disability'] < 0:
                            df.loc[person, 'rt_disability'] = 0
                        if df.loc[person, 'rt_disability'] > 1:
                            df.loc[person, 'rt_disability'] = 1
                        # If they don't have a swappable code, schedule the healing of the injury
                        # get the persons injuries
                        persons_injuries = df.loc[[person], RTI.INJURY_COLUMNS]
                        non_empty_injuries = persons_injuries[persons_injuries != "none"]
                        non_empty_injuries = non_empty_injuries.dropna(axis=1)
                        injury_columns = non_empty_injuries.columns
                        # create a dictionary to store recovery times in
                        no_treatment_recovery_times_in_days = {
                            '112': 49,
                            '211': 49,
                            '212': 49,
                            '412': 35,
                            '612': 63,
                            '712a': 70,
                            '712a': 70,
                            '712b': 70,
                            '712c': 70,
                            '811': 70,
                            '812': 70,
                            '322': 42,
                            '722': 84,
                            '822a': 60,
                            '822b': 180,
                            '241': 7,
                            '342': 42,
                            '441': 14,
                            '442': 14,
                            '552': 90,
                            '361': 7,
                            '461': 7,
                            '291': 7,
                            '1101': 7,
                            '2101': 7,
                            '3101': 7,
                            '4101': 7,
                            '5101': 7,
                            '6101': 7,
                            '7101': 7,
                            '8101': 7
                        }

                        columns = \
                            injury_columns.get_loc(road_traffic_injuries.rti_find_injury_column(person, [code])[0])
                        # assign a recovery date
                        # not all injuries have an assigned duration of recovery. These are more serious injuries that
                        # would normally be sent directly to the health system. In the instance that a serious injury
                        # occurs and no treatment is recieved but the person survives assume they will be disabled for
                        # the duration of the simulation
                        if code in no_treatment_recovery_times_in_days.keys():
                            df.loc[person, 'rt_date_to_remove_daly'][columns] = \
                                self.sim.date + DateOffset(days=no_treatment_recovery_times_in_days[code])
                        else:
                            df.loc[person, 'rt_date_to_remove_daly'][columns] = self.sim.end_date + DateOffset(days=1)
                        assert df.loc[person, 'rt_date_to_remove_daly'][columns] > self.sim.date


class RTI_Recovery_Event(RegularEvent, PopulationScopeEventMixin):
    """
    A regular event which checks the recovery date determined by each injury in columns rt_injury_1 through
    rt_injury_8, which is being stored in rt_date_to_remove_daly, a list property with 8 entries. This event
    checks the dates stored in rt_date_to_remove_daly property, when the date matches one of the entries,
    the daly weight is removed and the injury is fully healed.

    The properties changed in this functions is:

    1) rt_date_to_remove_daly - resetting the date to remove the daly weight for each injury once the date is
                                reached in the sim

    2) rt_inj_severity - resetting the person's injury severity once and injury is healed

    3) rt_injuries_to_heal_with_time - resetting the list of injuries that are due to heal over time once healed

    4) rt_injuries_for_minor_surgery - resetting the list of injuries that are treated with minor surgery once
                                       healed
    5) rt_injuries_for_major_surgery - resetting the list of injuries that are treated with major surgery once
                                       healed
    6) rt_injuries_for_open_fracture_treatment - resetting the list of injuries that are treated with open fracture
                                                 treatment once healed
    7) rt_injuries_to_cast - resetting the list of injuries that are treated with fracture cast treatment once healed
    """

    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(days=1))
        assert isinstance(module, RTI)

    def apply(self, population):
        road_traffic_injuries = self.module
        df = population.props
        now = self.sim.date
        # # Isolate the relevant population
        any_not_null = df.loc[df.is_alive, 'rt_date_to_remove_daly'].apply(lambda x: pd.notnull(x).any())
        relevant_population = any_not_null.index[any_not_null]
        # Isolate the relevant information
        recovery_dates = df.loc[relevant_population]['rt_date_to_remove_daly']
        default_recovery = [pd.NaT, pd.NaT, pd.NaT, pd.NaT, pd.NaT, pd.NaT, pd.NaT, pd.NaT]
        # Iterate over all the injured people who are having medical treatment
        for person in recovery_dates.index:
            # Iterate over all the dates in 'rt_date_to_remove_daly'
            for date in df.loc[person, 'rt_date_to_remove_daly']:
                # check that a recovery date hasn't been assigned to the past
                if not pd.isnull(date):
                    assert date >= self.sim.date, 'recovery date assigned to past'
                # check if the recovery date is today
                if date == now:
                    # find the index for the injury which the person has recovered from
                    dateindex = df.loc[person, 'rt_date_to_remove_daly'].index(date)
                    # find the injury code associated with the healed injury
                    code_to_remove = [df.loc[person, f'rt_injury_{dateindex + 1}']]
                    # Set the healed injury recovery data back to the default state
                    df.loc[person, 'rt_date_to_remove_daly'][dateindex] = pd.NaT
                    # Remove the daly weight associated with the healed injury code
                    person_injuries = df.loc[[person], RTI.INJURY_COLUMNS]
                    _, counts = RTI.rti_find_and_count_injuries(person_injuries, self.module.INJURY_CODES[1:])
                    if counts == 0:
                        pass
                    else:
                        road_traffic_injuries.rti_alter_daly_post_treatment(person, code_to_remove)
                    # Check whether all their injuries are healed so the injury properties can be reset
                    if df.loc[person, 'rt_date_to_remove_daly'] == default_recovery:
                        # remove the injury severity as person is uninjured
                        df.loc[person, 'rt_inj_severity'] = "none"
                        untreated_injuries = (
                            df.loc[person, 'rt_injuries_to_heal_with_time'] +
                            df.loc[person, 'rt_injuries_for_minor_surgery'] +
                            df.loc[person, 'rt_injuries_for_major_surgery'] +
                            df.loc[person, 'rt_injuries_for_open_fracture_treatment'] +
                            df.loc[person, 'rt_injuries_to_cast']
                        )
                        assert untreated_injuries == [], f"not every injury removed from dataframe when treated " \
                                                         f"{untreated_injuries}"
            # Check that the date to remove dalys is removed if the date to remove the daly is today
            assert now not in df.loc[person, 'rt_date_to_remove_daly']
            # finally ensure the reported disability burden is an appropriate value
            if df.loc[person, 'rt_disability'] < 0:
                df.loc[person, 'rt_disability'] = 0
            if df.loc[person, 'rt_disability'] > 1:
                df.loc[person, 'rt_disability'] = 1


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

    The properties changed in this event are:

    rt_injuries_for_major_surgery - the injuries that are determined to be treated by major surgery are stored in
                                    this list property
    rt_injuries_for_minor_surgery - the injuries that are determined to be treated by minor surgery are stored in
                                    this list property
    rt_injuries_to_cast - the injuries that are determined to be treated with fracture casts are stored in this list
                          property
    rt_injuries_for_open_fracture_treatment - the injuries that are determined to be treated with open fractre treatment
                                              are stored in this list property
    rt_injuries_to_heal_with_time - the injuries that are determined to heal with time are stored in this list property

    rt_date_to_remove_daly - recovery dates for the heal with time injuries are set here

    rt_date_death_no_med - the date to check mortality without medical intervention is removed as this person has
                           sought medical care
    rt_med_int - the bool property that shows whether a person has sought medical care or not
    """

    # TODO: include treatment or at least transfer between facilities, e.g. at KCH "Most patients transferred from
    #  either a health center, 2463 (47.2%), or district hospital, 1996 (38.3%)"

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        road_traffic_injuries = self.sim.modules['RTI']

        df = self.sim.population.props
        p = module.parameters
        person = df.loc[person_id]
        # Load the parameters used in this event
        self.prob_depressed_skull_fracture = p['prob_depressed_skull_fracture']  # proportion of depressed skull
        # fractures in https://doi.org/10.1016/j.wneu.2017.09.084
        self.prob_mild_burns = p['prob_mild_burns']  # proportion of burns accross SSA with TBSA < 10
        # https://doi.org/10.1016/j.burns.2015.04.006
        self.prob_TBI_require_craniotomy = p['prob_TBI_require_craniotomy']
        self.prob_exploratory_laparotomy = p['prob_exploratory_laparotomy']
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
        person_injuries = df.loc[[person_id], RTI.INJURY_COLUMNS]

        # todo: work out if the amputations need to be included as a swap or if they already exist

        # create a dictionary to store the probability of each possible treatment for applicable injuries, we are
        # assuming that any amputation treatment plan will just be a major surgery for now
        treatment_plans = {
            # Treatment plan options for skull fracture
            '112': [[self.prob_depressed_skull_fracture, 1 - self.prob_depressed_skull_fracture], ['major', 'HWT']],
            '113': [[1], ['HWT']],
            # Treatment plan for facial fractures
            '211': [[1], ['minor']],
            '212': [[1], ['minor']],
            # Treatment plan for rib fractures
            '412': [[1], ['HWT']],
            # Treatment plan for flail chest
            '414': [[1], ['major']],
            # Treatment plan options for foot fractures
            '811': [[p['prob_foot_frac_require_cast'], p['prob_foot_frac_require_maj_surg'],
                     p['prob_foot_frac_require_min_surg'], p['prob_foot_frac_require_amp']],
                    ['cast', 'major', 'minor', 'major']],
            # Treatment plan options for lower leg fractures
            '812': [[p['prob_tib_fib_frac_require_cast'], p['prob_tib_fib_frac_require_maj_surg'],
                     p['prob_tib_fib_frac_require_min_surg'], p['prob_tib_fib_frac_require_traction'],
                     p['prob_tib_fib_frac_require_amp']],
                    ['cast', 'major', 'minor', 'HWT', 'major']],
            # Treatment plan options for femur/hip fractures
            '813a': [[p['prob_femural_fracture_require_major_surgery'],
                      p['prob_femural_fracture_require_minor_surgery'], p['prob_femural_fracture_require_cast'],
                      p['prob_femural_fracture_require_traction'], p['prob_femural_fracture_require_amputation']],
                     ['major', 'minor', 'cast', 'HWT', 'major']],
            # Treatment plan options for femur/hip fractures
            '813c': [[p['prob_femural_fracture_require_major_surgery'],
                      p['prob_femural_fracture_require_minor_surgery'], p['prob_femural_fracture_require_cast'],
                      p['prob_femural_fracture_require_traction'], p['prob_femural_fracture_require_amputation']],
                     ['major', 'minor', 'cast', 'HWT', 'major']],
            # Treatment plan options for pelvis fractures
            '813b': [[p['prob_pelvis_fracture_traction'], p['prob_pelvis_frac_major_surgery'],
                      p['prob_pelvis_frac_minor_surgery'], p['prob_pelvis_frac_cast']],
                     ['HWT', 'major', 'minor', 'cast']],
            # Treatment plan options for open fractures
            '813bo': [[1], ['open']],
            '813co': [[1], ['open']],
            '813do': [[1], ['open']],
            '813eo': [[1], ['open']],
            # Treatment plan options for traumatic brain injuries
            '133a': [[self.prob_TBI_require_craniotomy, 1 - self.prob_TBI_require_craniotomy], ['major', 'HWT']],
            '133b': [[self.prob_TBI_require_craniotomy, 1 - self.prob_TBI_require_craniotomy], ['major', 'HWT']],
            '133c': [[self.prob_TBI_require_craniotomy, 1 - self.prob_TBI_require_craniotomy], ['major', 'HWT']],
            '133d': [[self.prob_TBI_require_craniotomy, 1 - self.prob_TBI_require_craniotomy], ['major', 'HWT']],
            '134a': [[self.prob_TBI_require_craniotomy, 1 - self.prob_TBI_require_craniotomy], ['major', 'HWT']],
            '134b': [[self.prob_TBI_require_craniotomy, 1 - self.prob_TBI_require_craniotomy], ['major', 'HWT']],
            '135': [[self.prob_TBI_require_craniotomy, 1 - self.prob_TBI_require_craniotomy], ['major', 'HWT']],
            # Treatment plan options for abdominal injuries
            '552': [[self.prob_exploratory_laparotomy, 1 - self.prob_exploratory_laparotomy], ['major', 'HWT']],
            '553': [[self.prob_exploratory_laparotomy, 1 - self.prob_exploratory_laparotomy], ['major', 'HWT']],
            '554': [[self.prob_exploratory_laparotomy, 1 - self.prob_exploratory_laparotomy], ['major', 'HWT']],
            # Treatment plan for vertebrae fracture
            '612': [[1], ['HWT']],
            # Treatment plan for dislocations
            '822a': [[p['prob_dis_hip_require_maj_surg'], p['prob_hip_dis_require_traction'],
                      p['prob_dis_hip_require_cast']], ['major', 'HWT', 'cast']],
            '322': [[self.prob_dislocation_requires_surgery, 1 - self.prob_dislocation_requires_surgery],
                    ['minor', 'HWT']],
            '323': [[self.prob_dislocation_requires_surgery, 1 - self.prob_dislocation_requires_surgery],
                    ['minor', 'HWT']],
            '722': [[self.prob_dislocation_requires_surgery, 1 - self.prob_dislocation_requires_surgery],
                    ['minor', 'HWT']],
            # Soft tissue injury in neck treatment plan
            '342': [[1], ['major']],
            '343': [[1], ['major']],
            # Treatment plan for surgical emphysema
            '442': [[1], ['HWT']],
            # Treatment plan for internal bleeding
            '361': [[1], ['major']],
            '363': [[1], ['major']],
            '461': [[1], ['HWT']],
            # Treatment plan for amputations
            '782a': [[1], ['major']],
            '782b': [[1], ['major']],
            '782c': [[1], ['major']],
            '783': [[1], ['major']],
            '882': [[1], ['major']],
            '883': [[1], ['major']],
            '884': [[1], ['major']],
            # Treatment plan for eye injury
            '291': [[1], ['minor']],
            # Treatment plan for soft tissue injury
            '241': [[1], ['minor']],
            # treatment plan for simple fractures and dislocations
            '712a': [[1], ['cast']],
            '712b': [[1], ['cast']],
            '712c': [[1], ['cast']],
            '822b': [[1], ['cast']]

        }
        # store number of open fractures for use later
        self.open_fractures = 0
        # check if they have an injury for which we need to find the treatment plan for

        for code in treatment_plans.keys():
            _, counts = road_traffic_injuries.rti_find_and_count_injuries(person_injuries, [code])
            if counts > 0:
                treatment_choice = self.module.rng.choice(treatment_plans[code][1], p=treatment_plans[code][0])
                if treatment_choice == 'cast':
                    df.loc[person_id, 'rt_injuries_to_cast'].append(code)
                if treatment_choice == 'major':
                    df.loc[person_id, 'rt_injuries_for_major_surgery'].append(code)
                    self.major_surgery_counts += 1
                if treatment_choice == 'minor':
                    df.loc[person_id, 'rt_injuries_for_minor_surgery'].append(code)
                    self.minor_surgery_counts += 1
                if treatment_choice == 'HWT':
                    df.loc[person_id, 'rt_injuries_to_heal_with_time'].append(code)
                if treatment_choice == 'open':
                    self.open_fractures += 1
                    df.loc[person_id, 'rt_injuries_for_open_fracture_treatment'].append(code)

        # -------------------------------- Spinal cord injury requirements --------------------------------------------
        # Check whether they have a spinal cord injury, if we allow spinal cord surgery capacilities here, ask for a
        # surgery, otherwise make the injury permanent
        codes = ['673', '673a', '673b', '674', '674a', '674b', '675', '675a', '675b', '676']
        # Ask if this person has a spinal cord injury
        _, counts = road_traffic_injuries.rti_find_and_count_injuries(person_injuries, codes)
        if (counts > 0) & ('include_spine_surgery' in self.allowed_interventions):
            # if this person has a spinal cord injury and we allow surgeries, determine their exact injury
            actual_injury = np.intersect1d(codes, person_injuries.values)
            # update the number of major surgeries
            self.major_surgery_counts += 1
            # add the injury to the injuries to be treated by major surgery
            df.loc[person_id, 'rt_injuries_for_major_surgery'].append(actual_injury[0])
        elif counts > 0:
            # if no surgery assume that the person will be permanently disabled
            df.at[person_id, 'rt_perm_disability'] = True
            # Find the column and code where the permanent injury is stored
            column, code = road_traffic_injuries.rti_find_injury_column(person_id=person_id, codes=codes)
            # make the injury permanent by adding a 'P' before the code
            df.loc[person_id, column] = "P" + code
            code = df.loc[person_id, column]
            # find which property the injury is stored in
            columns, codes = road_traffic_injuries.rti_find_all_columns_of_treated_injuries(person_id, [code])
            for col in columns:
                # schedule the recovery date for the permanent injury for beyond the end of the simulation (making
                # it permanent)
                df.loc[person_id, 'rt_date_to_remove_daly'][int(col[-1]) - 1] = self.sim.end_date + \
                                                                                DateOffset(days=1)
                assert df.loc[person_id, 'rt_date_to_remove_daly'][int(col[-1]) - 1] > self.sim.date

        # --------------------------------- Soft tissue injury in thorax/ lung injury ----------------------------------
        # Check whether they have any soft tissue injuries in the thorax, if so schedule surgery if required else make
        # the injuries heal over time without further medical care
        codes = ['441', '443', '453', '453a', '453b']
        # check if they have chest traume
        _, counts = road_traffic_injuries.rti_find_and_count_injuries(person_injuries, codes)
        if (counts > 0) & ('include_thoroscopy' in self.allowed_interventions):
            # work out the exact injury they have
            actual_injury = np.intersect1d(codes, person_injuries.values)
            # update the number of major surgeries required
            self.major_surgery_counts += 1
            # add the injury to the injuries to be treated with major surgery so they aren't treated elsewhere
            df.loc[person_id, 'rt_injuries_for_major_surgery'].append(actual_injury[0])

        # -------------------------------- Internal bleeding -----------------------------------------------------------
        # check if they have internal bleeding in the thorax, and if the surgery is available, schedule a major surgery
        codes = ['463']
        _, counts = road_traffic_injuries.rti_find_and_count_injuries(person_injuries, codes)
        if (counts > 0) & ('include_thoroscopy' in self.allowed_interventions):
            # update the number of major surgeries needed
            self.major_surgery_counts += 1
            # add the injury to the injuries to be treated with major surgery.
            df.loc[person_id, 'rt_injuries_for_major_surgery'].append('463')
        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'RTI_MedicalIntervention'  # This must begin with the module name
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = the_accepted_facility_level
        self.ALERT_OTHER_DISEASES = []

        # ================ Determine how long the person will be in hospital based on their ISS score ==================
        self.inpatient_days = road_traffic_injuries.rti_determine_LOS(person_id)
        # If the patient needs skeletal traction for their injuries they need to stay at minimum 6 weeks,
        # average length of stay for those with femur skeletal traction found from Kramer et al. 2016:
        # https://doi.org/10.1007/s00264-015-3081-3
        # todo: put in complications from femur fractures
        self.femur_fracture_skeletal_traction_mean_los = p['femur_fracture_skeletal_traction_mean_los']
        self.other_skeletal_traction_los = p['other_skeletal_traction_los']
        min_los_for_traction = {
            '813c': self.femur_fracture_skeletal_traction_mean_los,
            '813b': self.other_skeletal_traction_los,
            '813a': self.other_skeletal_traction_los,
            '812': self.other_skeletal_traction_los,
        }
        traction_injuries = [injury for injury in df.loc[person_id, 'rt_injuries_to_heal_with_time'] if injury in
                             min_los_for_traction.keys()]
        if len(traction_injuries) > 0:
            if self.inpatient_days < min_los_for_traction[traction_injuries[0]]:
                self.inpatient_days = min_los_for_traction[traction_injuries[0]]

        # Specify the type of bed days needed? not sure if necessary
        self.BEDDAYS_FOOTPRINT.update({'general_bed': self.inpatient_days})
        # update the expected appointment foortprint
        if self.inpatient_days > 0:
            self.EXPECTED_APPT_FOOTPRINT.update({'InpatientDays': self.inpatient_days})
        # ================ Determine whether the person will require ICU days =========================================
        # Percentage of RTIs that required ICU stay 2.7% at KCH : https://doi.org/10.1007/s00268-020-05853-z
        # Percentage of RTIs that require HDU stay 3.3% at KCH
        # Assume for now that ICU admission is entirely dependent on injury severity so that only the 2.7% of most
        # severe injuries get admitted to ICU and the following 3.3% of most severe injuries get admitted to HDU
        # NOTE: LEAVING INPATIENT DAYS IN PLACE TEMPORARILY
        # Seems only one level of care above normal so adjust accordingly
        # self.icu_cut_off_iss_score = 38
        self.hdu_cut_off_iss_score = p['hdu_cut_off_iss_score']
        # Malawi ICU data: doi: 10.1177/0003134820950282
        # General length of stay from Malawi source, not specifically for injuries though
        # mean = 4.8, s.d. = 6, TBI admission mean = 8.4, s.d. = 6.4
        # mortality percentage = 51.2 overall, 50% for TBI admission and 49% for hemorrhage
        # determine the number of ICU days used to treat patient

        if df.loc[person_id, 'rt_ISS_score'] > self.hdu_cut_off_iss_score:
            mean_icu_days = p['mean_icu_days']
            sd_icu_days = p['sd_icu_days']
            mean_tbi_icu_days = p['mean_tbi_icu_days']
            sd_tbi_icu_days = p['sd_tbi_icu_days']
            codes = ['133', '133a', '133b', '133c', '133d' '134', '134a', '134b', '135']
            _, counts = road_traffic_injuries.rti_find_and_count_injuries(person_injuries, codes)
            if counts > 0:
                self.icu_days = int(self.module.rng.normal(mean_tbi_icu_days, sd_tbi_icu_days, 1))
            else:
                self.icu_days = int(self.module.rng.normal(mean_icu_days, sd_icu_days, 1))
            # if the number of ICU days is less than zero make it zero
            if self.icu_days < 0:
                self.icu_days = 0
            # update the property showing if a person is in ICU
            df.loc[person_id, 'rt_in_icu_or_hdu'] = True
            # update the bed days footprint
            self.BEDDAYS_FOOTPRINT.update({'high_dependency_bed': self.icu_days})
            # store the injury information of patients in ICU
            logger.info(key='ICU_patients',
                        data=person_injuries,
                        description='The injuries of ICU patients')
        # Check that each injury has only one treatment plan assigned to it
        treatment_plan = \
            person['rt_injuries_for_minor_surgery'] + person['rt_injuries_for_major_surgery'] + \
            person['rt_injuries_to_heal_with_time'] + person['rt_injuries_for_open_fracture_treatment'] + \
            person['rt_injuries_to_cast']
        assert len(treatment_plan) == len(set(treatment_plan))

        # Other test admission protocol. Basing ICU admission of whether they have a TBI
        # 17.3% of head injury patients in KCH were admitted to ICU/HDU (7.9 and 9.4% respectively)

        # Injury characteristics of patients admitted to ICU in Tanzania:
        # 97.8% had lacerations
        # 32.4% had fractures
        # 21.5% had TBI
        # 13.1% had abdominal injuries
        # 2.9% had burns
        # 3.8% had 'other' injuries
        # https://doi.org/10.1186/1757-7241-19-61

    def apply(self, person_id, squeeze_factor):
        road_traffic_injuries = self.sim.modules['RTI']
        df = self.sim.population.props
        hs = self.sim.modules["HealthSystem"]

        if not df.at[person_id, 'is_alive']:
            self.EXPECTED_APPT_FOOTPRINT = hs.get_blank_appt_footprint()
            return
        # Remove the scheduled death without medical intervention
        df.loc[person_id, 'rt_date_death_no_med'] = pd.NaT
        # Isolate relevant injury information
        person = df.loc[person_id]
        person_injuries = df.loc[[person_id], RTI.INJURY_COLUMNS]
        non_empty_injuries = person_injuries[person_injuries != "none"]
        non_empty_injuries = non_empty_injuries.dropna(axis=1)
        injury_columns = non_empty_injuries.columns
        # Check that those who arrive here are alive and have been through the first generic appointment, and didn't
        # die due to rti
        assert person['rt_diagnosed'], 'person sent here has not been through A and E'
        # Check that those who arrive here have at least one injury
        _, counts = RTI.rti_find_and_count_injuries(person_injuries,
                                                    self.module.PROPERTIES.get('rt_injury_1').categories[1:-1])
        assert counts > 0, 'This person has asked for medical treatment despite not being injured'
        # update the model's properties to reflect that this person has sought medical care
        df.at[person_id, 'rt_med_int'] = True
        # =============================== Make 'healed with time' injuries disappear ===================================
        # these are the injuries actually treated in this HSI
        heal_with_time_recovery_times_in_days = {
            # using estimated 6 weeks PLACEHOLDER FOR neck dislocations
            '322': 42,
            '323': 42,
            # using estimated 12 weeks placeholder for dislocated shoulders
            '722': 84,
            # using estimated 2 month placeholder for dislocated knees
            '822a': 60,
            # using estimated 7 weeks PLACEHOLDER FOR SKULL FRACTURE
            '112': 49,
            '113': 49,
            # using estimated 5 weeks PLACEHOLDER FOR rib FRACTURE
            '412': 35,
            # using estimated 9 weeks PLACEHOLDER FOR Vertebrae FRACTURE
            '612': 63,
            # using estimated 9 weeks PLACEHOLDER FOR skeletal traction for tibia/fib
            '812': 63,
            # using estimated 9 weeks PLACEHOLDER FOR skeletal traction for hip
            '813a': 63,
            # using estimated 9 weeks PLACEHOLDER FOR skeletal traction for pelvis
            '813b': 63,
            # using estimated 9 weeks PLACEHOLDER FOR skeletal traction for femur
            '813c': 63,
            # using estimated 3 month PLACEHOLDER FOR abdominal trauma
            '552': 90,
            '553': 90,
            '554': 90,
            # using 1 week placeholder for surgical emphysema
            '442': 7,
            # 2 week placeholder for chest wall bruising
            '461': 14

        }
        tbi = ['133', '133a', '133b', '133c', '133d', '134', '134a', '134b', '135']
        if len(df.at[person_id, 'rt_injuries_to_heal_with_time']) > 0:
            # check whether the heal with time injuries include dislocations, which may have been sent to surgery
            for code in person['rt_injuries_to_heal_with_time']:
                # temporarily dealing with TBI heal dates seporately
                if code in tbi:
                    pass
                else:
                    columns = injury_columns.get_loc(road_traffic_injuries.rti_find_injury_column(person_id, [code])[0])
                    df.loc[person_id, 'rt_date_to_remove_daly'][columns] = \
                        self.sim.date + DateOffset(days=heal_with_time_recovery_times_in_days[code])
                    assert df.loc[person_id, 'rt_date_to_remove_daly'][columns] > self.sim.date
            heal_with_time_codes = []
            # Check whether the heal with time injury is a skull fracture, which may have been sent to surgery
            tbi = ['133', '133a', '133b', '133c', '133d', '134', '134a', '134b', '135']
            tbi_injury = [injury for injury in tbi if injury in person['rt_injuries_to_heal_with_time']]
            if len(tbi_injury) > 0:
                columns = injury_columns.get_loc(road_traffic_injuries.rti_find_injury_column(person_id, tbi_injury)[0])
                # ask if this injury will be permanent
                perm_injury = self.module.rng.random_sample(size=1)
                if perm_injury < self.prob_perm_disability_with_treatment_severe_TBI:
                    column, code = road_traffic_injuries.rti_find_injury_column(person_id=person_id, codes=tbi_injury)
                    df.loc[person_id, column] = "P" + code
                    heal_with_time_codes.append("P" + code)
                    df.loc[person_id, 'rt_date_to_remove_daly'][columns] = self.sim.end_date + DateOffset(days=1)
                    assert df.loc[person_id, 'rt_date_to_remove_daly'][columns] > self.sim.date
                else:
                    heal_with_time_codes.append(tbi_injury[0])
                    # using estimated 6 months PLACEHOLDER FOR TRAUMATIC BRAIN INJURY
                    df.loc[person_id, 'rt_date_to_remove_daly'][columns] = self.sim.date + DateOffset(months=6)
                    assert df.loc[person_id, 'rt_date_to_remove_daly'][columns] > self.sim.date
            # swap potentially swappable codes
            swapping_codes = RTI.SWAPPING_CODES[:]
            # remove codes that will be treated elsewhere
            for code in person['rt_injuries_for_minor_surgery']:
                if code in swapping_codes:
                    swapping_codes.remove(code)
            for code in person['rt_injuries_for_major_surgery']:
                if code in swapping_codes:
                    swapping_codes.remove(code)
            for code in person['rt_injuries_to_cast']:
                if code in swapping_codes:
                    swapping_codes.remove(code)
            for code in person['rt_injuries_for_open_fracture_treatment']:
                if code in swapping_codes:
                    swapping_codes.remove(code)
            # drop injuries potentially treated elsewhere
            codes_to_swap = [code for code in heal_with_time_codes if code in swapping_codes]
            if len(codes_to_swap) > 0:
                road_traffic_injuries.rti_swap_injury_daly_upon_treatment(person_id, codes_to_swap)
            # check every heal with time injury has a recovery date associated with it
            for code in person['rt_injuries_to_heal_with_time']:
                columns = injury_columns.get_loc(road_traffic_injuries.rti_find_injury_column(person_id, [code])
                                                 [0])
                assert not pd.isnull(df.loc[person_id, 'rt_date_to_remove_daly'][columns]), \
                    'no recovery date given for this injury' + code
                # check injury heal time is in the future
                assert df.loc[person_id, 'rt_date_to_remove_daly'][columns] > self.sim.date
                # remove code from heal with time injury list

            df.loc[person_id, 'rt_injuries_to_heal_with_time'].clear()
        # schedule treatments of all injuries here

        # ======================================= Schedule surgeries ==================================================
        # Schedule the surgeries by calling the functions rti_do_for_major/minor_surgeries which in turn schedules the
        # surgeries, people can have multiple surgeries scheduled so schedule surgeries seperate to the rest of the
        # treatment plans
        # Check they haven't died from another source
        if not pd.isnull(df.loc[person_id, 'cause_of_death']):
            pass
        else:
            if self.major_surgery_counts > 0:
                # schedule major surgeries
                for count in range(0, self.major_surgery_counts):
                    road_traffic_injuries.rti_do_for_major_surgeries(person_id=person_id, count=count)
            if self.minor_surgery_counts > 0:
                # shedule minor surgeries
                for count in range(0, self.minor_surgery_counts):
                    road_traffic_injuries.rti_do_for_minor_surgeries(person_id=person_id, count=count)
        # Schedule all other treatments here
        # Fractures are sometimes treated via major/minor surgeries. Need to establish which injuries are due to be
        # treated via fracture cast
        frac_codes = ['712', '712a', '712b', '712c', '811', '812', '813a', '813b', '813c', '822a', '822b']
        p = df.loc[person_id]
        codes_treated_elsewhere = \
            p['rt_injuries_for_minor_surgery'] + p['rt_injuries_for_major_surgery'] + \
            p['rt_injuries_to_heal_with_time'] + p['rt_injuries_for_open_fracture_treatment']
        frac_codes = [code for code in frac_codes if code not in codes_treated_elsewhere]
        # Create a lookup table for treatment methods and the injuries that they are due to treat
        single_option_treatments = {
            'suture': ['1101', '2101', '3101', '4101', '5101', '7101', '8101'],
            'burn': ['1114', '2114', '3113', '4113', '5113', '7113', '8113'],
            'fracture': frac_codes,
            'tetanus': ['1101', '2101', '3101', '4101', '5101', '7101', '8101', '1114', '2114', '3113', '4113', '5113',
                        '7113', '8113'],
            'pain': self.module.PROPERTIES.get('rt_injury_1').categories[1:],
            'open': ['813bo', '813co', '813do', '813eo']
        }
        # find this person's untreated injuries
        untreated_injury_cols = []
        idx_for_untreated_injuries = []
        for index, time in enumerate(df.loc[person_id, 'rt_date_to_remove_daly']):
            if pd.isnull(time):
                idx_for_untreated_injuries.append(index)
        for idx in idx_for_untreated_injuries:
            untreated_injury_cols.append(RTI.INJURY_COLUMNS[idx])
        person_untreated_injuries = df.loc[[person_id], untreated_injury_cols]

        for treatment in single_option_treatments:
            _, inj_counts = road_traffic_injuries.rti_find_and_count_injuries(person_untreated_injuries,
                                                                              single_option_treatments[treatment])
            if inj_counts > 0 & df.loc[person_id, 'is_alive']:
                if treatment == 'suture':
                    road_traffic_injuries.rti_ask_for_suture_kit(person_id=person_id)
                if treatment == 'burn':
                    road_traffic_injuries.rti_ask_for_burn_treatment(person_id=person_id)
                if treatment == 'fracture':
                    road_traffic_injuries.rti_ask_for_fracture_casts(person_id=person_id)
                if treatment == 'tetanus':
                    road_traffic_injuries.rti_ask_for_tetanus(person_id=person_id)
                if treatment == 'pain':
                    road_traffic_injuries.rti_acute_pain_management(person_id=person_id)
                if treatment == 'open':
                    road_traffic_injuries.rti_ask_for_open_fracture_treatment(person_id=person_id,
                                                                              counts=self.open_fractures)

        treatment_plan = \
            p['rt_injuries_for_minor_surgery'] + p['rt_injuries_for_major_surgery'] +  \
            p['rt_injuries_to_heal_with_time'] + p['rt_injuries_for_open_fracture_treatment'] +  \
            p['rt_injuries_to_cast']
        # make sure injuries are treated in one place only
        assert len(treatment_plan) == len(set(treatment_plan))
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
        injurycodes = {'First injury': df.loc[person_id, 'rt_injury_1'],
                       'Second injury': df.loc[person_id, 'rt_injury_2'],
                       'Third injury': df.loc[person_id, 'rt_injury_3'],
                       'Fourth injury': df.loc[person_id, 'rt_injury_4'],
                       'Fifth injury': df.loc[person_id, 'rt_injury_5'],
                       'Sixth injury': df.loc[person_id, 'rt_injury_6'],
                       'Seventh injury': df.loc[person_id, 'rt_injury_7'],
                       'Eight injury': df.loc[person_id, 'rt_injury_8']}
        logger.debug(f'Injury profile of person %d, {injurycodes}', person_id)


class HSI_RTI_Shock_Treatment(HSI_Event, IndividualScopeEventMixin):
    """
    This HSI event handles the process of treating hypovolemic shock, as recommended by the pediatric
    handbook for Malawi and (TODO: FIND ADULT REFERENCE)
    Currently this HSI_Event is described only and not used, as I still need to work out how to model the occurrence
    of shock
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, RTI)
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        self.is_child = False
        # create placeholder footprint requirements
        df = self.sim.population.props
        if df.loc[person_id, 'age_years'] < 5:
            the_appt_footprint['Under5OPD'] = 1  # Placeholder requirement
        else:
            the_appt_footprint['Over5OPD'] = 1  # Placeholder requirement
        # determine if this is a child
        if df.loc[person_id, 'age_years'] < 15:
            self.is_child = True
        the_accepted_facility_level = 1
        self.TREATMENT_ID = 'RTI_Shock_Treatment'  # This must begin with the module name
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = the_accepted_facility_level
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        hs = self.sim.modules["HealthSystem"]

        if not df.at[person_id, 'is_alive']:
            return hs.get_blank_appt_footprint()
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        consumables_shock = {'Intervention_Package_Code': dict(), 'Item_Code': dict()}
        # TODO: find a more complete list of required consumables for adults
        if self.is_child:
            item_code_fluid_replacement = pd.unique(
                consumables.loc[consumables['Items'] ==
                                "Sodium lactate injection (Ringer's), 500 ml, with giving set", 'Item_Code'])[0]
            item_code_dextrose = pd.unique(consumables.loc[consumables['Items'] ==
                                                           "Dextrose (glucose) 5%, 1000ml_each_CMST", 'Item_Code'])[0]
            item_code_cannula = pd.unique(consumables.loc[consumables['Items'] ==
                                                          'Cannula iv  (winged with injection pot) 20_each_CMST',
                                                          'Item_Code'])[0]
            item_code_blood = pd.unique(consumables.loc[consumables['Items'] == 'Blood, one unit', 'Item_Code'])[0]
            item_code_oxygen = pd.unique(consumables.loc[consumables['Items'] ==
                                                         "Oxygen, 1000 liters, primarily with oxygen cylinders",
                                                         'Item_Code'])[0]
            consumables_shock['Item_Code'].update({item_code_cannula: 1, item_code_fluid_replacement: 1,
                                                   item_code_dextrose: 1, item_code_blood: 1, item_code_oxygen: 1})
        else:
            item_code_fluid_replacement = pd.unique(
                consumables.loc[consumables['Items'] ==
                                "Sodium lactate injection (Ringer's), 500 ml, with giving set", 'Item_Code'])[0]
            item_code_oxygen = pd.unique(consumables.loc[consumables['Items'] ==
                                                         "Oxygen, 1000 liters, primarily with oxygen cylinders",
                                                         'Item_Code'])[0]
            item_code_cannula = pd.unique(consumables.loc[consumables['Items'] ==
                                                          'Cannula iv  (winged with injection pot) 20_each_CMST',
                                                          'Item_Code'])[0]
            item_code_blood = pd.unique(consumables.loc[consumables['Items'] == 'Blood, one unit', 'Item_Code'])[0]
            consumables_shock['Item_Code'].update({item_code_fluid_replacement: 1, item_code_cannula: 1,
                                                   item_code_oxygen: 1, item_code_blood: 1})
        is_cons_available = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=self,
            cons_req_as_footprint=consumables_shock,
            to_log=True)
        if is_cons_available:
            logger.debug("Hypovolemic shock treatment available for person %d",
                         person_id)
            df.at[person_id, 'rt_in_shock'] = False

    def did_not_run(self, person_id):
        # Assume that untreated shock leads to death for now
        # Schedule the death
        df = self.sim.population.props
        df.at[person_id, 'rt_death_from_shock'] = True
        self.sim.modules['Demography'].do_death(individual_id=person_id, cause="RTI_death_shock",
                                                originating_module=self.module)
        # Log the death
        logger.debug('This is RTI_Shock_Treatment scheduling a death for person %d who did not recieve treatment'
                     'for shock',
                     person_id, self.sim.date)


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
    '813a' - Broken hip
    '813b' - broken pelvis
    '813c' - broken femur

    '822a' - dislocated hip
    '822b' - dislocated knee

    The properties altered by this function are
    rt_date_to_remove_daly - setting recovery dates for injuries treated with fracture casts
    rt_injuries_to_cast - once treated the codes used to denote injuries to be treated by fracture casts are removed
                          from the list of injuries due to be treated with fracture casts
    rt_med_int - the property used to denote whether a person getting treatment for road traffic injuries

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
        # Get the population and health system
        df = self.sim.population.props
        hs = self.sim.modules["HealthSystem"]
        # if the person isn't alive return a blank footprint
        if not df.at[person_id, 'is_alive']:
            return hs.get_blank_appt_footprint()
        # get a shorthand reference to RTI and consumables modules
        road_traffic_injuries = self.sim.modules['RTI']
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        # isolate the relevant injury information
        # Find the untreated injuries
        p = df.loc[person_id]
        assigned_injury_recovery_time = list(pd.isnull(p['rt_date_to_remove_daly']))
        idx_for_untreated_injuries = np.where(assigned_injury_recovery_time)[0]
        untreated_injury_cols = [RTI.INJURY_COLUMNS[idx] for idx in idx_for_untreated_injuries]
        person_injuries = df.loc[[person_id], untreated_injury_cols]
        # check if they have a fracture that requires a cast
        codes = ['712b', '712c', '811', '812', '813a', '813b', '813c', '822a', '822b']
        _, fracturecastcounts = road_traffic_injuries.rti_find_and_count_injuries(person_injuries, codes)
        # check if they have a fracture that requires a sling
        codes = ['712a']
        _, slingcounts = road_traffic_injuries.rti_find_and_count_injuries(person_injuries, codes)
        consumables_fractures = {'Intervention_Package_Code': dict(), 'Item_Code': dict()}
        # Check the person sent here is alive, been through the generic first appointment,
        # been through the RTI med intervention
        assert p['rt_diagnosed'], 'person sent here has not been diagnosed'
        assert p['rt_med_int'], 'person sent here has not been treated'
        # Check that the person sent here has an injury treated by this module
        assert fracturecastcounts + slingcounts > 0
        # Check this person has an injury intended to be treated here
        assert len(p['rt_injuries_to_cast']) > 0
        # Check this injury assigned to be treated here is actually had by the person
        assert all(injuries in person_injuries.values for injuries in p['rt_injuries_to_cast'])
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
            to_log=True)
        # if the consumables are available then the appointment can run
        if is_cons_available:
            logger.debug(f"Fracture casts available for person %d's {fracturecastcounts + slingcounts} fractures",
                         person_id)
            # update the property rt_med_int to indicate they are recieving treatment
            df.at[person_id, 'rt_med_int'] = True
            # Find the persons injuries
            non_empty_injuries = person_injuries[person_injuries != "none"]
            non_empty_injuries = non_empty_injuries.dropna(axis=1)
            # Find the injury codes treated by fracture casts/slings
            codes = ['712a', '712b', '712c', '811', '812', '813a', '813b', '813c', '822a', '822b']
            # Some TLO codes have daly weights associated with treated and non-treated injuries, copy the list of
            # swapping codes
            swapping_codes = RTI.SWAPPING_CODES[:]
            # find the relevant swapping codes for this treatment
            swapping_codes = [code for code in swapping_codes if code in codes]
            # remove codes that will be treated elsewhere
            injuries_treated_elsewhere = \
                p['rt_injuries_for_minor_surgery'] + p['rt_injuries_for_major_surgery'] + \
                p['rt_injuries_to_heal_with_time'] + p['rt_injuries_for_open_fracture_treatment']
            # remove codes that are being treated elsewhere
            swapping_codes = [code for code in swapping_codes if code not in injuries_treated_elsewhere]
            # find any potential codes this person has that are due to be swapped and then swap with
            # rti_swap_injury_daly_upon_treatment
            relevant_codes = np.intersect1d(non_empty_injuries.values, swapping_codes)
            if len(relevant_codes) > 0:
                road_traffic_injuries.rti_swap_injury_daly_upon_treatment(person_id, relevant_codes)
            # Find the injuries that have been treated and then schedule a recovery date
            columns, codes = \
                road_traffic_injuries.rti_find_all_columns_of_treated_injuries(person_id, df.loc[person_id,
                                                                                                 'rt_injuries_to_cast'])
            # check that for each injury to be treated by this event we have a corresponding column
            assert len(columns) == len(df.loc[person_id, 'rt_injuries_to_cast'])
            # iterate over the columns of injuries treated here and assign a recovery date
            for col in columns:
                # todo: update this with recovery times for casted broken hips/pelvis/femurs
                # todo: update this with recovery times for casted dislocated hip
                df.loc[person_id, 'rt_date_to_remove_daly'][int(col[-1]) - 1] = self.sim.date + \
                                                                                DateOffset(weeks=7)
                # make sure the assigned injury recovery date is in the future
                assert df.loc[person_id, 'rt_date_to_remove_daly'][int(col[-1]) - 1] > self.sim.date
            person_injuries = df.loc[person_id, RTI.INJURY_COLUMNS]
            non_empty_injuries = person_injuries[person_injuries != "none"]
            injury_columns = non_empty_injuries.keys()
            for code in df.loc[person_id, 'rt_injuries_to_cast']:
                columns = injury_columns.get_loc(road_traffic_injuries.rti_find_injury_column(person_id, [code])[0])
                assert not pd.isnull(df.loc[person_id, 'rt_date_to_remove_daly'][columns]), \
                    'no recovery date given for this injury'
                df.loc[person_id, 'rt_injuries_to_cast']
            # remove codes from fracture cast list
            df.loc[person_id, 'rt_injuries_to_cast'].clear()
        else:
            logger.debug(f"Person %d's has {fracturecastcounts + slingcounts} fractures without treatment",
                         person_id)

    def did_not_run(self, person_id):
        logger.debug('Fracture casts unavailable for person %d', person_id)


class HSI_RTI_Open_Fracture_Treatment(HSI_Event, IndividualScopeEventMixin):
    """
    This HSI event handles fracture casts/giving slings for those who need it. The HSI event tests whether the injured
    person has an appropriate injury code, determines how many fractures the person and then requests fracture
    treatment as required.

    The injury codes dealt with in this HSI event are:
    '813bo' - Open fracture of the pelvis
    '813co' - Open fracture of the femur
    '813do' - Open fracture of the foot
    '813eo' - Open fracture of the tibia/fibula/ankle/patella

    The properties altered by this function are:
    rt_med_int - to denote that this person is recieving treatment
    rt_injuries_for_open_fracture_treatment - removing codes that have been treated by open fracture treatment
    rt_date_to_remove_daly - to schedule recovery dates for open fractures that have recieved treatment
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        assert isinstance(module, RTI)
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        the_appt_footprint['Over5OPD'] = 1  # Placeholder requirement
        the_appt_footprint['MinorSurg'] = 1  # wound debridement requires minor surgery
        the_accepted_facility_level = 1
        self.TREATMENT_ID = 'RTI_Open_Fracture_Treatment'  # This must begin with the module name
        self.EXPECTED_APPT_FOOTPRINT = the_appt_footprint
        self.ACCEPTED_FACILITY_LEVEL = the_accepted_facility_level
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        hs = self.sim.modules["HealthSystem"]

        if not df.at[person_id, 'is_alive']:
            return hs.get_blank_appt_footprint()
        road_traffic_injuries = self.sim.modules['RTI']
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        # isolate the relevant injury information
        person_injuries = df.loc[[person_id], RTI.INJURY_COLUMNS]
        # check if they have a fracture that requires a cast
        codes = ['813bo', '813co', '813do', '813eo']
        _, open_fracture_counts = road_traffic_injuries.rti_find_and_count_injuries(person_injuries, codes)
        assert open_fracture_counts > 0
        consumables_fractures = {'Intervention_Package_Code': dict(), 'Item_Code': dict()}
        # Check the person sent here is alive, been through the generic first appointment,
        # been through the RTI med intervention
        assert df.loc[person_id, 'rt_diagnosed'], 'person sent here has not been diagnosed'
        assert df.loc[person_id, 'rt_med_int'], 'person sent here has not been treated'

        # If they have an open fracture, ask for consumables to treat fracture
        if open_fracture_counts > 0:
            # Ask for ceftriaxon antibiotics as first choice.
            first_choice_antibiotic_code = pd.unique(
                consumables.loc[consumables['Items'] ==
                                'ceftriaxon 500 mg, powder for injection_10_IDA',
                                'Item_Code'])[0]
            consumables_fractures['Item_Code'].update({first_choice_antibiotic_code: 1})
            # Ask for sterilized gauze
            item_code_cetrimide_chlorhexidine = pd.unique(
                consumables.loc[consumables['Items'] ==
                                'Cetrimide 15% + chlorhexidine 1.5% solution.for dilution _5_CMST', 'Item_Code'])[0]
            consumables_fractures['Item_Code'].update({item_code_cetrimide_chlorhexidine: 1})
            item_code_gauze = pd.unique(
                consumables.loc[
                    consumables['Items'] == "Dressing, paraffin gauze 9.5cm x 9.5cm (square)_packof 36_CMST",
                    'Item_Code'])[0]
            consumables_fractures['Item_Code'].update({item_code_gauze: 1})
            # Ask for suture kit
            item_code_suture_kit = pd.unique(
                consumables.loc[consumables['Items'] == 'Suture pack', 'Item_Code'])[0]
            consumables_fractures['Item_Code'].update({item_code_suture_kit: 1})
            # If wound is "grossly contaminated" administer Metronidazole
            # todo: parameterise the probability of wound contamination
            p = self.module.parameters
            prob_open_fracture_contaminated = p['prob_open_fracture_contaminated']
            rand_for_contamination = self.module.rng.random_sample(size=1)
            if rand_for_contamination < prob_open_fracture_contaminated:
                conaminated_wound_metronidazole_code = pd.unique(
                    consumables.loc[consumables['Items'] ==
                                    'Metronidazole, injection, 500 mg in 100 ml vial',
                                    'Item_Code'])[0]
                consumables_fractures['Item_Code'].update({conaminated_wound_metronidazole_code: 1})

        # Check that there are enough consumables to treat this person's fractures
        is_cons_available = self.sim.modules['HealthSystem'].request_consumables(
            hsi_event=self,
            cons_req_as_footprint=consumables_fractures,
            to_log=True)
        if is_cons_available:
            logger.debug(f"Fracture casts available for person %d's {open_fracture_counts} open fractures",
                         person_id)
            person = df.loc[person_id]
            # update the dataframe to show this person is recieving treatment
            df.loc[person_id, 'rt_med_int'] = True
            # Find the persons injuries to be treated
            non_empty_injuries = person['rt_injuries_for_open_fracture_treatment']
            columns, code = road_traffic_injuries.rti_find_all_columns_of_treated_injuries(
                person_id, non_empty_injuries
            )
            # Some TLO codes have daly weights associated with treated and non-treated injuries
            if code[0] == '813bo':
                road_traffic_injuries.rti_swap_injury_daly_upon_treatment(person_id, code[0])
            # Schedule a recovery date for the injury
            # estimated 6-9 months recovery times for open fractures
            df.loc[person_id, 'rt_date_to_remove_daly'][int(columns[0][-1]) - 1] = self.sim.date + DateOffset(months=7)
            assert df.loc[person_id, 'rt_date_to_remove_daly'][int(columns[0][-1]) - 1] > self.sim.date
            assert not pd.isnull(df.loc[person_id, 'rt_date_to_remove_daly'][int(columns[0][-1]) - 1]), \
                'no recovery date given for this injury'
            # remove code from open fracture list
            if code[0] in df.loc[person_id, 'rt_injuries_for_open_fracture_treatment']:
                df.loc[person_id, 'rt_injuries_for_open_fracture_treatment'].remove(code[0])
        else:
            logger.debug(f"Person %d's has {open_fracture_counts} open fractures without treatment",
                         person_id)

    def did_not_run(self, person_id):
        logger.debug('Open fracture treatment unavailable for person %d', person_id)


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

    The properties altered by this function are:
    rt_med_int - to denote that this person is recieving treatment
    rt_date_to_remove_daly - to schedule recovery dates for lacerations treated in this hsi
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
        hs = self.sim.modules["HealthSystem"]

        if not df.at[person_id, 'is_alive']:
            return hs.get_blank_appt_footprint()
        road_traffic_injuries = self.sim.modules['RTI']

        person_injuries = df.loc[[person_id], RTI.INJURY_COLUMNS]
        codes = ['1101', '2101', '3101', '4101', '5101', '7101', '8101']
        _, lacerationcounts = road_traffic_injuries.rti_find_and_count_injuries(person_injuries, codes)
        # Check the person sent here didn't die due to rti, has been through A&E, through Med int
        assert df.loc[person_id, 'rt_diagnosed'], 'person sent here has not been through A and E'
        assert df.loc[person_id, 'rt_med_int'], 'person sent here has not been treated'
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
                to_log=True)['Item_Code']

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
                        assert df.loc[person_id, 'rt_date_to_remove_daly'][int(col[-1]) - 1] > self.sim.date

                else:
                    logger.debug("This laceration wasn't cleaned before stitching, person %d is at risk of infection",
                                 person_id)
                    df.at[person_id, 'rt_med_int'] = True
                    columns, codes = road_traffic_injuries.rti_find_all_columns_of_treated_injuries(person_id, codes)
                    for col in columns:
                        df.loc[person_id, 'rt_date_to_remove_daly'][int(col[-1]) - 1] = self.sim.date + \
                                                                                        DateOffset(days=14)
                        assert df.loc[person_id, 'rt_date_to_remove_daly'][int(col[-1]) - 1] > self.sim.date

            else:
                logger.debug('This facility has no treatment for open wounds available.')

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

    The properties treated by this module are:
    rt_med_int - to denote that this person is recieving treatment for their injuries
    rt_date_to_remove_daly - to schedule recovery dates for injuries treated here
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
        hs = self.sim.modules["HealthSystem"]

        if not df.at[person_id, 'is_alive']:
            return hs.get_blank_appt_footprint()
        road_traffic_injuries = self.sim.modules['RTI']

        person_injuries = df.loc[[person_id], RTI.INJURY_COLUMNS]
        codes = ['1114', '2114', '3113', '4113', '5113', '7113', '8113']
        _, burncounts = road_traffic_injuries.rti_find_and_count_injuries(person_injuries, codes)
        # check the person sent here has an injury treated by this module
        assert burncounts > 0
        # check the person sent here didn't die due to rti, has been through A and E and had RTI_med_int
        assert df.loc[person_id, 'rt_diagnosed'], 'this person has not been through a and e'
        assert df.loc[person_id, 'rt_med_int'], 'this person has not been treated'
        if burncounts > 0:
            # Request materials for burn treatment
            item_code_cetrimide_chlorhexidine = pd.unique(
                consumables.loc[consumables['Items'] ==
                                'Cetrimide 15% + chlorhexidine 1.5% solution.for dilution _5_CMST', 'Item_Code'])[0]
            item_code_gauze = pd.unique(
                consumables.loc[
                    consumables['Items'] == "Dressing, paraffin gauze 9.5cm x 9.5cm (square)_packof 36_CMST",
                    'Item_Code'])[0]
            possible_large_TBSA_burn_codes = ['7113', '8113', '4113', '5113']
            idx2, bigburncounts = \
                road_traffic_injuries.rti_find_and_count_injuries(person_injuries, possible_large_TBSA_burn_codes)
            random_for_severe_burn = self.module.rng.random_sample(size=1)
            # ======================== If burns severe enough then give IV fluid replacement ===========================
            if (burncounts > 1) or ((len(idx2) > 0) & (random_for_severe_burn > self.prob_mild_burns)):
                # check if they have multiple burns, which implies a higher burned total body surface area (TBSA) which
                # will alter the treatment plan

                item_code_fluid_replacement = pd.unique(
                    consumables.loc[consumables['Items'] ==
                                    "Sodium lactate injection (Ringer's), 500 ml, with giving set", 'Item_Code'])[0]
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
                to_log=True)['Item_Code']
            if all(value for value in is_cons_available.values()):
                logger.debug('This facility has burn treatment available which has been used for person %d.',
                             person_id)
                logger.debug(f'This facility treated their {burncounts} burns')
                df.at[person_id, 'rt_med_int'] = True
                person = df.loc[person_id]
                non_empty_injuries = person_injuries[person_injuries != "none"]
                injury_columns = non_empty_injuries.columns
                columns = injury_columns.get_loc(road_traffic_injuries.rti_find_injury_column(person_id, codes)[0])
                # estimate burns take 4 weeks to heal
                df.loc[person_id, 'rt_date_to_remove_daly'][columns] = self.sim.date + DateOffset(weeks=4)
                assert df.loc[person_id, 'rt_date_to_remove_daly'][columns] > self.sim.date
                persons_injuries = df.loc[[person_id], RTI.INJURY_COLUMNS]
                non_empty_injuries = persons_injuries[persons_injuries != "none"]
                non_empty_injuries = non_empty_injuries.dropna(axis=1)
                swapping_codes = RTI.SWAPPING_CODES[:]
                swapping_codes = [code for code in swapping_codes if code in codes]
                # remove codes that will be treated elsewhere
                treatment_plan = (
                    person['rt_injuries_for_major_surgery'] + person['rt_injuries_for_minor_surgery'] +
                    person['rt_injuries_for_minor_surgery'] + person['rt_injuries_to_cast'] +
                    person['rt_injuries_to_heal_with_time'] + person['rt_injuries_for_open_fracture_treatment']
                )
                swapping_codes = [code for code in swapping_codes if code not in treatment_plan]
                relevant_codes = np.intersect1d(non_empty_injuries.values, swapping_codes)
                if len(relevant_codes) > 0:
                    road_traffic_injuries.rti_swap_injury_daly_upon_treatment(person_id, relevant_codes)

                assert df.loc[person_id, 'rt_date_to_remove_daly'][columns] > self.sim.date, \
                    'recovery date assigned to past'
            else:
                logger.debug('This facility has no treatment for burns available.')

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
        hs = self.sim.modules["HealthSystem"]

        if not df.at[person_id, 'is_alive']:
            return hs.get_blank_appt_footprint()
        person_injuries = df.loc[[person_id], RTI.INJURY_COLUMNS]
        # check the person sent here hasn't died due to rti, has been through A and E and had RTI_med_int
        assert df.loc[person_id, 'rt_diagnosed'], 'This person has not been through a and e'
        assert df.loc[person_id, 'rt_med_int'], 'This person has not been through rti med int'
        # check the person sent here has an injury treated by this module
        codes_for_tetanus = ['1101', '2101', '3101', '4101', '5101', '7101', '8101',
                             '1114', '2114', '3113', '4113', '5113', '7113', '8113']
        _, counts = RTI.rti_find_and_count_injuries(person_injuries, codes_for_tetanus)
        assert counts > 0
        # If they have a laceration/burn ask request the tetanus vaccine
        if counts > 0:
            consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
            item_code_tetanus = pd.unique(
                consumables.loc[consumables['Items'] == 'Tetanus toxoid, injection', 'Item_Code'])[0]
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
     SEE doi: 10.2105/AJPH.2018.304747
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
        hs = self.sim.modules["HealthSystem"]

        if not df.at[person_id, 'is_alive']:
            return hs.get_blank_appt_footprint()
        # Check that the person sent here is alive, has been through A&E and RTI_Med_int
        assert df.loc[person_id, 'rt_diagnosed'], 'This person has not been through a and e'
        assert df.loc[person_id, 'rt_med_int'], 'This person has not been through rti med int'
        person_injuries = df.loc[[person_id], RTI.INJURY_COLUMNS]
        consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
        road_traffic_injuries = self.sim.modules['RTI']
        pain_level = "none"
        # create a dictionary to associate the level of pain to the codes
        pain_dict = {
            'severe': ['1114', '2114', '3113', '4113', '5113', '7113', '8113',  # burns
                       'P782', 'P782a', 'P782b', 'P782c', 'P783', 'P882', 'P883', 'P884',  # amputations
                       '673', '673a', '673b', '674', '674a', '674b', '675', '675a', '675b', '676',
                       'P673', 'P673a', 'P673b', 'P674', 'P674a', 'P674b', 'P675', 'P675a', 'P675b', 'P676',  # SCI
                       '552', '553', '554',  # abdominal trauma
                       '463', '453', '453a', '453b', '441', '443'  # severe chest trauma
                       ],
            'moderate': ['112', '113', '211', '212', '412', '414', '612', '712', '712a', '712b', '712c',
                         '811', '812', '813', '813a', '813b', '813c',  # fractures
                         '322', '323', '722', '822', '822a', '822b',  # dislocations
                         '342', '343', '361', '363',  # neck trauma
                         '461',  # chest wall bruising
                         '813bo', '813co', '813do', '813eo'  # open fractures
                         ],
            'mild': ['1101', '2101', '3101', '4101', '5101', '7101', '8101',  # lacerations
                     '241',  # Minor soft tissue injuries
                     '133', '133a', '133b', '133c', '133d', '134', '134a', '134b', '135',  # TBI
                     'P133', 'P133a', 'P133b', 'P133c', 'P133d', 'P134', 'P134a', 'P134b', 'P135',  # Perm TBI
                     '291',  # Eye injury
                     '442'
                     ]
        }
        # iterate over the dictionary to find the pain level, going from highest pain to lowest pain in a for loop,
        # then find the highest level of pain this person has by breaking the for loop
        for severity in pain_dict.keys():
            _, counts = road_traffic_injuries.rti_find_and_count_injuries(person_injuries, pain_dict[severity])
            if counts > 0:
                pain_level = severity
                break
        # check that the people here have at least one injury
        assert counts > 0
        if pain_level == "mild":
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

            if df.loc[person_id, 'age_years'] < 16:
                # or df.iloc[person_id]['is_pregnant']
                # If they are under 16 or pregnant only give them paracetamol
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

        if pain_level == "moderate":
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
                to_log=True)['Item_Code'][item_code_tramadol]
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

        if pain_level == "severe":
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
                to_log=True)
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
        '811' - fractured foot - reported use of surgery in Chagomerana et al. 2017
        '812' - fracture tibia/fibula - reported use of surgery in Chagomerana et al. 2017
        '813a' - Fractured hip - reported use of surgery and Lavy et al. 2007
        '813b' - Fractured pelvis - reported use of surgery and Lavy et al. 2007
        '813c' - Fractured femur - reported use of surgery and Lavy et al. 2007
        '414' - Flail chest - https://www.sciencedirect.com/science/article/abs/pii/S0020138303002900

        SOFT TISSUE INJURIES:
        '342' - Soft tissue injury of the neck
        '343' - Soft tissue injury of the neck

        Thoroscopy treated injuries:
        https://www.ncbi.nlm.nih.gov/nlmcatalog/101549743
        Ref from pediatric handbook for Malawi
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

        Dislocations - Reported in Chagomerana et al. 2017
        '822a' Hip dislocation

        The properties altered in this function are:
        rt_injury_1 through rt_injury_8 - in the incidence that despite treatment the person treated is left
                                          permanently disabled we need to update the injury code to inform the
                                          model that the disability burden associated with the permanently
                                          disabling injury shouldn't be removed
        rt_perm_disability - when a person is decided to be permanently disabled we update this property to reflect this
        rt_date_to_remove_daly - assign recovery dates for the injuries treated with the surgery
        rt_injuries_for_major_surgery - to remove codes due to be treated by major surgery when that injury recieves
                                        a treatment.
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
        self.allowed_interventions = p['allowed_interventions']
        self.treated_code = 'none'

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        hs = self.sim.modules["HealthSystem"]

        if not df.at[person_id, 'is_alive']:
            return hs.get_blank_appt_footprint()
        rng = self.module.rng
        road_traffic_injuries = self.sim.modules['RTI']
        # check the people sent here hasn't died due to rti, have had their injuries diagnosed and been through RTI_Med
        assert df.loc[person_id, 'rt_diagnosed'], 'This person has not been through a and e'
        assert df.loc[person_id, 'rt_med_int'], 'This person has not been through rti med int'
        # Isolate the relevant injury information
        surgically_treated_codes = ['112', '811', '812', '813a', '813b', '813c', '133a', '133b', '133c', '133d', '134a',
                                    '134b', '135', '552', '553', '554', '342', '343', '414', '361', '363',
                                    '782', '782a', '782b', '782c', '783', '822a', '882', '883', '884',
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
        persons_injuries = df.loc[[person_id], RTI.INJURY_COLUMNS]
        injuries_to_be_treated = df.loc[person_id, 'rt_injuries_for_major_surgery']
        assert len(set(injuries_to_be_treated) & set(surgically_treated_codes)) > 0, \
            'This person has asked for surgery but does not have an appropriate injury'
        # check the people sent here have at least one injury treated by this HSI event
        _, counts = road_traffic_injuries.rti_find_and_count_injuries(persons_injuries, surgically_treated_codes)
        assert counts > 0, (persons_injuries.to_dict(), surgically_treated_codes)
        # People can be sent here for multiple surgeries, but only one injury can be treated at a time. Decide which
        # injury is being treated in this surgery
        # find index for untreated injuries
        idx_for_untreated_injuries = np.where(pd.isnull(df.loc[person_id, 'rt_date_to_remove_daly']))
        # find untreated injury codes that are treated with major surgery
        relevant_codes = np.intersect1d(injuries_to_be_treated, surgically_treated_codes)
        # check that the person sent here has an appropriate code(s)
        assert len(relevant_codes) > 0, (persons_injuries.values[0], idx_for_untreated_injuries, person_id,
                                         persons_injuries.values[0][idx_for_untreated_injuries])
        # choose a code at random
        self.treated_code = rng.choice(relevant_codes)
        # ------------------------ Track permanent disabilities with treatment ----------------------------------------
        # --------------------------------- Perm disability from TBI --------------------------------------------------
        codes = ['133', '133a', '133b', '133c', '133d', '134', '134a', '134b', '135']

        """ Of patients that survived, 80.1% (n 148) had a good recovery with no appreciable clinical neurologic
        deficits, 13.1% (n 24) had a moderate disability with deficits that still allowed the patient to live
        independently, 4.9% (n 9) had severe disability which will require assistance with activities of daily life,
        and 1.1% (n 2) were in a vegetative state
        """
        # Check whether the person having treatment for their tbi will be left permanently disabled
        if self.treated_code in codes:
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
                code_to_drop_index = injuries_to_be_treated.index(self.treated_code)
                injuries_to_be_treated.pop(code_to_drop_index)
                self.treated_code = "P" + self.treated_code
                df.loc[person_id, column] = self.treated_code
                injuries_to_be_treated.append(self.treated_code)
                assert len(injuries_to_be_treated) == len(df.loc[person_id, 'rt_injuries_for_major_surgery'])

            columns, codes = road_traffic_injuries.rti_find_all_columns_of_treated_injuries(person_id,
                                                                                            [self.treated_code])

            # schedule the recovery date for the permanent injury for beyond the end of the simulation (making
            # it permanent)
            df.loc[person_id, 'rt_date_to_remove_daly'][int(columns[0][-1]) - 1] = \
                self.sim.end_date + DateOffset(days=1)
            assert df.loc[person_id, 'rt_date_to_remove_daly'][int(columns[0][-1]) - 1] > self.sim.date
        # ------------------------------------- Perm disability from SCI ----------------------------------------------
        if 'include_spine_surgery' in self.allowed_interventions:
            codes = ['673', '673a', '673b', '674', '674a', '674b', '675', '675a', '675b', '676']
            if self.treated_code in codes:
                # Track whether they are permanently disabled
                df.at[person_id, 'rt_perm_disability'] = True
                # Find the column and code where the permanent injury is stored
                column, code = road_traffic_injuries.rti_find_injury_column(person_id=person_id,
                                                                            codes=[self.treated_code])
                logger.debug('@@@@@@@@@@ Person %d had intervention for SCI on %s but still disabled!!!!!!',
                             person_id, self.sim.date)
                code_to_drop_index = injuries_to_be_treated.index(self.treated_code)
                injuries_to_be_treated.pop(code_to_drop_index)
                self.treated_code = "P" + self.treated_code
                df.loc[person_id, column] = self.treated_code
                injuries_to_be_treated.append(self.treated_code)
                for injury in injuries_to_be_treated:
                    if injury not in df.loc[person_id, 'rt_injuries_for_major_surgery']:
                        df.loc[person_id, 'rt_injuries_for_major_surgery'].append(injury)
                assert len(injuries_to_be_treated) == len(df.loc[person_id, 'rt_injuries_for_major_surgery'])
                columns, codes = road_traffic_injuries.rti_find_all_columns_of_treated_injuries(person_id,
                                                                                                [self.treated_code])

                # schedule the recovery date for the permanent injury for beyond the end of the simulation (making
                # it permanent)
                df.loc[person_id, 'rt_date_to_remove_daly'][int(columns[0][-1]) - 1] = \
                    self.sim.end_date + DateOffset(days=1)
                assert df.loc[person_id, 'rt_date_to_remove_daly'][int(columns[0][-1]) - 1] > self.sim.date

        # ------------------------------------- Perm disability from amputation ----------------------------------------
        codes = ['782', '782a', '782b', '782c', '783', '882', '883', '884']
        if self.treated_code in codes:
            # Track whether they are permanently disabled
            df.at[person_id, 'rt_perm_disability'] = True
            # Find the column and code where the permanent injury is stored
            column, code = road_traffic_injuries.rti_find_injury_column(person_id=person_id, codes=[self.treated_code])
            logger.debug('@@@@@@@@@@ Person %d had intervention for an amputation on %s but still disabled!!!!!!',
                         person_id, self.sim.date)
            # Update the code to make the injury permanent, so it will not have the associated daly weight removed
            # later on
            code_to_drop_index = injuries_to_be_treated.index(self.treated_code)
            injuries_to_be_treated.pop(code_to_drop_index)
            self.treated_code = "P" + self.treated_code
            df.loc[person_id, column] = self.treated_code
            injuries_to_be_treated.append(self.treated_code)
            for injury in injuries_to_be_treated:
                if injury not in df.loc[person_id, 'rt_injuries_for_major_surgery']:
                    df.loc[person_id, 'rt_injuries_for_major_surgery'].append(injury)
            assert len(injuries_to_be_treated) == len(df.loc[person_id, 'rt_injuries_for_major_surgery'])
            columns, codes = road_traffic_injuries.rti_find_all_columns_of_treated_injuries(person_id,
                                                                                            [self.treated_code])
            # Schedule recovery for the end of the simulation, thereby making the injury permanent

            df.loc[person_id, 'rt_date_to_remove_daly'][int(columns[0][-1]) - 1] = \
                self.sim.end_date + DateOffset(days=1)
            assert df.loc[person_id, 'rt_date_to_remove_daly'][int(columns[0][-1]) - 1] > self.sim.date

        # ============================== Schedule the recovery dates for the non-permanent injuries ==================
        non_empty_injuries = persons_injuries[persons_injuries != "none"]
        non_empty_injuries = non_empty_injuries.dropna(axis=1)
        injury_columns = non_empty_injuries.columns
        maj_surg_recovery_time_in_days = {
            '112': 42,
            '552': 90,
            '553': 90,
            '554': 90,
            '822a': 270,
            '811': 63,
            '812': 63,
            '813a': 270,
            '813b': 70,
            '813c': 120,
            '133a': 42,
            '133b': 42,
            '133c': 42,
            '133d': 42,
            '134a': 42,
            '134b': 42,
            '135': 42,
            '342': 42,
            '343': 42,
            '414': 365,
            '441': 14,
            '443': 14,
            '453a': 42,
            '453b': 42,
            '361': 7,
            '363': 7,
            '463': 7,
        }
        # find the column of the treated injury
        if self.treated_code in maj_surg_recovery_time_in_days.keys():
            columns = injury_columns.get_loc(road_traffic_injuries.rti_find_injury_column(person_id,
                                                                                          [self.treated_code])[0])
            if pd.isnull(df.loc[person_id, 'rt_date_to_remove_daly'][columns]):
                df.loc[person_id, 'rt_date_to_remove_daly'][columns] = \
                    self.sim.date + DateOffset(days=maj_surg_recovery_time_in_days[self.treated_code])
                assert df.loc[person_id, 'rt_date_to_remove_daly'][columns] > self.sim.date

        # some injuries have a daly weight that swaps upon treatment, get list of those codes
        swapping_codes = RTI.SWAPPING_CODES[:]
        # isolate that swapping codes that will be treated here
        swapping_codes = [code for code in swapping_codes if code in surgically_treated_codes]
        # find the injuries this person will have treated in other forms of treatment
        person = df.loc[person_id]
        treatment_plan = (
            person['rt_injuries_for_minor_surgery'] + person['rt_injuries_to_cast'] +
            person['rt_injuries_to_heal_with_time'] + person['rt_injuries_for_open_fracture_treatment']
        )
        # remove codes that will be treated elsewhere
        swapping_codes = [code for code in swapping_codes if code not in treatment_plan]
        # swap the daly weight for any applicable injuries
        if self.treated_code in swapping_codes:
            road_traffic_injuries.rti_swap_injury_daly_upon_treatment(person_id, [self.treated_code])
        # Check that every injury treated has a recovery time
        columns = injury_columns.get_loc(road_traffic_injuries.rti_find_injury_column(person_id,
                                                                                      [self.treated_code])[0])
        assert not pd.isnull(df.loc[person_id, 'rt_date_to_remove_daly'][columns]), \
            'no recovery date given for this injury'
        assert df.loc[person_id, 'rt_date_to_remove_daly'][columns] > self.sim.date
        logger.debug('This is RTI_Major_Surgeries supplying surgery for person %d on date %s!!!!!!, removing code %s',
                     person_id, self.sim.date)
        # remove code from major surgeries list
        if self.treated_code in df.loc[person_id, 'rt_injuries_for_major_surgery']:
            df.loc[person_id, 'rt_injuries_for_major_surgery'].remove(self.treated_code)

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
        https://link.springer.com/article/10.1007%2Fs11999-008-0366-5
        '211' - Facial fractures
        '212' - Facial fractures
        '291' - Injury to the eye
        '241' - Soft tissue injury of the face

        '322' - Dislocation in the neck
        '323' - Dislocation in the neck

        '722' - Dislocated shoulder

        External fixation of fractures
        '811' - fractured foot
        '812' - fractures tibia/fibula
        '813a' - Fractured hip
        '813b' - Fractured pelvis
        '813C' - Fractured femur
        The properties altered in this function are:
        rt_med_int - update to show this person is being treated for their injuries.
        rt_date_to_remove_daly - assign recovery dates for the injuries treated with the surgery
        rt_injuries_for_minor_surgery - to remove codes due to be treated by minor surgery when that injury recieves
                                        a treatment.
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
        hs = self.sim.modules["HealthSystem"]

        if not df.at[person_id, 'is_alive']:
            return hs.get_blank_appt_footprint()
        rng = self.module.rng
        road_traffic_injuries = self.sim.modules['RTI']
        surgically_treated_codes = ['322', '211', '212', '323', '722', '291', '241', '811', '812', '813a', '813b',
                                    '813c']
        persons_injuries = df.loc[[person_id], RTI.INJURY_COLUMNS]
        person = df.loc[person_id]
        # =========================================== Tests ============================================================
        # check the people sent here hasn't died due to rti, have had their injuries diagnosed and been through RTI_Med
        assert person['rt_diagnosed'], 'This person has not been through a and e'
        assert person['rt_med_int'], 'This person has not been through rti med int'
        # check they have at least one injury treated by minor surgery
        _, counts = road_traffic_injuries.rti_find_and_count_injuries(persons_injuries, surgically_treated_codes)
        assert counts > 0
        # find the injuries this person has
        non_empty_injuries = persons_injuries[persons_injuries != "none"]
        non_empty_injuries = non_empty_injuries.dropna(axis=1)
        # find the injuries which will be treated here
        relevant_codes = np.intersect1d(df.loc[person_id, 'rt_injuries_for_minor_surgery'], surgically_treated_codes)
        # Check that a code has been selected to be treated
        assert len(relevant_codes) > 0
        # choose an injury to treat
        treated_code = rng.choice(relevant_codes)
        injury_columns = persons_injuries.columns
        # create a dictionary to store the recovery times for each injury in days
        minor_surg_recov_time_days = {
            '322': 180,
            '323': 180,
            '722': 49,
            '211': 49,
            '212': 49,
            '291': 7,
            '241': 7,
            '811': 63,
            '812': 63,
            '813a': 63,
            '813b': 63,
            '813c': 63,
        }
        # need to determine whether this person has an injury which will treated with external fixation
        external_fixation = False
        external_fixation_codes = ['811', '812', '813a', '813b', '813c']
        if treated_code in external_fixation_codes:
            external_fixation = True
        # assign a recovery time for the treated person from the dictionary, get the column which the injury is stored
        # in
        columns = injury_columns.get_loc(road_traffic_injuries.rti_find_injury_column(person_id, [treated_code])[0])
        # assign a recovery date
        df.loc[person_id, 'rt_date_to_remove_daly'][columns] = \
            self.sim.date + DateOffset(days=minor_surg_recov_time_days[treated_code])
        # make sure the injury recovery date is in the future
        assert df.loc[person_id, 'rt_date_to_remove_daly'][columns] > self.sim.date
        # If surgery requires external fixation, request the materials as part of the appointment footprint
        if external_fixation:
            consumables = self.sim.modules['HealthSystem'].parameters['Consumables']
            item_code_external_fixator = pd.unique(
                consumables.loc[consumables['Items'] == 'External fixator', 'Item_Code'])[0]
            consumables_external_fixation = {
                'Intervention_Package_Code': dict(),
                'Item_Code': {item_code_external_fixator: 1}
            }
            is_external_fixator_available = self.sim.modules['HealthSystem'].request_consumables(
                hsi_event=self,
                cons_req_as_footprint=consumables_external_fixation,
                to_log=True)
            if is_external_fixator_available:
                logger.debug('An external fixator is available for this minor surgery'
                             'person %d.', person_id)

        # some injuries have a change in daly weight if they are treated, find all possible swappable codes
        swapping_codes = RTI.SWAPPING_CODES[:]
        # exclude any codes that could be swapped but are due to be treated elsewhere
        treatment_plan = (
            person['rt_injuries_for_minor_surgery'] + person['rt_injuries_to_cast'] +
            person['rt_injuries_to_heal_with_time'] + person['rt_injuries_for_open_fracture_treatment']
        )
        swapping_codes = [code for code in swapping_codes if code not in treatment_plan]
        if treated_code in swapping_codes:
            road_traffic_injuries.rti_swap_injury_daly_upon_treatment(person_id, [treated_code])
        logger.debug('This is RTI_Minor_Surgeries supplying minor surgeries for person %d on date %s!!!!!!',
                     person_id, self.sim.date)
        # update the dataframe to reflect that this person is recieving medical care
        df.at[person_id, 'rt_med_int'] = True
        # Check if the injury has been given a recovery date
        columns = injury_columns.get_loc(road_traffic_injuries.rti_find_injury_column(person_id, [treated_code])[0])
        assert not pd.isnull(df.loc[person_id, 'rt_date_to_remove_daly'][columns]), \
            'no recovery date given for this injury'
        # remove code from minor surgeries list as it has now been treated
        if treated_code in df.loc[person_id, 'rt_injuries_for_minor_surgery']:
            df.loc[person_id, 'rt_injuries_for_minor_surgery'].remove(treated_code)

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

     The properties altered by this event are:
     rt_post_med_death - updated to reflect when a person dies from their injuries
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        p = self.module.parameters
        self.prob_death_iss_less_than_9 = p['prob_death_iss_less_than_9']
        # self.prob_death_iss_less_than_9 = 0.0211 * 0.571
        self.prob_death_iss_10_15 = p['prob_death_iss_10_15']
        # self.prob_death_iss_10_15 = 0.0306 * 0.571
        self.prob_death_iss_16_24 = p['prob_death_iss_16_24']
        # self.prob_death_iss_16_24 = 0.0870573 * 0.571
        self.prob_death_iss_25_35 = p['prob_death_iss_25_35']
        # self.prob_death_iss_25_35 = 0.376464 * 0.571
        self.prob_death_iss_35_plus = p['prob_death_iss_25_35']
        # self.prob_death_iss_35_plus = 0.6399888 * 0.571

    def apply(self, person_id):
        df = self.sim.population.props

        randfordeath = self.module.rng.random_sample(size=1)
        # ======================================== Tests ==============================================================
        assert df.loc[person_id, 'rt_ISS_score'] > 0
        mortality_checked = False
        probabilities_of_death = {
            '1-4': [range(1, 5), 0],
            '5-9': [range(5, 10), self.prob_death_iss_less_than_9],
            '10-15': [range(10, 16), self.prob_death_iss_10_15],
            '16-24': [range(16, 25), self.prob_death_iss_16_24],
            '25-35': [range(25, 36), self.prob_death_iss_25_35],
            '35-75': [range(25, 76), self.prob_death_iss_35_plus]
        }
        # Schedule death for those who died from their injuries despite medical intervention
        if df.loc[person_id, 'cause_of_death'] == 'Other':
            pass
        for range_boundaries in probabilities_of_death.keys():
            if df.loc[person_id].rt_ISS_score in probabilities_of_death[range_boundaries][0]:
                if randfordeath < probabilities_of_death[range_boundaries][1]:
                    mortality_checked = True
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
                    self.sim.modules['Demography'].do_death(individual_id=person_id, cause="RTI_death_with_med",
                                                            originating_module=self.module)
                    # Log the death
                    logger.debug('This is RTIMedicalInterventionDeathEvent scheduling a death for person %d who was '
                                 'treated for their injuries but still died on date %s',
                                 person_id, self.sim.date)
                else:
                    mortality_checked = True

        assert mortality_checked, 'Something missing in criteria'


class RTI_No_Lifesaving_Medical_Intervention_Death_Event(Event, IndividualScopeEventMixin):
    """This is the NoMedicalInterventionDeathEvent. It is scheduled by the MedicalInterventionEvent which determines the
    resources required to treat that person and if they aren't present, the person is sent here. This function is also
    called by the did not run function for rti_major_surgeries for certain injuries, implying that if life saving
    surgery is not available for the person, then we have to ask the probability of them dying without having this life
    saving surgery.

    some information on time to craniotomy here:
    https://thejns.org/focus/view/journals/neurosurg-focus/45/6/article-pE2.xml?body=pdf-10653


    The properties altered by this event are:
    rt_unavailable_med_death - to denote that this person has died due to medical interventions not being available
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)
        p = self.module.parameters
        # load the parameteres used for this event
        self.prob_death_TBI_SCI_no_treatment = p['prob_death_TBI_SCI_no_treatment']
        self.prob_death_fractures_no_treatment = p['prob_death_fractures_no_treatment']
        self.prop_death_burns_no_treatment = p['prop_death_burns_no_treatment']

    def apply(self, person_id):
        # self.scheduled_death = 0
        df = self.sim.population.props
        # create a dictionary to store the injuries and corresponding probability of death for untreated injuries
        untreated_dict = {'non-lethal': [['241', '291', '322', '323', '461', '442', '1101', '2101', '3101', '4101',
                                         '5101', '7101', '8101', '722', '822a', '822b'], 0],
                          'severe': [['133', '133a', '133b', '133c', '133d', '134', '134a', '134b', '135', '342', '343',
                                      '361', '363', '414', '441', '443', '453a', '453b', '463', '552', '553', '554',
                                      '673', '673a', '673b', '674', '674a', '674b', '675', '675a', '675b', '676',
                                      '782a', '782b', '782c', '783', '882', '883', '884', 'P133', 'P133a', 'P133b',
                                      'P133c', 'P133d', 'P134', 'P134a', 'P134b', 'P135', 'P673', 'P673a', 'P673b',
                                      'P674', 'P674a', 'P674b', 'P675', 'P675a', 'P675b', 'P676', 'P782a', 'P782b',
                                      'P782c', 'P783', 'P882', 'P883', 'P884', '813bo', '813co', '813do', '813eo'],
                                     self.prob_death_TBI_SCI_no_treatment],
                          'fracture': [['112', '113', '211', '212', '412', '612', '712', '712a', '712b', '712c', '811',
                                        '812', '813', '813a', '813b', '813c'], self.prob_death_fractures_no_treatment],
                          'burn': [['1114', '2114', '3113', '4113', '5113', '7113', '8113'],
                                   self.prop_death_burns_no_treatment]
                          }

        persons_injuries = df.loc[[person_id], RTI.INJURY_COLUMNS]
        non_empty_injuries = persons_injuries[persons_injuries != "none"]
        non_empty_injuries = non_empty_injuries.dropna(axis=1)
        untreated_injuries = []
        prob_death = 0
        # Find which injuries are left untreated by finding injuries which haven't been set a recovery time
        for col in non_empty_injuries:
            if pd.isnull(df.loc[person_id, 'rt_date_to_remove_daly'][int(col[-1]) - 1]):
                untreated_injuries.append(df.at[person_id, col])
        # people can have multiple untreated injuries, most serious injury will be used to determine the likelihood of
        # their passing, so create an empty list to store probabilities of death associated with this person's untreated
        # injury and take the max value
        prob_deaths = []
        for injury in untreated_injuries:
            for severity_level in untreated_dict:
                if injury in untreated_dict[severity_level][0]:
                    prob_deaths.append(untreated_dict[severity_level][1])
        prob_death = max(prob_deaths)
        randfordeath = self.module.rng.random_sample(size=1)
        if randfordeath < prob_death:
            df.loc[person_id, 'rt_unavailable_med_death'] = True
            self.sim.modules['Demography'].do_death(individual_id=person_id, cause="RTI_unavailable_med",
                                                    originating_module=self.module)
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
        # Injury location on body data
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
        # More model progression data
        self.totinjured = 0
        self.deathonscene = 0
        self.soughtmedcare = 0
        self.deathaftermed = 0
        self.deathwithoutmed = 0
        self.permdis = 0
        self.ISSscore = []
        self.severe_pain = 0
        self.moderate_pain = 0
        self.mild_pain = 0
        # Create variables for averages over time in the model
        self.numerator = 0
        self.denominator = 0
        self.death_inc_numerator = 0
        self.death_in_denominator = 0
        self.fracdenominator = 0
        # Create variables to measure where certain injuries are located on the body
        self.fracdist = [0, 0, 0, 0, 0, 0, 0, 0]
        self.openwounddist = [0, 0, 0, 0, 0, 0, 0, 0]
        self.burndist = [0, 0, 0, 0, 0, 0, 0, 0]

    def apply(self, population):
        # Make some summary statistics
        # Get the dataframe and isolate the important information
        df = population.props
        # dump dataframe each month if population size is large (used to find the minimum viable population size)
        time_stamped_file_name = "df_at_" + str(self.sim.date.month) + "_" + str(self.sim.date.year)
        if len(df.loc[df.is_alive]) > 750000:
            df.to_csv(f"C:/Users/Robbie Manning Smith/Documents/Dataframe_dump/{time_stamped_file_name}.csv")
        thoseininjuries = df.loc[df.rt_road_traffic_inc]
        # ================================= Injury severity ===========================================================
        sev = thoseininjuries['rt_inj_severity']
        rural_injuries = df.loc[df.rt_road_traffic_inc & ~df.li_urban]
        if len(rural_injuries) > 0:
            percent_sev_rural = \
                len(rural_injuries.loc[rural_injuries['rt_inj_severity'] == 'severe']) / len(rural_injuries)
        else:
            percent_sev_rural = 'none_injured'
        urban_injuries = df.loc[df.rt_road_traffic_inc & df.li_urban]
        if len(urban_injuries) > 0:
            percent_sev_urban = \
                len(urban_injuries.loc[urban_injuries['rt_inj_severity'] == 'severe']) / len(urban_injuries)
        else:
            percent_sev_urban = 'none_injured'
        severity, severitycount = np.unique(sev, return_counts=True)
        if 'mild' in severity:
            idx = np.where(severity == 'mild')
            self.totmild += len(idx)
        if 'severe' in severity:
            idx = np.where(severity == 'severe')
            self.totsevere += len(idx)
        dict_to_output = {
            'total_mild_injuries': self.totmild,
            ''
            '_severe_injuries': self.totsevere,
            'Percent_severe_rural': percent_sev_rural,
            'Percent_severe_urban': percent_sev_urban
        }
        logger.info(key='injury_severity',
                    data=dict_to_output,
                    description='severity of injuries in simulation')
        # ==================================== Incidence ==============================================================
        # How many were involved in a RTI
        n_in_RTI = df.rt_road_traffic_inc.sum()
        children_in_RTI = len(df.loc[df.rt_road_traffic_inc & (df['age_years'] < 19)])
        children_alive = len(df.loc[df['age_years'] < 19])
        self.numerator += n_in_RTI
        self.totinjured += n_in_RTI
        # How many were disabled
        n_perm_disabled = (df.is_alive & df.rt_perm_disability).sum()
        # self.permdis += n_perm_disabled
        n_alive = df.is_alive.sum()
        self.denominator += (n_alive - n_in_RTI) * (1 / 12)
        n_immediate_death = (df.rt_road_traffic_inc & df.rt_imm_death).sum()
        self.deathonscene += n_immediate_death
        diedfromrtiidx = df.index[df.rt_imm_death | df.rt_post_med_death | df.rt_no_med_death | df.rt_death_from_shock |
                                  df.rt_unavailable_med_death]
        n_sought_care = (df.rt_road_traffic_inc & df.rt_med_int).sum()
        self.soughtmedcare += n_sought_care
        n_death_post_med = df.rt_post_med_death.sum()
        self.deathaftermed += n_death_post_med
        self.deathwithoutmed += df.rt_no_med_death.sum()
        self.death_inc_numerator += n_immediate_death + n_death_post_med + len(df.loc[df.rt_no_med_death])
        self.death_in_denominator += (n_alive - (n_immediate_death + n_death_post_med + len(df.loc[df.rt_no_med_death])
                                                 )) * \
                                     (1 / 12)
        if self.numerator > 0:
            percent_accidents_result_in_death = \
                (self.deathonscene + self.deathaftermed + self.deathwithoutmed) / self.numerator
        else:
            percent_accidents_result_in_death = 'none injured'
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
        if (n_in_RTI - len(df.loc[df.rt_imm_death])) > 0:
            percent_sought_care = n_sought_care / (n_in_RTI - len(df.loc[df.rt_imm_death]))
        else:
            percent_sought_care = 'none_injured'

        if n_sought_care > 0:
            percent_died_post_care = n_death_post_med / n_sought_care
        else:
            percent_died_post_care = 'none_injured'

        if n_sought_care > 0:
            percentage_admitted_to_ICU_or_HDU = len(df.loc[df.rt_med_int & df.rt_in_icu_or_hdu]) / n_sought_care
        else:
            percentage_admitted_to_ICU_or_HDU = 'none_injured'
        if (n_alive - n_in_RTI) > 0:
            inc_rti = (n_in_RTI / ((n_alive - n_in_RTI) * (1 / 12))) * 100000
        else:
            inc_rti = 0
        if (children_alive - children_in_RTI) > 0:
            inc_rti_in_children = (children_in_RTI / ((children_alive - children_in_RTI) * (1 / 12))) * 100000
        else:
            inc_rti_in_children = 0
        if (n_alive - len(diedfromrtiidx)) > 0:
            inc_rti_death = (len(diedfromrtiidx) / ((n_alive - len(diedfromrtiidx)) * (1 / 12))) * 100000
        else:
            inc_rti_death = 0
        if (n_alive - len(df.loc[df.rt_post_med_death])) > 0:
            inc_post_med_death = (len(df.loc[df.rt_post_med_death]) / ((n_alive - len(df.loc[df.rt_post_med_death])) *
                                                                       (1 / 12))) * 100000
        else:
            inc_post_med_death = 0
        if (n_alive - len(df.loc[df.rt_imm_death])) > 0:
            inc_imm_death = (len(df.loc[df.rt_imm_death]) / ((n_alive - len(df.loc[df.rt_imm_death])) * (1 / 12))) * \
                            100000
        else:
            inc_imm_death = 0
        if (n_alive - len(df.loc[df.rt_no_med_death])) > 0:
            inc_death_no_med = (len(df.loc[df.rt_no_med_death]) /
                                ((n_alive - len(df.loc[df.rt_no_med_death])) * (1 / 12))) * 100000
        else:
            inc_death_no_med = 0
        if (n_alive - len(df.loc[df.rt_unavailable_med_death])) > 0:
            inc_death_unavailable_med = (len(df.loc[df.rt_unavailable_med_death]) /
                                         ((n_alive - len(df.loc[df.rt_unavailable_med_death])) * (1 / 12))) * 100000
        else:
            inc_death_unavailable_med = 0
        if self.fracdenominator > 0:
            frac_incidence = (self.totfracnumber / self.fracdenominator) * 100000
        else:
            frac_incidence = 0
        # calculate case fatality ratio for those injured who don't seek healthcare
        did_not_seek_healthcare = len(df.loc[df.rt_road_traffic_inc & ~df.rt_med_int & ~df.rt_diagnosed])
        died_no_healthcare = \
            len(df.loc[df.rt_road_traffic_inc & df.rt_no_med_death & ~df.rt_med_int & ~df.rt_diagnosed])
        if did_not_seek_healthcare > 0:
            cfr_no_med = died_no_healthcare / did_not_seek_healthcare
        else:
            cfr_no_med = 'all_sought_care'
        dict_to_output = {
            'number involved in a rti': n_in_RTI,
            'incidence of rti per 100,000': inc_rti,
            'incidence of rti per 100,000 in children': inc_rti_in_children,
            'incidence of rti death per 100,000': inc_rti_death,
            'incidence of death post med per 100,000': inc_post_med_death,
            'incidence of prehospital death per 100,000': inc_imm_death,
            'incidence of death without med per 100,000': inc_death_no_med,
            'incidence of death due to unavailable med per 100,000': inc_death_unavailable_med,
            'incidence of fractures per 100,000': frac_incidence,
            'number alive': n_alive,
            'number immediate deaths': n_immediate_death,
            'number deaths post med': n_death_post_med,
            'number deaths without med': len(df.loc[df.rt_no_med_death]),
            'number deaths unavailable med': len(df.loc[df.rt_unavailable_med_death]),
            'number rti deaths': len(diedfromrtiidx),
            'number permanently disabled': n_perm_disabled,
            'percent of crashes that are fatal': percent_accidents_result_in_death,
            'male:female ratio': mfratio,
            'percent sought healthcare': percent_sought_care,
            'percentage died after med': percent_died_post_care,
            'percent admitted to ICU or HDU': percentage_admitted_to_ICU_or_HDU,
            'cfr_no_med': cfr_no_med,
        }
        logger.info(key='summary_1m',
                    data=dict_to_output,
                    description='Summary of the rti injuries in the last month')
        # =========================== Get population demographics of those with RTIs ==================================
        columnsOfInterest = ['sex', 'age_years', 'li_ex_alc']
        injuredDemographics = df.loc[df.rt_road_traffic_inc]

        injuredDemographics = injuredDemographics.loc[:, columnsOfInterest]
        try:
            percent_related_to_alcohol = len(injuredDemographics.loc[injuredDemographics.li_ex_alc]) / \
                                         len(injuredDemographics)
        except ZeroDivisionError:
            percent_related_to_alcohol = 0
        injured_demography_summary = {
            'males_in_rti': injuredDemographics['sex'].value_counts()['M'],
            'females_in_rti': injuredDemographics['sex'].value_counts()['F'],
            'age': injuredDemographics['age_years'].values.tolist(),
            'male_age': injuredDemographics.loc[injuredDemographics['sex'] == 'M', 'age_years'].values.tolist(),
            'female_age': injuredDemographics.loc[injuredDemographics['sex'] == 'F', 'age_years'].values.tolist(),
            'percent_related_to_alcohol': percent_related_to_alcohol,
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
