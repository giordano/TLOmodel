"""This script is used to verify the output values from the natural history."""

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from tlo import Date, Simulation
from tlo.methods import (
    alri,
    demography,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    simplified_births,
    symptommanager,
)
from tlo.methods.alri import (
    AlriIncidentCase,
    AlriPollingEvent,
    AlriPropertiesOfOtherModules,
    HSI_Alri_Treatment,
    _make_hw_diagnosis_perfect,
    _make_treatment_and_diagnosis_perfect,
)
from tlo.util import sample_outcome

MODEL_POPSIZE = 15_000
MIN_SAMPLE_OF_NEW_CASES = 200
NUM_REPS_FOR_EACH_CASE = 20

_facility_level = '2'  # <-- assumes that the diagnosis/treatment occurs at this level


def get_sim(popsize):
    """Return a simulation (composed of only <5 years old) that has run for 0 days."""
    resourcefilepath = Path('./resources')
    start_date = Date(2010, 1, 1)
    sim = Simulation(start_date=start_date, seed=1)

    sim.register(
        demography.Demography(resourcefilepath=resourcefilepath),
        simplified_births.SimplifiedBirths(resourcefilepath=resourcefilepath),
        enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
        symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
        healthseekingbehaviour.HealthSeekingBehaviour(
            resourcefilepath=resourcefilepath,
            force_any_symptom_to_lead_to_healthcareseeking=True,
        ),
        healthburden.HealthBurden(resourcefilepath=resourcefilepath),
        healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                  cons_availability='all',
                                  ),
        alri.Alri(resourcefilepath=resourcefilepath),
        AlriPropertiesOfOtherModules(),
    )
    sim.modules['Demography'].parameters['max_age_initial'] = 5
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=sim.start_date)
    return sim


# Create simulation (This is needed to generate a population with representative characteristics and to initialise the
# Alri module.)

# Alri module with default values
sim0 = get_sim(popsize=MODEL_POPSIZE)
alri_module_with_imperfect_diagnosis_and_imperfect_treatment = sim0.modules['Alri']
hsi_with_imperfect_diagnosis_and_imperfect_treatment = HSI_Alri_Treatment(
    module=alri_module_with_imperfect_diagnosis_and_imperfect_treatment, person_id=None)

# Alri module with perfect diagnosis (and imperfect treatment)
sim1 = get_sim(popsize=MODEL_POPSIZE)
alri_module_with_perfect_diagnosis = sim1.modules['Alri']
_make_hw_diagnosis_perfect(alri_module_with_perfect_diagnosis)
hsi_with_perfect_diagnosis = HSI_Alri_Treatment(module=alri_module_with_perfect_diagnosis, person_id=None)


def generate_case_mix() -> pd.DataFrame:
    """Generate table of all the cases that may be created"""

    def get_incident_case_mix() -> pd.DataFrame:
        """Return a representative mix of new incidence alri cases (age, sex of person and the pathogen)."""

        alri_polling_event = AlriPollingEvent(module=alri_module_with_perfect_diagnosis)

        # Get probabilities for each person of being infected with each pathogen
        probs_of_acquiring_pathogen = alri_polling_event.get_probs_of_acquiring_pathogen(
            interval_as_fraction_of_a_year=1.0)

        # Sample who is infected and with what pathogen & Repeat 10 times with replacement to generate larger numbers
        new_alri = []
        while len(new_alri) < MIN_SAMPLE_OF_NEW_CASES:
            new_alri.extend(
                [(k, v) for k, v in
                 sample_outcome(probs=probs_of_acquiring_pathogen, rng=alri_module_with_perfect_diagnosis.rng).items()]
            )

        # Return dataframe in which person_id is replaced with age and sex (ignoring variation in vaccine /
        # under-nutrition).
        return pd.DataFrame(columns=['person_id', 'pathogen'], data=new_alri) \
            .merge(sim1.population.props[['age_exact_years', 'sex']],
                   right_index=True, left_on=['person_id'], how='left') \
            .drop(columns=['person_id'])

    def char_of_incident_case(sex,
                              age_exact_years,
                              pathogen,
                              va_hib_all_doses=False,
                              va_pneumo_all_doses=False,
                              un_clinical_acute_malnutrition="well",
                              ) -> dict:
        """Return the characteristics that are determined by IncidentCase (over 1000 iterations), given an infection
        caused by the pathogen"""
        incident_case = AlriIncidentCase(module=alri_module_with_perfect_diagnosis, person_id=None, pathogen=None)

        samples = []
        for _ in range(NUM_REPS_FOR_EACH_CASE):
            nature_of_char = incident_case.determine_nature_of_the_case(
                age_exact_years=age_exact_years,
                sex=sex,
                pathogen=pathogen,
                va_hib_all_doses=va_hib_all_doses,
                va_pneumo_all_doses=va_pneumo_all_doses,
                un_clinical_acute_malnutrition=un_clinical_acute_malnutrition,
            )

            samples.append({**nature_of_char, **{'sex': sex,
                                                 'age_exact_years': age_exact_years,
                                                 'pathogen': pathogen,
                                                 'va_hib_all_doses': va_hib_all_doses,
                                                 'va_pneumo_all_doses': va_pneumo_all_doses,
                                                 'un_clinical_acute_malnutrition': un_clinical_acute_malnutrition,
                                                 }
                            })

        return samples

    incident_case_mix = get_incident_case_mix()
    overall_case_mix = []  # This will be the mix of cases, representing the characteristics of the case as well as the
    #                        pathogen and who they are incident to.

    for x in incident_case_mix.itertuples():
        overall_case_mix.extend(
            char_of_incident_case(sex=x.sex, age_exact_years=x.age_exact_years, pathogen=x.pathogen)
        )

    return pd.DataFrame(overall_case_mix)


df = generate_case_mix()

# ------------------------------------------------------------------
# Verify the number of complications

# group by disease type: pneumonia vs other_alri
df_by_group = df.groupby('disease_type')
df_pneumonia = df_by_group.get_group('pneumonia')
df_other_alri = df_by_group.get_group('other_alri')

# total complications per group
df_pneumonia['complications'].apply(lambda x: 1 if len(x) != 0 else 0).sum()
df_other_alri['complications'].apply(lambda x: 1 if len(x) != 0 else 0).sum()

# total cases per group
total_pneumonia_cases = df_pneumonia.index.size
total_other_alri_cases = df_other_alri.index.size
total_alri_cases = df.index.size

# total complications in all ALRIs
total_alri_complications = df['complications'].apply(lambda x: 1 if len(x) != 0 else 0).sum()
prop_complicated_alri = total_alri_complications / total_alri_cases

# pulmonary complications in pneumonia group
total_pulmonary_complicated_pneum = df_pneumonia['complications'].apply(lambda x: 1 if any(
    e in ['pleural_effusion', 'empyema', 'lung_abscess', 'pmeumothorax'] for e in x) else 0).sum()
prop_pulmonary_complicated_pneumonia = total_pulmonary_complicated_pneum / total_pneumonia_cases

# sepsis in pneumonia group
total_sepsis = df_pneumonia['complications'].apply(lambda x: 1 if 'sepsis' in x else 0).sum()

# hypoxaemia in pneumonia group
tot_hypox_pneumonia = df_pneumonia['complications'].apply(lambda x: 1 if 'hypoxaemia' in x else 0).sum()
prop_hypox_pneumonia = tot_hypox_pneumonia / total_pneumonia_cases

# hypoxaemia in pneumonia group
tot_hypox_other_alri = df_other_alri['complications'].apply(lambda x: 1 if 'hypoxaemia' in x else 0).sum()
prop_hypox_other_alri = tot_hypox_other_alri / total_other_alri_cases

# all hypoxaemia in ALRI
tot_hypox_all_alri = df['complications'].apply(lambda x: 1 if 'hypoxaemia' in x else 0).sum()
prop_hypox_alri = tot_hypox_all_alri / total_alri_cases

# no sepsis and pulmonary complications in other_alri
assert df_other_alri['complications'].apply(lambda x: 1 if 'sepsis' in x else 0).sum() == 0
assert df_other_alri['complications'].apply(lambda x: 1 if any(
    e in ['pleural_effusion', 'empyema', 'lung_abscess', 'pmeumothorax'] for e in x) else 0).sum() == 0

# # # # # # # # # # # # # # # # # Symptoms # # # # # # # # # # # # # # # # #

# check if the symptoms match the classification

# proportion of danger signs

danger_signs_symptom = df['symptoms'].apply(lambda x: 1 if 'danger_signs' in x else 0).sum()
chest_indrawing_symptom = df['symptoms'].apply(lambda x: 1 if 'chest_indrawing' in x and
                                                              'danger_signs' not in x else 0).sum()
fast_breathing_symptom = df['symptoms'].apply(lambda x: 1 if 'tachypnoea' in x and 'danger_signs' not in x and
                                                             'chest_indrawing' not in x else 0).sum()
prop_danger_signs_alri = danger_signs_symptom / total_alri_cases
prop_chest_indrawing_alri = chest_indrawing_symptom / total_alri_cases
prop_fast_breathing_alri = fast_breathing_symptom / total_alri_cases

# danger signs in CXR+
danger_signs_in_pneumonia = df_pneumonia['symptoms'].apply(lambda x: 1 if 'danger_signs' in x else 0).sum()
prop_danger_signs_in_pneumonia = danger_signs_in_pneumonia / total_pneumonia_cases

# danger signs in CXR-
danger_signs_in_other_alri = df_other_alri['symptoms'].apply(lambda x: 1 if 'danger_signs' in x else 0).sum()
prop_danger_signs_in_other_alri = danger_signs_in_other_alri / total_other_alri_cases

# check the proportions of dangers signs in complications

# proportion danger signs in hypoxaemia in pneumonia (CXR+)
total_ds_in_hypoxaemia_pneumonia = df_pneumonia[['complications', 'symptoms']].apply(lambda x: 1 if ('hypoxaemia' in x[0] and 'danger_signs' in x[1]) else 0, axis=1).sum()
prop_total_ds_hypoxaemia_pneum = total_ds_in_hypoxaemia_pneumonia / tot_hypox_pneumonia
total_ds_in_no_hypoxaemia_pneumonia = df_pneumonia[['complications', 'symptoms']].apply(lambda x: 1 if ('hypoxaemia' not in x[0] and 'danger_signs' in x[1]) else 0, axis=1).sum()

# danger signs in sepsis
total_ds_in_sepsis_pneumonia = df_pneumonia[['complications', 'symptoms']].apply(lambda x: 1 if ('sepsis' in x[0] and 'danger_signs' in x[1]) else 0, axis=1).sum()
assert total_ds_in_sepsis_pneumonia == total_sepsis

# proportion danger signs in hypoxaemia in pneumonia (CXR+)
total_ds_in_hypoxaemia_other_alri = df_other_alri[['complications', 'symptoms']].apply(lambda x: 1 if ('hypoxaemia' in x[0] and 'danger_signs' in x[1]) else 0, axis=1).sum()
prop_total_ds_hypoxaemia_other_alri = total_ds_in_hypoxaemia_other_alri / tot_hypox_other_alri


# proportion danger signs in hypoxaemia in all ALRI
total_ds_in_hypoxaemia = df[['complications', 'symptoms']].apply(lambda x: 1 if ('hypoxaemia' in x[0] and 'danger_signs' in x[1]) else 0, axis=1).sum()
prop_total_ds_hypoxaemia_alri = total_ds_in_hypoxaemia / tot_hypox_all_alri

total_ds_in_spo2_90 = df[['oxygen_saturation', 'symptoms']].apply(lambda x: 1 if (x[0] == '<90%' and 'danger_signs' in x[1]) else 0, axis=1).sum()
total_no_ds_spo2_90 = df[['oxygen_saturation', 'symptoms']].apply(lambda x: 1 if (x[0] == '<90%' and not 'danger_signs' in x[1]) else 0, axis=1).sum()
total_SpO2_90 = df['oxygen_saturation'].apply(lambda x: 1 if '<90%' in x else 0).sum()
prop_total_ds_in_spo2_90 = total_ds_in_spo2_90 / total_SpO2_90  # 0.44 --> this means only 44% of SpO2<90% displayed danger signs, while 56% do Not have danger signs


total_danger_signs_and_SpO2_90 = df[['oxygen_saturation', 'symptoms']].apply(lambda x: 1 if (x[0] == '<90%' or 'danger_signs' in x[1]) else 0, axis=1).sum()
prop_SpO2_90_in_all_ds_pneumo = total_SpO2_90 / total_danger_signs_and_SpO2_90  # 0.42
prop_SpO2_90_with_ds_in_all_ds_pneumo = total_ds_in_spo2_90 / total_danger_signs_and_SpO2_90  # 0.1879
prop_SpO2_90_without_ds_in_all_ds_pneumo = total_no_ds_spo2_90 / total_danger_signs_and_SpO2_90  # 0.237  ---> this means 23.7% of SpO2<90% without danger signs would not be detected by perfect hw performance

total_danger_signs_and_without_SpO2_90 = df[['oxygen_saturation', 'symptoms']].apply(lambda x: 1 if (x[0] != '<90%' and 'danger_signs' in x[1]) else 0, axis=1).sum()
prop_no_SpO2_90_in_all_ds_pneumo = total_danger_signs_and_without_SpO2_90 / total_danger_signs_and_SpO2_90  # 0.575

# 0.188 (SpO2 with danger signs) + 0.237 (SpO2 without danger signs) + 0.575 (danger signs without SpO2) = 1

total_cold = df['symptoms'].apply(lambda x: 1 if 'danger_signs' not in x and 'tachypnoea' not in x and 'chest_indrawing' not in x else 0).sum()
prop_cold = total_cold / total_alri_cases

total_cold_in_complications = df[['complications', 'symptoms']].apply(
    lambda x: 1 if len(x[0]) != 0 and 'danger_signs' not in x[1] and 'tachypnoea'
                   not in x[1] and 'chest_indrawing' not in x[1] else 0, axis=1).sum()

prop_cold_in_complications = total_cold_in_complications / total_alri_complications  # 0.0408

total_cold_in_hypoxaemia = df[['complications', 'symptoms']].apply(
    lambda x: 1 if ('hypoxaemia' in x[0] and 'danger_signs' not in x[1] and 'tachypnoea'
                   not in x[1] and 'chest_indrawing' not in x[1]) else 0, axis=1).sum()

prop_cold_in_hypoxaemia = total_cold_in_hypoxaemia / tot_hypox_all_alri  # 0.043


# todo: what is the ovrall treatement failure in danger signs pneumonia - check for spo2<90 vs >90%
