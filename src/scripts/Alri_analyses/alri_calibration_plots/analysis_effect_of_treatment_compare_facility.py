"""This script will generate a table that describes a representative mix of all the IncidentCases that are created, and
 the associated diagnosis and risk of death for each under various conditions of treatments/non-treatment."""
import random
from pathlib import Path
from typing import List
import datetime

from _pytest.mark.expression import expression

from tlo.util import random_date, sample_outcome

import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors

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

# Helper function for conversion between odds and probabilities
to_odds = lambda pr: pr / (1.0 - pr)  # noqa: E731
to_prob = lambda odds: odds / (1.0 + odds)  # noqa: E731

# Date for saving the image for log-file
datestamp = datetime.date.today().strftime("__%Y_%m_%d")


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

# Alri module with perfect diagnosis and perfect treatment
sim2 = get_sim(popsize=MODEL_POPSIZE)
alri_module_with_perfect_treatment_and_diagnosis = sim2.modules['Alri']
_make_treatment_and_diagnosis_perfect(alri_module_with_perfect_treatment_and_diagnosis)


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


def treatment_efficacy(
    age_exact_years,
    symptoms,
    oxygen_saturation,
    disease_type,
    complications,
    oximeter_available,
    oxygen_available,
    treatment_perfect,
    hw_dx_perfect,
    facility_level
):
    """Return the percentage by which the treatment reduce the risk of death"""
    # Decide which hsi configuration to use:
    if hw_dx_perfect:
        hsi = hsi_with_perfect_diagnosis
        alri_module = alri_module_with_perfect_diagnosis
    else:
        hsi = hsi_with_imperfect_diagnosis_and_imperfect_treatment
        alri_module = alri_module_with_imperfect_diagnosis_and_imperfect_treatment

    # Get Treatment classification
    classification_for_treatment_decision = hsi._get_disease_classification_for_treatment_decision(
        age_exact_years=age_exact_years,
        symptoms=symptoms,
        oxygen_saturation=oxygen_saturation,
        facility_level=facility_level,
        use_oximeter=oximeter_available,
    )

    imci_symptom_based_classification = alri_module_with_perfect_diagnosis.get_imci_classification_based_on_symptoms(
        child_is_younger_than_2_months=(age_exact_years < 2.0 / 12.0),
        symptoms=symptoms, facility_level='2'
    )

    # Get the treatment selected based on classification given
    ultimate_treatment = alri_module._ultimate_treatment_indicated_for_patient(
        classification_for_treatment_decision=classification_for_treatment_decision,
        age_exact_years=age_exact_years,
        use_oximeter=oximeter_available,
        oxygen_saturation=oxygen_saturation,
    )

    # Decide which alri_module configuration to use:
    if treatment_perfect:
        alri_module = alri_module_with_perfect_treatment_and_diagnosis
    else:
        alri_module = alri_module_with_imperfect_diagnosis_and_imperfect_treatment

    # "Treatment Fails" is the probability that a death is averted (if one is schedule)
    treatment_fails = alri_module.models._prob_treatment_fails(
        antibiotic_provided=ultimate_treatment['antibiotic_indicated'][0],
        oxygen_provided=ultimate_treatment['oxygen_indicated'] if oxygen_available else False,
        imci_symptom_based_classification=imci_symptom_based_classification,
        SpO2_level=oxygen_saturation,
        disease_type=disease_type,
        age_exact_years=age_exact_years,
        symptoms=symptoms,
        complications=complications,
        hiv_infected_and_not_on_art=False,
        un_clinical_acute_malnutrition='well',
    )

    # for inpatients provide 2nd line IV antibiotic if 1st line failed
    if ultimate_treatment['antibiotic_indicated'][0].startswith('1st_line_IV'):
        treatment_fails = treatment_fails * (alri_module.models._prob_treatment_fails(
            antibiotic_provided='2nd_line_IV_flucloxacillin_gentamicin',
            oxygen_provided=ultimate_treatment['oxygen_indicated'] if oxygen_available else False,
            imci_symptom_based_classification=imci_symptom_based_classification,
            SpO2_level=oxygen_saturation,
            disease_type=disease_type,
            age_exact_years=age_exact_years,
            symptoms=symptoms,
            complications=complications,
            hiv_infected_and_not_on_art=False,
            un_clinical_acute_malnutrition='well',
        ))

    # Return percentage probability of treatment success
    return 100.0 * (1.0 - treatment_fails)


def generate_table():
    """Return table providing a representative case mix of persons with Alri, the intrinsic risk of death and the
    efficacy of treatment under different conditions."""

    # Get Case Mix
    df = generate_case_mix()

    # Consider risk of death for this person, intrinsically and under different conditions of treatments
    risk_of_death = list()
    for x in df.itertuples():
        risk_of_death.append({
            'prob_die_if_no_treatment': alri_module_with_perfect_diagnosis.models.prob_die_of_alri(
                age_exact_years=x.age_exact_years,
                sex=x.sex,
                bacterial_infection=x.pathogen,
                disease_type=x.disease_type,
                SpO2_level=x.oxygen_saturation,
                complications=x.complications,
                all_symptoms=x.symptoms,
                un_clinical_acute_malnutrition=x.un_clinical_acute_malnutrition,
            ),

            # Treatment Efficacy with * PERFECT HW Diagnosis * LEVEL 2 *
            f'treatment_efficacy_if_perfect_treatment_and_with_oximeter_and_oxygen_perfect_hw_dx_level2':
                treatment_efficacy(
                    # Information about the patient:
                    age_exact_years=x.age_exact_years,
                    symptoms=x.symptoms,
                    oxygen_saturation=x.oxygen_saturation,
                    disease_type=x.disease_type,
                    complications=x.complications,

                    # Information about the care that can be provided:
                    oximeter_available=True,
                    oxygen_available=True,
                    treatment_perfect=True,
                    hw_dx_perfect=True,
                    facility_level='2'
                ),

            f'treatment_efficacy_if_normal_treatment_and_with_oximeter_and_oxygen_perfect_hw_dx_level2':
                treatment_efficacy(
                    # Information about the patient:
                    age_exact_years=x.age_exact_years,
                    symptoms=x.symptoms,
                    oxygen_saturation=x.oxygen_saturation,
                    disease_type=x.disease_type,
                    complications=x.complications,

                    # Information about the care that can be provided:
                    oximeter_available=True,
                    oxygen_available=True,
                    treatment_perfect=False,
                    hw_dx_perfect=True,
                    facility_level='2'
                ),

            f'treatment_efficacy_if_normal_treatment_but_without_oximeter_or_oxygen_perfect_hw_dx_level2':
                treatment_efficacy(
                    # Information about the patient:
                    age_exact_years=x.age_exact_years,
                    symptoms=x.symptoms,
                    oxygen_saturation=x.oxygen_saturation,
                    disease_type=x.disease_type,
                    complications=x.complications,

                    # Information about the care that can be provided:
                    oximeter_available=False,
                    oxygen_available=False,
                    treatment_perfect=False,
                    hw_dx_perfect=True,
                    facility_level='2'
                ),

            f'treatment_efficacy_if_normal_treatment_and_with_oximeter_but_without_oxygen_perfect_hw_dx_level2':
                treatment_efficacy(
                    # Information about the patient:
                    age_exact_years=x.age_exact_years,
                    symptoms=x.symptoms,
                    oxygen_saturation=x.oxygen_saturation,
                    disease_type=x.disease_type,
                    complications=x.complications,

                    # Information about the care that can be provided:
                    oximeter_available=True,
                    oxygen_available=False,
                    treatment_perfect=False,
                    hw_dx_perfect=True,
                    facility_level='2'
                ),

            f'treatment_efficacy_if_normal_treatment_and_with_oxygen_but_without_oximeter_perfect_hw_dx_level2':
                treatment_efficacy(
                    # Information about the patient:
                    age_exact_years=x.age_exact_years,
                    symptoms=x.symptoms,
                    oxygen_saturation=x.oxygen_saturation,
                    disease_type=x.disease_type,
                    complications=x.complications,

                    # Information about the care that can be provided:
                    oximeter_available=False,
                    oxygen_available=True,
                    treatment_perfect=False,
                    hw_dx_perfect=True,
                    facility_level='2'
                ),

            # Treatment Efficacy with * IMPERFECT HW Diangosis * LEVEL 2 *
            f'treatment_efficacy_if_normal_treatment_and_with_oximeter_and_oxygen_imperfect_hw_dx_level2':
                treatment_efficacy(
                    # Information about the patient:
                    age_exact_years=x.age_exact_years,
                    symptoms=x.symptoms,
                    oxygen_saturation=x.oxygen_saturation,
                    disease_type=x.disease_type,
                    complications=x.complications,

                    # Information about the care that can be provided:
                    oximeter_available=True,
                    oxygen_available=True,
                    treatment_perfect=False,
                    hw_dx_perfect=False,
                    facility_level='2'
                ),

            f'treatment_efficacy_if_normal_treatment_but_without_oximeter_or_oxygen_imperfect_hw_dx_level2':
                treatment_efficacy(
                    # Information about the patient:
                    age_exact_years=x.age_exact_years,
                    symptoms=x.symptoms,
                    oxygen_saturation=x.oxygen_saturation,
                    disease_type=x.disease_type,
                    complications=x.complications,

                    # Information about the care that can be provided:
                    oximeter_available=False,
                    oxygen_available=False,
                    treatment_perfect=False,
                    hw_dx_perfect=False,
                    facility_level='2'
                ),

            f'treatment_efficacy_if_normal_treatment_and_with_oximeter_but_without_oxygen_imperfect_hw_dx_level2':
                treatment_efficacy(
                    # Information about the patient:
                    age_exact_years=x.age_exact_years,
                    symptoms=x.symptoms,
                    oxygen_saturation=x.oxygen_saturation,
                    disease_type=x.disease_type,
                    complications=x.complications,

                    # Information about the care that can be provided:
                    oximeter_available=True,
                    oxygen_available=False,
                    treatment_perfect=False,
                    hw_dx_perfect=False,
                    facility_level='2'
                ),

            f'treatment_efficacy_if_normal_treatment_and_with_oxygen_but_without_oximeter_imperfect_hw_dx_level2':
                treatment_efficacy(
                    # Information about the patient:
                    age_exact_years=x.age_exact_years,
                    symptoms=x.symptoms,
                    oxygen_saturation=x.oxygen_saturation,
                    disease_type=x.disease_type,
                    complications=x.complications,

                    # Information about the care that can be provided:
                    oximeter_available=False,
                    oxygen_available=True,
                    treatment_perfect=False,
                    hw_dx_perfect=False,
                    facility_level='2'
                ),

            # # # # # APPLY FOR LEVEL 1A # # # # #
            # Treatment Efficacy with * PERFECT HW Diagnosis * LEVEL 1A *
            f'treatment_efficacy_if_perfect_treatment_and_with_oximeter_and_oxygen_perfect_hw_dx_level1a':
                treatment_efficacy(
                    # Information about the patient:
                    age_exact_years=x.age_exact_years,
                    symptoms=x.symptoms,
                    oxygen_saturation=x.oxygen_saturation,
                    disease_type=x.disease_type,
                    complications=x.complications,

                    # Information about the care that can be provided:
                    oximeter_available=True,
                    oxygen_available=True,
                    treatment_perfect=True,
                    hw_dx_perfect=True,
                    facility_level='1a'
                ),

            f'treatment_efficacy_if_normal_treatment_and_with_oximeter_and_oxygen_perfect_hw_dx_level1a':
                treatment_efficacy(
                    # Information about the patient:
                    age_exact_years=x.age_exact_years,
                    symptoms=x.symptoms,
                    oxygen_saturation=x.oxygen_saturation,
                    disease_type=x.disease_type,
                    complications=x.complications,

                    # Information about the care that can be provided:
                    oximeter_available=True,
                    oxygen_available=True,
                    treatment_perfect=False,
                    hw_dx_perfect=True,
                    facility_level='1a'
                ),

            f'treatment_efficacy_if_normal_treatment_but_without_oximeter_or_oxygen_perfect_hw_dx_level1a':
                treatment_efficacy(
                    # Information about the patient:
                    age_exact_years=x.age_exact_years,
                    symptoms=x.symptoms,
                    oxygen_saturation=x.oxygen_saturation,
                    disease_type=x.disease_type,
                    complications=x.complications,

                    # Information about the care that can be provided:
                    oximeter_available=False,
                    oxygen_available=False,
                    treatment_perfect=False,
                    hw_dx_perfect=True,
                    facility_level='1a'
                ),

            f'treatment_efficacy_if_normal_treatment_and_with_oximeter_but_without_oxygen_perfect_hw_dx_level1a':
                treatment_efficacy(
                    # Information about the patient:
                    age_exact_years=x.age_exact_years,
                    symptoms=x.symptoms,
                    oxygen_saturation=x.oxygen_saturation,
                    disease_type=x.disease_type,
                    complications=x.complications,

                    # Information about the care that can be provided:
                    oximeter_available=True,
                    oxygen_available=False,
                    treatment_perfect=False,
                    hw_dx_perfect=True,
                    facility_level='1a'
                ),

            f'treatment_efficacy_if_normal_treatment_and_with_oxygen_but_without_oximeter_perfect_hw_dx_level1a':
                treatment_efficacy(
                    # Information about the patient:
                    age_exact_years=x.age_exact_years,
                    symptoms=x.symptoms,
                    oxygen_saturation=x.oxygen_saturation,
                    disease_type=x.disease_type,
                    complications=x.complications,

                    # Information about the care that can be provided:
                    oximeter_available=False,
                    oxygen_available=True,
                    treatment_perfect=False,
                    hw_dx_perfect=True,
                    facility_level='1a'
                ),

            # Treatment Efficacy with * IMPERFECT HW Diangosis * LEVEL 1A *
            f'treatment_efficacy_if_normal_treatment_and_with_oximeter_and_oxygen_imperfect_hw_dx_level1a':
                treatment_efficacy(
                    # Information about the patient:
                    age_exact_years=x.age_exact_years,
                    symptoms=x.symptoms,
                    oxygen_saturation=x.oxygen_saturation,
                    disease_type=x.disease_type,
                    complications=x.complications,

                    # Information about the care that can be provided:
                    oximeter_available=True,
                    oxygen_available=True,
                    treatment_perfect=False,
                    hw_dx_perfect=False,
                    facility_level='1a'
                ),

            f'treatment_efficacy_if_normal_treatment_but_without_oximeter_or_oxygen_imperfect_hw_dx_level1a':
                treatment_efficacy(
                    # Information about the patient:
                    age_exact_years=x.age_exact_years,
                    symptoms=x.symptoms,
                    oxygen_saturation=x.oxygen_saturation,
                    disease_type=x.disease_type,
                    complications=x.complications,

                    # Information about the care that can be provided:
                    oximeter_available=False,
                    oxygen_available=False,
                    treatment_perfect=False,
                    hw_dx_perfect=False,
                    facility_level='1a'
                ),

            f'treatment_efficacy_if_normal_treatment_and_with_oximeter_but_without_oxygen_imperfect_hw_dx_level1a':
                treatment_efficacy(
                    # Information about the patient:
                    age_exact_years=x.age_exact_years,
                    symptoms=x.symptoms,
                    oxygen_saturation=x.oxygen_saturation,
                    disease_type=x.disease_type,
                    complications=x.complications,

                    # Information about the care that can be provided:
                    oximeter_available=True,
                    oxygen_available=False,
                    treatment_perfect=False,
                    hw_dx_perfect=False,
                    facility_level='1a'
                ),

            f'treatment_efficacy_if_normal_treatment_and_with_oxygen_but_without_oximeter_imperfect_hw_dx_level1a':
                treatment_efficacy(
                    # Information about the patient:
                    age_exact_years=x.age_exact_years,
                    symptoms=x.symptoms,
                    oxygen_saturation=x.oxygen_saturation,
                    disease_type=x.disease_type,
                    complications=x.complications,

                    # Information about the care that can be provided:
                    oximeter_available=False,
                    oxygen_available=True,
                    treatment_perfect=False,
                    hw_dx_perfect=False,
                    facility_level='1a'
                ),

            # # # # # # * Classifications * # # # # # #
            # * CLASSIFICATION BY LEVEL 2 *
            'classification_for_treatment_decision_with_oximeter_perfect_accuracy_level2':
                hsi_with_perfect_diagnosis._get_disease_classification_for_treatment_decision(
                    age_exact_years=x.age_exact_years,
                    symptoms=x.symptoms,
                    oxygen_saturation=x.oxygen_saturation,
                    facility_level='2',
                    use_oximeter=True,
                ),

            'classification_for_treatment_decision_without_oximeter_perfect_accuracy_level2':
                hsi_with_perfect_diagnosis._get_disease_classification_for_treatment_decision(
                    age_exact_years=x.age_exact_years,
                    symptoms=x.symptoms,
                    oxygen_saturation=x.oxygen_saturation,
                    facility_level='2',
                    use_oximeter=False,
                ),

            'classification_for_treatment_decision_with_oximeter_imperfect_accuracy_level2':
                hsi_with_imperfect_diagnosis_and_imperfect_treatment._get_disease_classification_for_treatment_decision(
                    age_exact_years=x.age_exact_years,
                    symptoms=x.symptoms,
                    oxygen_saturation=x.oxygen_saturation,
                    facility_level='2',
                    use_oximeter=True,
                ),

            'classification_for_treatment_decision_without_oximeter_imperfect_accuracy_level2':
                hsi_with_imperfect_diagnosis_and_imperfect_treatment._get_disease_classification_for_treatment_decision(
                    age_exact_years=x.age_exact_years,
                    symptoms=x.symptoms,
                    oxygen_saturation=x.oxygen_saturation,
                    facility_level='2',
                    use_oximeter=False,
                ),

            # * CLASSIFICATION BY LEVEL 1a *
            'classification_for_treatment_decision_with_oximeter_perfect_accuracy_level1a':
                hsi_with_perfect_diagnosis._get_disease_classification_for_treatment_decision(
                    age_exact_years=x.age_exact_years,
                    symptoms=x.symptoms,
                    oxygen_saturation=x.oxygen_saturation,
                    facility_level='1a',
                    use_oximeter=True,
                ),

            'classification_for_treatment_decision_without_oximeter_perfect_accuracy_level1a':
                hsi_with_perfect_diagnosis._get_disease_classification_for_treatment_decision(
                    age_exact_years=x.age_exact_years,
                    symptoms=x.symptoms,
                    oxygen_saturation=x.oxygen_saturation,
                    facility_level='1a',
                    use_oximeter=False,
                ),

            'classification_for_treatment_decision_with_oximeter_imperfect_accuracy_level1a':
                hsi_with_imperfect_diagnosis_and_imperfect_treatment._get_disease_classification_for_treatment_decision(
                    age_exact_years=x.age_exact_years,
                    symptoms=x.symptoms,
                    oxygen_saturation=x.oxygen_saturation,
                    facility_level='1a',
                    use_oximeter=True,
                ),

            'classification_for_treatment_decision_without_oximeter_imperfect_accuracy_level1a':
                hsi_with_imperfect_diagnosis_and_imperfect_treatment._get_disease_classification_for_treatment_decision(
                    age_exact_years=x.age_exact_years,
                    symptoms=x.symptoms,
                    oxygen_saturation=x.oxygen_saturation,
                    facility_level='1a',
                    use_oximeter=False,
                ),

        })
    return df.join(pd.DataFrame(risk_of_death))


if __name__ == "__main__":
    table = generate_table()
    table = table.assign(
        has_danger_signs=lambda df: df['symptoms'].apply(lambda x: 'danger_signs' in x),
        needs_oxygen=lambda df: df['oxygen_saturation'] == "<90%",
    )


    def summarize_by(df: pd.DataFrame, by: List[str], columns: [List[str]]) -> pd.DataFrame:
        """Helper function returns dataframe that summarizes the dataframe provided using groupby, with by arguements,
        and provides columns as follows: [size-of-the-group, mean-of-column-1, mean-of-column-2, ...]"""
        return pd.DataFrame({'fraction': df.groupby(by=by).size()}).apply(lambda x: x / x.sum(), axis=0) \
            .join(df.groupby(by=by)[columns].mean())


    # Examine case mix
    case_mix_by_disease_and_pathogen = summarize_by(table,
                                                    by=['disease_type', 'pathogen'],
                                                    columns=['has_danger_signs', 'needs_oxygen',
                                                             'prob_die_if_no_treatment']
                                                    ).reset_index()
    print(f"{case_mix_by_disease_and_pathogen=}")

    # Average risk of death by disease type
    table.groupby(by=['disease_type'])['prob_die_if_no_treatment'].mean()

    # Examine danger_signs as a predictor of SpO2 < 90%
    print(pd.crosstab(
        pd.Series(table['oxygen_saturation'] == "<90%", name='SpO2<90%'),
        table['has_danger_signs'],
        normalize="index", margins=True
    ))

    # Look at diagnosis errors (assuming that the "truth" is
    # "classification_for_treatment_decision_with_oximeter_perfect_accuracy")
    truth = table["classification_for_treatment_decision_with_oximeter_perfect_accuracy_level2"]


    def cross_tab(truth: pd.Series, dx: pd.Series):
        """Return cross-tab between truth and dx and count number of incongruent rows."""
        print(f"% cases mis-classified is {100 * (truth != dx).mean()}%")
        return pd.crosstab(truth, dx, normalize='index')

    # Examine risk of death and treatment effectiveness by disease classification and oxygen saturation
    # And include estimate of % of deaths by scaling using "prob_die_if_no_treatment".
    # UNDER PERFECT HW DX
    risk_of_death = summarize_by(
        df=table,
        by=['oxygen_saturation', 'classification_for_treatment_decision_without_oximeter_perfect_accuracy_level2'],
        columns=[
            'prob_die_if_no_treatment',
            'treatment_efficacy_if_normal_treatment_and_with_oximeter_and_oxygen_perfect_hw_dx_level2',
            'treatment_efficacy_if_normal_treatment_but_without_oximeter_or_oxygen_perfect_hw_dx_level2',
            'treatment_efficacy_if_normal_treatment_and_with_oximeter_but_without_oxygen_perfect_hw_dx_level2',
            'treatment_efficacy_if_normal_treatment_and_with_oxygen_but_without_oximeter_perfect_hw_dx_level2',
            'treatment_efficacy_if_normal_treatment_and_with_oximeter_and_oxygen_perfect_hw_dx_level1a',
            'treatment_efficacy_if_normal_treatment_but_without_oximeter_or_oxygen_perfect_hw_dx_level1a',
            'treatment_efficacy_if_normal_treatment_and_with_oximeter_but_without_oxygen_perfect_hw_dx_level1a',
            'treatment_efficacy_if_normal_treatment_and_with_oxygen_but_without_oximeter_perfect_hw_dx_level1a',
        ]
    ).assign(
        fraction_of_deaths=lambda df: (
            (df.fraction * df.prob_die_if_no_treatment) / (df.fraction * df.prob_die_if_no_treatment).sum()
        )
    )
    print(f"{risk_of_death=}")

    # risk of deaths if no treatment
    (risk_of_death.fraction * risk_of_death.prob_die_if_no_treatment).sum()  # 0.0342 -- 3.42%

    # risk of death with treatment with oximeter_and_oxygen_perfect_hw_dx
    (risk_of_death.fraction * risk_of_death.prob_die_if_no_treatment * (
        (
                100 - risk_of_death.treatment_efficacy_if_normal_treatment_and_with_oximeter_and_oxygen_perfect_hw_dx_level2) / 100)
     ).sum()  # 0.00604 -- 0.06% deaths

    # Overall summary figure: Number of deaths in the cohort Deaths broken down by ( disease / oxygen_saturation) when
    # * No Treatment
    # * Treatment antibiotics only (no oxygen, no oximeter)
    # * Treatment with antibiotics and oxygen (no oximeter)
    # * Treatment with antibiotics and oxygen and use of oximeter
    # ... under assumptions of (i) normal treatment and normal dx accuracy; (ii) normal treatment and perfect dx
    # accuracy

    # disease_classification2 = table["classification_for_treatment_decision_with_oximeter_perfect_accuracy_level2"]
    disease_classification1a = table["classification_for_treatment_decision_with_oximeter_perfect_accuracy_level1a"]
    disease_classification2 = table['classification_for_treatment_decision_without_oximeter_perfect_accuracy_level2']
    low_oxygen = (table["oxygen_saturation"] == "<90%").replace({True: '<90%', False: ">=90%"})
    # low_oxygen = (table["oxygen_saturation"])
    fraction = risk_of_death['fraction']
    number_cases = table.groupby(by=[disease_classification2, low_oxygen]).size()
    dx_accuracy = 'perfect'

    res = {
        "Facility level 2": {
            # "No Treatment": table['prob_die_if_no_treatment'].groupby(by=[disease_classification2, low_oxygen]).sum(),
            "Antiobiotics only": (
                table['prob_die_if_no_treatment'] *
                (1.0 -
                 table[
                     f'treatment_efficacy_if_normal_treatment_but_without_oximeter_or_oxygen_{dx_accuracy}_hw_dx_level2'] / 100.0
                 )).groupby(by=[disease_classification2, low_oxygen]).sum(),
            "+ oxygen": (
                table['prob_die_if_no_treatment'] *
                (1.0 -
                 table[
                     f'treatment_efficacy_if_normal_treatment_and_with_oxygen_but_without_oximeter_{dx_accuracy}_hw_dx_level2']
                 / 100.0
                 )).groupby(by=[disease_classification2, low_oxygen]).sum(),
            "+ oximeter": (
                table['prob_die_if_no_treatment'] *
                (1.0 -
                 table[
                     f'treatment_efficacy_if_normal_treatment_and_with_oximeter_but_without_oxygen_{dx_accuracy}_hw_dx_level2'] / 100.0
                 )).groupby(by=[disease_classification2, low_oxygen]).sum(),
            "+ oximeter & oxygen": (
                table['prob_die_if_no_treatment'] *
                (1.0 -
                 table[
                     f'treatment_efficacy_if_normal_treatment_and_with_oximeter_and_oxygen_{dx_accuracy}_hw_dx_level2'] / 100.0
                 )).groupby(by=[disease_classification2, low_oxygen]).sum(),
        },

        "Facility level 1a": {
            # "No Treatment": table['prob_die_if_no_treatment'].groupby(by=[disease_classification2, low_oxygen]).sum(),
            "Antiobiotics only": (
                table['prob_die_if_no_treatment'] *
                (1.0 -
                 table[
                     f'treatment_efficacy_if_normal_treatment_but_without_oximeter_or_oxygen_{dx_accuracy}_hw_dx_level1a'] / 100.0
                 )).groupby(by=[disease_classification2, low_oxygen]).sum(),
            "+ oxygen": (
                table['prob_die_if_no_treatment'] *
                (1.0 -
                 table[
                     f'treatment_efficacy_if_normal_treatment_and_with_oxygen_but_without_oximeter_{dx_accuracy}_hw_dx_level1a']
                 / 100.0
                 )).groupby(by=[disease_classification2, low_oxygen]).sum(),
            "+ oximeter": (
                table['prob_die_if_no_treatment'] *
                (1.0 -
                 table[
                     f'treatment_efficacy_if_normal_treatment_and_with_oximeter_but_without_oxygen_{dx_accuracy}_hw_dx_level1a'] / 100.0
                 )).groupby(by=[disease_classification2, low_oxygen]).sum(),
            "+ oximeter & oxygen": (
                table['prob_die_if_no_treatment'] *
                (1.0 -
                 table[
                     f'treatment_efficacy_if_normal_treatment_and_with_oximeter_and_oxygen_{dx_accuracy}_hw_dx_level1a'] / 100.0
                 )).groupby(by=[disease_classification2, low_oxygen]).sum(),
        },
    }

    results = (100_000 / len(table)) * pd.concat({k: pd.DataFrame(v) for k, v in res.items()}, axis=1)

    # # Plot graph by * classification differences *
    # # compare the classification differences
    # res1 = {
    #     "By hospital classification": {
    #         "No Treatment": table['prob_die_if_no_treatment'].groupby(by=[disease_classification2, low_oxygen]).sum(),
    #         "Antiobiotics only": (
    #             table['prob_die_if_no_treatment'] *
    #             (1.0 -
    #              table[
    #                  'treatment_efficacy_if_normal_treatment_but_without_oximeter_or_oxygen_perfect_hw_dx_level1a'] / 100.0
    #              )).groupby(by=[disease_classification2, low_oxygen]).sum(),
    #         "+ oxygen": (
    #             table['prob_die_if_no_treatment'] *
    #             (1.0 -
    #              table[
    #                  'treatment_efficacy_if_normal_treatment_and_with_oxygen_but_without_oximeter_perfect_hw_dx_level1a']
    #              / 100.0
    #              )).groupby(by=[disease_classification2, low_oxygen]).sum(),
    #         "+ oximeter": (
    #             table['prob_die_if_no_treatment'] *
    #             (1.0 -
    #              table[
    #                  'treatment_efficacy_if_normal_treatment_and_with_oximeter_but_without_oxygen_perfect_hw_dx_level1a'] / 100.0
    #              )).groupby(by=[disease_classification2, low_oxygen]).sum(),
    #         "+ oximeter & oxygen": (
    #             table['prob_die_if_no_treatment'] *
    #             (1.0 -
    #              table[
    #                  'treatment_efficacy_if_normal_treatment_and_with_oximeter_and_oxygen_perfect_hw_dx_level1a'] / 100.0
    #              )).groupby(by=[disease_classification2, low_oxygen]).sum(),
    #     },
    #
    #     "By health centre classification": {
    #         "No Treatment": table['prob_die_if_no_treatment'].groupby(by=[disease_classification1a, low_oxygen]).sum(),
    #         "Antiobiotics only": (
    #             table['prob_die_if_no_treatment'] *
    #             (1.0 -
    #              table[
    #                  'treatment_efficacy_if_normal_treatment_but_without_oximeter_or_oxygen_perfect_hw_dx_level1a'] / 100.0
    #              )).groupby(by=[disease_classification1a, low_oxygen]).sum(),
    #         "+ oxygen": (
    #             table['prob_die_if_no_treatment'] *
    #             (1.0 -
    #              table[
    #                  'treatment_efficacy_if_normal_treatment_and_with_oxygen_but_without_oximeter_perfect_hw_dx_level1a']
    #              / 100.0
    #              )).groupby(by=[disease_classification1a, low_oxygen]).sum(),
    #         "+ oximeter": (
    #             table['prob_die_if_no_treatment'] *
    #             (1.0 -
    #              table[
    #                  'treatment_efficacy_if_normal_treatment_and_with_oximeter_but_without_oxygen_perfect_hw_dx_level1a'] / 100.0
    #              )).groupby(by=[disease_classification1a, low_oxygen]).sum(),
    #         "+ oximeter & oxygen": (
    #             table['prob_die_if_no_treatment'] *
    #             (1.0 -
    #              table[
    #                  'treatment_efficacy_if_normal_treatment_and_with_oximeter_and_oxygen_perfect_hw_dx_level1a'] / 100.0
    #              )).groupby(by=[disease_classification1a, low_oxygen]).sum(),
    #     },
    # }
    #
    # results = (100_000 / len(table)) * pd.concat({k: pd.DataFrame(v) for k, v in res1.items()}, axis=1)

    # reorder the index:
    reorderlist = list()
    if results.index.size == 12:  # broken down by 3 SpO2 levels
        # rename the index - to shorten the labels
        results.index = pd.MultiIndex.from_tuples(
            [(x[0], f'SpO2_{x[1]}') if x[1] == '90-92%' else (x[0], x[1]) for x in results.index],
            names=("classification_for_treatment_decision_without_oximeter_perfect_accuracy", "oxygen_saturation"))
        results.index = pd.MultiIndex.from_tuples(
            [('cough/cold', f'SpO2{x[1]}') if x[0] == 'cough_or_cold' else (x[0], x[1]) for x in results.index],
            names=("classification_for_treatment_decision_without_oximeter_perfect_accuracy", "oxygen_saturation"))
        results.index = pd.MultiIndex.from_tuples(
            [('fb-pneumonia', f'SpO2{x[1]}') if x[0] == 'fast_breathing_pneumonia' else (x[0], x[1]) for x in results.index],
            names=("classification_for_treatment_decision_without_oximeter_perfect_accuracy", "oxygen_saturation"))
        results.index = pd.MultiIndex.from_tuples(
            [('ci-pneumonia', f'SpO2{x[1]}') if x[0] == 'chest_indrawing_pneumonia' else (x[0], x[1]) for x in results.index],
            names=("classification_for_treatment_decision_without_oximeter_perfect_accuracy", "oxygen_saturation"))
        results.index = pd.MultiIndex.from_tuples(
            [('ds-pneumonia', f'SpO2{x[1]}') if x[0] == 'danger_signs_pneumonia' else (x[0], x[1]) for x in results.index],
            names=("classification_for_treatment_decision_without_oximeter_perfect_accuracy", "oxygen_saturation"))

        reorderlist = [('cough/cold', 'SpO2>=93%'),
                       ('cough/cold', 'SpO2_90-92%'),
                       ('cough/cold', 'SpO2<90%'),
                       ('fb-pneumonia', 'SpO2>=93%'),
                       ('fb-pneumonia', 'SpO2_90-92%'),
                       ('fb-pneumonia', 'SpO2<90%'),
                       ('ci-pneumonia', 'SpO2>=93%'),
                       ('ci-pneumonia', 'SpO2_90-92%'),
                       ('ci-pneumonia', 'SpO2<90%'),
                       ('ds-pneumonia', 'SpO2>=93%'),
                       ('ds-pneumonia', 'SpO2_90-92%'),
                       ('ds-pneumonia', 'SpO2<90%'),
                       ]

    elif results.index.size == 5:
        # rename the index for danger_signs_pneumonia with SpO2<90% to any_severity,
        # only applicable to current policy (inpatient care to SpO2<90%) results
        idx_to_change = ('danger_signs_pneumonia', '<90%')
        results.index = pd.MultiIndex.from_tuples(
            [('any severity', f'SpO2{x[1]}') if x == idx_to_change else (x[0], x[1]) for x in results.index],
            names=("classification_for_treatment_decision_with_oximeter_perfect_accuracy", "oxygen_saturation"))
        results.index = pd.MultiIndex.from_tuples(
            [('cough/cold', f'SpO2{x[1]}') if x[0] == 'cough_or_cold' else (x[0], x[1]) for x in results.index],
            names=("classification_for_treatment_decision_with_oximeter_perfect_accuracy", "oxygen_saturation"))
        results.index = pd.MultiIndex.from_tuples(
            [('fb-pneumonia', f'SpO2{x[1]}') if x[0] == 'fast_breathing_pneumonia' else (x[0], x[1]) for x in results.index],
            names=("classification_for_treatment_decision_with_oximeter_perfect_accuracy", "oxygen_saturation"))
        results.index = pd.MultiIndex.from_tuples(
            [('ci-pneumonia', f'SpO2{x[1]}') if x[0] == 'chest_indrawing_pneumonia' else (x[0], x[1]) for x in results.index],
            names=("classification_for_treatment_decision_with_oximeter_perfect_accuracy", "oxygen_saturation"))
        results.index = pd.MultiIndex.from_tuples(
            [('ds-pneumonia', f'SpO2{x[1]}') if x[0] == 'danger_signs_pneumonia' else (x[0], x[1]) for x in results.index],
            names=("classification_for_treatment_decision_with_oximeter_perfect_accuracy", "oxygen_saturation"))

        reorderlist = [('cough/cold', 'SpO2>=90%'),
                       ('fb-pneumonia', 'SpO2>=90%'),
                       ('ci-pneumonia', 'SpO2>=90%'),
                       ('ds-pneumonia', 'SpO2>=90%'),
                       ('any severity', 'SpO2<90%')
                       ]

    elif results.index.size == 8:
        results.index = pd.MultiIndex.from_tuples(
            [(x[0], '>=90%') if x[1] == '90-92%' else (x[0], x[1]) for x in results.index],
            names=("classification_for_treatment_decision_without_oximeter_perfect_accuracy", "oxygen_saturation"))
        results.index = pd.MultiIndex.from_tuples(
            [('cough/cold', f'SpO2{x[1]}') if x[0] == 'cough_or_cold' else (x[0], x[1]) for x in results.index],
            names=("classification_for_treatment_decision_without_oximeter_perfect_accuracy", "oxygen_saturation"))
        results.index = pd.MultiIndex.from_tuples(
            [('fb-pneumonia', f'SpO2{x[1]}') if x[0] == 'fast_breathing_pneumonia' else (x[0], x[1]) for x in results.index],
            names=("classification_for_treatment_decision_without_oximeter_perfect_accuracy", "oxygen_saturation"))
        results.index = pd.MultiIndex.from_tuples(
            [('ci-pneumonia', f'SpO2{x[1]}') if x[0] == 'chest_indrawing_pneumonia' else (x[0], x[1]) for x in results.index],
            names=("classification_for_treatment_decision_without_oximeter_perfect_accuracy", "oxygen_saturation"))
        results.index = pd.MultiIndex.from_tuples(
            [('ds-pneumonia', f'SpO2{x[1]}') if x[0] == 'danger_signs_pneumonia' else (x[0], x[1]) for x in results.index],
            names=("classification_for_treatment_decision_without_oximeter_perfect_accuracy", "oxygen_saturation"))

        reorderlist = [('cough/cold', 'SpO2>=90%'),
                       ('cough/cold', 'SpO2<90%'),
                       ('fb-pneumonia', 'SpO2>=90%'),
                       ('fb-pneumonia', 'SpO2<90%'),
                       ('ci-pneumonia', 'SpO2>=90%'),
                       ('ci-pneumonia', 'SpO2<90%'),
                       ('ds-pneumonia', 'SpO2>=90%'),
                       ('ds-pneumonia', 'SpO2<90%')
                       ]
    else:
        raise ValueError(f'Index size not recognised {results.index.size}')

    results.reindex(reorderlist)
    results = results.iloc[pd.Categorical(results.index, reorderlist).argsort(ascending=False)]

    # index = results.index.tolist()  -- > copy this to match the multiindex

    assign_colors = dict()

    if results.index.size == 12:
        assign_colors = {('cough/_cold', 'SpO2>=93%'): '#fa8405',
                         ('cough/cold', 'SpO2_90-92%'): 'gold',
                         ('cough/cold', 'SpO2<90%'): 'navajowhite',
                         ('fb-pneumonia', 'SpO2>=93%'): 'navy',
                         ('fb-pneumonia', 'SpO2_90-92%'): 'blue',
                         ('fb-pneumonia', 'SpO2<90%'): 'deepskyblue',
                         ('ci-pneumonia', 'SpO2>=93%'): 'darkgreen',
                         ('ci-pneumonia', 'SpO2_90-92%'): 'seagreen',
                         ('ci-pneumonia', 'SpO2<90%'): 'lightgreen',
                         ('ds-pneumonia', 'SpO2>=93%'): 'darkred',
                         ('ds-pneumonia', 'SpO2_90-92%'): 'orangered',
                         ('ds-pneumonia', 'SpO2<90%'): 'darksalmon',
                         }
    elif results.index.size == 5:
        assign_colors = {('cough/cold', 'SpO2>=90%'): 'tab:orange',
                         ('fb-pneumonia', 'SpO2>=90%'): 'mediumblue',
                         ('ci-pneumonia', 'SpO2>=90%'): 'tab:green',
                         ('ds-pneumonia', 'SpO2>=90%'): 'firebrick',
                         ('any severity', 'SpO2<90%'): 'palevioletred'
                         }

    elif results.index.size == 8:
        assign_colors = {('cough/cold', 'SpO2>=90%'): 'tab:orange',
                         ('cough/cold', 'SpO2<90%'): 'navajowhite',
                         ('fb-pneumonia', 'SpO2>=90%'): 'mediumblue',
                         ('fb-pneumonia', 'SpO2<90%'): 'deepskyblue',
                         ('ci-pneumonia', 'SpO2>=90%'): 'tab:green',
                         ('ci-pneumonia', 'SpO2<90%'): 'lightgreen',
                         ('ds-pneumonia', 'SpO2>=90%'): 'firebrick',
                         ('ds-pneumonia', 'SpO2<90%'): 'darksalmon'
                         }

    else:
        raise ValueError(f'Index size not recognised {results.index.size}')

    # PLOTS!!!
    fig, axs = plt.subplots(ncols=2, nrows=1, sharey=True, constrained_layout=True)
    for i, ix in enumerate(results.columns.levels[0]):
        ax = axs[i]
        # results.loc[:, (ix, slice(None))].T.plot.bar(stacked=True, ax=ax, legend=False)
        results.loc[:, (ix, slice(None))].T.plot.bar(stacked=True, ax=ax, width=0.6, legend=False,
                                                     color=results.index.map(assign_colors))
        ax.set_xticklabels(results.loc[:, (ix, slice(None))].columns.levels[1])
        ax.set_ylabel('Deaths per 100,000 cases of ALRI')
        ax.set_title(f"{ix}", fontsize=10)
        ax.grid(axis='y')
    handles, labels = axs[-1].get_legend_handles_labels()
    ax.legend(reversed(handles), reversed(labels), title='Case Type', loc='upper left', bbox_to_anchor=(1, 1),
              fontsize=7)
    # fig.suptitle('Deaths Under Different Interventions Combinations', fontsize=14, fontweight='semibold')
    fig.suptitle('Scenario: Perfect health worker diagnostic accuracy', fontsize=12, fontweight='semibold')
    fig.show()
    # fig.savefig(Path('./outputs') / ('perfect dx - intervention bars hosp vs hc current policy' + datestamp + ".pdf"), format='pdf')
    plt.close(fig)

    # # # # Calculations # # # #
    # get the stats of impact between interventions
    # base comparator - natural history mortality
    natural_deaths = (table['prob_die_if_no_treatment']).groupby(by=[disease_classification2, low_oxygen]).sum()
    number_cases = table.groupby(by=[disease_classification2, low_oxygen]).size()
    cfr = natural_deaths / number_cases
    # overall_cfr = (cfr * fraction).sum()  # only works for 3 spo2 breakdowns
    # prop_death = natural_deaths / natural_deaths.sum()

    # base comparator - antibiotics only under perfect hw dx
    deaths_antibiotics_level2 = (table['prob_die_if_no_treatment'] * (
        1.0 - table[f'treatment_efficacy_if_normal_treatment_but_without_oximeter_or_oxygen_{dx_accuracy}_hw_dx_level2'] / 100.0)
                                     ).groupby(by=[disease_classification2, low_oxygen]).sum()

    # pulse oximeter only under perfect hw dx
    deaths_po_only_level2 = (table['prob_die_if_no_treatment'] * (
        1.0 - table[f'treatment_efficacy_if_normal_treatment_and_with_oximeter_but_without_oxygen_{dx_accuracy}_hw_dx_level2'] / 100.0)
                                 ).groupby(by=[disease_classification2, low_oxygen]).sum()

    # oxygen only under perfect hw dx
    deaths_ox_only_level2 = (table['prob_die_if_no_treatment'] * (
        1.0 - table[f'treatment_efficacy_if_normal_treatment_and_with_oxygen_but_without_oximeter_{dx_accuracy}_hw_dx_level2'] / 100.0)
                                 ).groupby(by=[disease_classification2, low_oxygen]).sum()

    # oxygen only under perfect hw dx
    deaths_po_and_ox_level2 = (table['prob_die_if_no_treatment'] * (
        1.0 - table[f'treatment_efficacy_if_normal_treatment_and_with_oximeter_and_oxygen_{dx_accuracy}_hw_dx_level2'] / 100.0)
                                   ).groupby(by=[disease_classification2, low_oxygen]).sum()

    impact_antibiotics = 1 - (deaths_antibiotics_level2.mean() / natural_deaths.mean())
    impact_po = 1 - (deaths_po_only_level2.mean() / natural_deaths.mean())  #deaths_antibiotics_perfect_hw.mean()  # 0.887704
    impact_ox = 1 - (deaths_ox_only_level2.mean() / natural_deaths.mean()) #deaths_antibiotics_perfect_hw.mean()  # 0.76990223%
    impact_po_and_ox = 1 - (deaths_po_and_ox_level2.mean() / natural_deaths.mean())  # deaths_antibiotics_perfect_hw.mean()  # 0.671578

    cfr_antibiotics = deaths_antibiotics_level2 / number_cases
    cfr_po = deaths_po_only_level2 / number_cases
    cfr_ox = deaths_ox_only_level2 / number_cases
    cfr_po_and_ox = deaths_po_and_ox_level2 / number_cases


    # Repeat for FACILITY LEVEL 1A
    # base comparator - antibiotics only under normal hw dx
    deaths_antibiotics_level1a = (table['prob_die_if_no_treatment'] * (
        1.0 - table[f'treatment_efficacy_if_normal_treatment_but_without_oximeter_or_oxygen_{dx_accuracy}_hw_dx_level1a'] / 100.0)
                                    ).groupby(by=[disease_classification2, low_oxygen]).sum()

    # pulse oximeter only under perfect hw dx
    deaths_po_only_level1a = (table['prob_die_if_no_treatment'] * (
        1.0 - table[f'treatment_efficacy_if_normal_treatment_and_with_oximeter_but_without_oxygen_{dx_accuracy}_hw_dx_level1a'] / 100.0)
                                ).groupby(by=[disease_classification2, low_oxygen]).sum()

    # oxygen only under perfect hw dx
    deaths_ox_only_level1a = (table['prob_die_if_no_treatment'] * (
        1.0 - table[f'treatment_efficacy_if_normal_treatment_and_with_oxygen_but_without_oximeter_{dx_accuracy}_hw_dx_level1a'] / 100.0)
                                ).groupby(by=[disease_classification2, low_oxygen]).sum()

    # oxygen only under perfect hw dx
    deaths_po_and_ox_level1a = (table['prob_die_if_no_treatment'] * (
        1.0 - table[f'treatment_efficacy_if_normal_treatment_and_with_oximeter_and_oxygen_{dx_accuracy}_hw_dx_level1a'] / 100.0)
                                  ).groupby(by=[disease_classification2, low_oxygen]).sum()

    # impact reduction on mortality
    impact_antibiotics_level1a = 1 - (deaths_antibiotics_level1a.mean() / natural_deaths.mean())
    impact_po_level1a = 1 - (deaths_po_only_level1a.mean() / natural_deaths.mean())  #deaths_antibiotics_perfect_hw.mean()  # 0.887704
    impact_ox_level1a = 1 - (deaths_ox_only_level1a.mean() / natural_deaths.mean()) #deaths_antibiotics_perfect_hw.mean()  # 0.76990223%
    impact_po_and_ox_level1a = 1 - (deaths_po_and_ox_level1a.mean() / natural_deaths.mean())  # deaths_antibiotics_perfect_hw.mean()  # 0.671578

    cfr_antibiotics_level1a = deaths_antibiotics_level1a / number_cases
    cfr_po_level1a = deaths_po_only_level1a / number_cases
    cfr_ox_level1a = deaths_ox_only_level1a / number_cases
    cfr_po_and_ox_level1a = deaths_po_and_ox_level1a / number_cases
