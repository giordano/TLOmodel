"""
The file contains the event HSI_GenericFirstApptAtFacilityLevel1, which describes the first interaction with
the health system following the onset of acute generic symptoms.

This file contains the HSI events that represent the first contact with the Health System, which are triggered by
the onset of symptoms. Non-emergency symptoms lead to `HSI_GenericFirstApptAtFacilityLevel0` and emergency symptoms
lead to `HSI_GenericEmergencyFirstApptAtFacilityLevel1`.
"""
import pandas as pd

from tlo import logging
from tlo.events import IndividualScopeEventMixin
from tlo.methods.bladder_cancer import (
    HSI_BladderCancer_Investigation_Following_Blood_Urine,
    HSI_BladderCancer_Investigation_Following_pelvic_pain,
)
from tlo.methods.breast_cancer import (
    HSI_BreastCancer_Investigation_Following_breast_lump_discernible,
)
from tlo.methods.care_of_women_during_pregnancy import (
    HSI_CareOfWomenDuringPregnancy_PostAbortionCaseManagement,
    HSI_CareOfWomenDuringPregnancy_TreatmentForEctopicPregnancy,
)
from tlo.methods.chronicsyndrome import HSI_ChronicSyndrome_SeeksEmergencyCareAndGetsTreatment
from tlo.methods.epilepsy import HSI_Epilepsy_Start_Anti_Epileptic
from tlo.methods.healthsystem import HSI_Event
from tlo.methods.hiv import HSI_Hiv_TestAndRefer
from tlo.methods.labour import HSI_Labour_ReceivesSkilledBirthAttendanceDuringLabour
from tlo.methods.malaria import (
    HSI_Malaria_complicated_treatment_adult,
    HSI_Malaria_complicated_treatment_child,
    HSI_Malaria_non_complicated_treatment_adult,
    HSI_Malaria_non_complicated_treatment_age0_5,
    HSI_Malaria_non_complicated_treatment_age5_15,
)
from tlo.methods.measles import HSI_Measles_Treatment
from tlo.methods.mockitis import HSI_Mockitis_PresentsForCareWithSevereSymptoms
from tlo.methods.oesophagealcancer import HSI_OesophagealCancer_Investigation_Following_Dysphagia
from tlo.methods.other_adult_cancers import (
    HSI_OtherAdultCancer_Investigation_Following_early_other_adult_ca_symptom,
)
from tlo.methods.prostate_cancer import (
    HSI_ProstateCancer_Investigation_Following_Pelvic_Pain,
    HSI_ProstateCancer_Investigation_Following_Urinary_Symptoms,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class HSI_GenericFirstApptAtFacilityLevel0(HSI_Event, IndividualScopeEventMixin):
    """This is a Health System Interaction Event that represents the first interaction with the health system following
     the onset of non-emergency symptom(s). It occurs at level 0. This is the HSI that is generated by the
     HealthCareSeekingBehaviour module by default."""

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        assert module is self.sim.modules['HealthSeekingBehaviour']

        self.TREATMENT_ID = 'FirstAttendance_NonEmergency'
        self.ACCEPTED_FACILITY_LEVEL = '0'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'ConWithDCSA': 1})

    def apply(self, person_id, squeeze_factor):
        """Run the actions required during the HSI."""
        df = self.sim.population.props

        if not df.at[person_id, 'is_alive']:
            return

        do_at_generic_first_appt_non_emergency(hsi_event=self, squeeze_factor=squeeze_factor)


class HSI_GenericEmergencyFirstApptAtFacilityLevel1(HSI_Event, IndividualScopeEventMixin):
    """This is a Health System Interaction Event that represents the generic appointment which is the first interaction
    with the health system following the onset of emergency symptom(s). It occurs at level 1."""

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        assert module.name in ['HealthSeekingBehaviour', 'Labour', 'PregnancySupervisor', 'RTI']

        self.TREATMENT_ID = 'FirstAttendance_Emergency'
        self.ACCEPTED_FACILITY_LEVEL = '1b'
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({
            ('Under5OPD' if self.sim.population.props.at[person_id, "age_years"] < 5 else 'Over5OPD'): 1})

    def apply(self, person_id, squeeze_factor):

        df = self.sim.population.props

        if not df.at[person_id, 'is_alive']:
            return

        do_at_generic_first_appt_emergency(hsi_event=self, squeeze_factor=squeeze_factor)


class HSI_EmergencyCare_SpuriousSymptom(HSI_Event, IndividualScopeEventMixin):
    """This is an HSI event that provides Accident & Emergency Care for a person that has spurious emergency symptom."""

    def __init__(self, module, person_id, accepted_facility_level='1a'):
        super().__init__(module, person_id=person_id)
        assert module is self.sim.modules['HealthSeekingBehaviour']

        self.TREATMENT_ID = "FirstAttendance_SpuriousEmergencyCare"
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({'AccidentsandEmerg': 1})
        self.ACCEPTED_FACILITY_LEVEL = accepted_facility_level  # '1a' in default or '1b' as an alternative

    def apply(self, person_id, squeeze_factor):
        df = self.sim.population.props
        if not df.at[person_id, 'is_alive']:
            return self.make_appt_footprint({})
        else:
            sm = self.sim.modules['SymptomManager']
            sm.change_symptom(person_id, "spurious_emergency_symptom", '-', sm)


def do_at_generic_first_appt_non_emergency(hsi_event, squeeze_factor):
    """The actions are taken during the non-emergency generic HSI, HSI_GenericFirstApptAtFacilityLevel0."""

    # Gather useful shortcuts
    sim = hsi_event.sim
    person_id = hsi_event.target
    df = hsi_event.sim.population.props
    symptoms = hsi_event.sim.modules['SymptomManager'].has_what(person_id=person_id)
    age = df.at[person_id, 'age_years']
    schedule_hsi = hsi_event.sim.modules["HealthSystem"].schedule_hsi_event

    # ----------------------------------- ALL AGES -----------------------------------
    # Consider Measles if rash.
    if 'Measles' in sim.modules:
        if "rash" in symptoms:
            schedule_hsi(
                HSI_Measles_Treatment(
                    person_id=person_id,
                    module=hsi_event.sim.modules['Measles']),
                priority=0,
                topen=hsi_event.sim.date,
                tclose=None)

    # 'Automatic' testing for HIV for everyone attending care with AIDS symptoms:
    #  - suppress the footprint (as it done as part of another appointment)
    #  - do not do referrals if the person is HIV negative (assumed not time for counselling etc).
    if 'Hiv' in sim.modules:
        if 'aids_symptoms' in symptoms:
            schedule_hsi(
                HSI_Hiv_TestAndRefer(
                    person_id=person_id,
                    module=hsi_event.sim.modules['Hiv'],
                    referred_from="hsi_generic_first_appt",
                    suppress_footprint=True,
                    do_not_refer_if_neg=True),
                topen=hsi_event.sim.date,
                tclose=None,
                priority=0)

    if 'injury' in symptoms:
        if 'RTI' in sim.modules:
            sim.modules['RTI'].do_rti_diagnosis_and_treatment(person_id)

    if 'Schisto' in sim.modules:
        sim.modules['Schisto'].do_on_presentation_with_symptoms(person_id=person_id, symptoms=symptoms)

    if age <= 5:
        # ----------------------------------- CHILD < 5 -----------------------------------
        if 'Diarrhoea' in sim.modules:
            if 'diarrhoea' in symptoms:
                sim.modules['Diarrhoea'].do_when_presentation_with_diarrhoea(
                    person_id=person_id, hsi_event=hsi_event)

        if 'Alri' in sim.modules:
            if ('cough' in symptoms) or ('difficult_breathing' in symptoms):
                sim.modules['Alri'].on_presentation(person_id=person_id, hsi_event=hsi_event)

        if "Malaria" in sim.modules:
            if 'fever' in symptoms:
                malaria_test_result = sim.modules['Malaria'].check_if_fever_is_caused_by_malaria(
                    person_id=person_id, hsi_event=hsi_event)

                # Treat / refer based on diagnosis
                if malaria_test_result == "severe_malaria":
                    schedule_hsi(
                        HSI_Malaria_complicated_treatment_child(
                            person_id=person_id,
                            module=sim.modules["Malaria"]),
                        priority=1,
                        topen=sim.date,
                        tclose=None)

                elif malaria_test_result == "clinical_malaria":
                    schedule_hsi(
                        HSI_Malaria_non_complicated_treatment_age0_5(
                            person_id=person_id,
                            module=sim.modules["Malaria"]),
                        priority=1,
                        topen=sim.date,
                        tclose=None)

        # Routine assessments
        if 'Stunting' in sim.modules:
            sim.modules['Stunting'].do_routine_assessment_for_chronic_undernutrition(person_id=person_id)

    elif age < 15:
        # ----------------------------------- CHILD 5-14 -----------------------------------
        if 'fever' in symptoms and "Malaria" in sim.modules:
            malaria_test_result = sim.modules['Malaria'].check_if_fever_is_caused_by_malaria(
                person_id=person_id, hsi_event=hsi_event)

            # Treat / refer based on diagnosis
            if malaria_test_result == "severe_malaria":
                schedule_hsi(
                    HSI_Malaria_complicated_treatment_child(
                        person_id=person_id,
                        module=sim.modules["Malaria"]),
                    priority=1,
                    topen=sim.date,
                    tclose=None)

            elif malaria_test_result == "clinical_malaria":
                schedule_hsi(
                    HSI_Malaria_non_complicated_treatment_age5_15(
                        person_id=person_id,
                        module=sim.modules["Malaria"]),
                    priority=1,
                    topen=sim.date,
                    tclose=None)

    else:
        # ----------------------------------- ADULT -----------------------------------
        if 'OesophagealCancer' in sim.modules:
            # If the symptoms include dysphagia, then begin investigation for Oesophageal Cancer:
            if 'dysphagia' in symptoms:
                schedule_hsi(
                    HSI_OesophagealCancer_Investigation_Following_Dysphagia(
                        person_id=person_id,
                        module=sim.modules['OesophagealCancer']),
                    priority=0,
                    topen=sim.date,
                    tclose=None
                )

        if 'BladderCancer' in sim.modules:
            # If the symptoms include blood_urine, then begin investigation for Bladder Cancer:
            if 'blood_urine' in symptoms:
                schedule_hsi(
                    HSI_BladderCancer_Investigation_Following_Blood_Urine(
                        person_id=person_id,
                        module=sim.modules['BladderCancer']),
                    priority=0,
                    topen=sim.date,
                    tclose=None
                )

            # If the symptoms include pelvic_pain, then begin investigation for Bladder Cancer:
            if 'pelvic_pain' in symptoms:
                schedule_hsi(
                    HSI_BladderCancer_Investigation_Following_pelvic_pain(
                        person_id=person_id,
                        module=sim.modules['BladderCancer']),
                    priority=0,
                    topen=sim.date,
                    tclose=None)

        if 'ProstateCancer' in sim.modules:
            # If the symptoms include urinary, then begin investigation for prostate cancer:
            if 'urinary' in symptoms:
                schedule_hsi(
                    HSI_ProstateCancer_Investigation_Following_Urinary_Symptoms(
                        person_id=person_id,
                        module=sim.modules['ProstateCancer']),
                    priority=0,
                    topen=sim.date,
                    tclose=None)

            if 'pelvic_pain' in symptoms:
                schedule_hsi(
                    HSI_ProstateCancer_Investigation_Following_Pelvic_Pain(
                        person_id=person_id,
                        module=sim.modules['ProstateCancer']),
                    priority=0,
                    topen=sim.date,
                    tclose=None)

        if 'OtherAdultCancer' in sim.modules:
            if 'early_other_adult_ca_symptom' in symptoms:
                schedule_hsi(
                    HSI_OtherAdultCancer_Investigation_Following_early_other_adult_ca_symptom(
                        person_id=person_id,
                        module=sim.modules['OtherAdultCancer']
                    ),
                    priority=0,
                    topen=sim.date,
                    tclose=None)

        if 'BreastCancer' in sim.modules:
            # If the symptoms include breast lump discernible:
            if 'breast_lump_discernible' in symptoms:
                schedule_hsi(
                    HSI_BreastCancer_Investigation_Following_breast_lump_discernible(
                        person_id=person_id,
                        module=sim.modules['BreastCancer'],
                    ),
                    priority=0,
                    topen=sim.date,
                    tclose=None)

        if 'Depression' in sim.modules:
            sim.modules['Depression'].do_on_presentation_to_care(person_id=person_id,
                                                                 hsi_event=hsi_event,
                                                                 squeeze_factor=squeeze_factor)

        if "Malaria" in sim.modules:
            if 'fever' in symptoms:
                malaria_test_result = sim.modules['Malaria'].check_if_fever_is_caused_by_malaria(
                    person_id=person_id, hsi_event=hsi_event)

                if malaria_test_result == "severe_malaria":
                    schedule_hsi(
                        HSI_Malaria_complicated_treatment_adult(
                            person_id=person_id,
                            module=sim.modules["Malaria"]),
                        priority=1,
                        topen=sim.date,
                        tclose=None)

                elif malaria_test_result == "clinical_malaria":
                    schedule_hsi(
                        HSI_Malaria_non_complicated_treatment_adult(
                            person_id=person_id,
                            module=sim.modules["Malaria"]),
                        priority=1,
                        topen=sim.date,
                        tclose=None)

        if 'CardioMetabolicDisorders' in sim.modules:
            # Take a blood pressure measurement for proportion of individuals who have not been diagnosed and
            # are either over 50 or younger than 50 but are selected to get tested.
            sim.modules['CardioMetabolicDisorders'].determine_if_will_be_investigated(person_id=person_id)

        if 'Copd' in sim.modules:
            if ('breathless_moderate' in symptoms) or ('breathless_severe' in symptoms):
                sim.modules['Copd'].do_when_present_with_breathless(person_id=person_id, hsi_event=hsi_event)


def do_at_generic_first_appt_emergency(hsi_event, squeeze_factor):
    """The actions are taken during the non-emergency generic HSI, HSI_GenericEmergencyFirstApptAtFacilityLevel1."""

    # Gather useful shortcuts
    sim = hsi_event.sim
    rng = hsi_event.module.rng
    person_id = hsi_event.target
    df = hsi_event.sim.population.props
    symptoms = hsi_event.sim.modules['SymptomManager'].has_what(person_id=person_id)
    schedule_hsi = hsi_event.sim.modules["HealthSystem"].schedule_hsi_event
    age = df.at[person_id, 'age_years']

    if 'PregnancySupervisor' in sim.modules:

        # -----  ECTOPIC PREGNANCY  -----
        if df.at[person_id, 'ps_ectopic_pregnancy'] != 'none':
            event = HSI_CareOfWomenDuringPregnancy_TreatmentForEctopicPregnancy(
                module=sim.modules['CareOfWomenDuringPregnancy'], person_id=person_id)
            schedule_hsi(event, priority=0, topen=sim.date, tclose=sim.date + pd.DateOffset(days=1))

        # -----  COMPLICATIONS OF ABORTION  -----
        abortion_complications = sim.modules['PregnancySupervisor'].abortion_complications
        if abortion_complications.has_any([person_id], 'sepsis', 'injury', 'haemorrhage', first=True):
            event = HSI_CareOfWomenDuringPregnancy_PostAbortionCaseManagement(
                module=sim.modules['CareOfWomenDuringPregnancy'], person_id=person_id)
            schedule_hsi(event, priority=0, topen=sim.date, tclose=sim.date + pd.DateOffset(days=1))

    if 'Labour' in sim.modules:
        mni = sim.modules['PregnancySupervisor'].mother_and_newborn_info
        labour_list = sim.modules['Labour'].women_in_labour

        if person_id in labour_list:
            la_currently_in_labour = df.at[person_id, 'la_currently_in_labour']
            if (
                la_currently_in_labour &
                mni[person_id]['sought_care_for_complication'] &
                (mni[person_id]['sought_care_labour_phase'] == 'intrapartum')
            ):
                event = HSI_Labour_ReceivesSkilledBirthAttendanceDuringLabour(
                    module=sim.modules['Labour'], person_id=person_id,
                    facility_level_of_this_hsi=rng.choice(['1a', '1b']))
                schedule_hsi(event, priority=0, topen=sim.date, tclose=sim.date + pd.DateOffset(days=1))

    if "Depression" in sim.modules:
        sim.modules['Depression'].do_on_presentation_to_care(person_id=person_id,
                                                             hsi_event=hsi_event,
                                                             squeeze_factor=squeeze_factor)

    if "Malaria" in sim.modules:
        # Quick diagnosis algorithm - just perfectly recognises the symptoms of severe malaria
        sev_set = {"acidosis",
                   "coma_convulsions",
                   "renal_failure",
                   "shock",
                   "jaundice",
                   "anaemia"}

        # if person's symptoms are on severe malaria list then consider treatment for malaria
        any_symptoms_indicative_of_severe_malaria = len(sev_set.intersection(symptoms)) > 0

        if any_symptoms_indicative_of_severe_malaria:
            # Check if malaria parasitaemia:
            malaria_test_result = sim.modules["Malaria"].check_if_fever_is_caused_by_malaria(
                person_id=person_id, hsi_event=hsi_event)

            # if any symptoms indicative of malaria and they have parasitaemia (would return a positive rdt)
            if malaria_test_result in ("severe_malaria", "clinical_malaria"):
                # Launch the HSI for treatment for Malaria - choosing the right one for adults/children
                if df.at[person_id, 'age_years'] < 5.0:
                    schedule_hsi(
                        hsi_event=HSI_Malaria_complicated_treatment_child(
                            sim.modules["Malaria"], person_id=person_id
                        ),
                        priority=0,
                        topen=sim.date,
                    )
                else:
                    schedule_hsi(
                        hsi_event=HSI_Malaria_complicated_treatment_adult(
                            sim.modules["Malaria"], person_id=person_id
                        ),
                        priority=0,
                        topen=sim.date,
                    )

    # ------ CARDIO-METABOLIC DISORDERS ------
    if 'CardioMetabolicDisorders' in sim.modules:
        sim.modules['CardioMetabolicDisorders'].determine_if_will_be_investigated_events(person_id=person_id)

    if "Epilepsy" in sim.modules:
        if 'seizures' in symptoms:
            schedule_hsi(HSI_Epilepsy_Start_Anti_Epileptic(person_id=person_id,
                                                           module=sim.modules['Epilepsy']),
                         priority=0,
                         topen=sim.date,
                         tclose=None)

    # -----  EXAMPLES FOR MOCKITIS AND CHRONIC SYNDROME  -----
    if 'craving_sandwiches' in symptoms:
        event = HSI_ChronicSyndrome_SeeksEmergencyCareAndGetsTreatment(
            module=sim.modules['ChronicSyndrome'],
            person_id=person_id
        )
        schedule_hsi(event, priority=1, topen=sim.date)

    if 'extreme_pain_in_the_nose' in symptoms:
        event = HSI_Mockitis_PresentsForCareWithSevereSymptoms(
            module=sim.modules['Mockitis'],
            person_id=person_id
        )
        schedule_hsi(event, priority=1, topen=sim.date)

    if 'severe_trauma' in symptoms:
        if 'RTI' in sim.modules:
            sim.modules['RTI'].do_rti_diagnosis_and_treatment(person_id=person_id)

    if 'Alri' in sim.modules:
        if (age <= 5) and (('cough' in symptoms) or ('difficult_breathing' in symptoms)):
            sim.modules['Alri'].on_presentation(person_id=person_id, hsi_event=hsi_event)

    # ----- spurious emergency symptom -----
    if 'spurious_emergency_symptom' in symptoms:
        event = HSI_EmergencyCare_SpuriousSymptom(
            module=sim.modules['HealthSeekingBehaviour'],
            person_id=person_id
        )
        schedule_hsi(event, priority=0, topen=sim.date)

    if 'Copd' in sim.modules:
        if ('breathless_moderate' in symptoms) or ('breathless_severe' in symptoms):
            sim.modules['Copd'].do_when_present_with_breathless(person_id=person_id, hsi_event=hsi_event)
