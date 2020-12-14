"""
The joint NCDs model by Tim Hallett and Britta Jewell, October 2020

"""
from pathlib import Path

from tlo import DateOffset, Module, Parameter, Property, Types, logging
from tlo.events import IndividualScopeEventMixin, PopulationScopeEventMixin, RegularEvent, Event
from tlo.methods import Metadata, demography
from tlo.methods.demography import InstantaneousDeath
import tlo.methods.demography as de
from tlo.lm import LinearModel, LinearModelType, Predictor
from tlo.methods.healthsystem import HSI_Event

import pandas as pd
import numpy as np
import copy
import math

# ---------------------------------------------------------------------------------------------------------
#   MODULE DEFINITIONS
# ---------------------------------------------------------------------------------------------------------

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Ncds(Module):
    """
    One line summary goes here...

    """
    # Declare Metadata (this is for a typical 'Disease Module')
    METADATA = {
        Metadata.DISEASE_MODULE,
        Metadata.USES_SYMPTOMMANAGER,
        Metadata.USES_HEALTHSYSTEM,
        Metadata.USES_HEALTHBURDEN
    }

    PARAMETERS = {
        'interval_between_polls': Parameter(Types.INT, 'months between the main polling event'),
        'baseline_annual_probability': Parameter(Types.REAL,
                                                 'baseline annual probability of acquiring/losing condition'),
        'rr_male': Parameter(Types.REAL, 'rr if male'),
        'rr_0_4': Parameter(Types.REAL, 'rr if 0-4'),
        'rr_5_9': Parameter(Types.REAL, 'rr if 5-9'),
        'rr_10_14': Parameter(Types.REAL, 'rr if 10-14'),
        'rr_15_19': Parameter(Types.REAL, 'rr if 15-19'),
        'rr_20_24': Parameter(Types.REAL, 'rr if 20-24'),
        'rr_25_29': Parameter(Types.REAL, 'rr if 25-29'),
        'rr_30_34': Parameter(Types.REAL, 'rr if 30-34'),
        'rr_35_39': Parameter(Types.REAL, 'rr if 35-39'),
        'rr_40_44': Parameter(Types.REAL, 'rr if 40-44'),
        'rr_45_49': Parameter(Types.REAL, 'rr if 45-49'),
        'rr_50_54': Parameter(Types.REAL, 'rr if 50-54'),
        'rr_55_59': Parameter(Types.REAL, 'rr if 55-59'),
        'rr_60_64': Parameter(Types.REAL, 'rr if 60-64'),
        'rr_65_69': Parameter(Types.REAL, 'rr if 65-69'),
        'rr_70_74': Parameter(Types.REAL, 'rr if 70-74'),
        'rr_75_79': Parameter(Types.REAL, 'rr if 75-79'),
        'rr_80_84': Parameter(Types.REAL, 'rr if 80-84'),
        'rr_85_89': Parameter(Types.REAL, 'rr if 85-89'),
        'rr_90_94': Parameter(Types.REAL, 'rr if 90-94'),
        'rr_95_99': Parameter(Types.REAL, 'rr if 95-99'),
        'rr_100': Parameter(Types.REAL, 'rr if 100+'),
        'rr_urban': Parameter(Types.REAL, 'rr if living in an urban area'),
        'rr_wealth_1': Parameter(Types.REAL, 'rr if wealth 1'),
        'rr_wealth_2': Parameter(Types.REAL, 'rr if wealth 2'),
        'rr_wealth_3': Parameter(Types.REAL, 'rr if wealth 3'),
        'rr_wealth_4': Parameter(Types.REAL, 'rr if wealth 4'),
        'rr_wealth_5': Parameter(Types.REAL, 'rr if wealth 5'),
        'rr_bmi_1': Parameter(Types.REAL, 'rr if bmi 1'),
        'rr_bmi_2': Parameter(Types.REAL, 'rr if bmi 2'),
        'rr_bmi_3': Parameter(Types.REAL, 'rr if bmi 3'),
        'rr_bmi_4': Parameter(Types.REAL, 'rr if bmi 4'),
        'rr_bmi_5': Parameter(Types.REAL, 'rr if bmi 5'),
        'rr_low_exercise': Parameter(Types.REAL, 'rr if low exercise'),
        'rr_high_salt': Parameter(Types.REAL, 'rr if high salt'),
        'rr_high_sugar': Parameter(Types.REAL, 'rr if high sugar'),
        'rr_tobacco': Parameter(Types.REAL, 'rr if tobacco'),
        'rr_alcohol': Parameter(Types.REAL, 'rr if alcohol'),
        'rr_marital_status_1': Parameter(Types.REAL, 'rr if never married'),
        'rr_marital_status_2': Parameter(Types.REAL, 'rr if currently married'),
        'rr_marital_status_3': Parameter(Types.REAL, 'rr if widowed or divorced'),
        'rr_in_education': Parameter(Types.REAL, 'rr if in education'),
        'rr_current_education_level_1': Parameter(Types.REAL, 'rr if education level 1'),
        'rr_current_education_level_2': Parameter(Types.REAL, 'rr if education level 2'),
        'rr_current_education_level_3': Parameter(Types.REAL, 'rr if education level 3'),
        'rr_unimproved_sanitation': Parameter(Types.REAL, 'rr if unimproved sanitation'),
        'rr_no_access_handwashing': Parameter(Types.REAL, 'rr if no access to handwashing'),
        'rr_no_clean_drinking_water': Parameter(Types.REAL, 'rr if no access to drinking water'),
        'rr_wood_burning_stove': Parameter(Types.REAL, 'rr if wood-burning stove'),
        'rr_diabetes': Parameter(Types.REAL, 'rr if currently has diabetes'),
        'rr_hypertension': Parameter(Types.REAL, 'rr if currently has hypertension'),
        'rr_depression': Parameter(Types.REAL, 'rr if currently has depression'),
        'rr_chronic_kidney_disease': Parameter(Types.REAL, 'rr if currently has chronic kidney disease'),
        'rr_chronic_lower_back_pain': Parameter(Types.REAL, 'rr if currently has chronic lower back pain'),
        'rr_chronic_ischemic_heart_disease': Parameter(Types.REAL,
                                                       'rr if currently has chronic ischemic heart disease'),
        'rr_cancers': Parameter(Types.REAL, 'rr if currently has cancers'),
        'rr_ever_stroke': Parameter(Types.REAL, 'rr if has ever had stroke'),
        'rr_ever_heart_attack': Parameter(Types.REAL,
                                       'rr of has ever had heart attack'),
        'r_death_nc_hypertension': Parameter(Types.REAL, 'baseline annual probability of dying if has hypertension'),
        'r_death_nc_diabetes': Parameter(Types.REAL, 'baseline annual probability of dying if has diabetes'),
        'r_death_nc_depression': Parameter(Types.REAL, 'baseline annual probability of dying if has depression'),
        'r_death_nc_chronic_lower_back_pain': Parameter(Types.REAL,
                                                        'baseline annual probability of dying if has chronic lower back pain'),
        'r_death_nc_chronic_kidney_disease': Parameter(Types.REAL, 'baseline annual probability of dying if has CKD'),
        'r_death_nc_chronic_ischemic_hd': Parameter(Types.REAL, 'baseline annual probability of dying if has CIHD'),
        'r_death_nc_cancers': Parameter(Types.REAL, 'baseline annual probability of dying if has cancers'),
        'r_death_nc_stroke': Parameter(Types.REAL, 'baseline annual probability of dying if has ever had a stroke'),
        'r_death_nc_heart_attack': Parameter(Types.REAL,
                                             'baseline annual probability of dying if has ever had a heart attack')
    }

    # Note that all properties must have a two letter prefix that identifies them to this module.

    PROPERTIES = {
        # These are all the states:
        'nc_diabetes': Property(Types.BOOL, 'Whether or not someone currently has diabetes'),
        'nc_hypertension': Property(Types.BOOL, 'Whether or not someone currently has hypertension'),
        'nc_depression': Property(Types.BOOL, 'Whether or not someone currently has depression'),
        # 'nc_muscoskeletal': Property(Types.BOOL, 'Whether or not someone currently has muscoskeletal conditions'),
        # 'nc_frailty': Property(Types.BOOL, 'Whether or not someone currently has frailty'),
        'nc_chronic_lower_back_pain': Property(Types.BOOL,
                                               'Whether or not someone currently has chronic lower back pain'),
        # 'nc_arthritis': Property(Types.BOOL, 'Whether or not someone currently has arthritis'),
        # 'nc_vision_disorders': Property(Types.BOOL, 'Whether or not someone currently has vision disorders'),
        # 'nc_chronic_liver_disease': Property(Types.BOOL, 'Whether or not someone currently has chronic liver disease'),
        'nc_chronic_kidney_disease': Property(Types.BOOL,
                                              'Whether or not someone currently has chronic kidney disease'),
        'nc_chronic_ischemic_hd': Property(Types.BOOL,
                                           'Whether or not someone currently has chronic ischemic heart disease'),
        # 'nc_lower_extremity_disease': Property(Types.BOOL, 'Whether or not someone currently has lower extremity disease'),
        # 'nc_dementia': Property(Types.BOOL, 'Whether or not someone currently has dementia'),
        # 'nc_bladder_cancer': Property(Types.BOOL, 'Whether or not someone currently has bladder cancer'),
        # 'nc_oesophageal_cancer': Property(Types.BOOL, 'Whether or not someone currently has oesophageal cancer'),
        # 'nc_breast_cancer': Property(Types.BOOL, 'Whether or not someone currently has breast cancer'),
        # 'nc_prostate_cancer': Property(Types.BOOL, 'Whether or not someone currently has prostate cancer'),
        'nc_cancers': Property(Types.BOOL, 'Whether or not someone currently has cancers'),
        # 'nc_chronic_respiratory_disease': Property(Types.BOOL, 'Whether or not someone currently has chronic respiratory disease'),
        # 'nc_other_infections': Property(Types.BOOL, 'Whether or not someone currently has other infections'),
        'nc_ever_stroke': Property(Types.BOOL, 'Whether or not someone has ever had a stroke'),
        'nc_ever_heart_attack': Property(Types.BOOL, 'Whether or not someone has ever had a heart attack')
    }

    # TODO: we will have to later gather from the others what the symptoms are in each state - for now leave blank
    SYMPTOMS = {}

    def __init__(self, name=None, resourcefilepath=None):
        # NB. Parameters passed to the module can be inserted in the __init__ definition.

        super().__init__(name)
        self.resourcefilepath = resourcefilepath

        # save a list of the conditions that covered in this module (extracted from PROPERTIES)
        self.conditions = list(self.PROPERTIES.keys())
        self.conditions.remove('nc_ever_stroke')
        self.conditions.remove('nc_ever_heart_attack')

        # save a list of the events that are covered in this module (created manually for now)
        self.events = ['nc_ever_stroke', 'nc_ever_heart_attack']

    def read_parameters(self, data_folder):
        """Read parameter values from file, if required.
        To access files use: Path(self.resourcefilepath) / file_name
        """
        self.age_index = self.sim.modules['Demography'].AGE_RANGE_CATEGORIES

        # dict to hold counters for the number of episodes by condition-type and age-group
        blank_counter = dict(zip(self.conditions, [list() for _ in self.conditions]))
        self.incident_case_tracker_blank = dict()
        for age_range in self.age_index:
            self.incident_case_tracker_blank[f'{age_range}'] = copy.deepcopy(blank_counter)

        self.incident_case_tracker = copy.deepcopy(self.incident_case_tracker_blank)

        zeros_counter = dict(zip(self.conditions, [0] * len(self.conditions)))
        self.incident_case_tracker_zeros = dict()
        for age_range in self.age_index:
            self.incident_case_tracker_zeros[f'{age_range}'] = copy.deepcopy(zeros_counter)

        self.params_dict_onset = dict()
        self.params_dict_removal = dict()
        self.params_dict_events = dict()

        for condition in self.conditions:
            params_onset = pd.read_excel(Path(self.resourcefilepath) / "ResourceFile_NCDs_condition_onset.xlsx",
                                         sheet_name=f"{condition}")
            # replace NaNs with 1
            params_onset['value'] = params_onset['value'].replace(np.nan, 1)
            self.params_dict_onset[condition] = params_onset

            # Get the death rates from a params_dict
            self.parameters[f'r_death_{condition}'] = params_onset.loc[params_onset['parameter_name'] == f'r_death_{condition}', 'value'].values[0]

            params_removal = pd.read_excel(Path(self.resourcefilepath) / "ResourceFile_NCDs_condition_onset.xlsx",
                                           sheet_name=f"{condition}")
            # replace NaNs with 1
            params_removal['value'] = params_removal['value'].replace(np.nan, 1)
            self.params_dict_removal[condition] = params_removal

        for event in self.events:
            params_events = pd.read_excel(Path(self.resourcefilepath) / "ResourceFile_NCDs_events.xlsx",
                                          sheet_name=f"{event}")
            # replace NaNs with 1
            params_events['value'] = params_events['value'].replace(np.nan, 1)
            self.params_dict_events[event] = params_events

        # Set the interval (in months) between the polls
        self.parameters['interval_between_polls'] = 3


    def initialise_population(self, population):
        """Set our property values for the initial population.
        """
        # TODO: @britta - we might need to gather this info from the others too: or it might that we have to find
        #  this through fitting. For now, let there be no conditions for anyone

        df = population.props

        for condition in self.conditions:
            df[condition] = False

    def initialise_simulation(self, sim):
        """Schedule:
        * Main Polling Event
        * Main Logging Event
        * Build the LinearModels for the onset/removal of each condition:
        """
        sim.schedule_event(Ncds_MainPollingEvent(self, self.parameters['interval_between_polls']), sim.date)
        sim.schedule_event(Ncds_LoggingEvent(self), sim.date)

        # Create Tracker for the number of different types of events
        self.eventsTracker = {'StrokeEvents': 0, 'HeartAttackEvents': 0}

        # Build the LinearModel for onset/removal of each condition
        self.lms_onset = dict()
        self.lms_removal = dict()

        # Build the LinearModel for occurrence of events
        self.lms_event_occurrence = dict()

        for condition in self.conditions:
            self.lms_onset[condition] = self.build_linear_model(condition, self.parameters['interval_between_polls'])
            self.lms_removal[condition] = self.build_linear_model(condition, self.parameters['interval_between_polls'])

        for event in self.events:
            self.lms_event_occurrence[event] = self.build_linear_model_events(event,
                                                                              self.parameters['interval_between_polls'])

    def build_linear_model(self, condition, interval_between_polls):
        """
        :param_dict: the dict read in from the resourcefile
        :param interval_between_polls: the duration (in months) between the polls
        :return: a linear model
        """

        p = self.params_dict_onset[condition].set_index('parameter_name').T.to_dict('records')[0]
        p['baseline_annual_probability'] = p['baseline_annual_probability'] * (interval_between_polls / 12)

        self.lms_onset[condition] = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            p['baseline_annual_probability'],
            Predictor().when('(sex=="M")', p['rr_male']),
            Predictor('age_years').when('.between(0, 4)', p['rr_0_4'])
                .when('.between(5, 9)', p['rr_5_9'])
                .when('.between(10, 14)', p['rr_10_14'])
                .when('.between(15, 19)', p['rr_15_19'])
                .when('.between(20, 24)', p['rr_20_24'])
                .when('.between(25, 29)', p['rr_25_29'])
                .when('.between(30, 34)', p['rr_30_34'])
                .when('.between(35, 39)', p['rr_35_39'])
                .when('.between(40, 44)', p['rr_40_44'])
                .when('.between(45, 49)', p['rr_45_49'])
                .when('.between(50, 54)', p['rr_50_54'])
                .when('.between(55, 59)', p['rr_55_59'])
                .when('.between(60, 64)', p['rr_60_64'])
                .when('.between(65, 69)', p['rr_65_69'])
                .when('.between(70, 74)', p['rr_70_74'])
                .when('.between(75, 79)', p['rr_75_79'])
                .when('.between(80, 84)', p['rr_80_84'])
                .when('.between(85, 89)', p['rr_85_89'])
                .when('.between(90, 94)', p['rr_90_94'])
                .when('.between(95, 99)', p['rr_95_99'])
                .otherwise(p['rr_100']),
            Predictor('li_urban').when(True, p['rr_urban']),
            Predictor('li_wealth').when('==1', p['rr_wealth_1'])
                .when('2', p['rr_wealth_2'])
                .when('3', p['rr_wealth_3'])
                .when('4', p['rr_wealth_4'])
                .when('5', p['rr_wealth_5']),
            Predictor('li_bmi').when('==1', p['rr_bmi_1'])
                .when('2', p['rr_bmi_2'])
                .when('3', p['rr_bmi_3'])
                .when('4', p['rr_bmi_4'])
                .when('5', p['rr_bmi_5']),
            Predictor('li_low_ex').when(True, p['rr_low_exercise']),
            Predictor('li_high_salt').when(True, p['rr_high_salt']),
            Predictor('li_high_sugar').when(True, p['rr_high_sugar']),
            Predictor('li_tob').when(True, p['rr_tobacco']),
            Predictor('li_ex_alc').when(True, p['rr_alcohol']),
            Predictor('li_mar_stat').when('1', p['rr_marital_status_1'])
                .when('2', p['rr_marital_status_2'])
                .when('3', p['rr_marital_status_3']),
            Predictor('li_in_ed').when(True, p['rr_in_education']),
            Predictor('li_ed_lev').when('==1', p['rr_current_education_level_1'])
                .when('2', p['rr_current_education_level_2'])
                .when('3', p['rr_current_education_level_3']),
            Predictor('li_unimproved_sanitation').when(True, p['rr_unimproved_sanitation']),
            Predictor('li_no_access_handwashing').when(True, p['rr_no_access_handwashing']),
            Predictor('li_no_clean_drinking_water').when(True, p['rr_no_clean_drinking_water']),
            Predictor('li_wood_burn_stove').when(True, p['rr_wood_burning_stove']),
            Predictor('nc_diabetes').when(True, p['rr_diabetes']),
            Predictor('nc_hypertension').when(True, p['rr_hypertension']),
            Predictor('nc_depression').when(True, p['rr_depression']),
            Predictor('nc_chronic_kidney_disease').when(True, p['rr_chronic_kidney_disease']),
            Predictor('nc_chronic_lower_back_pain').when(True, p['rr_chronic_lower_back_pain']),
            Predictor('nc_chronic_ischemic_hd').when(True, p['rr_chronic_ischemic_heart_disease']),
            Predictor('nc_cancers').when(True, p['rr_cancers'])
        )

        p = self.params_dict_removal[condition].set_index('parameter_name').T.to_dict('records')[0]
        p['baseline_annual_probability'] = p['baseline_annual_probability'] * (interval_between_polls / 12)

        self.lms_removal[condition] = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            p['baseline_annual_probability'],
            Predictor().when('(sex=="M")', p['rr_male']),
            Predictor('age_years').when('.between(0, 4)', p['rr_0_4'])
                .when('.between(5, 9)', p['rr_5_9'])
                .when('.between(10, 14)', p['rr_10_14'])
                .when('.between(15, 19)', p['rr_15_19'])
                .when('.between(20, 24)', p['rr_20_24'])
                .when('.between(25, 29)', p['rr_25_29'])
                .when('.between(30, 34)', p['rr_30_34'])
                .when('.between(35, 39)', p['rr_35_39'])
                .when('.between(40, 44)', p['rr_40_44'])
                .when('.between(45, 49)', p['rr_45_49'])
                .when('.between(50, 54)', p['rr_50_54'])
                .when('.between(55, 59)', p['rr_55_59'])
                .when('.between(60, 64)', p['rr_60_64'])
                .when('.between(65, 69)', p['rr_65_69'])
                .when('.between(70, 74)', p['rr_70_74'])
                .when('.between(75, 79)', p['rr_75_79'])
                .when('.between(80, 84)', p['rr_80_84'])
                .when('.between(85, 89)', p['rr_85_89'])
                .when('.between(90, 94)', p['rr_90_94'])
                .when('.between(95, 99)', p['rr_95_99'])
                .otherwise(p['rr_100']),
            Predictor('li_urban').when(True, p['rr_urban']),
            Predictor('li_wealth').when('==1', p['rr_wealth_1'])
                .when('2', p['rr_wealth_2'])
                .when('3', p['rr_wealth_3'])
                .when('4', p['rr_wealth_4'])
                .when('5', p['rr_wealth_5']),
            Predictor('li_bmi').when('==1', p['rr_bmi_1'])
                .when('2', p['rr_bmi_2'])
                .when('3', p['rr_bmi_3'])
                .when('4', p['rr_bmi_4'])
                .when('5', p['rr_bmi_5']),
            Predictor('li_low_ex').when(True, p['rr_low_exercise']),
            Predictor('li_high_salt').when(True, p['rr_high_salt']),
            Predictor('li_high_sugar').when(True, p['rr_high_sugar']),
            Predictor('li_tob').when(True, p['rr_tobacco']),
            Predictor('li_ex_alc').when(True, p['rr_alcohol']),
            Predictor('li_mar_stat').when('1', p['rr_marital_status_1'])
                .when('2', p['rr_marital_status_2'])
                .when('3', p['rr_marital_status_3']),
            Predictor('li_in_ed').when(True, p['rr_in_education']),
            Predictor('li_ed_lev').when('==1', p['rr_current_education_level_1'])
                .when('2', p['rr_current_education_level_2'])
                .when('3', p['rr_current_education_level_3']),
            Predictor('li_unimproved_sanitation').when(True, p['rr_unimproved_sanitation']),
            Predictor('li_no_access_handwashing').when(True, p['rr_no_access_handwashing']),
            Predictor('li_no_clean_drinking_water').when(True, p['rr_no_clean_drinking_water']),
            Predictor('li_wood_burn_stove').when(True, p['rr_wood_burning_stove']),
            Predictor('nc_diabetes').when(True, p['rr_diabetes']),
            Predictor('nc_hypertension').when(True, p['rr_hypertension']),
            Predictor('nc_depression').when(True, p['rr_depression']),
            Predictor('nc_chronic_kidney_disease').when(True, p['rr_chronic_kidney_disease']),
            Predictor('nc_chronic_lower_back_pain').when(True, p['rr_chronic_lower_back_pain']),
            Predictor('nc_chronic_ischemic_hd').when(True, p['rr_chronic_ischemic_heart_disease']),
            Predictor('nc_cancers').when(True, p['rr_cancers'])
        )

        return self.lms_onset[condition], self.lms_removal[condition]

    def build_linear_model_events(self, event, interval_between_polls):

        p = self.params_dict_events[event].set_index('parameter_name').T.to_dict('records')[0]
        p['baseline_annual_probability'] = p['baseline_annual_probability'] * (interval_between_polls / 12)

        self.lms_event_occurrence[event] = LinearModel(
            LinearModelType.MULTIPLICATIVE,
            p['baseline_annual_probability'],
            Predictor().when('(sex=="M")', p['rr_male']),
            Predictor('age_years').when('.between(0, 4)', p['rr_0_4'])
                .when('.between(5, 9)', p['rr_5_9'])
                .when('.between(10, 14)', p['rr_10_14'])
                .when('.between(15, 19)', p['rr_15_19'])
                .when('.between(20, 24)', p['rr_20_24'])
                .when('.between(25, 29)', p['rr_25_29'])
                .when('.between(30, 34)', p['rr_30_34'])
                .when('.between(35, 39)', p['rr_35_39'])
                .when('.between(40, 44)', p['rr_40_44'])
                .when('.between(45, 49)', p['rr_45_49'])
                .when('.between(50, 54)', p['rr_50_54'])
                .when('.between(55, 59)', p['rr_55_59'])
                .when('.between(60, 64)', p['rr_60_64'])
                .when('.between(65, 69)', p['rr_65_69'])
                .when('.between(70, 74)', p['rr_70_74'])
                .when('.between(75, 79)', p['rr_75_79'])
                .when('.between(80, 84)', p['rr_80_84'])
                .when('.between(85, 89)', p['rr_85_89'])
                .when('.between(90, 94)', p['rr_90_94'])
                .when('.between(95, 99)', p['rr_95_99'])
                .otherwise(p['rr_100']),
            Predictor('li_urban').when(True, p['rr_urban']),
            Predictor('li_wealth').when('==1', p['rr_wealth_1'])
                .when('2', p['rr_wealth_2'])
                .when('3', p['rr_wealth_3'])
                .when('4', p['rr_wealth_4'])
                .when('5', p['rr_wealth_5']),
            Predictor('li_bmi').when('==1', p['rr_bmi_1'])
                .when('2', p['rr_bmi_2'])
                .when('3', p['rr_bmi_3'])
                .when('4', p['rr_bmi_4'])
                .when('5', p['rr_bmi_5']),
            Predictor('li_low_ex').when(True, p['rr_low_exercise']),
            Predictor('li_high_salt').when(True, p['rr_high_salt']),
            Predictor('li_high_sugar').when(True, p['rr_high_sugar']),
            Predictor('li_tob').when(True, p['rr_tobacco']),
            Predictor('li_ex_alc').when(True, p['rr_alcohol']),
            Predictor('li_mar_stat').when('1', p['rr_marital_status_1'])
                .when('2', p['rr_marital_status_2'])
                .when('3', p['rr_marital_status_3']),
            Predictor('li_in_ed').when(True, p['rr_in_education']),
            Predictor('li_ed_lev').when('==1', p['rr_current_education_level_1'])
                .when('2', p['rr_current_education_level_2'])
                .when('3', p['rr_current_education_level_3']),
            Predictor('li_unimproved_sanitation').when(True, p['rr_unimproved_sanitation']),
            Predictor('li_no_access_handwashing').when(True, p['rr_no_access_handwashing']),
            Predictor('li_no_clean_drinking_water').when(True, p['rr_no_clean_drinking_water']),
            Predictor('li_wood_burn_stove').when(True, p['rr_wood_burning_stove']),
            Predictor('nc_diabetes').when(True, p['rr_diabetes']),
            Predictor('nc_hypertension').when(True, p['rr_hypertension']),
            Predictor('nc_depression').when(True, p['rr_depression']),
            Predictor('nc_chronic_kidney_disease').when(True, p['rr_chronic_kidney_disease']),
            Predictor('nc_chronic_lower_back_pain').when(True, p['rr_chronic_lower_back_pain']),
            Predictor('nc_chronic_ischemic_hd').when(True, p['rr_chronic_ischemic_heart_disease']),
            Predictor('nc_cancers').when(True, p['rr_cancers']),
            Predictor('nc_ever_stroke').when(True, p['rr_ever_stroke']),
            Predictor('nc_ever_heart_attack').when(True, p['rr_ever_heart_attack'])
        )

        return self.lms_event_occurrence[event]

    def on_birth(self, mother_id, child_id):
        """Initialise our properties for a newborn individual.

        :param mother_id: the mother for this child
        :param child_id: the new child
        """
        # TODO: @britta - assuming that the all children have nothing when they are born
        df = self.sim.population.props
        for condition in self.conditions:
            df.at[child_id, condition] = False

    def report_daly_values(self):
        """Report DALY values to the HealthBurden module"""
        # This must send back a pd.Series or pd.DataFrame that reports on the average daly-weights that have been
        # experienced by persons in the previous month. Only rows for alive-persons must be returned.
        # The names of the series of columns is taken to be the label of the cause of this disability.
        # It will be recorded by the healthburden module as <ModuleName>_<Cause>.

        # To return a value of 0.0 (fully health) for everyone, use:
        # df = self.sim.popultion.props
        # return pd.Series(index=df.index[df.is_alive],data=0.0)

        # TODO: @britta - we will also have to gather information for daly weights for each condition and do a simple
        #  mapping to them from the properties. For now, anyone who has anything has a daly_wt of 0.1

        df = self.sim.population.props
        any_condition = df.loc[df.is_alive, self.conditions].any(axis=1)

        return any_condition * 0.1

        pass

    def on_hsi_alert(self, person_id, treatment_id):
        """
        This is called whenever there is an HSI event commissioned by one of the other disease modules.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------------------------------------
#   DISEASE MODULE EVENTS
#
#   These are the events which drive the simulation of the disease. It may be a regular event that updates
#   the status of all the population of subsections of it at one time. There may also be a set of events
#   that represent disease events for particular persons.
# ---------------------------------------------------------------------------------------------------------

class Ncds_MainPollingEvent(RegularEvent, PopulationScopeEventMixin):
    """The Main Polling Event.
    * Establishes onset of each condition
    * Establishes removal of each condition
    * Schedules events that arise, according the condition.
    """

    def __init__(self, module, interval_between_polls):
        """The Main Polling Event of the NCDs Module

        :param module: the module that created this event
        """
        super().__init__(module, frequency=DateOffset(months=interval_between_polls))
        assert isinstance(module, Ncds)

    def apply(self, population):
        """Apply this event to the population.

        :param population: the current population
        """
        df = population.props
        m = self.module
        rng = m.rng

        # Determine onset/removal of conditions
        for condition in self.module.conditions:

            # onset:
            eligible_population = df.is_alive & ~df[condition]
            acquires_condition = self.module.lms_onset[condition].predict(df.loc[eligible_population], rng)
            idx_acquires_condition = acquires_condition[acquires_condition].index
            df.loc[idx_acquires_condition, condition] = True

            # -------------------------------------------------------------------------------------------
            # Add this incident case to the tracker

            for personal_idx in idx_acquires_condition:
                age_range = df.loc[personal_idx, ['age_range']].age_range
                self.module.incident_case_tracker[age_range][condition].append(self.sim.date)
            # -------------------------------------------------------------------------------------------

            # removal:
            # df.loc[
            # self.module.lms_removal[condition].predict(df.loc[df.is_alive & df[condition]
            # ],
            # self.module.rng), condition] = False

            # -------------------- DEATH FROM NCD CONDITION ---------------------------------------
            # There is a risk of death for those who have an NCD condition. Death is assumed to happen instantly.

            # Strip leading 'nc_' from condition name
            condition_name = condition.replace('nc_', '')

            condition_idx = df.index[df.is_alive & (df[f'{condition}'])]
            selected_to_die = condition_idx[
                rng.random_sample(size=len(condition_idx)) < self.module.parameters[f'r_death_{condition}']]

            for person_id in selected_to_die:
                self.sim.schedule_event(
                    InstantaneousDeath(self.module, person_id, f"{condition_name}"), self.sim.date
                )

        # Determine occurrence of events
        for event in self.module.events:

            eligible_population_for_event = df.is_alive
            has_event = self.module.lms_event_occurrence[event].predict(df.loc[eligible_population_for_event], rng)
            idx_has_event = has_event[has_event].index

            if event == 'nc_ever_stroke':
                for person_id in idx_has_event:
                    self.sim.schedule_event(NcdStrokeEvent(self.module, person_id),
                                            self.sim.date + DateOffset(days=self.module.rng.randint(0, 90)))
            elif event == 'nc_ever_heart_attack':
                for person_id in idx_has_event:
                    self.sim.schedule_event(NcdHeartAttackEvent(self.module, person_id),
                                            self.sim.date + DateOffset(days=self.module.rng.randint(0, 90)))


class NcdStrokeEvent(Event, IndividualScopeEventMixin):
    """
    This is a Stroke event. It has been scheduled to occur by the Ncds_MainPollingEvent.
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        if not self.sim.population.props.at[person_id, 'is_alive']:
            return

        self.module.eventsTracker['StrokeEvents'] += 1
        self.sim.population.props.at[person_id, 'nc_ever_stroke'] = True

        ## Add the outward symptom to the SymptomManager. This will result in emergency care being sought
        # self.sim.modules['SymptomManager'].change_symptom(
        #    person_id=person_id,
        #    disease_module=self.module,
        #    add_or_remove='+',
        #    symptom_string='Damage_From_Stroke'
        # )

class NcdHeartAttackEvent(Event, IndividualScopeEventMixin):
    """
    This is a Stroke event. It has been scheduled to occur by the Ncds_MainPollingEvent.
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

    def apply(self, person_id):
        if not self.sim.population.props.at[person_id, 'is_alive']:
            return

        self.module.eventsTracker['HeartAttackEvents'] += 1
        self.sim.population.props.at[person_id, 'nc_ever_heart_attack'] = True

        ## Add the outward symptom to the SymptomManager. This will result in emergency care being sought
        # self.sim.modules['SymptomManager'].change_symptom(
        #    person_id=person_id,
        #    disease_module=self.module,
        #    add_or_remove='+',
        #    symptom_string='Damage_From_Heart_Attack'
        # )


# ---------------------------------------------------------------------------------------------------------
#   LOGGING EVENTS
#
#   Put the logging events here. There should be a regular logger outputting current states of the
#   population. There may also be a loggig event that is driven by particular events.
# ---------------------------------------------------------------------------------------------------------

class Ncds_LoggingEvent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        """Produce a summary of the numbers of people with respect to the action of this module.
        """

        self.repeat = 12
        super().__init__(module, frequency=DateOffset(months=self.repeat))
        self.date_last_run = self.sim.date
        self.AGE_RANGE_LOOKUP = de.Demography.AGE_RANGE_LOOKUP
        assert isinstance(module, Ncds)

    def apply(self, population):

        # Convert the list of timestamps into a number of timestamps
        # and check that all the dates have occurred since self.date_last_run
        counts = copy.deepcopy(self.module.incident_case_tracker_zeros)

        for age_grp in self.module.incident_case_tracker.keys():
            for condition in self.module.conditions:
                list_of_times = self.module.incident_case_tracker[age_grp][condition]
                counts[age_grp][condition] = len(list_of_times)
                for t in list_of_times:
                    assert self.date_last_run <= t <= self.sim.date

        logger.info(key='incidence_count_by_condition', data=counts)

        # Reset the counters and the date_last_run
        self.module.incident_case_tracker = copy.deepcopy(self.module.incident_case_tracker_blank)
        self.date_last_run = self.sim.date

        # Output the person-years lived by single year of age in the past year
        # py = self.module.calc_py_lived_in_last_year()
        # logger.info(key='person_years', data=py.to_dict())

        df = self.sim.population.props
        delta = pd.DateOffset(years=1)
        for cond in self.module.conditions:
            # mask is a Series restricting dataframe to individuals who do not have the condition, which is passed to
            # demography module to calculate person-years lived without the condition
            mask = (df.is_alive & ~df[f'{cond}'])
            py = de.Demography.calc_py_lived_in_last_year(self, delta, mask)
            logger.info(key=f'person_years_{cond}', data=py.to_dict())

        # Make some summary statistics for prevalence by age/sex for each condition
        df = population.props

        def proportion_of_something_in_a_groupby_ready_for_logging(df, something, groupbylist):
            dfx = df.groupby(groupbylist).apply(lambda dft: pd.Series(
                {'something': dft[something].sum(), 'not_something': (~dft[something]).sum()}))
            pr = dfx['something'] / dfx.sum(axis=1)

            # create into a dict with keys as strings
            pr = pr.reset_index()
            pr['flat_index'] = ''
            for i in range(len(pr)):
                pr.at[i, 'flat_index'] = '__'.join([f"{col}={pr.at[i, col]}" for col in groupbylist])
            pr = pr.set_index('flat_index', drop=True)
            pr = pr.drop(columns=groupbylist)
            return pr[0].to_dict()

        # Prevalence of conditions broken down by sex and age

        for condition in self.module.conditions:
            # Strip leading 'nc_' from condition name
            condition_name = condition.replace('nc_', '')

            # Prevalence of conditions broken down by sex and age
            logger.info(
                key=f'{condition_name}_prevalence_by_age_and_sex',
                description='current fraction of the population classified as having condition, by sex and age',
                data={'data': proportion_of_something_in_a_groupby_ready_for_logging(df, f'{condition}',
                                                                                     ['sex', 'age_range'])}
            )

            # Prevalence of conditions by adults aged 20 or older
            adult_prevalence = {
                'prevalence': len(df[df[f'{condition}'] & df.is_alive & (df.age_years >= 20)]) / len(
                    df[df.is_alive & (df.age_years >= 20)])}

            logger.info(
                key=f'{condition_name}_prevalence',
                description='current fraction of the adult population classified as having condition',
                data=adult_prevalence
            )

        # Counter for number of co-morbidities
        df = population.props
        # restrict df to alive and aged >=20
        df_comorbidities = df[df.is_alive & (df.age_years >= 20)]
        # restrict df to list of conditions
        df_comorbidities = df_comorbidities[
            df_comorbidities.columns[df_comorbidities.columns.isin(self.module.conditions)]]
        # calculate number of conditions by row
        df_comorbidities['n_conditions'] = df_comorbidities.sum(axis=1)
        n_comorbidities = df_comorbidities['n_conditions'].value_counts()
        prop_comorbidities = n_comorbidities / len(df[(df.is_alive & (df.age_years >= 20))])

        logger.info(key='mm_prevalence',
                    data=prop_comorbidities,
                    description='annual summary of multi-morbidities')

        # NB. logging like this as cannot do directly as a dict [logger requires key in a dict to be strings]
