import os
from pathlib import Path

import numpy as np
import pandas as pd

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    contraception,
    demography,
    depression,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    labour,
    pregnancy_supervisor,
    symptommanager, ncds,
)

try:
    resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
except NameError:
    # running interactively
    resourcefilepath = Path('./resources')

output_files = dict()


def routine_checks(sim):
    """
    Insert checks here:
    """

    # Check types of columns
    df = sim.population.props
    orig = sim.population.new_row
    #assert (df.dtypes == orig.dtypes).all()

    # check that someone has had onset of each condition

    df = sim.population.props
    assert df.nc_diabetes.any()
    assert df.nc_hypertension.any()
    assert df.nc_depression.any()
    assert df.nc_chronic_lower_back_pain.any()
    assert df.nc_chronic_kidney_disease.any()
    assert df.nc_chronic_ischemic_hd.any()

    # check that someone has had onset of each event

    assert df.nc_ever_stroke.any()
    assert df.nc_ever_heart_attack.any()

    # check that someone dies of each condition that has a death rate associated with it

    assert df.cause_of_death.loc[~df.is_alive].str.startswith('diabetes').any()
    assert df.cause_of_death.loc[~df.is_alive].str.startswith('chronic_ischemic_hd').any()
    assert df.cause_of_death.loc[~df.is_alive].str.startswith('chronic_kidney_disease').any()
    assert df.cause_of_death.loc[~df.is_alive].str.startswith('stroke').any()
    assert df.cause_of_death.loc[~df.is_alive].str.startswith('heart_attack').any()

    # check that no one dies of each condition that has a death rate of zero
    assert not df.cause_of_death.loc[~df.is_alive].str.startswith('chronic_lower_back_pain').any()
    assert not df.cause_of_death.loc[~df.is_alive].str.startswith('hypertension').any()


def test_basic_run():
    # --------------------------------------------------------------------------
    # Create and run a short but big population simulation for use in the tests
    sim = Simulation(start_date=Date(year=2010, month=1, day=1), seed=0)

    # Register the appropriate modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 contraception.Contraception(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath, disable=True),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 labour.Labour(resourcefilepath=resourcefilepath),
                 pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
                 depression.Depression(resourcefilepath=resourcefilepath),
                 ncds.Ncds(resourcefilepath=resourcefilepath)
                 )

    # Set incidence/death rates of conditions to high values to decrease run time

    p = sim.modules['Ncds'].parameters

    p['nc_diabetes_onset'].loc[
        p['nc_diabetes_onset'].parameter_name == "baseline_annual_probability", "value"] = 0.75
    p['nc_hypertension_onset'].loc[
        p['nc_hypertension_onset'].parameter_name == "baseline_annual_probability", "value"] = 0.75
    p['nc_chronic_lower_back_pain_onset'].loc[
        p['nc_chronic_lower_back_pain_onset'].parameter_name == "baseline_annual_probability", "value"] = 0.75
    p['nc_chronic_kidney_disease_onset'].loc[
        p['nc_chronic_kidney_disease_onset'].parameter_name == "baseline_annual_probability", "value"] = 0.75
    p['nc_ever_stroke_onset'].loc[
        p['nc_ever_stroke_onset'].parameter_name == "baseline_annual_probability", "value"] = 0.75
    p['nc_ever_heart_attack_onset'].loc[
        p['nc_ever_heart_attack_onset'].parameter_name == "baseline_annual_probability", "value"] = 0.75
    p['nc_diabetes_death'].loc[
        p['nc_diabetes_death'].parameter_name == "baseline_annual_probability", "value"] = 0.75
    p['nc_chronic_kidney_disease_death'].loc[
        p['nc_chronic_kidney_disease_death'].parameter_name == "baseline_annual_probability", "value"] = 0.75
    p['nc_ever_stroke_death'].loc[
        p['nc_ever_stroke_death'].parameter_name == "baseline_annual_probability", "value"] = 0.75
    p['nc_ever_heart_attack_death'].loc[
        p['nc_ever_heart_attack_death'].parameter_name == "baseline_annual_probability", "value"] = 0.75

    sim.make_initial_population(n=5000)
    sim.simulate(end_date=Date(year=2011, month=1, day=1))

    routine_checks(sim)


def test_basic_run_with_high_incidence_hypertension():
    # Create and run a short but big population simulation for use in the tests
    """make one ncd very common and the others non-existent to check basic functions for prevalence and death"""

    sim = Simulation(start_date=Date(year=2010, month=1, day=1), seed=0)

    # Register the appropriate modules
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 contraception.Contraception(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath, disable=True),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 labour.Labour(resourcefilepath=resourcefilepath),
                 pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
                 depression.Depression(resourcefilepath=resourcefilepath),
                 ncds.Ncds(resourcefilepath=resourcefilepath)
                 )

    # Set incidence of hypertension to 1 and incidence of all other conditions to 0

    p = sim.modules['Ncds'].parameters

    p['nc_hypertension_onset'].loc[
        p['nc_hypertension_onset'].parameter_name == "baseline_annual_probability", "value"] = 1
    p['nc_diabetes_onset'].loc[
        p['nc_diabetes_onset'].parameter_name == "baseline_annual_probability", "value"] = 0
    p['nc_chronic_lower_back_pain_onset'].loc[
        p['nc_chronic_lower_back_pain_onset'].parameter_name == "baseline_annual_probability", "value"] = 0
    p['nc_chronic_kidney_disease_onset'].loc[
        p['nc_chronic_kidney_disease_onset'].parameter_name == "baseline_annual_probability", "value"] = 0

    # Increase RR of heart disease very high if individual has hypertension
    p['nc_chronic_ischemic_hd_onset'].loc[
        p['nc_chronic_ischemic_hd_onset'].parameter_name == "rr_hypertension", "value"] = 1000

    sim.make_initial_population(n=2000)
    sim.simulate(end_date=Date(year=2020, month=1, day=1))

    df = sim.population.props

    # check that no one has any conditions that were set to zero incidence
    assert ~df.nc_diabetes.all()
    assert ~df.nc_chronic_lower_back_pain.all()
    assert ~df.nc_chronic_kidney_disease.all()

    # check that no one has died from conditions that were set to zero incidence
    assert not df.loc[~df.is_alive & ~pd.isnull(df.date_of_birth), 'cause_of_death'].str.startswith('diabetes').any()
    assert not df.loc[~df.is_alive & ~pd.isnull(df.date_of_birth), 'cause_of_death'].str.startswith(
        'chronic_lower_back_pain').any()
    assert not df.loc[~df.is_alive & ~pd.isnull(df.date_of_birth), 'cause_of_death'].str.startswith(
        'chronic_kidney_disease').any()

    # restrict population to individuals aged >=20 at beginning of sim
    start_date = pd.Timestamp(year=2010, month=1, day=1)
    df['start_date'] = pd.to_datetime(start_date)
    df['diff_years'] = df.start_date - df.date_of_birth
    df['diff_years'] = df.diff_years / np.timedelta64(1, 'Y')
    df = df[df['diff_years'] >= 20]
    df = df[df.is_alive]

    hypertension_prev = (len(df[df.nc_hypertension & df.is_alive & (df.age_years >= 20)])) / \
                        (len(df[df.is_alive & (df.age_years >= 20)]))
    cihd_prev = (len(df[df.nc_chronic_ischemic_hd & df.is_alive & (df.age_years >= 20)])) / \
                (len(df[df.is_alive & (df.age_years >= 20)]))

    # check that everyone has hypertension and CIHD by end
    assert df.loc[df.is_alive & (df.age_years >= 20)].nc_hypertension.all()
    assert df.loc[df.is_alive & (df.age_years >= 20)].nc_chronic_ischemic_hd.all()
