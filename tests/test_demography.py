import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pytest import approx

from tlo import Date, Module, Simulation, logging
from tlo.analysis.utils import compare_number_of_deaths, parse_log_file
from tlo.methods import Metadata, demography
from tlo.methods.causes import Cause
from tlo.methods.demography import AgeUpdateEvent
from tlo.methods.diarrhoea import increase_risk_of_death, make_treatment_perfect
from tlo.methods.fullmodel import fullmodel

start_date = Date(2010, 1, 1)
end_date = Date(2015, 1, 1)
popsize = 500


@pytest.fixture
def simulation(seed):
    resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
    sim = Simulation(start_date=start_date, seed=seed)
    core_module = demography.Demography(resourcefilepath=resourcefilepath)
    sim.register(core_module)
    return sim


def check_dtypes(simulation):
    # check types of columns
    df = simulation.population.props
    orig = simulation.population.new_row
    assert (df.dtypes == orig.dtypes).all()


def test_run_dtypes_and_mothers_female(simulation):
    simulation.make_initial_population(n=popsize)
    simulation.simulate(end_date=end_date)
    assert set(['Other']) == set(simulation.population.props['cause_of_death'].cat.categories)
    check_dtypes(simulation)
    # check all mothers are female
    df = simulation.population.props
    mothers = df.loc[df.mother_id >= 0, 'mother_id']
    is_female = mothers.apply(lambda mother_id: df.at[mother_id, 'sex'] == 'F')
    assert is_female.all()


def test_storage_of_cause_of_death(seed):
    rfp = Path(os.path.dirname(__file__)) / '../resources'

    class DummyModule(Module):
        METADATA = {Metadata.DISEASE_MODULE}
        CAUSES_OF_DEATH = {'a_cause': Cause(label='a_cause')}

        def read_parameters(self, data_folder):
            pass

        def initialise_population(self, population):
            pass

        def initialise_simulation(self, sim):
            pass

    sim = Simulation(start_date=Date(2010, 1, 1), seed=seed)
    sim.register(
        demography.Demography(resourcefilepath=rfp),
        DummyModule()
    )
    sim.make_initial_population(n=20)
    df = sim.population.props
    orig = df.dtypes
    assert type(orig['cause_of_death']) == pd.CategoricalDtype
    assert set(['Other', 'a_cause']) == set(df['cause_of_death'].cat.categories)

    # Cause a person to die by the DummyModule
    person_id = 0
    sim.modules['Demography'].do_death(
        individual_id=person_id,
        originating_module=sim.modules['DummyModule'],
        cause='a_cause'
    )

    person = df.loc[person_id]
    assert not person.is_alive
    assert person.cause_of_death == 'a_cause'
    assert (df.dtypes == orig).all()
    check_dtypes(sim)


@pytest.mark.slow
def test_cause_of_death_being_registered(tmpdir, seed):
    """Test that the modules can declare causes of death, that the mappers between tlo causes of death and gbd
    causes of death can be created correctly and that the analysis helper scripts can be used to produce comparisons
    between model outputs and GBD data."""
    rfp = Path(os.path.dirname(__file__)) / '../resources'

    sim = Simulation(start_date=Date(2010, 1, 1), seed=seed, log_config={
        'filename': 'temp',
        'directory': tmpdir,
        'custom_levels': {
            "*": logging.WARNING,
            'tlo.methods.demography': logging.INFO
        }
    })

    sim.register(*fullmodel(resourcefilepath=rfp, healthsystem_disable=True))

    # Increase risk of death of Diarrhoea to ensure that are at least some deaths
    increase_risk_of_death(sim.modules['Diarrhoea'])
    make_treatment_perfect(sim.modules['Diarrhoea'])

    sim.make_initial_population(n=1000)
    sim.simulate(end_date=Date(2010, 12, 31))
    check_dtypes(sim)

    mapper_from_tlo_causes, mapper_from_gbd_causes = \
        sim.modules['Demography'].create_mappers_from_causes_of_death_to_label()

    assert set(mapper_from_tlo_causes.keys()) == set(sim.modules['Demography'].causes_of_death)
    assert set(mapper_from_gbd_causes.keys()) == sim.modules['Demography'].gbd_causes_of_death
    assert set(mapper_from_gbd_causes.values()) == set(mapper_from_tlo_causes.values())

    # check that these mappers come out in the log correctly
    output = parse_log_file(sim.log_filepath)
    demoglog = output['tlo.methods.demography']
    assert mapper_from_tlo_causes == \
           pd.Series(demoglog['mapper_from_tlo_cause_to_common_label'].drop(columns={'date'}).loc[0]).to_dict()
    assert mapper_from_gbd_causes == \
           pd.Series(demoglog['mapper_from_gbd_cause_to_common_label'].drop(columns={'date'}).loc[0]).to_dict()

    # Check that the mortality risks being used in Other Death Poll have been reduced from the 'all-cause' rates
    odp = sim.modules['Demography'].other_death_poll
    all_cause_risk = odp.get_all_cause_mort_risk_per_poll()
    actual_risk_per_poll = odp.mort_risk_per_poll
    assert (
        actual_risk_per_poll['prob_of_dying_before_next_poll'] < all_cause_risk['prob_of_dying_before_next_poll']
    ).all()

    # check that can recover from the log the proportion of deaths represented by the OtherDeaths
    logged_prop_of_death_by_odp = demoglog['other_deaths'][['Sex', 'Age_Grp', '0']].to_dict()
    dict_of_ser = {k: pd.DataFrame(v)[0] for k, v in logged_prop_of_death_by_odp.items()}
    log_odp = pd.concat(dict_of_ser, axis=1).set_index(['Sex', 'Age_Grp'])['0']
    assert (log_odp < 1.0).all()

    # Run the analysis file:
    results = compare_number_of_deaths(logfile=sim.log_filepath, resourcefilepath=rfp)
    # Check the number of deaths in model represented is right (allowing for the scaling factor)
    assert (results['model'].sum() * 5.0) == approx(len(output['tlo.methods.demography']['death'])
                                                    / sim.modules['Demography'].initial_model_to_data_popsize_ratio
                                                    )


@pytest.mark.slow
def test_calc_of_scaling_factor(tmpdir, seed):
    """Test that the scaling factor is computed and put out to the log"""
    rfp = Path(os.path.dirname(__file__)) / '../resources'
    popsize = 10_000
    sim = Simulation(start_date=Date(2010, 1, 1), seed=seed, log_config={
        'filename': 'temp',
        'directory': tmpdir,
        'custom_levels': {
            "*": logging.INFO,
        }
    })
    sim.register(
        demography.Demography(resourcefilepath=rfp),
    )
    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=sim.start_date)

    # Check that the scaling factor is calculated in the log correctly:
    output = parse_log_file(sim.log_filepath)
    sf = output['tlo.methods.demography']['scaling_factor'].at[0, 'scaling_factor']
    assert sf == approx(14.5e6 / popsize, rel=0.10)

    # Check that the scaling factor is also logged in `tlo.methods.population`
    assert output['tlo.methods.demography']['scaling_factor'].at[0, 'scaling_factor'] == \
           output['tlo.methods.population']['scaling_factor'].at[0, 'scaling_factor']


def test_py_calc(simulation):
    # make population of one person:
    simulation.make_initial_population(n=1)

    df = simulation.population.props
    df.sex = 'M'
    simulation.date += pd.DateOffset(days=1)
    age_update = AgeUpdateEvent(simulation.modules['Demography'], simulation.modules['Demography'].AGE_RANGE_LOOKUP)
    now = simulation.date
    one_year = np.timedelta64(1, 'Y')
    one_month = np.timedelta64(1, 'M')

    calc_py_lived_in_last_year = simulation.modules['Demography'].calc_py_lived_in_last_year

    # calc py: person is born and died before sim.date
    df.date_of_birth = now - (one_year * 10)
    df.date_of_death = now - (one_year * 9)
    df.is_alive = False
    age_update.apply(simulation.population)
    assert (0 == calc_py_lived_in_last_year(delta=one_year)['M']).all()

    # calc py of person who is not yet born:
    df.date_of_birth = pd.NaT
    df.date_of_death = pd.NaT
    df.is_alive = False
    age_update.apply(simulation.population)
    assert (0 == calc_py_lived_in_last_year(delta=one_year)['M']).all()

    # calc person who is alive and aged 20, with birthdays on today's date and lives throughout the period
    df.date_of_birth = now - (one_year * 20)
    df.date_of_death = pd.NaT
    df.is_alive = True
    age_update.apply(simulation.population)
    np.testing.assert_almost_equal(calc_py_lived_in_last_year(delta=one_year)['M'][19], 1.0)

    # calc person who is alive and aged 20, with birthdays on today's date, and dies 3 months ago
    df.date_of_birth = now - (one_year * 20)
    df.date_of_death = now - pd.Timedelta(one_year) * 0.25
    # we have to set the age at time of death - usually this would have been set by the AgeUpdateEvent
    df.age_exact_years = (df.date_of_death - df.date_of_birth) / one_year
    df.age_years = df.age_exact_years.astype('int64')
    df.is_alive = False
    age_update.apply(simulation.population)
    df_py = calc_py_lived_in_last_year(delta=one_year)
    np.testing.assert_almost_equal(0.75, df_py['M'][19])
    assert df_py['M'][20] == 0.0

    # calc person who is alive and aged 19, has birthday mid-way through the last year, and lives throughout
    df.date_of_birth = now - (one_year * 20) - (one_month * 6)
    df.date_of_death = pd.NaT
    df.is_alive = True
    age_update.apply(simulation.population)
    df_py = calc_py_lived_in_last_year(delta=one_year)
    np.testing.assert_almost_equal(0.5, df_py['M'][19])
    np.testing.assert_almost_equal(0.5, df_py['M'][20])

    # calc person who is alive and aged 19, has birthday mid-way through the last year, and died 3 months ago
    df.date_of_birth = now - (one_year * 20) - (one_month * 6)
    df.date_of_death = now - np.timedelta64(3, 'M')
    # we have to set the age at time of death - usually this would have been set by the AgeUpdateEvent
    df.age_exact_years = (df.date_of_death - df.date_of_birth) / one_year
    df.age_years = df.age_exact_years.astype('int64')
    df.is_alive = False
    age_update.apply(simulation.population)
    df_py = calc_py_lived_in_last_year(delta=one_year)
    assert 0.75 == df_py['M'].sum()
    assert 0.5 == df_py['M'][19]
    assert 0.25 == df_py['M'][20]

    # 0/1 year-old with first birthday during the last year
    df.date_of_birth = now - (one_month * 15)
    df.date_of_death = pd.NaT
    df.is_alive = True
    age_update.apply(simulation.population)
    df_py = calc_py_lived_in_last_year(delta=one_year)
    assert 0.75 == df_py['M'][0]
    assert 0.25 == df_py['M'][1]

    # 0 year born in the last year
    df.date_of_birth = now - (one_month * 9)
    df.date_of_death = pd.NaT
    df.is_alive = True
    age_update.apply(simulation.population)
    df_py = calc_py_lived_in_last_year(delta=one_year)
    assert 0.75 == df_py['M'][0]
    assert (0 == df_py['M'][1:]).all()

    # 99 years-old turning 100 in the last year
    df.date_of_birth = now - (one_year * 100) - (one_month * 6)
    df.date_of_death = pd.NaT
    df.is_alive = True
    age_update.apply(simulation.population)
    df_py = calc_py_lived_in_last_year(delta=one_year)
    assert 0.5 == df_py['M'][99]
    assert 1 == df_py['M'].sum()


def test_py_calc_w_mask(simulation):
    """test that function calc_py_lived_in_last_year works to calculate PY lived without a given condition """

    # make population of two people
    simulation.make_initial_population(n=2)

    df = simulation.population.props
    df.sex = 'M'
    simulation.date += pd.DateOffset(days=1)
    age_update = AgeUpdateEvent(simulation.modules['Demography'], simulation.modules['Demography'].AGE_RANGE_LOOKUP)
    now = simulation.date
    one_year = np.timedelta64(1, 'Y')

    calc_py_lived_in_last_year = simulation.modules['Demography'].calc_py_lived_in_last_year

    # calc two people who are alive and aged 20, with birthdays on today's date and live throughout the period,
    # neither has hypertension

    df.date_of_birth = now - (one_year * 20)
    df.date_of_death = pd.NaT
    df['nc_hypertension'] = False
    mask = (df.is_alive & ~df['nc_hypertension'])
    df = df[mask]
    age_update.apply(simulation.population)
    df_py = calc_py_lived_in_last_year(delta=one_year, mask=mask)
    np.testing.assert_almost_equal(2.0, df_py['M'][19])

    # calc two people who are alive and aged 20, with birthdays on today's date and live throughout the period,
    # one has hypertension

    df.date_of_birth = now - (one_year * 20)
    df.date_of_death = pd.NaT
    df['nc_hypertension'].iloc[0] = True
    mask = (df.is_alive & ~df['nc_hypertension'])
    df = df[mask]
    age_update.apply(simulation.population)
    df_py = calc_py_lived_in_last_year(delta=one_year, mask=mask)
    np.testing.assert_almost_equal(1.0, df_py['M'][19])


def test_max_age_initial(seed):
    """Check that the parameter in the `Demography` module, `max_age_initial`, works as expected
     * `max_age_initial=X`: only persons up to and including age_years (age in whole years) up to X are included in the
      initial population.
     * `max_age_initial=0` or `>MAX_AGE`: results in an error being thrown.
    """

    from tlo.methods.demography import MAX_AGE

    def max_age_in_sim_with_max_age_initial_argument(_max_age_initial):
        """Return the greatest value of `age_years` in a population that is created."""
        resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
        sim = Simulation(start_date=start_date, seed=seed)
        sim.register(
            demography.Demography(resourcefilepath=resourcefilepath)
        )
        sim.modules['Demography'].parameters['max_age_initial'] = _max_age_initial
        sim.make_initial_population(n=50_000)
        return sim.population.props.age_years.max()

    # `max_age_initial=5` (using integer)
    assert max_age_in_sim_with_max_age_initial_argument(5) <= 5

    # `max_age_initial=5.5` (using float)
    assert max_age_in_sim_with_max_age_initial_argument(5.5) <= int(5.5)

    # `max_age_initial=0`
    with pytest.raises(ValueError):
        max_age_in_sim_with_max_age_initial_argument(0)

    # `max_age_initial>MAX_AGE`
    with pytest.raises(ValueError):
        max_age_in_sim_with_max_age_initial_argument(MAX_AGE + 1)


def test_can_turn_off_the_detailed_logger_when_using_custom_log_after_registering(seed, tmpdir):
    """Check that the simulation can be set-up to get only the usual demography logger and not the detailed logger,
    when the custom_log information is given after the models are registered."""

    log_config = {
        'filename': 'temp',
        'directory': tmpdir,
        'custom_levels': {
            "*": logging.WARNING,
            'tlo.methods.demography.detail': logging.WARNING,  # <-- Explicitly turning off the detailed logger
            'tlo.methods.demography': logging.INFO,  # <-- Turning on the normal logger
        }
    }

    rfp = Path(os.path.dirname(__file__)) / '../resources'

    sim = Simulation(start_date=Date(2010, 1, 1), seed=seed)
    sim.register(demography.Demography(resourcefilepath=rfp))
    sim.configure_logging(**log_config)
    sim.make_initial_population(n=100)
    sim.simulate(end_date=sim.start_date)

    # Cause one death to occur
    sim.modules['Demography'].do_death(
        individual_id=0,
        originating_module=sim.modules['Demography'],
        cause='Other'
    )

    log = parse_log_file(sim.log_filepath)

    # Check the usual `tlo.methods.demography' log is created and that check persons have died (which would be when the
    # detailed logger would be used).
    assert 'tlo.methods.demography' in log.keys()
    assert 1 == len(log['tlo.methods.demography']['death'])

    # Check that the detailed logger is not created.
    assert 'tlo.methods.demography.detail' not in log.keys()


def test_can_turn_off_the_detailed_logger_when_using_custom_log_when_registering(seed, tmpdir):
    """Check that the simulation can be set-up to get only the usual demography logger and not the detailed logger,
    when providing the config_log information when the simulation is initialised."""

    log_config = {
        'filename': 'temp',
        'directory': tmpdir,
        'custom_levels': {
            "*": logging.WARNING,
            'tlo.methods.demography.detail': logging.WARNING,  # <-- Explicitly turning off the detailed logger
            'tlo.methods.demography': logging.INFO,  # <-- Turning on the normal logger
        }
    }

    rfp = Path(os.path.dirname(__file__)) / '../resources'

    sim = Simulation(start_date=Date(2010, 1, 1), seed=seed, log_config=log_config)

    sim.register(demography.Demography(resourcefilepath=rfp))
    sim.make_initial_population(n=100)
    sim.simulate(end_date=sim.start_date)

    # Cause one death to occur
    sim.modules['Demography'].do_death(
        individual_id=0,
        originating_module=sim.modules['Demography'],
        cause='Other'
    )

    log = parse_log_file(sim.log_filepath)

    # Check the usual `tlo.methods.demography' log is created and that check persons have died (which would be when the
    # detailed logger would be used).
    assert 'tlo.methods.demography' in log.keys()
    assert 1 == len(log['tlo.methods.demography']['death'])

    # Check that the detailed logger is not created.
    assert 'tlo.methods.demography.detail' not in log.keys()
