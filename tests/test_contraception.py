import os
from pathlib import Path

import pandas as pd

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    care_of_women_during_pregnancy,
    contraception,
    demography,
    enhanced_lifestyle,
    healthsystem,
    labour,
    newborn_outcomes,
    postnatal_supervisor,
    pregnancy_supervisor,
    symptommanager,
)
from tlo.methods.hiv import DummyHivModule


def run_sim(tmpdir,
            use_healthsystem=False,
            disable=False,
            healthsystem_disable_and_reject_all=False,
            consumables_available=True,
            run=True,
            no_discontinuation=False,
            popsize=1000,
            end_date=Date(2011, 12, 31)
            ):
    """Run basic checks on function of contraception module"""

    def __check_properties(df):
        """basic checks on configuration of properties"""
        assert not ((~df.date_of_birth.isna()) & (df.sex == 'M') & (df.co_contraception != 'not_using')).any()
        assert not ((~df.date_of_birth.isna()) & (df.age_years < 15) & (df.co_contraception != 'not_using')).any()
        assert not ((~df.date_of_birth.isna()) & (df.sex == 'M') & df.is_pregnant).any()

    def __check_dtypes(simulation):
        """check types of columns"""
        df = simulation.population.props
        orig = simulation.population.new_row
        assert (df.dtypes == orig.dtypes).all()

    resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'

    start_date = Date(2010, 1, 1)

    log_config = {
        'filename': 'temp',
        'directory': tmpdir,
        'custom_levels': {
            "*": logging.WARNING,
            'tlo.methods.contraception': logging.INFO
        }
    }

    sim = Simulation(start_date=start_date, log_config=log_config, seed=0)

    sim.register(
        # - core modules:
        demography.Demography(resourcefilepath=resourcefilepath),
        enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
        symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
        healthsystem.HealthSystem(resourcefilepath=resourcefilepath,
                                  disable=disable,
                                  disable_and_reject_all=healthsystem_disable_and_reject_all,
                                  ignore_cons_constraints=consumables_available,
                                  ),

        # - modules for mechanistic representation of contraception -> pregnancy -> labour -> delivery etc.
        contraception.Contraception(resourcefilepath=resourcefilepath, use_healthsystem=use_healthsystem),
        pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
        care_of_women_during_pregnancy.CareOfWomenDuringPregnancy(resourcefilepath=resourcefilepath),
        labour.Labour(resourcefilepath=resourcefilepath),
        newborn_outcomes.NewbornOutcomes(resourcefilepath=resourcefilepath),
        postnatal_supervisor.PostnatalSupervisor(resourcefilepath=resourcefilepath),

        # - Dummy HIV module (as contraception requires the property hv_inf)
        DummyHivModule()
    )

    if not consumables_available:
        # Make consumables not available
        sim.modules['HealthSystem'].prob_item_codes_available.loc[:] = 0.0

    if no_discontinuation:
        sim.modules['Contraception'].parameters['Discontinuation'] *= 0.0

    sim.make_initial_population(n=popsize)
    __check_dtypes(sim)
    __check_properties(sim.population.props)

    # Make most of the population women
    df = sim.population.props
    df.loc[df.is_alive, 'sex'] = sim.modules['Demography'].rng.choice(['M', 'F'], p=[0.5, 0.5],
                                                                      size=df.is_alive.sum())
    df.loc[(df.sex == 'M'), "co_contraception"] = "not_using"

    if not run:
        return sim

    sim.simulate(end_date=end_date)
    __check_dtypes(sim)
    __check_properties(sim.population.props)

    return sim


def __check_some_starting_swtiching_and_stopping(sim):
    """Check that there is at least some usage and some starting, switching and stopping"""

    logs = parse_log_file(sim.log_filepath)

    # Check that yearly-summary logs are as expected and that some use of contraception is happening:
    ys = logs['tlo.methods.contraception']['contraception_use_yearly_summary']
    ys = ys.set_index('date')
    assert set(ys.columns) == sim.modules['Contraception'].all_contraception_states
    assert (ys.drop(columns=['not_using']).sum(axis=1) > 0).all()

    # Check that there is some starting, switching and stopping:
    contraception_change = logs['tlo.methods.contraception']['contraception_change']

    # some starting
    assert len(contraception_change.loc[contraception_change.switch_from == "not_using"])

    # some stopping
    assert len(contraception_change.loc[contraception_change.switch_to == "not_using"])

    # some switching
    assert len(contraception_change.loc[
                   (contraception_change.switch_from != "not_using") & (contraception_change.switch_to != "not_using")])


def __check_no_illegal_switches(sim):
    """Check that there are no illegal switches happening (from `female_sterilization` or to if if age <30).
    This is especially important when the configuration is such that people are defaulting from methods when
    HealthSystem/consumables are not available."""
    logs = parse_log_file(sim.log_filepath)

    # Check for no illegal changes
    if 'tlo.methods.contraception' in logs:
        if 'contraception_change' in logs['tlo.methods.contraception']:
            con = logs['tlo.methods.contraception']['contraception_change']
            assert not (con.switch_from == 'female_sterilization').any()  # no switching from female_sterilization
            assert not (con.loc[con['age_years'] <= 30, 'switch_to'] == 'female_sterilization').any()  # no switching to
            # female_sterilization if age less than 30 (or equal to, in case they have aged since an HSI was scheduled)


def test_contraception_use_and_not_using_healthsystem(tmpdir):
    """Test that the contraception module function and that what comes out in log is as expected when initiation and
    switching is NOT going through the HealthSystem."""

    # Run basic check, for the case when the model is using the healthsystem and when not and check the logs
    sim_does_not_use_healthsystem = run_sim(tmpdir=tmpdir, use_healthsystem=False, disable=True)
    __check_no_illegal_switches(sim_does_not_use_healthsystem)
    __check_some_starting_swtiching_and_stopping(sim_does_not_use_healthsystem)

    sim_uses_healthsystem = run_sim(tmpdir=tmpdir, use_healthsystem=True, disable=True)
    __check_no_illegal_switches(sim_uses_healthsystem)
    __check_some_starting_swtiching_and_stopping(sim_uses_healthsystem)

    # Check that the output of these two simulations are the same (apart from day of the month, which may change as
    # HSI dates are intentionally scattered over the month.)

    def format_log(_log):
        """Format the log so that date is replaced with the only the month and year"""
        _log["year_month"] = pd.to_datetime(_log['date']).dt.to_period('M')
        return _log.drop(columns=['date', 'age_years']).sort_values(['year_month', 'woman_id']).reset_index(drop=True)

    for key in {'pregnancy', 'contraception_change'}:
        pd.testing.assert_frame_equal(
            format_log(parse_log_file(sim_uses_healthsystem.log_filepath)['tlo.methods.contraception'][key]),
            format_log(parse_log_file(sim_does_not_use_healthsystem.log_filepath)['tlo.methods.contraception'][key])
        )

    # Equality of 'contraception_use_yearly_summary':
    pd.testing.assert_frame_equal(
        parse_log_file(sim_uses_healthsystem.log_filepath)['tlo.methods.contraception'][
            'contraception_use_yearly_summary'],
        parse_log_file(sim_does_not_use_healthsystem.log_filepath)['tlo.methods.contraception'][
            'contraception_use_yearly_summary']
    )


def test_contraception_using_healthsystem_but_no_capability(tmpdir):
    """Check that if switching and initiation use the HealthSystem but no HSI occur that there is no initiation or
     switching to anything that requires an HSI."""

    # Run simulation whereby contraception requires HSI but the HealthSystem prevent HSI occurring
    sim = run_sim(tmpdir=tmpdir, use_healthsystem=True, healthsystem_disable_and_reject_all=True)
    __check_no_illegal_switches(sim)

    log = parse_log_file(sim.log_filepath)['tlo.methods.contraception']

    # No record of starting/switching-to contraception of anything that requires an HSI
    states_that_may_require_HSI_to_switch_to = sim.modules['Contraception'].states_that_may_require_HSI_to_switch_to

    changes = log["contraception_change"]
    assert not changes["switch_to"].isin(states_that_may_require_HSI_to_switch_to).any()
    assert (changes.loc[changes["switch_from"].isin(states_that_may_require_HSI_to_switch_to), "switch_to"]
            == "not_using"
            ).all()

    # todo - could add that no one is maintained on it either.


def test_HSI_or_consumable_not_available_leads_to_defauling_to_not_using(tmpdir):
    """Check that if someone is on a method that requires an HSI, if consumable is not available and/or the health
    system cannot do the appointment that the person defaults to not using after they become due for a `maintenance`
    appointment"""

    def check_that_persons_on_contraceptive_default(sim):
        """Edit parameters, run simulation and do checks; women start on a contraceptive, and those who are on a
        contraceptive that requires HSI and consumables default by the end of the simulation."""

        # Let there be no chance of starting, switching or discontinuing (everyone would maintain if HSI/cons available)
        p = sim.modules['Contraception'].parameters
        p['contraception_initiation1'] *= 0.0
        p['contraception_discontinuation'] *= 0.0
        p['contraception_switching']['probability'] *= 0.0
        p['contraception_initiation2'] *= 0.0

        df = sim.population.props
        contraceptives = list(sim.modules['Contraception'].all_contraception_states)

        # Set that person_id=0-10 are woman on each of the contraceptive and due an appointment next month (these women
        # will default if on a contraceptive that requires a consumable).
        person_ids_due_appt = list(range(len(contraceptives)))

        # Set that person_id=12-25 are women each of the contraceptives and not due an appointment during the simulation
        # These women will not default.
        person_ids_not_due_appt = [i + len(contraceptives) for i in person_ids_due_appt]

        for i, contraceptive in enumerate(contraceptives):
            original_props = {
                'sex': 'F',
                'age_years': 30,
                'date_of_birth': sim.date - pd.DateOffset(years=30),
                'co_contraception': contraceptive,
                'is_pregnant': False,
                'date_of_last_pregnancy': pd.NaT,
                'co_unintended_preg': False,
                'co_date_of_last_fp_appt': sim.date - pd.DateOffset(months=5)
                # <-- due for an appointment in 1 mo
            }
            df.loc[person_ids_due_appt[i], original_props.keys()] = original_props.values()

            original_props['co_date_of_last_fp_appt'] = sim.date - pd.DateOffset(days=1)
            # <--not due an appointment
            df.loc[person_ids_not_due_appt[i], original_props.keys()] = original_props.values()

        # Check they are using the correct contraceptive
        for i, _c in enumerate(contraceptives):
            assert df.at[person_ids_due_appt[i], "co_contraception"] == _c
            assert df.at[person_ids_not_due_appt[i], "co_contraception"] == _c

        # Run simulation
        sim.simulate(end_date=sim.start_date+pd.DateOffset(months=3))
        __check_no_illegal_switches(sim)

        # Those on a contraceptive that requires HSI for maintenance should have defaulted to "not_using"
        for i, _c in enumerate(contraceptives):
            # These due an appointment will have defaulted if they are on a contraceptive that requires HSI/consumables

            if _c in sim.modules['Contraception'].states_that_may_require_HSI_to_maintain_on:
                assert df.at[person_ids_due_appt[i], "co_contraception"] == 'not_using'
            else:
                assert df.at[person_ids_due_appt[i], "co_contraception"] == _c

            # Those not due an appointment will not have defaulted (were not due an appointment)
            assert df.at[person_ids_not_due_appt[i], "co_contraception"] == _c

    # Check when no HSI occur
    sim = run_sim(tmpdir,
                  use_healthsystem=True,
                  healthsystem_disable_and_reject_all=True,
                  consumables_available=True,
                  run=False,
                  popsize=50
                  )
    check_that_persons_on_contraceptive_default(sim)

    # Check when HSI occur but consumables are not available
    sim = run_sim(tmpdir,
                  use_healthsystem=True,
                  disable=False,
                  consumables_available=False,
                  run=False,
                  popsize=50
                  )
    check_that_persons_on_contraceptive_default(sim)


def test_no_consumables_causes_no_use_of_contracptives(tmpdir):
    """Check that if no consumables are available then after six months (period between appointments for those
    maintaining), no one is using a contraceptive that requires a consumable: bec. initiation/switching are impossible
    when consumables are not available."""

    # Run simulation whereby contraception requires HSI, HSI run, but there are no consumables
    # Let there be no discontinuation (so that every would otherwise stay on contraception)
    sim = run_sim(tmpdir=tmpdir, use_healthsystem=True, disable=False, consumables_available=False,
                  no_discontinuation=True)
    __check_no_illegal_switches(sim)

    log = parse_log_file(sim.log_filepath)['tlo.methods.contraception']

    states_that_may_require_HSI_to_switch_to = sim.modules['Contraception'].states_that_may_require_HSI_to_switch_to
    states_that_may_require_HSI_to_maintain_on = sim.modules['Contraception'].states_that_may_require_HSI_to_maintain_on

    # Check that, after six months of simulation time, no one is on a contraceptive that requires a consumable for
    # maintenance.
    num_on_contraceptives = log['contraception_use_yearly_summary']
    after_six_mo = pd.to_datetime(num_on_contraceptives['date']) > (sim.start_date + pd.DateOffset(months=6))
    assert (num_on_contraceptives.loc[after_six_mo, states_that_may_require_HSI_to_maintain_on] == 0).all().all()

    # Check that people are not switching to those contraceptives that require consumables to switch to.
    changes = log["contraception_change"]
    assert not changes["switch_to"].isin(states_that_may_require_HSI_to_switch_to).any()

    # ... but are switching_from them to "not_using"
    assert changes["switch_from"].isin(states_that_may_require_HSI_to_maintain_on).any()
    assert (
        changes.loc[changes["switch_from"].isin(states_that_may_require_HSI_to_maintain_on), "switch_to"] == "not_using"
    ).all()


def test_occurence_of_HSI_for_maintain_and_switch(tmpdir):
    """Check HSI for the maintenance of a person on a contraceptive are scheduled as expected.."""

    # Create a simulation that has run for zero days and clear the event queue
    sim = run_sim(tmpdir,
                  use_healthsystem=True,
                  disable=False,
                  consumables_available=True,
                  end_date=Date(2010, 1, 1)
                  )
    sim.event_queue.queue = []
    sim.modules['HealthSystem'].reset_queue()

    # Let there be no chance of switching or discontinuing
    p = sim.modules['Contraception'].parameters
    p['contraception_discontinuation'] *= 0.0
    p['contraception_switching']['probability'] *= 0.0

    # Set that person_id=0 is a woman on a contraceptive for longer than six months
    person_id = 0
    df = sim.population.props
    original_props = {
        'sex': 'F',
        'age_years': 30,
        'date_of_birth': sim.date - pd.DateOffset(years=30),
        'co_contraception': 'pill',  # <-- requires appointments for maintenance
        'is_pregnant': False,
        'date_of_last_pregnancy': pd.NaT,
        'co_unintended_preg': False,
        'co_date_of_last_fp_appt': sim.date - pd.DateOffset(months=7)
    }
    df.loc[person_id, original_props.keys()] = original_props.values()

    # Run the ContraceptivePoll
    poll = contraception.ContraceptionPoll(module=sim.modules['Contraception'])
    poll.apply(sim.population)

    # Confirm that an HSI_FamilyPlanningAppt has been made for her (within 28 days as she is due an appointment already)
    events = sim.modules['HealthSystem'].find_events_for_person(person_id)
    assert 1 == len(events)
    ev = events[0]
    assert isinstance(ev[1], contraception.HSI_Contraception_FamilyPlanningAppt)

    date_of_hsi = ev[0]
    assert date_of_hsi <= (sim.date + pd.DateOffset(days=28))

    # Run that HSI_FamilyPlanningAppt and confirm there is no change in her state except that the date of last
    # appointment has been updated.
    sim.date = date_of_hsi
    ev[1].apply(person_id=person_id, squeeze_factor=0.0)

    df = sim.population.props  # update shortcut df
    props_to_be_same = [k for k in original_props.keys() if k != "co_date_of_last_fp_appt"]
    assert list(df.loc[person_id, props_to_be_same].values) == [original_props[p] for p in props_to_be_same]
    assert sim.population.props.at[person_id, "co_date_of_last_fp_appt"] == sim.date

    # CLear the HealthSystem queue and run the ContraceptivePoll again
    sim.modules['HealthSystem'].reset_queue()
    poll.apply(sim.population)

    # Confirm that no HSI_FamilyPlanningAppt has been scheduled (now that there is less time elapsed since her last
    # appointment)
    assert not len(sim.modules['HealthSystem'].find_events_for_person(person_id))


# Todo Items:
# * Refactoring - running now -- DONE
# * Change logic to allow for "switch_to" and "maintain_on" but let these be identical lists -- DONE
# * Changes so that female sterilization and condoms does not need HSI to be maintained (i.e. don't default) and edits to test file to accomodate -- DONE
# * Put the check_logs into all the checks & Check that the check on illegal moves operational (DONE

# * check that adding of sterilization to maintenance causing errors

# * typos line 523 and 627 and in file name in tests.
# * New test: __check_pregnancies_occuring




# ~~~
# * Check running of analysis file
# * Run long/large calibrations
# * Clean up parameters!
# ~~~


# def __check_pregnancies_occurring(sim):
#     """Check that pregnancies are occurring from those on and off contraception
#     """
#     logs = parse_log_file(sim.log_filepath)['tlo.methods.contraception']
#     assert set(logs.keys()) == {'contraception_use_yearly_summary', 'pregnancy', 'contraception_change'}
#
#     # Check that pregnancies are happening and that some of from those on a contraceptive and some from those not on
#     # contraception
#     assert len(logs['pregnancy'])
#     # assert (logs['pregnancy']['contraception'] != "not_using").any()  # <-- todo: only works on big enough runs
#     assert (logs['pregnancy']['contraception'] == "not_using").any()
