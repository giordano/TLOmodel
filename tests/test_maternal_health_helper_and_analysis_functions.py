import os
from pathlib import Path

import pandas as pd
import pytest

from tlo import Date, Simulation
from tlo.analysis.utils import parse_log_file
from tlo.methods.fullmodel import fullmodel
from tlo.methods import (labour, pregnancy_supervisor)


start_date = Date(2010, 1, 1)

# The resource files
try:
    resourcefilepath = Path(os.path.dirname(__file__)) / '../resources'
except NameError:
    # running interactively
    resourcefilepath = 'resources'


def test_analysis_analysis_events_run_as_expected_and_update_parameters(seed):
    sim = Simulation(start_date=start_date, seed=seed)
    sim.register(*fullmodel(resourcefilepath=resourcefilepath))
    sim.make_initial_population(n=100)
    lparams = sim.modules['Labour'].parameters
    pparams = sim.modules['PregnancySupervisor'].parameters

    lparams['analysis_date'] = Date(2010, 1, 2)
    pparams['analysis_date'] = Date(2010, 1, 2)

    new_avail_prob = 0.5

    # set variables that trigger updates within analysis events
    pparams['alternative_anc_coverage'] = True
    pparams['alternative_anc_quality'] = True
    pparams['anc_availability_probability'] = new_avail_prob

    lparams['alternative_bemonc_availability'] = True
    lparams['alternative_cemonc_availability'] = True
    lparams['bemonc_availability'] = new_avail_prob
    lparams['cemonc_availability'] = new_avail_prob
    lparams['alternative_pnc_coverage'] = True
    lparams['alternative_pnc_quality'] = True
    lparams['pnc_availability_probability'] = new_avail_prob

    unchanged_odds_anc = pparams['odds_early_init_anc4'][0]
    unchanged_odds_pnc = lparams['odds_will_attend_pnc'][0]

    sim.simulate(end_date=Date(2010, 1, 3))

    # Check antenatal parameters correctly updated
    p_current_params = sim.modules['PregnancySupervisor'].current_parameters

    assert p_current_params['prob_anc1_months_2_to_4'] == [1.0, 0, 0]
    assert p_current_params['prob_late_initiation_anc4'] == 0
    assert p_current_params['odds_early_init_anc4'] != unchanged_odds_anc

    for parameter in ['prob_intervention_delivered_urine_ds', 'prob_intervention_delivered_bp',
                      'prob_intervention_delivered_ifa', 'prob_intervention_delivered_llitn',
                      'prob_intervention_delivered_llitn', 'prob_intervention_delivered_tt',
                      'prob_intervention_delivered_poct', 'prob_intervention_delivered_syph_test',
                      'prob_intervention_delivered_iptp', 'prob_intervention_delivered_gdm_test']:
        assert p_current_params[parameter] == new_avail_prob

    # Check labour/postnatal/newborn parameters updated
    l_current_params = sim.modules['Labour'].current_parameters
    pn_current_params = sim.modules['PostnatalSupervisor'].current_parameters
    nbparams = sim.modules['NewbornOutcomes'].current_parameters

    assert l_current_params['squeeze_threshold_for_delay_three_bemonc'] == 10_000
    assert l_current_params['squeeze_threshold_for_delay_three_cemonc'] == 10_000
    assert l_current_params['squeeze_threshold_for_delay_three_pn'] == 10_000
    assert nbparams['squeeze_threshold_for_delay_three_nb_care'] == 10_000

    assert l_current_params['odds_will_attend_pnc'] != unchanged_odds_pnc
    assert l_current_params['prob_careseeking_for_complication_pn'] == new_avail_prob
    assert l_current_params['prob_timings_pnc'] == [new_avail_prob, (1 - new_avail_prob)]

    assert nbparams['prob_pnc_check_newborn'] == new_avail_prob
    assert nbparams['prob_care_seeking_for_complication'] == new_avail_prob

    assert pn_current_params['prob_care_seeking_postnatal_emergency_neonate'] == new_avail_prob
    assert pn_current_params['prob_care_seeking_postnatal_emergency'] == new_avail_prob


def test_analysis_events_force_availability_of_consumables(seed):
    pass


def test_analysis_events_force_availability_of_signal_function_interventions(seed):
    pass


def test_logic_of_mni_intervention_logging(seed, ):
    # check all deaths being counted
    pass


def test_logic_of_met_need_logging(tmpdir, seed):
    sim = Simulation(start_date=start_date, seed=seed, log_config={"filename": "log", "directory": tmpdir})
    sim.register(*fullmodel(resourcefilepath=resourcefilepath))
    sim.make_initial_population(n=1000)
    sim.simulate(end_date=Date(2011, 1, 1))

    pass



