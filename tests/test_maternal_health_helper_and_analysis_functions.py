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


def test_analysis_event_scheduling(seed):
    sim = Simulation(start_date=start_date, seed=seed)
    sim.register(*fullmodel(resourcefilepath=resourcefilepath))
    sim.make_initial_population(n=100)
    lparams = sim.modules['Labour'].parameters
    pparams = sim.modules['PregnancySupervisor'].current_parameters

    lparams['analysis_date'] = Date(2010, 5, 1)
    # todo: this wont work whilst we dont set analysis date in PS module without a parameter
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

    sim.simulate(end_date=sim.date + pd.DateOffset(years=1))

    assert pparams['prob_anc1_months_2_to_4'] == [0, 1.0, 0]
    for visit in [5, 6, 7, 8]:
        pparams[f'prob_seek_anc{visit}'] = new_avail_prob

    for parameter in ['prob_intervention_delivered_urine_ds', 'prob_intervention_delivered_bp',
                      'prob_intervention_delivered_ifa', 'prob_intervention_delivered_llitn',
                      'prob_intervention_delivered_llitn', 'prob_intervention_delivered_tt',
                      'prob_intervention_delivered_poct', 'prob_intervention_delivered_syph_test',
                      'prob_intervention_delivered_iptp', 'prob_intervention_delivered_gdm_test']:
        assert pparams[parameter] == new_avail_prob
test_analysis_event_scheduling(0)


def test_analysis_events_force_availability_of_consumables(seed):
    pass


def test_analysis_events_force_availability_of_signal_function_interventions(seed):
    pass


def test_logic_of_met_need_logging(seed):
    pass


def test_logic_of_mni_logging(seed):
    pass
