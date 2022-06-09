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


def test_analysis_events_update_expected_parameters(seed):
    """"""
    sim = Simulation(start_date=start_date, seed=seed)
    sim.register(*fullmodel(resourcefilepath=resourcefilepath))
    sim.make_initial_population(n=100)
    sim.simulate(end_date=sim.date + pd.DateOffset(days=0))

    new_avail_prob = 0.5

    # set variables that trigger updates within analysis events
    pparams = sim.modules['PregnancySupervisor'].current_parameters
    pparams['alternative_anc_coverage'] = True
    pparams['alternative_anc_quality'] = True
    pparams['anc_availability_probability'] = new_avail_prob

    lparams = sim.modules['Labour'].current_parameters
    lparams['alternative_bemonc_availability'] = True
    lparams['alternative_cemonc_availability'] = True
    lparams['bemonc_availability'] = new_avail_prob
    lparams['cemonc_availability'] = new_avail_prob
    lparams['alternative_pnc_coverage'] = True
    lparams['alternative_pnc_quality'] = True
    lparams['pnc_availability_probability'] = new_avail_prob

    # define pregnancy analysis event
    preg_analysis = pregnancy_supervisor.PregnancyAnalysisEvent(module=sim.modules['PregnancySupervisor'])
    preg_analysis.apply(sim.population)

    # define labour/postnatal analysis event
    lab_analysis = labour.LabourAndPostnatalCareAnalysisEvent(module=sim.modules['Labour'])
    lab_analysis.apply(sim.population)

    # assert that all parameters that should have been specifically over ridden have been, and with the correct value
    assert pparams['prob_anc1_months_2_to_4'] == [0, 1.0, 0]
    for visit in [5, 6, 7, 8]:
        pparams[f'prob_seek_anc{visit}'] = new_avail_prob

    for parameter in ['prob_intervention_delivered_urine_ds', 'prob_intervention_delivered_bp',
                      'prob_intervention_delivered_ifa', 'prob_intervention_delivered_llitn',
                      'prob_intervention_delivered_llitn', 'prob_intervention_delivered_tt',
                      'prob_intervention_delivered_poct', 'prob_intervention_delivered_syph_test',
                      'prob_intervention_delivered_iptp', 'prob_intervention_delivered_gdm_test']:
        assert pparams[parameter] == new_avail_prob





test_analysis_events_update_expected_parameters(0)
