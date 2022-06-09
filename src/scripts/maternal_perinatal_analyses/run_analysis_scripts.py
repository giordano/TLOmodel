from scripts.maternal_perinatal_analyses.sba_analysis.sba_scenario_analysis import run_sba_scenario_analysis
from scripts.maternal_perinatal_analyses.pnc_analysis.pnc_scenario_analysis import run_pnc_scenario_analysis
from scripts.maternal_perinatal_analyses.met_need import met_need_and_contributing_factors_for_deaths

from scripts.maternal_perinatal_analyses.compare_incidence_rates_between_scenarios import \
    compare_key_rates_between_multiple_scenarios


met_need_and_contributing_factors_for_deaths(scenario_file_dict={'Status Quo': 'standard_mph_calibration'},
                                             outputspath='./outputs/sejjj49@ucl.ac.uk/',
                                             intervention_years=list(range(2011, 2014)))

