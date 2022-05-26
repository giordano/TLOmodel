from scripts.maternal_perinatal_analyses.sba_analysis.sba_scenario_analysis import run_sba_scenario_analysis
from scripts.maternal_perinatal_analyses.compare_incidence_rates_between_scenarios import \
    compare_key_rates_between_two_scenarios


for script in ['bemonc', 'cemonc', 'pnc_coverage', 'pnc_quality']:
    compare_key_rates_between_two_scenarios(baseline_scenario_filename='baseline_sba_scenario',
                                            intervention_scenario_filename=script,
                                            outputspath='./outputs/sejjj49@ucl.ac.uk/',
                                            show_and_store_graphs=True,
                                            sim_years=list(range(2021, 2030)))


#run_sba_scenario_analysis(scenario_file_dict={'Status Quo': 'baseline_sba_scenario',
#                                              'Increased BEmONC': 'bemonc',
#                                              'Increased CEmONC': 'cemonc'},
#                          outputspath='./outputs/sejjj49@ucl.ac.uk/',
#                          show_and_store_graphs=True,
#                          intervention_years=list(range(2021, 2030)),
#                          do_cons_calculation=False)
