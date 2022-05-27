from scripts.maternal_perinatal_analyses.sba_analysis.sba_scenario_analysis import run_sba_scenario_analysis
from scripts.maternal_perinatal_analyses.pnc_analysis.pnc_scenario_analysis import run_pnc_scenario_analysis

from scripts.maternal_perinatal_analyses.compare_incidence_rates_between_scenarios import \
    compare_key_rates_between_multiple_scenarios

compare_key_rates_between_multiple_scenarios(scenario_file_dict={'Status Quo': 'baseline_sba_scenario',
                                                                 'Increased BEmONC': 'bemonc',
                                                                 'Increased CEmONC': 'cemonc'},
                                             identifier='sba',
                                             outputspath='./outputs/sejjj49@ucl.ac.uk/',
                                             intervention_years=list(range(2021, 2031)))

compare_key_rates_between_multiple_scenarios(scenario_file_dict={'Status Quo': 'baseline_sba_scenario',
                                                                 'Increased PNC cov.': 'pnc_coverage',
                                                                 'Increased PNC qual.': 'pnc_quality'},
                                             identifier='pnc',
                                             outputspath='./outputs/sejjj49@ucl.ac.uk/',
                                             intervention_years=list(range(2021, 2031)))

#run_sba_scenario_analysis(scenario_file_dict={'Status Quo': 'baseline_sba_scenario',
#                                                                 'Increased BEmONC': 'bemonc',
#                                                                 'Increased CEmONC': 'cemonc'},
#                                             show_and_store_graphs=True,
#                                             outputspath='./outputs/sejjj49@ucl.ac.uk/',
#                                             intervention_years=list(range(2021, 2031)),
#                                            do_cons_calculation=False)

#run_pnc_scenario_analysis(scenario_file_dict={'Status Quo': 'baseline_sba_scenario',
#                                                                 'Increased PNC cov.': 'pnc_coverage',
#                                                                 'Increased PNC qual.': 'pnc_quality'},
#                                             show_and_store_graphs=True,
#                                             outputspath='./outputs/sejjj49@ucl.ac.uk/',
#                                             intervention_years=list(range(2021, 2031)),
 #                                           do_cons_calculation=False)



#for script in ['bemonc', 'cemonc', 'pnc_coverage', 'pnc_quality']:
#    compare_key_rates_between_two_scenarios(baseline_scenario_filename='baseline_sba_scenario',
#                                            intervention_scenario_filename=script,
#                                            outputspath='./outputs/sejjj49@ucl.ac.uk/',
#                                            show_and_store_graphs=True,
#                                            sim_years=list(range(2021, 2031)))


#compare_key_rates_between_multiple_scenarios(scenario_file_dict={'Status Quo': 'baseline_sba_scenario',
#                                                                 'Increased BEmONC': 'bemonc',
#                                                                 'Increased CEmONC': 'cemonc'},
#                                             outputspath='./outputs/sejjj49@ucl.ac.uk/',
#                                             intervention_years=list(range(2021, 2031)))
