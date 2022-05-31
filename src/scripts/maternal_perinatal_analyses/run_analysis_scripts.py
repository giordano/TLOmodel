from scripts.maternal_perinatal_analyses.sba_analysis.sba_scenario_analysis import run_sba_scenario_analysis
from scripts.maternal_perinatal_analyses.pnc_analysis.pnc_scenario_analysis import run_pnc_scenario_analysis

from scripts.maternal_perinatal_analyses.compare_incidence_rates_between_scenarios import \
    compare_key_rates_between_multiple_scenarios

#run_pnc_scenario_analysis(scenario_file_dict={'Status Quo': 'baseline_sba_scenario',
#                                              'Increased PNC cov.': 'pnc_coverage',
 #                                             'Increased PNC qual.': 'pnc_quality',
 #                                             'PNC + CEMONC': 'combined_plus'},
 #                         show_and_store_graphs=True,
 #                         outputspath='./outputs/sejjj49@ucl.ac.uk/',
 #                         intervention_years=list(range(2021, 2031)),
 #                         do_cons_calculation=False)

#run_sba_scenario_analysis(scenario_file_dict={'Status Quo': 'baseline_sba_scenario',
 #                                              'Increased BEmONC': 'bemonc',
 #                                             'Increased CEmONC': 'cemonc',
 #                                             'CEmONC & PNC': 'combined_plus'},
 #                         show_and_store_graphs=True,
 #                         outputspath='./outputs/sejjj49@ucl.ac.uk/',
 #                         intervention_years=list(range(2021, 2031)),
  #                        do_cons_calculation=False)




compare_key_rates_between_multiple_scenarios(scenario_file_dict={'Status Quo': 'baseline_sba_scenario',
                                               'Increased BEmONC': 'bemonc',
                                              'Increased CEmONC': 'cemonc',
                                              'CEmONC & PNC': 'combined_plus'},
                                             identifier='sba',
                                             outputspath='./outputs/sejjj49@ucl.ac.uk/',
                                             intervention_years=list(range(2021, 2031)))

compare_key_rates_between_multiple_scenarios(scenario_file_dict={'Status Quo': 'baseline_sba_scenario',
                                              'Increased PNC cov.': 'pnc_coverage',
                                              'Increased PNC qual.': 'pnc_quality',
                                              'PNC + CEMONC': 'combined_plus'},
                                             identifier='pnc',
                                             outputspath='./outputs/sejjj49@ucl.ac.uk/',
                                             intervention_years=list(range(2021, 2031)))
