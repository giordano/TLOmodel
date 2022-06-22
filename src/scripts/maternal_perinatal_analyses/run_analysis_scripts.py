from scripts.maternal_perinatal_analyses.sba_analysis.sba_scenario_analysis import run_sba_scenario_analysis
from scripts.maternal_perinatal_analyses.pnc_analysis.pnc_scenario_analysis import run_pnc_scenario_analysis
from scripts.maternal_perinatal_analyses.met_need import met_need_and_contributing_factors_for_deaths
from scripts.maternal_perinatal_analyses.anc_analysis.anc_coverage_analysis import run_anc_scenario_analysis

from scripts.maternal_perinatal_analyses.compare_incidence_rates_between_scenarios import \
    compare_key_rates_between_multiple_scenarios

from scripts.maternal_perinatal_analyses.maternal_newborn_health_analysis import \
    run_maternal_newborn_health_analysis

run_maternal_newborn_health_analysis(scenario_file_dict={'Status Quo': 'baseline_anc_scenario',
                                                         '90% MPH services': 'full_uhc_mph_coverage'},
                                     outputspath='./outputs/sejjj49@ucl.ac.uk/',
                                     intervention_years=list(range(2010, 2026)),
                                     service_of_interest='full',
                                     show_all_results=True)

compare_key_rates_between_multiple_scenarios(scenario_file_dict={'Status Quo': 'baseline_anc_scenario',
                                                                 '90% MPH services': 'full_uhc_mph_coverage'},
                                             service_of_interest='full',
                                             outputspath='./outputs/sejjj49@ucl.ac.uk/',
                                             intervention_years=list(range(2010, 2026)))


met_need_and_contributing_factors_for_deaths(scenario_file_dict={'Status Quo': 'baseline_anc_scenario',
                                                                 '90% MPH services': 'full_uhc_mph_coverage'},
                                             outputspath='./outputs/sejjj49@ucl.ac.uk/',
                                             intervention_years=list(range(2010, 2026)),
                                             service_of_interest='full')

"""
run_maternal_newborn_health_analysis(scenario_file_dict={'Status Quo': 'baseline_anc_scenario',
                                                         'BEmONC (90%)': 'bemonc',
                                                         'CEmONC (90%)': 'cemonc',
                                                         },
                                     outputspath='./outputs/sejjj49@ucl.ac.uk/',
                                     intervention_years=list(range(2010, 2026)),
                                     service_of_interest='sba',
                                     show_all_results=False)

compare_key_rates_between_multiple_scenarios(scenario_file_dict={'Status Quo': 'baseline_anc_scenario',
                                                         'BEmONC (90%)': 'bemonc',
                                                         'CEmONC (90%)': 'cemonc',
                                                         },
                                             service_of_interest='sba',
                                             outputspath='./outputs/sejjj49@ucl.ac.uk/',
                                             intervention_years=list(range(2010, 2026)))


met_need_and_contributing_factors_for_deaths(scenario_file_dict={'Status Quo': 'baseline_anc_scenario',
                                                         'BEmONC (90%)': 'bemonc',
                                                         'CEmONC (90%)': 'cemonc',
                                                         },
                                             outputspath='./outputs/sejjj49@ucl.ac.uk/',
                                             intervention_years=list(range(2010, 2026)),
                                             service_of_interest='sba')


run_maternal_newborn_health_analysis(scenario_file_dict={'Status Quo': 'baseline_anc_scenario',
                                                         'Increased Coverage': 'increased_pnc_coverage',
                                                         'Increased Coverage and Quality': 'pnc_plus_cons_and_qual',
                                                         },
                                     outputspath='./outputs/sejjj49@ucl.ac.uk/',
                                     intervention_years=list(range(2010, 2026)),
                                     service_of_interest='pnc',
                                     show_all_results=True)

compare_key_rates_between_multiple_scenarios(scenario_file_dict={'Status Quo': 'baseline_anc_scenario',
                                                         'Increased Coverage': 'increased_pnc_coverage',
                                                         'Increased Coverage and Quality': 'pnc_plus_cons_and_qual',
                                                         },
                                             service_of_interest='pnc',
                                             outputspath='./outputs/sejjj49@ucl.ac.uk/',
                                             intervention_years=list(range(2010, 2026)))


met_need_and_contributing_factors_for_deaths(scenario_file_dict={'Status Quo': 'baseline_anc_scenario',
                                                         'Increased Coverage': 'increased_pnc_coverage',
                                                         'Increased Coverage and Quality': 'pnc_plus_cons_and_qual',
                                                         },
                                             outputspath='./outputs/sejjj49@ucl.ac.uk/',
                                             intervention_years=list(range(2010, 2026)),
                                             service_of_interest='pnc')"""
