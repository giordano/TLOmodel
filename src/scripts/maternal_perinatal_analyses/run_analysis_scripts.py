from scripts.maternal_perinatal_analyses.anc_analysis.anc_coverage_analysis import run_primary_analysis
from scripts.maternal_perinatal_analyses.anc_analysis.compare_incidence_rates_between_scenarios import \
    compare_key_rates_between_two_scenarios
from scripts.maternal_perinatal_analyses.anc_analysis.multiple_scenario_analysis import run_multiple_scenario_analysis
from scripts.maternal_perinatal_analyses.sba_analysis.sba_scenario_analysis import run_sba_scenario_analysis


run_sba_scenario_analysis(scenario_file_dict={'Status Quo': 'baseline_sba_scenario',
                                              'Perfect BEmONC': 'perfect_bemonc_sba_scenario',
                                              'Perfect CEmONC': 'perfect_cemonc_sba_scenario',
                                              'Perfect BEmONC+CEmONC': 'perfect_bemonc_cemonc_sba_scenario'},
                          outputspath='./outputs/sejjj49@ucl.ac.uk/',
                          show_and_store_graphs=True,
                          intervention_years=list(range(2021, 2030)),
                          do_cons_calculation=False)

#for intervention in ['perfect_bemonc_sba_scenario', 'perfect_cemonc_sba_scenario',
#                     'perfect_bemonc_cemonc_sba_scenario']:

#    compare_key_rates_between_two_scenarios('baseline_sba_scenario', intervention, './outputs/sejjj49@ucl.ac.uk/',
#                                            True, list(range(2010, 2030)))

"""
for intervention_file in ['increased_anc_scenario']:
    run_primary_analysis(baseline_scenario_filename='baseline_anc_scenario.py',
                         intervention_scenario_filename=f'{intervention_file}.py',
                         outputspath='./outputs/sejjj49@ucl.ac.uk/',
                         show_and_store_graphs=False,
                         anc_scenario=4,
                         intervention_years=list(range(2020, 2026)),
                         do_cons_calculation=True)

    compare_key_rates_between_two_scenarios(baseline_scenario_filename='baseline_anc_scenario.py',
                                            intervention_scenario_filename=f'{intervention_file}.py',
                                            outputspath='./outputs/sejjj49@ucl.ac.uk/',
                                            show_and_store_graphs=True,
                                            sim_years=list(range(2020, 2026)))
"""
