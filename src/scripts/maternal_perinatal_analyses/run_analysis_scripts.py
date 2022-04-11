from scripts.maternal_perinatal_analyses.anc_analysis.anc_coverage_analysis import run_primary_analysis
from scripts.maternal_perinatal_analyses.anc_analysis.compare_incidence_rates_between_scenarios import \
    compare_key_rates_between_two_scenarios
from scripts.maternal_perinatal_analyses.anc_analysis.multiple_scenario_analysis import run_multiple_scenario_analysis
from scripts.maternal_perinatal_analyses.sba_analysis.sba_scenario_analysis import run_sba_scenario_analysis


run_sba_scenario_analysis(scenario_file_dict={'Status Quo': 'baseline_anc_scenario',
                                                   'Intervention 1': 'increased_anc_scenario',
                                                   'Intervention 2': 'plus_cons',
                                                   'Intervention 3': 'and_qual'},
                               outputspath='./outputs/sejjj49@ucl.ac.uk/',
                               show_and_store_graphs=False,
                               intervention_years=list(range(2020, 2026)),
                               do_cons_calculation=True)

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
