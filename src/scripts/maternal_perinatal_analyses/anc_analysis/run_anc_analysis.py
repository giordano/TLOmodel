from scripts.maternal_perinatal_analyses.anc_analysis.anc_coverage_analysis import run_primary_analysis
from scripts.maternal_perinatal_analyses.anc_analysis.compare_incidence_rates_between_scenarios import \
    compare_key_rates_between_two_scenarios


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
