from scripts.maternal_perinatal_analyses.anc_analysis.anc_coverage_analysis import run_primary_analysis

for intervention_file in ['increased_anc_scenario']:

    run_primary_analysis(baseline_scenario_filename='baseline_anc_scenario.py',
                         intervention_scenario_filename=f'{intervention_file}.py',
                         outputspath='./outputs/sejjj49@ucl.ac.uk/',
                         show_and_store_graphs=True,
                         anc_scenario=4,
                         intervention_years=list(range(2020, 2026)),
                         do_cons_calculation=False)

