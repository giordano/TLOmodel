from scripts.maternal_perinatal_analyses.anc_analysis.anc_coverage_analysis import run_primary_analysis
#from scripts.maternal_perinatal_analyses.anc_analysis.compare_incidence_rates_between_scenarios import run_exploratory_analysis

run_primary_analysis(baseline_scenario_filename='baseline_anc_scenario.py',
                     intervention_scenario_filename='increased_anc_scenario.py',
                     outputspath='./outputs/sejjj49@ucl.ac.uk/',
                     plot_destination_folder='output_graphs_60k_increased_anc_scenario-2022-01-31T134117Z',
                     anc_scenario=4,
                     intervention_years=list(range(2020, 2026)),
                     do_cons_calculation=False)

#run_exploratory_analysis()
