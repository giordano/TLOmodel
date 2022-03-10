from src.scripts.maternal_perinatal_analyses.calibration.output_all_key_outcomes_per_year import \
    output_key_outcomes_from_scenario_file
# from src.scripts.maternal_perinatal_analyses.calibration.output_all_death_calibration_per_year import pass


for file in ['run_core_modules']:
    output_key_outcomes_from_scenario_file(scenario_filename=f'{file}.py',
                                           pop_size='10k',
                                           outputspath='./outputs/sejjj49@ucl.ac.uk/',
                                           sim_years=list(range(2010, 2021)),
                                           show_and_store_graphs=True)
