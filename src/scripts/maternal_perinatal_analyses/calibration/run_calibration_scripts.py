from src.scripts.maternal_perinatal_analyses.calibration.output_all_key_outcomes_per_year import \
    output_key_outcomes_from_scenario_file
from src.scripts.maternal_perinatal_analyses.calibration.output_all_death_calibration_per_year \
    import output_all_death_calibration_per_year


for file in ['long_run_all_diseases']:
    output_key_outcomes_from_scenario_file(scenario_filename=f'{file}.py',
                                           pop_size='20k',
                                           outputspath='./outputs/sejjj49@ucl.ac.uk/',
                                           sim_years=list(range(2010, 2030)),
                                           show_and_store_graphs=True)

    output_all_death_calibration_per_year(scenario_filename=f'{file}.py',
                                          outputspath='./outputs/sejjj49@ucl.ac.uk/',
                                          pop_size='20k',
                                          sim_years=list(range(2010, 2030)),
                                          daly_years=list(range(2010, 2020)))

