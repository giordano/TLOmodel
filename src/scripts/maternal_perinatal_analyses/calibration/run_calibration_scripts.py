from src.scripts.maternal_perinatal_analyses.calibration.output_all_death_calibration_per_year\
    import output_all_death_calibration_per_year
from src.scripts.maternal_perinatal_analyses.calibration.output_all_key_outcomes_per_year \
    import output_key_outcomes_from_scenario_file


for file in ['focused_anc_test']:
    output_key_outcomes_from_scenario_file(scenario_filename=f'{file}.py',
                                           pop_size='30',
                                           outputspath='./outputs/sejjj49@ucl.ac.uk/',
                                           sim_years=list(range(2010, 2021)),
                                           show_and_store_graphs=True)

    output_all_death_calibration_per_year(scenario_filename=f'{file}.py',
                                          outputspath='./outputs/sejjj49@ucl.ac.uk/',
                                          pop_size='30',
                                          sim_years=list(range(2010, 2021)),
                                          daly_years=list(range(2010, 2020)))
