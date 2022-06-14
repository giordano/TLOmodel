from src.scripts.maternal_perinatal_analyses.calibration.output_all_death_calibration_per_year\
    import output_all_death_calibration_per_year
from src.scripts.maternal_perinatal_analyses.calibration.output_all_key_outcomes_per_year \
    import output_key_outcomes_from_scenario_file


for file in ['anc_plus_cons_and_qual']:
    output_key_outcomes_from_scenario_file(scenario_filename=f'{file}.py',
                                           pop_size='60k',
                                           outputspath='./outputs/sejjj49@ucl.ac.uk/',
                                           sim_years=list(range(2010, 2025)),
                                           show_and_store_graphs=True)

    output_all_death_calibration_per_year(scenario_filename=f'{file}.py',
                                          outputspath='./outputs/sejjj49@ucl.ac.uk/',
                                          pop_size='60k',
                                          sim_years=list(range(2010, 2025)),
                                          daly_years=list(range(2010, 2020)))
