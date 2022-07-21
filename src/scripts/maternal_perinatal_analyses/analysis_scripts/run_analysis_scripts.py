from src.scripts.maternal_perinatal_analyses.analysis_scripts.test_outcome_combining import test

# from ..analysis_scripts import (
#    compare_incidence_rates_between_scenarios,
#    maternal_newborn_health_analysis,
#    met_need, test_outcome_combining
# )

# create dict of some scenario 'title' and the filename of the associated title
scenario_dict = {'Status Quo': 'baseline_anc_scenario',
                 '90% ANC4+': 'increased_anc_scenario',
                 '90% ANC4+ (and qual)': 'anc_scenario_plus_cons_and_qual'}

scenario_dict_sba = {'Status Quo': 'baseline_anc_scenario',
                     '90% BEmONC': 'bemonc',
                     '90% CEmONC': 'cemonc'}

scenario_dict_pnc = {'Status Quo': 'baseline_anc_scenario',
                     '90% PNC': 'increased_pnc_scenario',
                     '90% PNC (and qual)': 'pnc_scenario_plus_cons_and_qual'}

scenario_dict_all = {'Status Quo': 'baseline_anc_scenario',
                     '90% MPH Services': 'full_uhc_mph_coverage'}

# define key variables used within the analysis scripts
intervention_years = list(range(2020, 2031))
output_path = './outputs/sejjj49@ucl.ac.uk/'
# service_of_interest = 'anc'

for service_of_interest, dictionary in zip(['anc', 'sba', 'pnc', 'all'],
                                           [scenario_dict, scenario_dict_sba, scenario_dict_pnc, scenario_dict_all ]):
    test(
        scenario_file_dict=dictionary,
        outputspath=output_path,
        intervention_years=intervention_years,
        service_of_interest=service_of_interest)

# maternal_newborn_health_analysis.run_maternal_newborn_health_analysis(
#    scenario_file_dict=scenario_dict,
#    outputspath=output_path,
#    intervention_years=intervention_years,
#    service_of_interest=service_of_interest,
#    show_all_results=True)#

# compare_incidence_rates_between_scenarios.compare_key_rates_between_multiple_scenarios(
#    scenario_file_dict=scenario_dict,
#    service_of_interest=service_of_interest,
#    outputspath=output_path,
#    intervention_years=intervention_years)

# met_need.met_need_and_contributing_factors_for_deaths(
#    scenario_file_dict=scenario_dict,
#    outputspath=output_path,
#    intervention_years=intervention_years,
#    service_of_interest=service_of_interest)
