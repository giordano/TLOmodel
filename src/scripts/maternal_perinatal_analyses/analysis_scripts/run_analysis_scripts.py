import compare_incidence_rates_between_scenarios_v2
import maternal_newborn_health_analysis_v2
import met_need

# create dict of some scenario 'title' and the filename of the associated title
scenario_dict = {'Status Quo': 'baseline_anc_scenario',
                 'Increased ANC': 'increased_anc_scenario',
                 'Increased ANC Qual.': 'anc_scenario_plus_cons_and_qual'}

# define key variables used within the analysis scripts
intervention_years = list(range(2020, 2026))
output_path = './outputs/sejjj49@ucl.ac.uk/'
service_of_interest = 'anc'

maternal_newborn_health_analysis_v2.run_maternal_newborn_health_analysis(
     scenario_file_dict=scenario_dict,
     outputspath=output_path,
     intervention_years=intervention_years,
     service_of_interest=service_of_interest,
     show_all_results=True)

# compare_incidence_rates_between_scenarios_v2.compare_key_rates_between_multiple_scenarios(
#     scenario_file_dict=scenario_dict,
#     outputspath=output_path,
#     intervention_years=intervention_years,
#     service_of_interest=service_of_interest)



#compare_incidence_rates_between_scenarios.compare_key_rates_between_multiple_scenarios(
#    scenario_file_dict=scenario_dict,
#    service_of_interest=service_of_interest,
#    outputspath=output_path,
#    intervention_years=intervention_years)

#met_need.met_need_and_contributing_factors_for_deaths(
#    scenario_file_dict=scenario_dict,
#    outputspath=output_path,
#    intervention_years=intervention_years,
#    service_of_interest=service_of_interest)
