import compare_incidence_rates_between_scenarios_v2
import maternal_newborn_health_analysis_v2
import met_need

# create dict of some scenario 'title' and the filename of the associated title
scenario_dict = {'Status Quo': 'baseline_scenario',
                 'Intervention 1': 'increased_anc_scenario',
                 'Intervention 2': 'anc_scenario_plus_cons_and_qual',
                 'Sensitivity (min)': 'min_anc_sensitivity_scenario',
                 'Sensitivity (max)': 'max_anc_sensitivity_scenario',
                 }

# define key variables used within the analysis scripts
intervention_years = list(range(2022, 2031))
sim_years = list(range(2010, 2031))
output_path = './outputs/sejjj49@ucl.ac.uk/'
service_of_interest = 'anc'
scen_colours = ['rosybrown', 'lightcoral', 'pink', 'sandybrown', 'maroon']

met_need.met_need_and_contributing_factors_for_deaths(scenario_dict, output_path, intervention_years,
                                                      service_of_interest)
# maternal_newborn_health_analysis_v2.run_maternal_newborn_health_analysis(
#      scenario_file_dict=scenario_dict,
#      outputspath=output_path,
#      sim_years=sim_years,
#      intervention_years=intervention_years,
#      service_of_interest=service_of_interest,
#      show_all_results=True,
#      scen_colours=scen_colours)

# compare_incidence_rates_between_scenarios_v2.compare_key_rates_between_multiple_scenarios(
#     scenario_file_dict=scenario_dict,
#     outputspath=output_path,
#     sim_years=sim_years,
#     intervention_years=intervention_years,
#     service_of_interest=service_of_interest,
#     scen_colours=scen_colours)
#
#
