import compare_incidence_rates_between_scenarios_v2
import maternal_newborn_health_analysis_v2
import met_need

# create dict of some scenario 'title' and the filename of the associated title
scenario_dict = {'Status Quo': 'baseline_scenario',
                 'Increased ANC': 'increased_anc_scenario',
                 'Increased ANC and Qual': 'anc_scenario_plus_cons_and_qual',
                 'No ANC (sensitivity)': 'anc_min_sensitivity_scenario'}

# define key variables used within the analysis scripts
intervention_years = list(range(2010, 2015))
output_path = './outputs/sejjj49@ucl.ac.uk/'
service_of_interest = 'anc'
scen_colours = ['darkseagreen', 'steelblue', 'lightslategrey', 'orange']

maternal_newborn_health_analysis_v2.run_maternal_newborn_health_analysis(
     scenario_file_dict=scenario_dict,
     outputspath=output_path,
     intervention_years=intervention_years,
     service_of_interest=service_of_interest,
     show_all_results=True,
     scen_colours=scen_colours)

compare_incidence_rates_between_scenarios_v2.compare_key_rates_between_multiple_scenarios(
    scenario_file_dict=scenario_dict,
    outputspath=output_path,
    intervention_years=intervention_years,
    service_of_interest=service_of_interest,
    scen_colours=scen_colours)


