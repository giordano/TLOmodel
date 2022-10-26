import compare_incidence_rates_between_scenarios_v2
import maternal_newborn_health_analysis_v2
import met_need

# create dict of some scenario 'title' and the filename of the associated title
scenario_dict = {'Status Quo': 'baseline_scenario',
                 '90% BEmONC': 'bemonc',
                 '90% CEmONC': 'cemonc',
                 'No EmONC (sens. analysis)': 'sba_min_sensitivity_analysis-2022-10-24T075448Z',
                 }

# define key variables used within the analysis scripts
intervention_years = list(range(2010, 2015))
output_path = './outputs/sejjj49@ucl.ac.uk/'
service_of_interest = 'sba'
scen_colours = ['rosybrown', 'lightcoral', 'pink', 'red']

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


