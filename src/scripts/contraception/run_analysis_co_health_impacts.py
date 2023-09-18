import contraception_mnh_analysis_script

# create dict of some scenario 'title' and the filename of the associated title
scenario_dict1 = {'Status Quo': 'run_analysis_co_health_impacts_no_diseases-2023-05-25T161549Z',
                  'Intervention Scenario': 'run_analysis_co_health_impacts_no_diseases-2023-05-25T161708Z',
                 }


# define key variables used within the analysis scripts
intervention_years = list(range(2011, 2015))
sim_years = list(range(2010, 2015))
output_path = './outputs/sejjej5@ucl.ac.uk/'


for scenario_dict, colours in zip([scenario_dict1], [['cadetblue', 'midnightblue']]):

    scen_colours = colours

    contraception_mnh_analysis_script.run_maternal_newborn_health_analysis(
         scenario_file_dict=scenario_dict,
         outputspath=output_path,
         sim_years=sim_years,
         intervention_years=intervention_years,
         scen_colours=scen_colours)
