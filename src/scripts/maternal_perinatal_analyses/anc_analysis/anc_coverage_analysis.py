from pathlib import Path
import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
plt.style.use('seaborn-darkgrid')

from scripts.maternal_perinatal_analyses import analysis_utility_functions
from tlo.analysis.utils import extract_results, get_scenario_outputs, load_pickled_dataframes

# from tlo.methods.demography import get_scaling_factor


def run_anc_scenario_analysis(scenario_file_dict, outputspath, show_and_store_graphs, intervention_years,
                              service_structure):
    # Find results folder (most recent run generated using that scenario_filename)

    results_folders = {k: get_scenario_outputs(scenario_file_dict[k], outputspath)[-1] for k in scenario_file_dict}

    # Determine number of draw in this run (used in determining some output)
    run_path = os.listdir(f'{outputspath}/{results_folders["Status Quo"].name}/0')
    # todo: fix so doesnt rely on preset name
    runs = [int(item) for item in run_path]

    # Create folder to store graphs (if it hasnt already been created when ran previously)
    path = f'{outputspath}/anc_analysis_output_graphs_{results_folders["Status Quo"].name}'
    if not os.path.isdir(path):
        os.makedirs(f'{outputspath}/anc_analysis_output_graphs_{results_folders["Status Quo"].name}')

    plot_destination_folder = path

    # ========================================= BIRTHs PER SCENARIO ==================================================
    # Access birth data
    births_dict = analysis_utility_functions.return_birth_data_from_multiple_scenarios(results_folders,
                                                                                       intervention_years)

    # ========================================= COVERAGE ==============================================================
    def get_anc_coverage(folder):

        anc_coverage = extract_results(
            folder,
            module="tlo.methods.care_of_women_during_pregnancy",
            key="anc_count_on_birth",
            custom_generate_series=(
                lambda df: df.assign(
                    year=df['date'].dt.year).groupby(['year'])['person_id'].count()),
            do_scaling=True
        )

        anc_cov_of_interest = extract_results(
            folder,
            module="tlo.methods.care_of_women_during_pregnancy",
            key="anc_count_on_birth",
            custom_generate_series=(
                lambda df: df.loc[df['total_anc'] >= service_structure].assign(
                    year=df['date'].dt.year).groupby(['year'])['person_id'].count()),
            do_scaling=True
        )

        mean_total_anc = analysis_utility_functions.get_mean_and_quants(anc_coverage, intervention_years)
        mean_cov_anc = analysis_utility_functions.get_mean_and_quants(anc_cov_of_interest, intervention_years)

        result_m = [(x/y) * 100 for x, y in zip(mean_cov_anc[0], mean_total_anc[0])]
        result_lq = [(x/y) * 100 for x, y in zip(mean_cov_anc[1], mean_total_anc[1])]
        result_uq = [(x/y) * 100 for x, y in zip(mean_cov_anc[2], mean_total_anc[2])]

        return [result_m, result_lq, result_uq]

    cov_data = {k: get_anc_coverage(results_folders[k]) for k in results_folders}

    analysis_utility_functions.comparison_graph_multiple_scenarios(
        intervention_years, cov_data, '% Births',
        f'Proportion of women receiving {service_structure} (or more) ANC visits at birth',
        plot_destination_folder, f'anc{service_structure}_cov')

    death_data = analysis_utility_functions.return_death_data_from_multiple_scenarios(results_folders, births_dict,
                                                                                      intervention_years,
                                                                                      detailed_log=True)

    #  ================================== MATERNAL AND NEONATAL DEATH ===============================================
    def get_diff_direct_mmr(baseline_mmr_data, comparator):
        crude_diff = [x - y for x, y in zip(baseline_mmr_data[0], comparator[0])]

        avg_crude_diff = sum(crude_diff) / len(crude_diff)

        percentage_diff = [100 - ((x / y) * 100) for x, y in zip(comparator[0], baseline_mmr_data[0])]

        avg_percentage_diff = sum(percentage_diff) / len(percentage_diff)

        return {'crude': crude_diff,
                'crude_avg': avg_crude_diff,
                'percentage': percentage_diff,
                'percentage_avf': avg_percentage_diff}

    diff_data = {k: get_diff_direct_mmr(death_data['Status Quo']['direct_mmr'], death_data[k]['direct_mmr']) for k
                 in ['Increased Coverage', 'Increased Coverage and Quality']}

    if show_and_store_graphs:
        # Generate plots of yearly MMR and NMR
        analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
            intervention_years, death_data, 'direct_mmr',
            'Deaths per 100,000 live births',
            'MMR per Year at Baseline and Under Intervention (Direct only)', plot_destination_folder,
            'maternal_mr_direct')

        analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
            intervention_years, death_data, 'total_mmr',
            'Deaths per 100,000 live births',
            'MMR per Year at Baseline and Under Intervention (Total)',
            plot_destination_folder, 'maternal_mr_total')

        analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
            intervention_years, death_data, 'nmr',
            'Total Deaths per 1000 live births',
            'Neonatal Mortality Ratio per Year at Baseline and Under Intervention',
            plot_destination_folder, 'neonatal_mr_int')

        for group, l in zip(['Maternal', 'Neonatal'], ['m', 'n']):
            analysis_utility_functions.comparison_bar_chart_multiple_bars(
                death_data, f'crude_{l}_deaths', intervention_years,
                f'Total {group} Deaths (scaled)', f'Yearly Baseline {group} Deaths Compared to Intervention',
                plot_destination_folder, f'{group}_crude_deaths_comparison.png')

        # todo: make function that can be used for MMR, NMR, SBR
        N = len(intervention_years)
        ind = np.arange(N)
        width = 0.2

        for dict_name in ['crude', 'percentage']:
            for k, position, colour in zip(diff_data, [ind, ind + width],
                                           ['bisque', 'powderblue']):
                plt.bar(position, diff_data[k][dict_name], width, label=k, color=colour)

            plt.ylabel('Reduction From Baseline')
            plt.xlabel('Years')
            plt.title(f'Mean Reduction of MMR from Baseline per Year ({dict_name})')
            plt.legend(loc='best')
            plt.xticks([0., 1., 2., 3.],
                       labels=intervention_years)  # todo: has the be editied with number of years
            plt.savefig(f'{plot_destination_folder}/{dict_name}_diff_mmr.png')
            plt.show()
        #  ================================== STILLBIRTH  ===============================================
        # Get data
    sbr_data = analysis_utility_functions.return_stillbirth_data_from_multiple_scenarios(results_folders, births_dict,
                                                                                         intervention_years)

    if show_and_store_graphs:
        analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
            intervention_years, sbr_data, 'an_sbr',
            'Antenatal stillbirths per 1000 births',
            'Antenatal stillbirth Rate per Year at Baseline and Under Intervention',
            plot_destination_folder, 'an_sbr_int')

        analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
            intervention_years, sbr_data, 'ip_sbr',
            'Intrapartum stillbirths per 1000 births',
            'Intrapartum stillbirth Rate per Year at Baseline and Under Intervention',
            plot_destination_folder, 'ip_sbr_int')

        # Output SBR per year for scenario vs intervention
        analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
            intervention_years, sbr_data, 'sbr',
            'Stillbirths per 1000 births',
            'Stillbirth Rate per Year at Baseline and Under Intervention',
            plot_destination_folder, 'sbr_int')

        analysis_utility_functions.comparison_bar_chart_multiple_bars(
            sbr_data, 'crude_sb', intervention_years,
            'Total Stillbirths (scaled)', 'Yearly Baseline Stillbirths Compared to Intervention',
            plot_destination_folder, 'crude_stillbirths_comparison.png')

    # =================================================== DALYS =======================================================
    # Store DALYs data for baseline and intervention
    dalys_data = analysis_utility_functions.return_dalys_from_multiple_scenarios(results_folders, intervention_years)

    if show_and_store_graphs:

        for dict_key, axis, title, save_name in zip(['maternal_dalys_crude', 'maternal_dalys_rate',
                                                     'maternal_yll_crude', 'maternal_yll_rate',
                                                     'maternal_yld_crude', 'maternal_yld_rate',

                                                     'neonatal_dalys_crude', 'neonatal_dalys_rate',
                                                     'neonatal_yll_crude', 'neonatal_yll_rate',
                                                     'neonatal_yld_crude', 'neonatal_yld_rate'],

                                                    ['DALYs', 'DALYs per 100k Person-Years',
                                                     'YLL', 'YLL per 100k Person-Years',
                                                     'YLD', 'YLD per 100k Person-Years',

                                                     'DALYs', 'DALYs per 100k Person-Years',
                                                     'YLL', 'YLL per 100k Person-Years',
                                                     'YLD', 'YLD per 100k Person-Years'],

                                                    ['Crude Total DALYs per Year Attributable to Maternal disorders',
                                                     'DALYs per 100k Person-Years Attributable to Maternal disorders',
                                                     'Crude Total YLL per Year Attributable to Maternal disorders',
                                                     'YLL per 100k Person-Years Attributable to Maternal disorders',
                                                     'Crude Total YLD per Year Attributable to Maternal disorders',
                                                     'YLD per 100k Person-Years Attributable to Maternal disorders',

                                                     'Crude Total DALYs per Year Attributable to Neonatal disorders',
                                                     'DALYs per 100k Person-Years Attributable to Neonatal disorders',
                                                     'Crude Total YLL per Year Attributable to Neonatal disorders',
                                                     'YLL per 100k Person-Years Attributable to Neonatal disorders',
                                                     'Crude Total YLD per Year Attributable to Neonatal disorders',
                                                     'YLD per 100k Person-Years Attributable to Neonatal disorders'],

                                                    ['maternal_dalys_stacked', 'maternal_dalys_rate',
                                                     'maternal_yll', 'maternal_yll_rate',
                                                     'maternal_yld', 'maternal_yld_rate',
                                                     'neonatal_dalys_stacked', 'neonatal_dalys_rate',
                                                     'neonatal_yll', 'neonatal_yll_rate',
                                                     'neonatal_yld', 'neonatal_yld_rate']):
            analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
                intervention_years, dalys_data, dict_key, axis, title, plot_destination_folder, save_name)

    # =========================================== ADDITIONAL OUTCOMES ================================================
    # MALARIA
    # todo: what else? (proportion of infected women receiving iptp)
    def get_malaria_incidence_in_pregnancy(folder):
        # Number of clinical episodes in pregnant women per year
        preg_clin_counter_dates = extract_results(
            folder,
            module="tlo.methods.malaria",
            key="incidence",
            column='clinical_preg_counter',
            index='date',
            do_scaling=True
        )

        years = preg_clin_counter_dates.index.year
        preg_clin_counter_years = preg_clin_counter_dates.set_index(years)
        preg_clinical_counter = analysis_utility_functions.get_mean_and_quants(preg_clin_counter_years,
                                                                               intervention_years)

        incidence_dates = extract_results(
            folder,
            module="tlo.methods.malaria",
            key="incidence",
            column='inc_1000py',
            index='date',
            do_scaling=True
        )

        years = incidence_dates.index.year
        incidence_years = incidence_dates.set_index(years)
        incidence = analysis_utility_functions.get_mean_and_quants(incidence_years, intervention_years)

        return {'clin_counter': preg_clinical_counter,
                'incidence': incidence}

    mal_data = {k: get_malaria_incidence_in_pregnancy(results_folders[k]) for k in results_folders}

    analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
        intervention_years, mal_data, 'clin_counter',
        'Num. Clinical Cases',
        'Number of Clinical Cases of Malaria During Pregnancy Per Year Per Scenario',
        plot_destination_folder, 'mal_clinical_cases')

    analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
        intervention_years, mal_data, 'incidence',
        'Incidence per 1000 person years',
        'Incidence of Malaria Per Year Per Scenario',
        plot_destination_folder, 'mal_incidence')

    def get_hiv_information(folder):

        # Number of tests by sex- female
        hiv_tests = extract_results(
            folder,
            module="tlo.methods.malaria",
            key="incidence",
            column='clinical_preg_counter',
            index='date',
            do_scaling=True
        )

        years = preg_clin_counter_dates.index.year
        preg_clin_counter_years = preg_clin_counter_dates.set_index(years)
        preg_clinical_counter = analysis_utility_functions.get_mean_and_quants(preg_clin_counter_years,
                                                                               intervention_years)

        incidence_dates = extract_results(
            folder,
            module="tlo.methods.malaria",
            key="incidence",
            column='inc_1000py',
            index='date',
            do_scaling=True
        )

        years = incidence_dates.index.year
        incidence_years = incidence_dates.set_index(years)
        incidence = analysis_utility_functions.get_mean_and_quants(incidence_years, intervention_years)

        return {'clin_counter': preg_clinical_counter,
                'incidence': incidence}


    # HIV

