from pathlib import Path
import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
plt.style.use('seaborn-darkgrid')

from tlo.analysis.utils import (
    extract_results,
    get_scenario_outputs, load_pickled_dataframes
)
from scripts.maternal_perinatal_analyses import analysis_utility_functions


def run_sba_scenario_analysis(scenario_file_dict, outputspath, show_and_store_graphs, intervention_years,
                              do_cons_calculation):

    """
    """
    # Find results folder (most recent run generated using that scenario_filename)

    results_folders = {k: get_scenario_outputs(scenario_file_dict[k], outputspath)[-1] for k in scenario_file_dict}

    # Determine number of draw in this run (used in determining some output)
    run_path = os.listdir(f'{outputspath}/{results_folders["Status Quo"].name}/0')
    # todo: fix so doesnt rely on preset name
    runs = [int(item) for item in run_path]

    # Create folder to store graphs (if it hasnt already been created when ran previously)
    path = f'{outputspath}/sba_analysis_output_graphs_{results_folders["Status Quo"].name}'
    if not os.path.isdir(path):
        os.makedirs(f'{outputspath}/sba_analysis_output_graphs_{results_folders["Status Quo"].name}')

    plot_destination_folder = path

    # GET SCALING FACTORS FOR SCENARIOS
    scaling_factors = {k: load_pickled_dataframes(results_folders[k], 0, 0, 'tlo.methods.demography'
                            )['tlo.methods.demography']['scaling_factor']['scaling_factor'].values[0]
                       for k in results_folders}
    # todo: what leads to variation between scaling factors/runs/draws

    # ========================================= BIRTHs PER SCENARIO ==================================================
    # Access birth data
    births_dict = analysis_utility_functions.return_birth_data_from_multiple_scenarios(results_folders,
                                                                                       intervention_years)

    # ===============================================CHECK INTERVENTION ===============================================
    # todo: what can we output here to demonstrate the intervention has been applied?
    # met need? by intervention - ask Tim

    #  ================================== MATERNAL AND NEONATAL DEATH ===============================================
    # Access death data
    # Extract data from scenarios
    death_data = analysis_utility_functions.return_death_data_from_multiple_scenarios(results_folders, births_dict,
                                                                                      intervention_years,
                                                                                      detailed_log=True)
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

    # =============================================  COSTS/HEALTH SYSTEM ==============================================
    # =============================================  HCW TIME =========================================================

    def get_hcw_time_per_year(results_folder, sf):
        """
        This function simply uses HSI dataframe to determine the amount of time Antenatal Care takes to deliver for
        healthcare workers within a scenario
        :param results_folder:
        :return:
        """

        # Create df that replicates the 'extracted' df
        total_time_per_draw_per_year = pd.DataFrame(columns=[runs], index=[intervention_years])

        # Select data containing time by appt type
        resourcefilepath = Path("./resources/healthsystem/human_resources/definitions")
        time_by_appt = pd.read_csv(Path(resourcefilepath) / 'ResourceFile_Appt_Time_Table.csv')

        def get_hcw_time_by_appt_fp_and_fl(hcw):
            time_dict = {}
            for fl in ['1a', '1b', '2']:
                fl_dict = {}
                for at in ['NormalDelivery', 'CompDelivery', 'Csection', 'MajorSurg', 'InpatientDays']:
                    if fl not in time_by_appt.loc[(time_by_appt.Appt_Type_Code == at)]['Facility_Level'].values:
                        time = 0
                    elif hcw not in time_by_appt.loc[(time_by_appt.Facility_Level == fl) &
                                                     (time_by_appt.Appt_Type_Code == at)]['Officer_Category'].values:
                        time = 0
                    else:
                        time = float(time_by_appt.loc[
                            (time_by_appt.Appt_Type_Code == at) & (time_by_appt.Officer_Category == hcw) &
                            (time_by_appt.Facility_Level == fl), 'Time_Taken_Mins'].values)

                    fl_dict.update({at: time})
                time_dict.update({fl: fl_dict})

            return time_dict

        nm_time = get_hcw_time_by_appt_fp_and_fl('Nursing_and_Midwifery')
        clin_time = get_hcw_time_by_appt_fp_and_fl('Clinical')

        # TODO: add CSection code to CEmONC HSI
        # TODO: pharmacy time?

        # Loop over each draw
        for run in runs:
            # Load df, add year column and select only ANC interventions
            run_df = load_pickled_dataframes(results_folder, draw=0, run=run)

            hsi = run_df['tlo.methods.healthsystem']['HSI_Event']
            hsi['year'] = hsi['date'].dt.year

            # Get HSIs
            emonc_hsi = hsi.loc[hsi.TREATMENT_ID.str.contains('SkilledBirthAttendanceDuringLabour') & hsi.did_run]
            cemonc_hsi = hsi.loc[hsi.TREATMENT_ID.str.contains('ComprehensiveEmergencyObstetricCare') & hsi.did_run]
            newborn_hsi = hsi.loc[hsi.TREATMENT_ID.str.contains('CareOfTheNewbornBySkilledAttendant') &
                                  hsi.did_run]

            for year in intervention_years:
                # Get total number of anc1 visits (have a higher HCW time requirement)

                # todo: this wont work when the value of the dict changes...idk why it would though
                total_emonc_normal = len(
                    emonc_hsi.loc[(emonc_hsi.year == year) &
                                  (emonc_hsi.Number_By_Appt_Type_Code == {'NormalDelivery': 1})])

                total_emonc_comp = len(
                    emonc_hsi.loc[(emonc_hsi.year == year) &
                                  (emonc_hsi.Number_By_Appt_Type_Code == {'CompDelivery': 1})])

                assert total_emonc_normal + total_emonc_comp == len(emonc_hsi.loc[(emonc_hsi.year == year)])

                total_cemonc_surg = len(
                    cemonc_hsi.loc[(cemonc_hsi.year == year) &
                                  (cemonc_hsi.Number_By_Appt_Type_Code == {'MajorSurg': 1})])

                total_cemonc_other = len(
                    cemonc_hsi.loc[(cemonc_hsi.year == year) &
                                   (cemonc_hsi.Number_By_Appt_Type_Code == {'MajorSurg': 0})])

                assert total_cemonc_surg + total_cemonc_other == len(cemonc_hsi.loc[(cemonc_hsi.year == year)])

                total_newborn = len(newborn_hsi.loc[(newborn_hsi.year == year)])

                # TODO: THIS IS WRONG- WE NEED TO LOG FACILITY LEVEL OF HSIS
                total_bemonc_hcw_time = \
                    (total_emonc_normal * (nm_time['1b']['NormalDelivery'] +
                                           clin_time['1b']['NormalDelivery'])) + \
                    (total_emonc_comp * (nm_time['1b']['NormalDelivery'] +
                                         clin_time['1b']['NormalDelivery'])) + \
                    (total_newborn * (nm_time['1b']['InpatientDays'] +
                                      clin_time['1b']['InpatientDays']))

                total_cemonc_time = \
                    (total_cemonc_surg * (nm_time['1b']['MajorSurg'] +
                                           clin_time['1b']['MajorSurg'])) + \
                    (total_cemonc_other * (nm_time['1b']['InpatientDays'] +
                                           clin_time['1b']['InpatientDays']))

                total_time_per_draw_per_year.loc[year, run] = (total_bemonc_hcw_time + total_cemonc_time / 60) * sf
                # returns time in hours

                # todo: better to output for cemonc/emonc HSIs

        # TODO: not sure on the maths here (percentiles vs quantiles)
        mean_time = [total_time_per_draw_per_year.loc[year].to_numpy().mean() for year in intervention_years]
        lq_time = [np.percentile(total_time_per_draw_per_year.loc[year].to_numpy(), 2.5) for year in intervention_years]
        uq_time = [np.percentile(total_time_per_draw_per_year.loc[year].to_numpy(), 92.5) for year in intervention_years]

        return [mean_time, lq_time, uq_time]

    # Get the data
    hcw_data = {k: get_hcw_time_per_year(results_folders[k], scaling_factors[k]) for k in results_folders}

    # Output data
    if show_and_store_graphs:
        analysis_utility_functions.comparison_graph_multiple_scenarios(
            intervention_years, hcw_data, 'Total Time (hours)',
            'Total Healthcare Worker Time Requested to Deliver SBA Per Year (Scaled)',
            plot_destination_folder, 'hcw_time_sba')

    # =============================================  HCW COSTS =======================================================
    # todo: hrly rate? ((pa salary/working days) / avg working hrs per day)

    nurs_officer_salary_pa = 2978332
    assumed_working_days_in_a_year = 247  # todo: should this vary by year?
    nurse_hrly_rate = ((nurs_officer_salary_pa / assumed_working_days_in_a_year) / 7.5) * 0.0014
    # (working hrs in 5day working week)

    pay_list = [nurse_hrly_rate for years in intervention_years]

    # todo: this only makes sense because their salry is the same, if theres a difference you need to calculat the
    #  cost associated to different cadres
    hcw_cost = dict()
    for k in hcw_data:
        hcw_cost.update({k: {'cost': [[x * y for x, y in zip(hcw_data[k][0], pay_list)],
                                      [x * y for x, y in zip(hcw_data[k][1], pay_list)],
                                      [x * y for x, y in zip(hcw_data[k][2], pay_list)]]}})

    analysis_utility_functions.comparison_bar_chart_multiple_bars(
        hcw_cost, 'cost', intervention_years,
        'Cost USD', 'Yearly Cost of Requested HCW time to deliver SBA per Scenario',
        plot_destination_folder, 'hcw_time_cost.png')

    pd_dict = {k: {'pd': [[((x - y) / y) * 100 for x, y in zip(hcw_cost[k]['cost'][0],
                                                               hcw_cost['Status Quo']['cost'][0])],
                          [((x - y) / y) * 100 for x, y in zip(hcw_cost[k]['cost'][1],
                                                               hcw_cost['Status Quo']['cost'][1])],
                          [((x - y) / y) * 100 for x, y in zip(hcw_cost[k]['cost'][2],
                                                               hcw_cost['Status Quo']['cost'][2])]]}
                   for k in ['Perfect BEmONC', 'Perfect CEmONC', 'Perfect BEmONC+CEmONC']}

    analysis_utility_functions.comparison_bar_chart_multiple_bars(
        pd_dict, 'pd', intervention_years,
        'Percentage Difference', 'Percentage Difference in Cost for Requested HCW time to delivery SBA',
        plot_destination_folder, 'hcw_time_cost_pd.png')

    # ========================================== SQUEEZE =============================================================
    def get_squeeze_data(folder):
        squeeze_mean_dict = {}
        squeeze_prop_dict = {}

        for hsi in ['SkilledBirthAttendanceDuringLabour', 'ComprehensiveEmergencyObstetricCare',
                    'CareOfTheNewbornBySkilledAttendant']:
            df = extract_results(
                folder,
                module="tlo.methods.healthsystem",
                key="HSI_Event",
                custom_generate_series=(
                    lambda df: df.loc[df['TREATMENT_ID'].str.contains(hsi) & df['did_run']].assign(
                        year=df['date'].dt.year).groupby(['year'])['Squeeze_Factor'].mean()))

            squeeze_mean_dict.update({hsi: df})

        for k in squeeze_mean_dict:
            mean_squeeze_per_year = [squeeze_mean_dict[k].loc[year].to_numpy().mean() for year in
                                     intervention_years]

            lq_squeeze_per_year = [np.percentile(squeeze_mean_dict[k].loc[year].to_numpy(), 2.5) for year in
                                   intervention_years]

            uq_squeeze_per_year = [np.percentile(squeeze_mean_dict[k].loc[year].to_numpy(), 92.5) for year in
                                   intervention_years]

            squeeze_mean_dict[k] = [mean_squeeze_per_year, lq_squeeze_per_year, uq_squeeze_per_year]
            # squeeze_dict now contains means/lq/uq for squeeze factors for each HSI

        for hsi in ['SkilledBirthAttendanceDuringLabour', 'ComprehensiveEmergencyObstetricCare',
                    'CareOfTheNewbornBySkilledAttendant']:
            hsi_count = extract_results(
                folder,
                module="tlo.methods.healthsystem",
                key="HSI_Event",
                custom_generate_series=(
                    lambda df: df.loc[df['TREATMENT_ID'].str.contains(hsi) & df['did_run']].assign(
                        year=df['date'].dt.year).groupby(['year'])['year'].count()))

            hsi_squeeze = extract_results(
                folder,
                module="tlo.methods.healthsystem",
                key="HSI_Event",
                custom_generate_series=(
                    lambda df:
                    df.loc[(df['TREATMENT_ID'].str.contains(hsi)) & df['did_run'] & (df['Squeeze_Factor'] > 0)
                           ].assign(year=df['date'].dt.year).groupby(['year'])['year'].count()))

            prop_squeeze_year = [(hsi_squeeze.loc[year].to_numpy().mean()/hsi_count.loc[year].to_numpy().mean()) * 100
                                 for year in intervention_years]
            prop_squeeze_lq = [
                (np.percentile(hsi_squeeze.loc[year].to_numpy(), 2.5) /
                 np.percentile(hsi_count.loc[year].to_numpy(), 2.5)) * 100 for year in intervention_years]

            prop_squeeze_uq = [
                (np.percentile(hsi_squeeze.loc[year].to_numpy(), 92.5) /
                 np.percentile(hsi_count.loc[year].to_numpy(), 92.5)) * 100 for year in intervention_years]

            squeeze_prop_dict.update({hsi:[prop_squeeze_year, prop_squeeze_lq, prop_squeeze_uq]})

        return {'mean': squeeze_mean_dict,
                'proportion': squeeze_prop_dict}

    squeeze_data = {k: get_squeeze_data(results_folders[k]) for k in results_folders}
    # Output data

    # todo: PLOT

    # =================================================== CONSUMABLE COST =============================================
    if do_cons_calculation:  # only output if specified due to very long run tine
        resourcefilepath = Path("./resources/healthsystem/consumables/")
        consumables_df = pd.read_csv(Path(resourcefilepath) / 'ResourceFile_Consumables_Items_and_Packages.csv')

