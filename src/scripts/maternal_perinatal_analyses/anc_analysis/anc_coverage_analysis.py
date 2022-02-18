from pathlib import Path
import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt

from tlo.analysis.utils import (
    extract_results,
    get_scenario_outputs, load_pickled_dataframes
)
from scripts.maternal_perinatal_analyses import analysis_utility_functions


def run_primary_analysis(baseline_scenario_filename, intervention_scenario_filename, outputspath,
                         show_and_store_graphs, intervention_years,  anc_scenario,
                         do_cons_calculation):
    """
    This function outputs the primary outcomes for analysis relating to coverage of antenatal care. This includes
    outputting plots comparing two scenarios for the following outcomes; MMR, NMR, SBR, DALYs, HCW cost,
    HCW capaniltiies, Consumable cost and Incremental cost effectiveness analysis. Function is called by
    run_anc_analysis.py
    :param baseline_scenario_filename:
    :param intervention_scenario_filename:
    :param outputspath:
    :param plot_destination_folder:
    :param intervention_years:
    :param anc_scenario:
    :param do_cons_calculation:

    :return:
    """
    # Find results folder (most recent run generated using that scenario_filename)
    baseline_results_folder = get_scenario_outputs(baseline_scenario_filename, outputspath)[-1]
    intervention_results_folder = get_scenario_outputs(intervention_scenario_filename, outputspath)[-1]

    # Determine number of draw in this run (used in determining some output)
    run_path = os.listdir(f'{outputspath}/{baseline_results_folder.name}/0')
    runs = [int(item) for item in run_path]

    # Create folder to store graphs (if it hasnt already been created when ran previously)
    path = f'{outputspath}/analysis_output_graphs_{intervention_results_folder.name}'
    if not os.path.isdir(path):
        os.makedirs(f'{outputspath}/analysis_output_graphs_{intervention_results_folder.name}')

    plot_destination_folder = path

    # ========================================= BIRTHs PER SCENARIO ==================================================
    # First we extract the total births during each scenario and process this dataframe to provide the mean and lower/
    # upper quantiles of births per year of the simulation run

    def get_total_births_per_year(folder):
        """
        Extract mean, lower and upper quantile births per year for a given scenario
        :param folder: results folder for scenario
        :return: list of total births per year of pre defined intervention period (i.e. 2020-2030)
        """
        births_results = extract_results(
            folder,
            module="tlo.methods.demography",
            key="on_birth",
            custom_generate_series=(
                lambda df: df.assign(year=df['date'].dt.year).groupby(['year'])['year'].count()),
            do_scaling=True
        )
        total_births_per_year = analysis_utility_functions.get_mean_and_quants(births_results, intervention_years)[0]
        return total_births_per_year

    # Store data for later access
    baseline_births = get_total_births_per_year(baseline_results_folder)
    intervention_births = get_total_births_per_year(intervention_results_folder)

    # TODO: subtract intrapartum stillbirths (check being logged as births)

    # ===============================================CHECK INTERVENTION ===============================================
    # Next we extract data relating to ANC coverage from the log and plot coverage of the intervention in both scenarios
    # (following burn in period)
    def get_anc_coverage(folder):
        """
        Extracts yearly population mean, upper and lower quantile coverage of predefined number of ANC visits
        (i.e. >4 or >8) for a scenario
        :param folder: results folder for scenario
        :return: list of lists. Mean coverage, lower quantile coverage, upper quantile coverage of ANC scenario
        """
        results = extract_results(
            folder,
            module="tlo.methods.care_of_women_during_pregnancy",
            key="anc_count_on_birth",
            custom_generate_series=(
                lambda df_: df_.assign(year=df_['date'].dt.year).groupby(['year', 'total_anc'])['person_id'].count()),
            do_scaling=False
        )
        anc_count_df = pd.DataFrame(columns=intervention_years, index=[0, 1, 2, 3, 4, 5, 6, 7, 8])

        # get yearly outputs
        for year in intervention_years:
            for row in anc_count_df.index:
                if row in results.loc[year].index:
                    x = results.loc[year, row]
                    anc_count_df.at[row, year] = [x.mean(), x.quantile(0.025), x.quantile(0.925)]
                else:
                    anc_count_df.at[row, year] = [0, 0, 0]

        yearly_anc_rates = list()
        anc_lqs = list()
        anc_uqs = list()

        for year in intervention_years:
            anc_total = 0
            target_or_more_visits = 0

            for row in anc_count_df[year]:
                anc_total += row[0]

            if anc_scenario == 4:
                visits = anc_count_df.loc[anc_count_df.index > 3]
            else:
                visits = anc_count_df.loc[anc_count_df.index >= 8]

            f_lqs = 0
            f_uqs = 0
            for row in visits[year]:
                target_or_more_visits += row[0]
                f_lqs += row[1]
                f_uqs += row[2]

            yearly_anc_rates.append((target_or_more_visits / anc_total) * 100)
            anc_lqs.append((f_lqs / anc_total) * 100)
            anc_uqs.append((f_uqs / anc_total) * 100)

        return[yearly_anc_rates, anc_lqs, anc_uqs]

    # Extract data for each scenario
    baseline_anc_target_coverage = get_anc_coverage(baseline_results_folder)
    intervention_anc_target_coverage = get_anc_coverage(intervention_results_folder)

    # generate plots showing coverage of ANC intervention in the baseline and intervention scenarios
    if show_and_store_graphs:
        analysis_utility_functions.basic_comparison_graph(
            intervention_years, baseline_anc_target_coverage, intervention_anc_target_coverage,
            f'Coverage of ANC{anc_scenario}',
            f'ANC{anc_scenario} coverage in baseline and intervention scenarios from 2020',
            plot_destination_folder, f'anc{anc_scenario}_intervention_coverage')

    #  ================================== MATERNAL AND NEONATAL DEATH ===============================================
    def get_yearly_death_data(folder, births):
        """
        Extract mean, lower and upper quantile maternal mortality ratio, neonatal mortality ratio, crude maternal
        deaths and crude neonatal deaths per year for a given scenario
        :param folder: results folder for scenario
        :param births: list. mean number of births per year for a scenario (used as a denominator)
        :return: dict containing mean, LQ, UQ for MMR, NMR, maternal deaths and neonatal deaths
        """
        death_results_labels = extract_results(
            folder,
            module="tlo.methods.demography",
            key="death",
            custom_generate_series=(
                lambda df: df.assign(year=df['date'].dt.year).groupby(['year', 'label'])['year'].count()),
            do_scaling=True)

        other_preg_deaths = extract_results(
            folder,
            module="tlo.methods.demography",
            key="death",
            custom_generate_series=(
                lambda df: df.assign(year=df['date'].dt.year).groupby(['year', 'label', 'pregnancy'])['year'].count()),
            do_scaling=True
        )

        # Extract maternal mortality ratio from direct maternal causes
        mmr = analysis_utility_functions.get_comp_mean_and_rate('Maternal Disorders', births, death_results_labels,
                                                                100000, intervention_years)

        # Extract crude deaths due to direct maternal disorders
        crude_m_deaths = analysis_utility_functions.get_mean_and_quants_from_str_df(death_results_labels,
                                                                                    'Maternal Disorders',
                                                                                    intervention_years)

        # Extract crude deaths due to indirect causes in pregnant women
        indirect_causes = ['AIDS', 'Malaria', 'TB']
        indirect_deaths = list()
        id_lq = list()
        id_uq = list()
        for year in intervention_years:
            id_deaths_per_year = 0
            id_lq_py = 0
            id_uq_pu = 0
            for cause in indirect_causes:
                if cause in other_preg_deaths.loc[year, :, True].index:
                    id_deaths_per_year += other_preg_deaths.loc[year, cause, True].mean()
                    id_lq_py += other_preg_deaths.loc[year, cause, True].quantile(0.025)
                    id_uq_pu += other_preg_deaths.loc[year, cause, True].quantile(0.925)

            indirect_deaths.append(id_deaths_per_year)
            id_lq.append(id_deaths_per_year)
            id_uq.append(id_deaths_per_year)

        # Calculate total MMR (direct + indirect deaths)
        total_mmr = [[((x + y) / z) * 100000 for x, y, z in zip(indirect_deaths, crude_m_deaths[0], births)],
                     [((x + y) / z) * 100000 for x, y, z in zip(id_lq, crude_m_deaths[1], births)],
                     [((x + y) / z) * 100000 for x, y, z in zip(id_uq, crude_m_deaths[2], births)]
                     ]

        # Extract NMR
        nmr = analysis_utility_functions.get_comp_mean_and_rate('Neonatal Disorders', births, death_results_labels,
                                                                1000, intervention_years)

        # And crude neonatal deaths
        crude_n_deaths = analysis_utility_functions.get_mean_and_quants_from_str_df(death_results_labels,
                                                                                    'Neonatal Disorders',
                                                                                    intervention_years)

        return {'direct_mmr': mmr,
                'total_mmr': total_mmr,
                'nmr': nmr,
                'crude_m_deaths': crude_m_deaths,
                'crude_n_deaths': crude_n_deaths}

    def get_crude_death_graphs_graphs(b_deaths, i_deaths, group):
        """
        Generates bar chart with confidence intervals used to output crude number of deaths (maternal/neonatal) per year
         per scenario
        :param b_deaths: baseline death data
        :param i_deaths: intervention death data
        :param group: str. maternal or neonatal
        """
        b_deaths_ci = [(x - y) / 2 for x, y in zip(b_deaths[2], b_deaths[1])]
        i_deaths_ci = [(x - y) / 2 for x, y in zip(i_deaths[2], i_deaths[1])]

        N = len(b_deaths[0])
        ind = np.arange(N)
        width = 0.35
        plt.bar(ind, b_deaths[0], width, label='Baseline', yerr=b_deaths_ci, color='teal')
        plt.bar(ind + width, i_deaths[0], width, label='Intervention', yerr=i_deaths_ci, color='olivedrab')
        plt.ylabel(f'Total {group} Deaths (scaled)')
        plt.title(f'Yearly Baseline {group} Deaths Compared to Intervention')
        plt.xticks(ind + width / 2, intervention_years)
        plt.legend(loc='best')
        plt.savefig(f'{plot_destination_folder}/{group}_crude_deaths_comparison.png')
        plt.show()

    # Extract data from scenarios
    baseline_death_data = get_yearly_death_data(baseline_results_folder, baseline_births)
    intervention_death_data = get_yearly_death_data(intervention_results_folder, intervention_births)

    if show_and_store_graphs:
        # Generate plots of yearly MMR and NMR
        analysis_utility_functions.basic_comparison_graph(
            intervention_years, baseline_death_data['direct_mmr'], intervention_death_data['direct_mmr'],
            'Deaths per 100,000 live births',
            'MMR per Year at Baseline and Under Intervention (Direct only)',
            plot_destination_folder, 'maternal_mr_direct')

        analysis_utility_functions.basic_comparison_graph(
            intervention_years, baseline_death_data['total_mmr'], intervention_death_data['total_mmr'],
            'Deaths per 100,000 live births',
            'MMR per Year at Baseline and Under Intervention (Total)',
            plot_destination_folder, 'maternal_mr_total')

        analysis_utility_functions.basic_comparison_graph(
            intervention_years, baseline_death_data['nmr'], intervention_death_data['nmr'],
            'Total Deaths per 1000 live births',
            'Neonatal Mortality Ratio per Year at Baseline and Under Intervention',
            plot_destination_folder, 'neonatal_mr_int')

        # Generate plots for crude deaths
        get_crude_death_graphs_graphs(baseline_death_data['crude_m_deaths'], intervention_death_data['crude_m_deaths'],
                                      'Maternal')
        get_crude_death_graphs_graphs(baseline_death_data['crude_n_deaths'], intervention_death_data['crude_n_deaths'],
                                      'Neonatal')

    #  ================================== STILLBIRTH  ===============================================
    def get_yearly_sbr_data(folder, births):
        """
        Extract antenatal and intrapartum stillbirths from a scenario and return crude numbers and stillbirth rate per
        year
        :param folder: results folder for scenario
        :param births: list. mean number of births per year for a scenario (used as a denominator)
        """
        an_stillbirth_results = extract_results(
            folder,
            module="tlo.methods.pregnancy_supervisor",
            key="antenatal_stillbirth",
            custom_generate_series=(
                lambda df: df.assign(year=df['date'].dt.year).groupby(['year'])['year'].count()),
            do_scaling=True
        )
        ip_stillbirth_results = extract_results(
            folder,
            module="tlo.methods.labour",
            key="intrapartum_stillbirth",
            custom_generate_series=(
                lambda df: df.assign(year=df['date'].dt.year).groupby(['year'])['year'].count()),
            do_scaling=True
        )

        # Get stillbirths
        an_still_birth_data = analysis_utility_functions.get_mean_and_quants(an_stillbirth_results, intervention_years)
        ip_still_birth_data = analysis_utility_functions.get_mean_and_quants(ip_stillbirth_results, intervention_years)

        # Store mean number of stillbirths, LQ, UQ
        crude_sb = [[x + y for x, y in zip(an_still_birth_data[0], ip_still_birth_data[0])],
                    [x + y for x, y in zip(an_still_birth_data[1], ip_still_birth_data[1])],
                    [x + y for x, y in zip(an_still_birth_data[2], ip_still_birth_data[2])]]

        # Then generate SBR
        total_sbr = [[((x + y) / z) * 1000 for x, y, z in zip(an_still_birth_data[0], ip_still_birth_data[0], births)],
                     [((x + y) / z) * 1000 for x, y, z in zip(an_still_birth_data[1], ip_still_birth_data[1], births)],
                     [((x + y) / z) * 1000 for x, y, z in zip(an_still_birth_data[2], ip_still_birth_data[2], births)]]

        # Return as dict for graphs
        return {'sbr': total_sbr,
                'crude_sb': crude_sb}

    # Get data
    baseline_sb_data = get_yearly_sbr_data(baseline_results_folder, baseline_births)
    intervention_sb_data = get_yearly_sbr_data(intervention_results_folder, intervention_births)

    if show_and_store_graphs:
        # Output SBR per year for scenario vs intervention
        analysis_utility_functions.basic_comparison_graph(
            intervention_years, baseline_sb_data['sbr'], intervention_sb_data['sbr'],
            'Stillbirths per 1000 births',
            'Stillbirth Rate per Year at Baseline and Under Intervention',
            plot_destination_folder, 'sbr_int')

        # Create values for yerr to display on bars
        b_sb_ci = [(x - y) / 2 for x, y in zip(baseline_sb_data['crude_sb'][2], baseline_sb_data['crude_sb'][1])]
        i_sb_ci = [(x - y) / 2 for x, y in zip(intervention_sb_data['crude_sb'][2], intervention_sb_data['crude_sb'][1])]

        # output barchart
        N = len(baseline_sb_data['crude_sb'][0])
        ind = np.arange(N)
        width = 0.35
        plt.bar(ind, baseline_sb_data['crude_sb'][0], width, label='Baseline', yerr=b_sb_ci, color='teal')
        plt.bar(ind + width, intervention_sb_data['crude_sb'][0], width, label='Intervention', yerr=i_sb_ci,
                color='olivedrab')
        plt.ylabel('Total Stillbirths (scaled)')
        plt.title('Yearly Baseline Stillbirths Compared to Intervention')
        plt.xticks(ind + width / 2, intervention_years)
        plt.legend(loc='best')
        plt.savefig(f'{plot_destination_folder}/crude_stillbirths_comparison.png')
        plt.show()

    # =================================================== DALYS =======================================================
    # todo: express DALYs as a rate
    def get_dalys_from_scenario(results_folder):
        """
        Extracted stacked DALYs from logger for maternal and neonatal disorders
        :param results_folder: results folder for scenario
        :return: Maternal and neonatal dalys [Mean, LQ, UQ]
        """
        # Get DALY df
        dalys_stacked = extract_results(
            results_folder,
            module="tlo.methods.healthburden",
            key="dalys_stacked",
            custom_generate_series=(
                lambda df: df.drop(
                    columns='date').groupby(['year']).sum().stack()),
            do_scaling=True)

        def extract_dalys_tlo_model(group):
            """Extract mean, LQ, UQ DALYs for maternal or neonatal disorders"""
            stacked_dalys = [dalys_stacked.loc[year, f'{group} Disorders'].mean() for year in
                             intervention_years if year in intervention_years]

            stacked_dalys_lq = [dalys_stacked.loc[year, f'{group} Disorders'].quantile(0.025) for year in
                                intervention_years if year in intervention_years]

            stacked_dalys_uq = [dalys_stacked.loc[year, f'{group} Disorders'].quantile(0.925) for year in
                                intervention_years if year in intervention_years]

            return [stacked_dalys, stacked_dalys_lq, stacked_dalys_uq]

        return [extract_dalys_tlo_model('Maternal'), extract_dalys_tlo_model('Neonatal')]

    # Store DALYs data for baseline and intervention
    baseline_dalys = get_dalys_from_scenario(baseline_results_folder)
    baseline_maternal_dalys = baseline_dalys[0]
    baseline_neonatal_dalys = baseline_dalys[1]

    intervention_dalys = get_dalys_from_scenario(intervention_results_folder)
    intervention_maternal_dalys = intervention_dalys[0]
    intervention_neonatal_dalys = intervention_dalys[1]

    if show_and_store_graphs:
        # Output graphs
        analysis_utility_functions.basic_comparison_graph(
            intervention_years, baseline_maternal_dalys, intervention_maternal_dalys,
            'Disability Adjusted Life Years (stacked)',
            'Total DALYs per Year Attributable to Maternal disorders',
            plot_destination_folder, 'maternal_dalys_stacked')

        analysis_utility_functions.basic_comparison_graph(
            intervention_years, baseline_neonatal_dalys, intervention_neonatal_dalys,
            'Disability Adjusted Life Years (stacked)',
            'Total DALYs per Year Attributable to Neonatal disorders',
            plot_destination_folder, 'neonatal_dalys_stacked')

    # =============================================  COSTS/HEALTH SYSTEM ==============================================
    # =============================================  HCW TIME =========================================================

    def get_hcw_time_per_year(results_folder):
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

        cadre_time = {'Nursing_and_Midwifery': {'AntenatalFirst': 0,
                                                'ANCSubsequent': 0},
                      'Clinical': {'AntenatalFirst': 0,
                                   'ANCSubsequent': 0}}

        for k in cadre_time:
            for v in cadre_time[k].keys():
                time = time_by_appt.loc[
                    (time_by_appt.Appt_Type_Code == v) & (time_by_appt.Officer_Category == k) &
                    (time_by_appt.Facility_Level == '1a'), 'Time_Taken_Mins']
                cadre_time[k][v] += time.values

        # Loop over each draw
        for run in runs:
            # Load df, add year column and select only ANC interventions
            run_df = load_pickled_dataframes(results_folder, draw=0, run=run)

            hsi = run_df['tlo.methods.healthsystem']['HSI_Event']
            hsi['year'] = hsi['date'].dt.year
            anc_hsi = hsi.loc[hsi.TREATMENT_ID.str.contains('AntenatalCare') & hsi.did_run]

            for year in intervention_years:
                # Get total number of anc1 visits (have a higher HCW time requirement)
                total_anc_1_visits = len(anc_hsi.loc[(anc_hsi.year == year) &
                                                     (anc_hsi.TREATMENT_ID.str.contains('First'))])

                # Followed by the remaining 'subsequent' ANC visits
                total_anc_other_visits = len(anc_hsi.loc[(anc_hsi.year == year)]) - total_anc_1_visits
                assert total_anc_other_visits + total_anc_1_visits == len(anc_hsi.loc[(anc_hsi.year == year)])

                yearly_hcw_time = (total_anc_1_visits * (cadre_time['Nursing_and_Midwifery']['AntenatalFirst'] +
                                                         cadre_time['Clinical']['AntenatalFirst'])) + \
                                  (total_anc_other_visits * (cadre_time['Nursing_and_Midwifery']['ANCSubsequent'] +
                                                             cadre_time['Clinical']['ANCSubsequent']))

                total_time_per_draw_per_year.loc[year, run] = yearly_hcw_time / 60  # returns time in hours

        # TODO: not sure on the maths here (percentiles vs quantiles)
        mean_time = [total_time_per_draw_per_year.loc[year].to_numpy().mean() for year in intervention_years]
        lq_time = [np.percentile(total_time_per_draw_per_year.loc[year].to_numpy(), 2.5) for year in intervention_years]
        uq_time = [np.percentile(total_time_per_draw_per_year.loc[year].to_numpy(), 92.5) for year in intervention_years]

        return [mean_time, lq_time, uq_time]

    # Get the data
    b_hcw_time = get_hcw_time_per_year(baseline_results_folder)
    i_hcw_time = get_hcw_time_per_year(intervention_results_folder)

    # Output data
    if show_and_store_graphs:
        analysis_utility_functions.basic_comparison_graph(
            intervention_years, b_hcw_time, i_hcw_time,
            'Total Time (hours)', 'Total Nurse/Midwife Time Spent Delivering Antenatal Care Per Year (unscaled)',
            plot_destination_folder, 'hcw_time')

    # =============================================  HCW COSTS =======================================================
    # todo: use time (generated above) and salary of health care workers to determine salaried time-cost associated with
    #  each scenario
    # todo: worth considering costs associated with training (if known)
    # todo: plot...

    # ==========================================  HCW CAPABILITY ===================================================
    # todo: what fraction of total capabiltiies are taken up by scenario (split up by cadre? - will need new logging)

    # =================================================== CONSUMABLE COST ===========================================
    if do_cons_calculation:  # only output if specified due to very long run tine
        resourcefilepath = Path("./resources/healthsystem/consumables/")
        consumables_df = pd.read_csv(Path(resourcefilepath) / 'ResourceFile_Consumables.csv')

        # TODO: this should be scaled to the correct population size?
        # todo: also so slow...
        def get_cons_cost_per_year(results_folder):
            # Create df that replicates the 'extracted' df
            total_cost_per_draw_per_year = pd.DataFrame(columns=[runs], index=[intervention_years])

            # Loop over each draw
            for run in runs:
                # Load df, add year column and select only ANC interventions
                run_df = load_pickled_dataframes(results_folder, draw=0, run=run)

                cons = run_df['tlo.methods.healthsystem']['Consumables']
                cons['year'] = cons['date'].dt.year
                anc_cons = cons.loc[(cons.TREATMENT_ID.str.contains('AntenatalCare')) &
                                    (cons.year >= intervention_years[0])]
                # todo: is this right?
                anc_cons['Item_Available'].apply(lambda df: eval(df))

                cons_df_for_this_draw = pd.DataFrame(index=[intervention_years])

                # Loop over each year
                for year in intervention_years:
                    # Select the year of interest
                    year_df = anc_cons.loc[anc_cons.year == year]
                    anc_cons['Item_Available'].apply(
                        lambda x: ((cons_df_for_this_draw[k] == v) for k, v in x if k not in cons_df_for_this_draw.columns))

                    # For each row (hsi) in that year we unpack the dictionary
                    for row in year_df.index:
                        for k, v in year_df.at[row, 'Item_Available'].items():
                            if k in cons_df_for_this_draw.columns:
                                cons_df_for_this_draw.at[year, k] += v
                            elif k not in cons_df_for_this_draw.columns:
                                cons_df_for_this_draw[k] = v

                for row in cons_df_for_this_draw.index:
                    for column in cons_df_for_this_draw.columns:
                        cons_df_for_this_draw.at[row, column] =\
                            (cons_df_for_this_draw.at[row, column] *
                             (consumables_df[consumables_df.Item_Code == 0]['Unit_Cost'].iloc[0]))
                        cons_df_for_this_draw.at[row, column] = cons_df_for_this_draw.at[row, column] * 0.0014
                        # todo: this is usd conversion
                        # todo: account for inflation, and use 2010 rate

                for index in total_cost_per_draw_per_year.index:
                    total_cost_per_draw_per_year.at[index, run] = cons_df_for_this_draw.loc[index].sum()

            # todo: check maths
            mean_cost = [total_cost_per_draw_per_year.loc[year].to_numpy().mean() for year in intervention_years]

            lq_cost = [np.percentile(total_cost_per_draw_per_year.loc[year].to_numpy(), 2.5) for year in
                       intervention_years]

            uq_cost = [np.percentile(total_cost_per_draw_per_year.loc[year].to_numpy(), 92.5) for year in
                       intervention_years]

            return [mean_cost, lq_cost, uq_cost]

        baseline_cost_data = get_cons_cost_per_year(baseline_results_folder)
        intervention_cost_data = get_cons_cost_per_year(intervention_results_folder)

        if show_and_store_graphs:
            analysis_utility_functions.basic_comparison_graph(
                intervention_years, baseline_cost_data, intervention_cost_data,
                'Total Cost (USD)', 'Total Consumable Cost Attributable To ANC Per Year (in USD) (unscaled)',
                plot_destination_folder, 'cost')

        # ======================================== COST EFFECTIVENESS RATIO ===========================================
        # Cost (i) - Cost(b) / DALYs (i) - DALYs (b)
        # todo include healthcare worker cost

        cost_difference = [(x - y) for x, y in zip(intervention_cost_data[0], baseline_cost_data[0])]
        daly_difference = [(x - y) for x, y in zip(baseline_maternal_dalys[0], intervention_maternal_dalys[0])]
        ICR = [(x / y) for x, y in zip(cost_difference, daly_difference)]

        fig, ax = plt.subplots()
        ax.plot(intervention_years, ICR, label="Baseline (mean)", color='deepskyblue')
        plt.xlabel('Year')
        plt.ylabel("ICR")
        plt.title('Incremental Cost Effectiveness Ratio (maternal) (unscaled)')
        plt.legend()
        plt.savefig(f'{plot_destination_folder}/icr_maternal.png')
        plt.show()
