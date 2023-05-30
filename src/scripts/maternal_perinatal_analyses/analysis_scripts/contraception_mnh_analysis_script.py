import os

import analysis_utility_functions
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from tlo.analysis.utils import extract_results, get_scenario_outputs
from analysis_utility_functions import get_mean_from_columns, get_mean_95_CI_from_list, return_95_CI_across_runs
import scipy.stats

plt.style.use('seaborn-darkgrid')

def run_maternal_newborn_health_analysis(scenario_file_dict, outputspath, sim_years,
                                         intervention_years, scen_colours):
    """
    """
    results_folders = {k: get_scenario_outputs(scenario_file_dict[k], outputspath)[-1] for k in scenario_file_dict}
    scenario_names = list(results_folders.keys())

    # Create folder to store graphs (if it hasn't already been created when ran previously)
    path = f'{outputspath}/contraception_mnh_analysis_results_' \
           f'{results_folders[scenario_names[0]].name}'

    if not os.path.isdir(path):
        os.makedirs(f'{outputspath}/contraception_mnh_analysis_results_'
                    f'{results_folders[scenario_names[0]].name}')

    primary_oc_path = f'{path}/primary_outcomes'
    if not os.path.isdir(primary_oc_path):
        os.makedirs(f'{path}/primary_outcomes')

    secondary_oc_path = f'{path}/secondary_outcomes'
    if not os.path.isdir(secondary_oc_path):
        os.makedirs(f'{path}/secondary_outcomes')

    scenario_titles = list(results_folders.keys())

    output_df = pd.DataFrame(
        columns=['scenario',
                 'output',
                 'mean_95%CI_value_for_int_period',
                 'skew_for_int_data',
                 'mean_95%CI_diff_outcome_int_period',
                 'skew_for_diff_data',
                 'median_diff_outcome_int_period'])

    # DEFINE HELPER FUNCTIONs....
    def plot_agg_graph(data, key, y_label, title, save_name, save_location):
        labels = results_folders.keys()

        mean_vals = list()
        lq_vals = list()
        uq_vals = list()
        for k in data:
            mean_vals.append(data[k][key][0])
            lq_vals.append(data[k][key][1])
            uq_vals.append(data[k][key][2])

        width = 0.55  # the width of the bars: can also be len(x) sequence
        fig, ax = plt.subplots()

        ci = [(x - y) / 2 for x, y in zip(uq_vals, lq_vals)]
        ax.bar(labels, mean_vals, color=scen_colours, width=width, yerr=ci)
        ax.tick_params(axis='x', which='major', labelsize=8)

        ax.set_ylabel(y_label)
        ax.set_xlabel('Scenario')
        ax.set_title(title)
        plt.savefig(f'{save_location}/{save_name}.png')
        plt.show()

    def get_med_or_mean_from_columns(df, mean_or_med):
        values = list()
        for col in df:
            if mean_or_med == 'mean':
                values.append(np.mean(df[col]))
            elif mean_or_med == 'median':
                values.append(np.median(df[col]))
            else:
                values.append(sum(df[col]))
        return values

    def get_diff_between_runs(dfs, baseline, intervention, keys, intervention_years, output_df):

        def get_mean_and_confidence_interval(data, confidence=0.95):
            a = 1.0 * np.array(data)
            n = len(a)
            m, se = np.mean(a), scipy.stats.sem(a)
            h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)

            round(m, 3),
            round(h, 2)

            return m, m - h, m + h

        # TODO: replace with below, neater
        # st.t.interval(0.95, len(mean_diff_list) - 1, loc=np.mean(mean_diff_list), scale=st.sem(mean_diff_list))

        for k in keys:
            # Get DF which gives difference between outcomes for each run
            diff = dfs[baseline][k] - dfs[intervention][k]
            int_diff = diff.loc[intervention_years[0]: intervention_years[-1]]

            # Calculate, skew, mean and 95% CI for outcome in intervention
            if 'total' in k:
                operation = 'agg'
            else:
                operation = 'mean'

            mean_outcome_list_int = get_med_or_mean_from_columns(dfs[intervention][k].loc[intervention_years[0]:
                                                                                          intervention_years[-1]],
                                                                 operation)
            skew_mean_outcome_list_int = scipy.stats.skew(mean_outcome_list_int)
            mean_outcome_value_int = get_mean_and_confidence_interval(mean_outcome_list_int)

            # Calculate mean difference between outcome by run for intervention period, check skew and
            # calculate mean/95 % CI
            mean_diff_list = get_med_or_mean_from_columns(int_diff.loc[intervention_years[0]:
                                                                       intervention_years[-1]], operation)

            skew_diff_list = scipy.stats.skew(mean_diff_list)
            mean_outcome_diff = get_mean_and_confidence_interval(mean_diff_list)
            median_outcome_diff = [round(np.median(mean_diff_list), 2),
                                   round(np.quantile(mean_diff_list, 0.025), 2),
                                   round(np.quantile(mean_diff_list, 0.975), 2)]

            res_df = pd.DataFrame([(intervention,
                                    k,
                                    mean_outcome_value_int,
                                    skew_mean_outcome_list_int,
                                    mean_outcome_diff,
                                    skew_diff_list,
                                    median_outcome_diff
                                    )],
                                  columns=['scenario',
                                           'output',
                                           'mean_95%CI_value_for_int_period',
                                           'skew_for_int_data',
                                           'mean_95%CI_diff_outcome_int_period',
                                           'skew_for_diff_data',
                                           'median_diff_outcome_int_period'])

            output_df = output_df.append(res_df)

        return output_df

    def save_outputs(folder, keys, save_name, save_folder):
        dfs = []
        for k in scenario_names:
            scen_df = get_diff_between_runs(folder, scenario_names[0], k, keys, intervention_years, output_df)
            dfs.append(scen_df)

        final_df = pd.concat(dfs)
        final_df.to_csv(f'{save_folder}/{save_name}.csv')

    # Get denominator and complications folders
    births_dict = analysis_utility_functions.return_birth_data_from_multiple_scenarios(
        results_folders,  sim_years, intervention_years)

    preg_dict = analysis_utility_functions.return_pregnancy_data_from_multiple_scenarios(results_folders,
                                                                                         sim_years, intervention_years)

    comps_dfs = {k: analysis_utility_functions.get_modules_maternal_complication_dataframes(results_folders[k])
                 for k in results_folders}

    neo_comps_dfs = {k: analysis_utility_functions.get_modules_neonatal_complication_dataframes(results_folders[k])
                     for k in results_folders}

    comp_pregs_dict = {k: analysis_utility_functions.get_completed_pregnancies_from_multiple_scenarios(
        comps_dfs[k], births_dict[k], results_folders[k], sim_years, intervention_years) for k in results_folders}

    # --------------------------------------- DENOMINATOR OUTCOMES --------------------------------------------------
    analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
        scen_colours, sim_years, preg_dict, 'total_preg',
        'Total Pregnancies',
        'Total Number of Pregnancies Per Year By Scenario',
        primary_oc_path, 'preg')

    analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
        scen_colours, sim_years, births_dict, 'total_births',
        'Total Births',
        'Total Number of Births Per Year By Scenario',
        primary_oc_path, 'births')

    def bar_chart_from_dict(dict, y_title, title,plot_destination_folder, file_name):
        labels = dict.keys()
        mean_vals = list()
        lq_vals = list()
        uq_vals = list()
        for k in dict:
            mean_vals.append(dict[k][file_name][0])
            lq_vals.append(dict[k][file_name][1])
            uq_vals.append(dict[k][file_name][2])

        width = 0.55  # the width of the bars: can also be len(x) sequence
        fig, ax = plt.subplots()

        ci = [(x - y) / 2 for x, y in zip(uq_vals, lq_vals)]
        ax.bar(labels, mean_vals, color=scen_colours, width=width, yerr=ci)
        ax.tick_params(axis='x', which='major', labelsize=8)
        ax.set_ylabel(y_title)
        ax.set_xlabel('Scenario')
        ax.set_title(title)
        plt.savefig(f'{plot_destination_folder}/{file_name}.png')
        plt.show()

    bar_chart_from_dict(preg_dict, 'Pregnancies', 'Total Pregnancies by Scenario', primary_oc_path, 'agg_preg')
    bar_chart_from_dict(births_dict, 'Births', 'Total Births by Scenario', primary_oc_path, 'agg_births')

    # ------------------------------------ PRIMARY OUTCOMES... -------------------------------------------------------
    def extract_death_and_stillbirth_data_frames_and_summ_outcomes(folder, birth_df):
        # MATERNAL
        direct_deaths = extract_results(
            folder,
            module="tlo.methods.demography",
            key="death",
            custom_generate_series=(
                lambda df: df.loc[(df['label'] == 'Maternal Disorders')].assign(
                    year=df['date'].dt.year).groupby(['year'])['year'].count()),
            do_scaling=True)
        direct_deaths_final = direct_deaths.fillna(0)

        indirect_deaths = extract_results(
            folder,
            module="tlo.methods.demography.detail",
            key="properties_of_deceased_persons",
            custom_generate_series=(
                lambda df: df.loc[(df['is_pregnant'] | df['la_is_postpartum']) &
                                  df['cause_of_death'].str.contains(
                                      'AIDS_non_TB|AIDS_TB|TB|Malaria|Suicide|ever_stroke|diabetes|'
                                      'chronic_ischemic_hd|ever_heart_attack|chronic_kidney_disease')].assign(
                    year=df['date'].dt.year).groupby(['year'])['year'].count()),
            do_scaling=True
        )
        indirect_deaths_final = indirect_deaths.fillna(0)
        total_deaths = direct_deaths_final + indirect_deaths_final

        mmr = (total_deaths / birth_df) * 100_000
        total_mmr_by_year = return_95_CI_across_runs(mmr, sim_years)

        # TOTAL AVERAGE MMR DURING INTERVENTION
        mmr_df_int = mmr.loc[intervention_years[0]: intervention_years[-1]]
        mean_mmr_by_year_int = get_mean_from_columns(mmr_df_int, 'avg')
        total_mmr_aggregated = get_mean_95_CI_from_list(mean_mmr_by_year_int)

        # TOTAL MATERNAL DEATHS BY YEAR
        td_int = total_deaths.loc[intervention_years[0]: intervention_years[-1]]
        mean_total_deaths_by_year = return_95_CI_across_runs(total_deaths, sim_years)

        # TOTAL MATERNAL DEATHS DURING INTERVENTION PERIOD
        sum_mat_death_int_by_run = get_mean_from_columns(td_int, 'sum')
        total_deaths_by_scenario = get_mean_95_CI_from_list(sum_mat_death_int_by_run)

        # DIRECT MATERNAL DEATHS PER YEAR
        mean_direct_deaths_by_year = return_95_CI_across_runs(direct_deaths_final, sim_years)
        mean_direct_deaths_by_year_int = return_95_CI_across_runs(direct_deaths_final, intervention_years)

        # TOTAL DIRECT MATERNAL DEATHS DURING INTERVENTION
        dd_int = direct_deaths_final.loc[intervention_years[0]: intervention_years[-1]]
        sum_d_mat_death_int_by_run = get_mean_from_columns(dd_int, 'sum')
        total_direct_deaths_by_scenario = get_mean_95_CI_from_list(sum_d_mat_death_int_by_run)

        # DIRECT MMR BY YEAR
        d_mmr_df = (direct_deaths / birth_df) * 100_000
        total_direct_mmr_by_year = return_95_CI_across_runs(d_mmr_df, sim_years)

        # AVERAGE DIRECT MMR DURING INTERVENTION
        d_mmr_df_int = d_mmr_df.loc[intervention_years[0]: intervention_years[-1]]
        mean_d_mmr_by_year_int = get_mean_from_columns(d_mmr_df_int, 'avg')
        total_direct_mmr_aggregated = get_mean_95_CI_from_list(mean_d_mmr_by_year_int)

        # INDIRECT MATERNAL DEATHS PER YEAR
        mean_indirect_deaths_by_year = return_95_CI_across_runs(indirect_deaths, sim_years)
        mean_indirect_deaths_by_year_int = return_95_CI_across_runs(indirect_deaths, intervention_years)

        # TOTAL INDIRECT MATERNAL DEATHS DURING INTERVENTION
        in_int = indirect_deaths_final.loc[intervention_years[0]: intervention_years[-1]]
        sum_in_mat_death_int_by_run = get_mean_from_columns(in_int, 'sum')
        total_indirect_deaths_by_scenario = get_mean_95_CI_from_list(sum_in_mat_death_int_by_run)

        # INDIRECT MMR BY YEAR
        in_mmr_df = (indirect_deaths / birth_df) * 100_000
        total_indirect_mmr_by_year = return_95_CI_across_runs(in_mmr_df, sim_years)

        # AVERAGE INDIRECT MMR DURING INTERVENTION
        in_mmr_df_int = in_mmr_df.loc[intervention_years[0]: intervention_years[-1]]
        mean_in_mmr_by_year_int = get_mean_from_columns(in_mmr_df_int, 'avg')
        total_indirect_mmr_aggregated = get_mean_95_CI_from_list(mean_in_mmr_by_year_int)

        # NEONATAL
        nd = extract_results(
            folder,
            module="tlo.methods.demography.detail",
            key="properties_of_deceased_persons",
            custom_generate_series=(
                lambda df: df.loc[(df['age_days'] < 29)].assign(
                    year=df['date'].dt.year).groupby(['year'])['year'].count()),
            do_scaling=True)
        neo_deaths = nd.fillna(0)
        neo_deaths_int = neo_deaths.loc[intervention_years[0]: intervention_years[-1]]

        nmr = (neo_deaths / birth_df) * 1000
        total_nmr_by_year = return_95_CI_across_runs(nmr, sim_years)

        # AVERAGE NMR DURING INTERVENTION PERIOD
        nmr_df = nmr.loc[intervention_years[0]: intervention_years[-1]]
        mean_nmr_by_year_int = get_mean_from_columns(nmr_df, 'avg')
        total_nmr_aggregated = get_mean_95_CI_from_list(mean_nmr_by_year_int)

        # NEONATAL DEATHS PER YEAR
        mean_neonatal_deaths_by_year = return_95_CI_across_runs(neo_deaths, sim_years)
        mean_neonatal_deaths_by_year_int = return_95_CI_across_runs(neo_deaths_int, intervention_years)

        # TOTAL NEONATAL DEATHS DURING INTERVENTION PERIOD
        sum_neo_death_int_by_run = get_mean_from_columns(neo_deaths_int, 'sum')
        total_neonatal_deaths_by_scenario = get_mean_95_CI_from_list(sum_neo_death_int_by_run)

        # STILLBIRTH
        an_stillbirth_results = extract_results(
            folder,
            module="tlo.methods.pregnancy_supervisor",
            key="antenatal_stillbirth",
            custom_generate_series=(
                lambda df: df.assign(year=df['date'].dt.year).groupby(['year'])['year'].count()),
            do_scaling=True
        )
        an_stillbirth_results = an_stillbirth_results.fillna(0)

        ip_stillbirth_results = extract_results(
            folder,
            module="tlo.methods.labour",
            key="intrapartum_stillbirth",
            custom_generate_series=(
                lambda df: df.assign(year=df['date'].dt.year).groupby(['year'])['year'].count()),
            do_scaling=True
        )
        ip_stillbirth_results = ip_stillbirth_results.fillna(0)
        all_sb = an_stillbirth_results + ip_stillbirth_results

        an_sbr_df = (an_stillbirth_results / birth_df) * 1000
        ip_sbr_df = (ip_stillbirth_results / birth_df) * 1000
        sbr_df = (all_sb / birth_df) * 1000

        all_sb = an_stillbirth_results + ip_stillbirth_results
        all_sb_int = all_sb.loc[intervention_years[0]: intervention_years[-1]]
        # Store mean number of stillbirths, LQ, UQ
        crude_sb = return_95_CI_across_runs(all_sb, sim_years)

        def get_sbr(df):
            sbr_df = (df / birth_df) * 1000
            sbr_mean_quants = return_95_CI_across_runs(sbr_df, sim_years)
            return sbr_mean_quants

        an_sbr = get_sbr(an_stillbirth_results)
        ip_sbr = get_sbr(ip_stillbirth_results)
        total_sbr = get_sbr(all_sb)

        avg_sbr_df = (all_sb_int / birth_df.loc[intervention_years[0]:intervention_years[-1]]) * 1000
        avg_sbr_means = get_mean_from_columns(avg_sbr_df, 'avg')

        avg_isbr_df = (ip_stillbirth_results.loc[intervention_years[0]:intervention_years[-1]] /
                       birth_df.loc[intervention_years[0]:intervention_years[-1]]) * 1000
        ip_sbr_means = get_mean_from_columns(avg_isbr_df, 'avg')

        avg_asbr_df = (an_stillbirth_results.loc[intervention_years[0]:intervention_years[-1]] /
                       birth_df.loc[intervention_years[0]:intervention_years[-1]]) * 1000
        ap_sbr_means = get_mean_from_columns(avg_asbr_df, 'avg')

        avg_sbr = get_mean_95_CI_from_list(avg_sbr_means)
        avg_i_sbr = get_mean_95_CI_from_list(ip_sbr_means)
        avg_a_sbr = get_mean_95_CI_from_list(ap_sbr_means)

        return {'mmr_df': mmr,
                'mat_deaths_total_df': total_deaths,
                'nmr_df': nmr,
                'neo_deaths_total_df': neo_deaths,
                'sbr_df': sbr_df,
                'ip_sbr_df': ip_sbr_df,
                'an_sbr_df': an_sbr_df,
                'stillbirths_total_df': all_sb,

                'crude_t_deaths': mean_total_deaths_by_year,
                'agg_total': total_deaths_by_scenario,
                'total_mmr': total_mmr_by_year,
                'agg_total_mr': total_mmr_aggregated,

                'crude_dir_m_deaths': mean_direct_deaths_by_year,
                'agg_dir_m_deaths': total_direct_deaths_by_scenario,
                'direct_mmr': total_direct_mmr_by_year,
                'agg_dir_mr': total_direct_mmr_aggregated,

                'crude_ind_m_deaths': mean_indirect_deaths_by_year,
                'agg_ind_m_deaths': total_indirect_deaths_by_scenario,
                'indirect_mmr': total_indirect_mmr_by_year,
                'agg_ind_mr': total_indirect_mmr_aggregated,

                'crude_n_deaths': mean_neonatal_deaths_by_year,
                'agg_n_deaths': total_neonatal_deaths_by_scenario,
                'nmr': total_nmr_by_year,
                'agg_nmr': total_nmr_aggregated,

                'an_sbr': an_sbr,
                'ip_sbr': ip_sbr,
                'sbr': total_sbr,
                'crude_sb': crude_sb,
                'avg_sbr': avg_sbr,
                'avg_i_sbr': avg_i_sbr,
                'avg_a_sbr': avg_a_sbr}

    # Extract data from scenarios
    death_data = {k: extract_death_and_stillbirth_data_frames_and_summ_outcomes(results_folders[k],
                                                                                births_dict[k]['births_data_frame'])
                     for k in results_folders}

    #  ---------------- MMR/NMR GRAPHS ---------------
    for data, title, y_lable in \
        zip(['agg_dir_m_deaths',
             'agg_dir_mr',
             'agg_ind_m_deaths',
             'agg_ind_mr',
             'agg_total',
             'agg_total_mr',
             'agg_n_deaths',
             'agg_nmr'],
            ['Total Direct Maternal Deaths By Scenario',
             'Average Direct MMR by Scenario',
             'Total Indirect Maternal Deaths By Scenario',
             'Average Indirect MMR by Scenario',
             'Total Maternal Deaths By Scenario',
             'Average MMR by Scenario',
             'Total Neonatal Deaths By Scenario',
             'Average NMR by Scenario'],
            ['Total Direct Maternal Deaths',
             'Average MMR',
             'Total Indirect Maternal Deaths',
             'Average MMR',
             'Total Maternal Deaths',
             'Average MMR',
             'Total Neonatal Deaths',
             'Average NMR']):
        plot_agg_graph(death_data, data, y_lable, title, data, primary_oc_path)

    # 2.) TRENDS IN DEATHS
    # Output and save the relevant graphs
    analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
        scen_colours, sim_years, death_data, 'direct_mmr',
        'Deaths per 100,000 live births',
        'MMR per Year at Baseline and Under Intervention (Direct only)', primary_oc_path,
        'maternal_mr_direct')

    analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
        scen_colours,sim_years, death_data, 'total_mmr',
        'Deaths per 100,000 live births',
        'MMR per Year at Baseline and Under Intervention (Total)',
        primary_oc_path, 'maternal_mr_total')

    analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
        scen_colours, sim_years, death_data, 'nmr',
        'Total Deaths per 1000 live births',
        'Neonatal Mortality Ratio per Year at Baseline and Under Intervention',
        primary_oc_path, 'neonatal_mr_int')

    for group, l in zip(['Maternal', 'Neonatal'], ['dir_m', 'n']):
        analysis_utility_functions.comparison_bar_chart_multiple_bars(
            death_data, f'crude_{l}_deaths', sim_years, scen_colours,
            f'Total {group} Deaths (scaled)', f'Yearly Baseline {group} Deaths Compared to Intervention',
            primary_oc_path, f'{group}_crude_deaths_comparison.png')

    def extract_deaths_by_cause(results_folder, births_df, intervention_years):

        d_causes = ['ectopic_pregnancy', 'spontaneous_abortion', 'induced_abortion', 'severe_gestational_hypertension',
                    'severe_pre_eclampsia', 'eclampsia', 'antenatal_sepsis', 'uterine_rupture',
                    'intrapartum_sepsis', 'postpartum_sepsis','postpartum_haemorrhage',
                    'secondary_postpartum_haemorrhage','antepartum_haemorrhage']

        ind_causes = ['AIDS_non_TB', 'AIDS_TB', 'TB', 'Malaria', 'Suicide', 'ever_stroke', 'diabetes',
                      'chronic_ischemic_hd', 'ever_heart_attack',
                      'chronic_kidney_disease']

        def update_dfs_to_replace_missing_causes(df, causes):
            t = []
            for year in sim_years:
                for cause in causes:
                    if cause not in df.loc[year].index:
                        index = pd.MultiIndex.from_tuples([(year, cause)], names=["year", "cause_of_death"])
                        new_row = pd.DataFrame(columns=df.columns, index=index)
                        f_df = new_row.fillna(0.0)
                        t.append(f_df)

            causes_df = pd.concat(t)
            updated_df = df.append(causes_df)

            return updated_df

        dd = extract_results(
            results_folder,
            module="tlo.methods.demography.detail",
            key="properties_of_deceased_persons",
            custom_generate_series=(
                lambda df: df.assign(
                    year=df['date'].dt.year).groupby(['year', 'cause_of_death'])['year'].count()),
            do_scaling=True)
        direct_deaths = dd.fillna(0)

        id = extract_results(
            results_folder,
            module="tlo.methods.demography.detail",
            key="properties_of_deceased_persons",
            custom_generate_series=(
                lambda df: df.loc[(df['is_pregnant'] | df['la_is_postpartum'])].assign(
                    year=df['date'].dt.year).groupby(['year', 'cause_of_death'])['year'].count()),
            do_scaling=True)
        indirect_deaths = id.fillna(0)

        updated_dd = update_dfs_to_replace_missing_causes(direct_deaths, d_causes)
        updated_ind = update_dfs_to_replace_missing_causes(indirect_deaths, ind_causes)

        results = dict()

        def extract_mmr_data(cause, df):
            death_df = df.loc[(slice(None), cause), slice(None)].droplevel(1)
            mmr_df = (death_df / births_df) * 100_000
            results.update({f'{cause}_mmr_df': mmr_df})
            mmr_df_int = mmr_df.loc[intervention_years[0]:intervention_years[-1]]
            list_mmr = get_mean_from_columns(mmr_df_int, 'avg')
            results.update({f'{cause}_mmr_avg': get_mean_95_CI_from_list(list_mmr)})

        for cause in d_causes:
            extract_mmr_data(cause, updated_dd)

        for cause in ind_causes:
            extract_mmr_data(cause, updated_ind)

        return results

    def extract_neonatal_deaths_by_cause(results_folder, births_df, intervention_years):

        nd = extract_results(
            results_folder,
            module="tlo.methods.demography.detail",
            key="properties_of_deceased_persons",
            custom_generate_series=(
                lambda df: df.loc[(df['age_days'] < 29)].assign(
                    year=df['date'].dt.year).groupby(['year', 'cause_of_death'])['year'].count()),
            do_scaling=True)
        neo_deaths = nd.fillna(0)

        results = dict()
        n_causes = list(neo_deaths.loc[2010].index)

        for cause in n_causes:
            death_df = neo_deaths.loc[(slice(None), cause), slice(None)].droplevel(1)
            nmr_df = (death_df / births_df) * 1000
            nmr_df_final = nmr_df.fillna(0)
            results.update({f'{cause}_nmr_df': nmr_df_final})
            nmr_df_int = nmr_df_final.loc[intervention_years[0]:intervention_years[-1]]
            list_nmr = get_mean_from_columns(nmr_df_int, 'avg')
            results.update({f'{cause}_nmr_avg': get_mean_95_CI_from_list(list_nmr)})

        return results

    cod_data = {k: extract_deaths_by_cause(results_folders[k], births_dict[k]['births_data_frame'],
                                                intervention_years) for k in results_folders}

    cod_neo_data = {k: extract_neonatal_deaths_by_cause(results_folders[k], births_dict[k]['births_data_frame'],
                                                        intervention_years) for k in results_folders}
    def save_mr_by_cause_data_and_output_graphs(group, cause_d):
        if group == 'mat':
            d = ['m', 'MMR']
        else:
            d = ['n', 'NMR']

        cod_keys = list()
        for k in cause_d[scenario_titles[0]].keys():
            if 'df' in k:
                cod_keys.append(k)

        save_outputs(cause_d, cod_keys, f'diff_in_cause_specific_{d[0]}mr', primary_oc_path)

        labels = [l.replace(f'_{d[0]}mr_df', '') for l in cod_keys]

        for k, colour in zip(cause_d, scen_colours):
            mean_vals = list()
            lq_vals = list()
            uq_vals = list()
            for key in cause_d[k]:
                if 'avg' in key:
                    mean_vals.append(cause_d[k][key][0])
                    lq_vals.append(cause_d[k][key][1])
                    uq_vals.append(cause_d[k][key][2])

            width = 0.55  # the width of the bars: can also be len(x) sequence
            fig, ax = plt.subplots()

            ci = [(x - y) / 2 for x, y in zip(uq_vals, lq_vals)]
            ax.bar(labels, mean_vals, color=colour, width=width, yerr=ci)
            ax.tick_params(axis='x', which='major', labelsize=8, labelrotation=90)
            if group == 'mat':
                plt.gca().set_ylim(bottom=0, top=70)
            else:
                plt.gca().set_ylim(bottom=0, top=7)

            ax.set_ylabel(d[1])
            ax.set_xlabel('Complication')
            ax.set_title(f'Cause specific {d[1]} for {k} scenario')
            plt.savefig(f'{primary_oc_path}/{k}_{d[0]}mr_by_cause.png', bbox_inches='tight')
            plt.show()
    save_mr_by_cause_data_and_output_graphs('mat', cod_data)
    save_mr_by_cause_data_and_output_graphs('neo', cod_neo_data)

    def get_cause_spef_mmrs_on_same_graph(group, cod_data):
        csmmr_dict = dict()
        for k in cod_data:
            csmmr_dict.update({k: []})
            mean_vals = list()
            lq_vals = list()
            uq_vals = list()
            for key in cod_data[k]:
                if 'avg' in key:
                    mean_vals.append(cod_data[k][key][0])
                    lq_vals.append(cod_data[k][key][1])
                    uq_vals.append(cod_data[k][key][2])
            csmmr_dict[k] = [mean_vals, lq_vals, uq_vals]
        N = len(csmmr_dict['Status Quo'][0])
        ind = np.arange(N)
        if len(csmmr_dict.keys()) > 3:
            width = 0.15
        else:
            width = 0.35
        x_ticks = list()
        for x in range(len(csmmr_dict['Status Quo'][0])):
            x_ticks.append(x)
        for k, position, colour in zip(csmmr_dict, [ind - width, ind, ind + width, ind + width * 2, ind + width * 3],
                                       scen_colours):
            ci = [(x - y) / 2 for x, y in zip(csmmr_dict[k][2], csmmr_dict[k][1])]
            plt.bar(position, csmmr_dict[k][0], width, label=k, yerr=ci, color=colour)
        cod_keys = list()
        for k in cod_data[scenario_titles[0]].keys():
            if 'df' in k:
                cod_keys.append(k)
        labels = [l.replace('_mmr_df', '') for l in cod_keys]
        plt.gca().set_ylim(bottom=0, top=45)
        plt.ylabel('Average deaths per 100,000 live births')
        plt.xlabel('Cause of death')
        plt.title('Cause specific MMR by scenario')
        plt.legend(loc='best')
        plt.xticks(x_ticks, labels=labels, rotation=90, size=7)
        plt.savefig(f'{primary_oc_path}/cs_mmr_one_graph.png', bbox_inches='tight')
        plt.show()
        pass

    #  ---------------- STILLBIRTH GRAPHS ---------------
    analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
        scen_colours, sim_years, death_data, 'an_sbr',
        'Antenatal stillbirths per 1000 births',
        'Antenatal stillbirth Rate per Year at Baseline and Under Intervention',
        primary_oc_path, 'an_sbr_int')

    analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
        scen_colours, sim_years, death_data, 'ip_sbr',
        'Intrapartum stillbirths per 1000 births',
        'Intrapartum stillbirth Rate per Year at Baseline and Under Intervention',
        primary_oc_path, 'ip_sbr_int')

    # Output SBR per year for scenario vs intervention
    analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
        scen_colours, sim_years, death_data, 'sbr',
        'Stillbirths per 1000 births',
        'Stillbirth Rate per Year at Baseline and Under Intervention',
        primary_oc_path, 'sbr_int')

    analysis_utility_functions.comparison_bar_chart_multiple_bars(
        death_data, 'crude_sb', sim_years, scen_colours,
        'Total Stillbirths (scaled)', 'Yearly Baseline Stillbirths Compared to Intervention',
        primary_oc_path, 'crude_stillbirths_comparison.png')

    for data, title, y_lable in \
        zip(['avg_sbr',
             'avg_i_sbr',
             'avg_a_sbr'],
            ['Average Total Stillbirth Rate during the Intervention Period',
             'Average Intrapartum Stillbirth Rate during the Intervention Period',
             'Average Antenatal Stillbirth Rate during the Intervention Period'],
            ['Stillbirths per 1000 births',
             'Stillbirths per 1000 births',
             'Stillbirths per 1000 births']):

        labels = results_folders.keys()

        mean_vals = list()
        lq_vals = list()
        uq_vals = list()
        for k in death_data:
            mean_vals.append(death_data[k][data][0])
            lq_vals.append(death_data[k][data][1])
            uq_vals.append(death_data[k][data][2])

        width = 0.55  # the width of the bars: can also be len(x) sequence
        fig, ax = plt.subplots()

        ci = [(x - y) / 2 for x, y in zip(uq_vals, lq_vals)]
        ax.bar(labels, mean_vals, color=scen_colours, width=width, yerr=ci)
        ax.tick_params(axis='x', which='major', labelsize=8)

        ax.set_ylabel(y_lable)
        ax.set_xlabel('Scenario')
        ax.set_title(title)
        plt.savefig(f'{primary_oc_path}/{data}.png')
        plt.show()

    keys = ['mmr_df', 'nmr_df', 'sbr_df', 'an_sbr_df', 'ip_sbr_df', 'mat_deaths_total_df', 'neo_deaths_total_df',
            'stillbirths_total_df']
    save_outputs(death_data, keys, 'diff_in_mortality_outcomes', primary_oc_path)

    def extract_dalys(folder):
        results_dict = dict()

        # Get DALY df
        dalys = extract_results(
            folder,
            module="tlo.methods.healthburden",
            key="dalys",
            custom_generate_series=(
                lambda df: df.drop(
                    columns='date').groupby(['year']).sum().stack()),
            do_scaling=True)

        dalys_stacked = extract_results(
            folder,
            module="tlo.methods.healthburden",
            key="dalys_stacked",
            custom_generate_series=(
                lambda df: df.drop(
                    columns='date').groupby(['year']).sum().stack()),
            do_scaling=True)

        # todo - should this just be at risk or total (gbd suggests total, which this calibrates well with)
        person_years_total = extract_results(
            folder,
            module="tlo.methods.demography",
            key="person_years",
            custom_generate_series=(
                lambda df: df.assign(total=(df['M'].apply(lambda x: sum(x.values()))) + df['F'].apply(
                    lambda x: sum(x.values()))).assign(
                    year=df['date'].dt.year).groupby(['year'])['total'].sum()),
            do_scaling=True)

        for type, d in zip(['stacked', 'unstacked'], [dalys_stacked, dalys]):
            md = d.loc[(slice(None), 'Maternal Disorders'), slice(None)].droplevel(1)
            nd = d.loc[(slice(None), 'Neonatal Disorders'), slice(None)].droplevel(1)

            results_dict.update({f'maternal_total_dalys_df_{type}': md})
            results_dict.update({f'neonatal_total_dalys_df_{type}': nd})

            m_d_rate_df = (md / person_years_total) * 100_000
            n_d_rate_df = (nd / person_years_total) * 100_000

            results_dict.update({f'maternal_dalys_rate_df_{type}': m_d_rate_df})
            results_dict.update({f'neonatal_dalys_rate_df_{type}': n_d_rate_df})

            results_dict.update({f'maternal_dalys_rate_{type}': return_95_CI_across_runs(m_d_rate_df, sim_years)})
            results_dict.update({f'neonatal_dalys_rate_{type}': return_95_CI_across_runs(n_d_rate_df, sim_years)})

            m_int_period = m_d_rate_df.loc[intervention_years[0]: intervention_years[-1]]
            n_int_period = n_d_rate_df.loc[intervention_years[0]: intervention_years[-1]]

            m_int_means = get_mean_from_columns(m_int_period, 'avg')
            n_int_means = get_mean_from_columns(n_int_period, 'avg')

            results_dict.update({f'avg_mat_dalys_rate_{type}':  get_mean_95_CI_from_list(m_int_means)})
            results_dict.update({f'avg_neo_dalys_rate_{type}':  get_mean_95_CI_from_list(n_int_means)})

            # Get averages/sums
            results_dict.update({f'maternal_dalys_crude_{type}': return_95_CI_across_runs(md, sim_years)})
            results_dict.update({f'neonatal_dalys_crude_{type}': return_95_CI_across_runs(nd, sim_years)})

            m_int_agg = get_mean_from_columns(m_int_period, 'sum')
            n_int_agg = get_mean_from_columns(n_int_period, 'sum')

            results_dict.update({f'agg_mat_dalys_{type}': get_mean_95_CI_from_list(m_int_agg)})
            results_dict.update({f'agg_neo_dalys_{type}': get_mean_95_CI_from_list(n_int_agg)})

        mat_causes_death = ['ectopic_pregnancy',
                            'spontaneous_abortion',
                            'induced_abortion',
                            'severe_gestational_hypertension',
                            'severe_pre_eclampsia',
                            'eclampsia',
                            'antenatal_sepsis',
                            'uterine_rupture',
                            'intrapartum_sepsis',
                            'postpartum_sepsis',
                            'postpartum_haemorrhage',
                            'secondary_postpartum_haemorrhage',
                            'antepartum_haemorrhage']

        mat_causes_disab = ['maternal']

        neo_causes_death = ['early_onset_sepsis', 'late_onset_sepsis', 'encephalopathy', 'preterm_other',
                            'respiratory_distress_syndrome', 'neonatal_respiratory_depression']

        neo_causes_disab = ['Retinopathy of Prematurity', 'Neonatal Encephalopathy',
                            'Neonatal Sepsis Long term Disability', 'Preterm Birth Disability']

        yll = extract_results(
            folder,
            module="tlo.methods.healthburden",
            key="yll_by_causes_of_death",
            custom_generate_series=(
                lambda df: df.drop(
                    columns='date').groupby(['year']).sum().stack()),
            do_scaling=True)
        yll_final = yll.fillna(0)

        yll_stacked = extract_results(
            folder,
            module="tlo.methods.healthburden",
            key="yll_by_causes_of_death_stacked",
            custom_generate_series=(
                lambda df: df.drop(
                    columns='date').groupby(['year']).sum().stack()),
            do_scaling=True)
        yll_stacked_final = yll_stacked.fillna(0)

        yld = extract_results(
            folder,
            module="tlo.methods.healthburden",
            key="yld_by_causes_of_disability",
            custom_generate_series=(
                lambda df: df.drop(
                    columns='date').groupby(['year']).sum().stack()),
            do_scaling=True)
        yld_final = yld.fillna(0)

        def get_total_dfs(df, causes):
            dfs = []
            for k in causes:
                if k not in df.loc[(slice(None))]:
                    new_row = pd.DataFrame(columns=df.columns, index=sim_years)
                    scen_df = new_row.fillna(0.0)
                else:
                    scen_df = df.loc[(slice(None), k), slice(None)].droplevel(1)

                dfs.append(scen_df)

            final_df = sum(dfs)
            return final_df

        neo_yll_df = get_total_dfs(yll_final, neo_causes_death)
        mat_yll_df = get_total_dfs(yll_final, mat_causes_death)
        neo_yll_s_df = get_total_dfs(yll_stacked_final, neo_causes_death)
        mat_yll_s_df = get_total_dfs(yll_stacked_final, mat_causes_death)
        neo_yld_df = get_total_dfs(yld_final, neo_causes_disab)
        mat_yld_df = get_total_dfs(yld_final, mat_causes_disab)

        results_dict.update({'maternal_yll_crude_unstacked': return_95_CI_across_runs(mat_yll_df, sim_years)})
        results_dict.update({'maternal_yll_crude_stacked': return_95_CI_across_runs(mat_yll_s_df, sim_years)})

        mat_yll_df_rate = (mat_yll_df / person_years_total) * 100_000
        mat_yll_s_df_rate = (mat_yll_s_df / person_years_total) * 100_000

        results_dict.update({'maternal_yll_rate_unstacked': return_95_CI_across_runs(mat_yll_df_rate, sim_years)})
        results_dict.update({'maternal_yll_rate_stacked': return_95_CI_across_runs(mat_yll_s_df_rate, sim_years)})

        results_dict.update({'maternal_yld_crude_unstacked': return_95_CI_across_runs(mat_yld_df, sim_years)})

        mat_yld_df_rate = (mat_yld_df / person_years_total) * 100_000
        results_dict.update({'maternal_yld_rate_unstacked': return_95_CI_across_runs(mat_yld_df_rate, sim_years)})

        results_dict.update({'neonatal_yll_crude_unstacked': return_95_CI_across_runs(neo_yll_df, sim_years)})
        results_dict.update({'neonatal_yll_crude_stacked': return_95_CI_across_runs(neo_yll_s_df, sim_years)})

        neo_yll_df_rate = (neo_yll_df / person_years_total) * 100_000
        neo_yll_s_df_rate = (neo_yll_s_df / person_years_total) * 100_000

        results_dict.update({'neonatal_yll_rate_unstacked': return_95_CI_across_runs(neo_yll_df_rate, sim_years)})
        results_dict.update({'neonatal_yll_rate_stacked': return_95_CI_across_runs(neo_yll_s_df_rate, sim_years)})

        results_dict.update({'neonatal_yld_crude_unstacked': return_95_CI_across_runs(neo_yld_df, sim_years)})

        neo_yld_df_rate = (neo_yld_df / person_years_total) * 100_000
        results_dict.update({'neonatal_yld_rate_unstacked': return_95_CI_across_runs(neo_yld_df_rate, sim_years)})

        return results_dict

    dalys_folders = {k: extract_dalys(results_folders[k]) for k in results_folders}

    for data, title, y_lable in \
        zip(['agg_mat_dalys_stacked',
             'agg_neo_dalys_stacked',
             'avg_mat_dalys_rate_stacked',
             'avg_neo_dalys_rate_stacked'],
            ['Average Total Maternal DALYs (stacked) by Scenario',
             'Average Total Neonatal DALYs (stacked) by Scenario',
             'Average Total Maternal DALYs per 100k PY by Scenario',
             'Average Total Neonatal DALYs per 100k PY by Scenario'],
            ['DALYs',
             'DALYs',
             'DALYs per 100k PY',
             'DALYs per 100k PY']):
        labels = results_folders.keys()

        mean_vals = list()
        lq_vals = list()
        uq_vals = list()
        for k in dalys_folders:
            mean_vals.append(dalys_folders[k][data][0])
            lq_vals.append(dalys_folders[k][data][1])
            uq_vals.append(dalys_folders[k][data][2])

        width = 0.55  # the width of the bars: can also be len(x) sequence
        fig, ax = plt.subplots()

        ci = [(x - y) / 2 for x, y in zip(uq_vals, lq_vals)]
        ax.bar(labels, mean_vals, color=scen_colours, width=width, yerr=ci)
        ax.tick_params(axis='x', which='major', labelsize=8)

        ax.set_ylabel(y_lable)
        ax.set_xlabel('Scenario')
        ax.set_title(title)
        plt.savefig(f'{primary_oc_path}/{data}.png')
        plt.show()

    for dict_key, axis, title, save_name in zip(['maternal_dalys_crude_stacked', 'maternal_dalys_rate_stacked',
                                                 'maternal_yll_crude_stacked', 'maternal_yll_rate_stacked',
                                                 'maternal_yld_crude_unstacked', 'maternal_yld_rate_unstacked',

                                                 'neonatal_dalys_crude_stacked', 'neonatal_dalys_rate_stacked',
                                                 'neonatal_yll_crude_stacked', 'neonatal_yll_rate_stacked',
                                                 'neonatal_yld_crude_unstacked', 'neonatal_yld_rate_unstacked'],

                                                ['DALYs', 'DALYs per 100k Person-Years', 'YLL',
                                                 'YLL per 100k Person-Years', 'YLD', 'YLD per 100k Person-Years',

                                                 'DALYs', 'DALYs per 100k Person-Years', 'YLL',
                                                 'YLL per 100k Person-Years', 'YLD', 'YLD per 100k Person-Years'],

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
            scen_colours, sim_years, dalys_folders, dict_key, axis, title, primary_oc_path, save_name)

    keys = ['maternal_total_dalys_df_stacked', 'neonatal_total_dalys_df_stacked',
            'maternal_dalys_rate_df_stacked', 'neonatal_dalys_rate_df_stacked']
    save_outputs(dalys_folders, keys, 'diff_in_daly_outcomes', primary_oc_path)

    # ----------------------------------------- SECONDARY OUTCOMES --------------------------------------------------
    def get_coverage_of_key_maternity_services(folder, births_df):

        results = dict()

        # --- ANC ---
        anc_coverage = extract_results(
            folder,
            module="tlo.methods.care_of_women_during_pregnancy",
            key="anc_count_on_birth",
            custom_generate_series=(
                lambda df: df.assign(
                    year=df['date'].dt.year).groupby(['year'])['person_id'].count()),
            do_scaling=True
        )

        # Next get a version of that DF with women who attended >= 4/8 visits by birth
        an = extract_results(
            folder,
            module="tlo.methods.care_of_women_during_pregnancy",
            key="anc_count_on_birth",
            custom_generate_series=(
                lambda df: df.loc[df['total_anc'] >= 4].assign(
                    year=df['date'].dt.year).groupby(['year'])['person_id'].count()),
            do_scaling=True
        )
        anc_cov_of_interest = an.fillna(0)

        cd = (anc_cov_of_interest / anc_coverage) * 100
        coverage_df = cd.fillna(0)

        results.update({'anc_cov_df': coverage_df})
        results.update({'anc_cov_rate': return_95_CI_across_runs(coverage_df, sim_years)})

        # ---SBA--
        all_deliveries = extract_results(
            folder,
            module="tlo.methods.labour",
            key="delivery_setting_and_mode",
            custom_generate_series=(
                lambda df_: df_.assign(year=df_['date'].dt.year).groupby(['year'])[
                    'mother'].count()),
            do_scaling=True
        )

        for facility_type in ['home_birth', 'hospital', 'health_centre', 'facility']:
            if facility_type == 'facility':
                deliver_setting_results = extract_results(
                    folder,
                    module="tlo.methods.labour",
                    key="delivery_setting_and_mode",
                    custom_generate_series=(
                        lambda df: df.loc[df['facility_type'] != 'home_birth'].assign(
                            year=df['date'].dt.year).groupby(['year'])[
                            'mother'].count()),
                    do_scaling=True
                )

            else:
                deliver_setting_results = extract_results(
                    folder,
                    module="tlo.methods.labour",
                    key="delivery_setting_and_mode",
                    custom_generate_series=(
                        lambda df: df.loc[df['facility_type'] == facility_type].assign(
                            year=df['date'].dt.year).groupby(['year'])[
                            'mother'].count()),
                    do_scaling=True
                )
            rate_df = (deliver_setting_results / all_deliveries) * 100
            results.update({f'{facility_type}_df': rate_df})
            results.update({f'{facility_type}_rate': return_95_CI_across_runs(rate_df, sim_years)})

        # --- PNC ---
        all_surviving_mothers = extract_results(
            folder,
            module="tlo.methods.postnatal_supervisor",
            key="total_mat_pnc_visits",
            custom_generate_series=(
                lambda df: df.assign(year=df['date'].dt.year).groupby(['year'])['mother'].count()),
            do_scaling=True)

        # Extract data on all women with 1+ PNC visits
        pnc_results_maternal = extract_results(
            folder,
            module="tlo.methods.postnatal_supervisor",
            key="total_mat_pnc_visits",
            custom_generate_series=(
                lambda df: df.loc[df['visits'] > 0].assign(year=df['date'].dt.year).groupby(['year'])[
                    'mother'].count()),
            do_scaling=True
        )

        # Followed by newborns...
        all_surviving_newborns = extract_results(
            folder,
            module="tlo.methods.postnatal_supervisor",
            key="total_neo_pnc_visits",
            custom_generate_series=(
                lambda df: df.assign(year=df['date'].dt.year).groupby(['year'])[
                    'child'].count()),
            do_scaling=True
        )

        pnc_results_newborn = extract_results(
            folder,
            module="tlo.methods.postnatal_supervisor",
            key="total_neo_pnc_visits",
            custom_generate_series=(
                lambda df: df.loc[df['visits'] > 0].assign(year=df['date'].dt.year).groupby(['year'])[
                    'child'].count()),
            do_scaling=True
        )
        cov_mat_birth_df = (pnc_results_maternal / births_df) * 100
        cov_mat_surv_df = (pnc_results_maternal / all_surviving_mothers) * 100
        cov_neo_birth_df = (pnc_results_newborn / births_df) * 100
        cov_neo_surv_df = (pnc_results_newborn / all_surviving_newborns) * 100

        results.update({'pnc_mat_cov_birth_df': cov_mat_birth_df})
        results.update({'pnc_mat_cov_birth_rate': return_95_CI_across_runs(cov_mat_birth_df, sim_years)})

        results.update({'pnc_mat_cov_surv_df': cov_mat_surv_df})
        results.update({'pnc_mat_cov_surv_rate': return_95_CI_across_runs(cov_mat_surv_df, sim_years)})

        results.update({'pnc_neo_cov_birth_df': cov_neo_birth_df})
        results.update({'pnc_neo_cov_birth_rate': return_95_CI_across_runs(cov_neo_birth_df, sim_years)})

        results.update({'pnc_neo_cov_surv_df': cov_neo_surv_df})
        results.update({'pnc_neo_cov_surv_rate': return_95_CI_across_runs(cov_mat_surv_df, sim_years)})

        return results

    cov_data = {k: get_coverage_of_key_maternity_services(results_folders[k], births_dict[k]['births_data_frame'])
                for k in results_folders}

    cov_keys = list()
    for k in cov_data[scenario_titles[0]].keys():
        if 'df' in k:
            cov_keys.append(k)

    save_outputs(cov_data, cov_keys, 'diff_in_mat_service_coverage', secondary_oc_path)

    analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
        scen_colours, sim_years, cov_data, 'anc_cov_rate',
        '% Total Births',
        'Proportion of women receiving four (or more) ANC visits at birth',
        secondary_oc_path, 'anc4_cov')

    analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
        scen_colours, sim_years, cov_data, 'facility_rate',
        '% Total Births',
        'Facility Delivery Rate per Year Per Scenario',
        secondary_oc_path, 'fd_rate')

    analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
        scen_colours, sim_years, cov_data, 'home_birth_rate',
        '% Total Births',
        'Home birth Rate per Year Per Scenario',
        secondary_oc_path, 'hb_rate')

    analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
        scen_colours, sim_years, cov_data, 'hospital_rate',
        '% Total Births',
        'Hospital birth Rate per Year Per Scenario',
        secondary_oc_path, 'hp_rate')

    analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
        scen_colours, sim_years, cov_data, 'health_centre_rate',
        '% Total Births',
        'Health Centre Birth Rate per Year Per Scenario',
        secondary_oc_path, 'hc_rate')

    analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
        scen_colours, sim_years, cov_data, 'pnc_mat_cov_birth_rate',
        '% Total Births',
        'Maternal PNC Coverage as Proportion of Total Births',
        secondary_oc_path, 'mat_pnc_coverage_births')

    analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
        scen_colours, sim_years, cov_data, 'pnc_mat_cov_surv_rate',
        '% Total Survivors at Day 42',
        'Maternal PNC Coverage as Proportion of Postnatal Survivors',
        secondary_oc_path, 'mat_pnc_coverage_survivors')

    analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
        scen_colours, sim_years, cov_data, 'pnc_neo_cov_birth_rate',
        '% Total Births',
        'Neonatal PNC Coverage as Proportion of Total Births',
        secondary_oc_path, 'neo_pnc_coverage_births')

    analysis_utility_functions.comparison_graph_multiple_scenarios_multi_level_dict(
        scen_colours, sim_years, cov_data, 'pnc_neo_cov_surv_rate',
        '% Total Survivors at Day 28',
        'Neonatal PNC Coverage as Proportion of Neonatal Survivors',
        secondary_oc_path, 'neo_pnc_coverage_survivors')

