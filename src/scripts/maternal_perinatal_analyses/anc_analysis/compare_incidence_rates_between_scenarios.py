import pandas as pd
import os
from matplotlib import pyplot as plt

from tlo.analysis.utils import (
    extract_results,
    get_scenario_outputs,
)

from scripts.maternal_perinatal_analyses import analysis_utility_functions


# HELPER FUNCTION
def get_modules_maternal_complication_dataframes(results_folder):
    comp_dfs = dict()

    for module in ['pregnancy_supervisor', 'labour', 'postnatal_supervisor']:
        complications_df = extract_results(
            results_folder,
            module=f"tlo.methods.{module}",
            key="maternal_complication",
            custom_generate_series=(
                lambda df_: df_.assign(year=df_['date'].dt.year).groupby(['year', 'type'])['person'].count()),
            do_scaling=False
        )

        comp_dfs[module] = complications_df

    return comp_dfs


def compare_key_rates_between_two_scenarios(baseline_scenario_filename, intervention_scenario_filename, outputspath,
                                            show_and_store_graphs, sim_years):
    """
    This function outputs plots of incidence rates and complication level NMR/MMRs for a number of key complications
    within the model
    :param baseline_scenario_filename:
    :param intervention_scenario_filename:
    :param outputspath:
    :param show_and_store_graphs:
    :param sim_years:
    :return:
    """

    # Find results folder (most recent run generated using that scenario_filename)
    baseline_results_folder = get_scenario_outputs(baseline_scenario_filename, outputspath)[-1]
    intervention_results_folder = get_scenario_outputs(intervention_scenario_filename, outputspath)[-1]

    # Create folder to store graphs (if it hasnt already been created when ran previously)
    main_folder = f'{outputspath}/analysis_comparison_graphs_{baseline_results_folder.name}'
    path_prim = f'{main_folder}/comparison_with_{intervention_results_folder.name}'
    path_mmr = f'{path_prim}/mmr'
    path_nmr = f'{path_prim}/nmr'

    for path in [main_folder, path_prim, path_mmr, path_nmr]:
        if not os.path.isdir(path):
            os.makedirs(path)

    plot_destination_folder = path_prim

    # access complication dataframes
    b_comps_dfs = get_modules_maternal_complication_dataframes(baseline_results_folder)
    i_comps_dfs = get_modules_maternal_complication_dataframes(intervention_results_folder)

    # ============================================  DENOMINATORS... ==================================================
    # ---------------------------------------------Total_pregnancies...------------------------------------------------

    def get_pregnancies(results_folder):
        pregnancy_poll_results = extract_results(
            results_folder,
            module="tlo.methods.contraception",
            key="pregnancy",
            custom_generate_series=(
                lambda df: df.assign(year=pd.to_datetime(df['date']).dt.year).groupby(['year'])['year'].count()
            ))

        preg_data = analysis_utility_functions.get_mean_and_quants(pregnancy_poll_results, sim_years)

        return preg_data

    b_preg = get_pregnancies(baseline_results_folder)
    i_preg = get_pregnancies(intervention_results_folder)

    if show_and_store_graphs:
        analysis_utility_functions.basic_comparison_graph(
            sim_years, b_preg, i_preg, 'Pregnancies (mean)', 'Mean number of pregnancies for scenarios',
            plot_destination_folder, 'pregnancies')

    # -----------------------------------------------------Total births...---------------------------------------------
    def get_births(results_folder):
        births_results = extract_results(
            results_folder,
            module="tlo.methods.demography",
            key="on_birth",
            custom_generate_series=(
                lambda df: df.assign(year=df['date'].dt.year).groupby(['year'])['year'].count()
            ),
        )

        birth_data = analysis_utility_functions.get_mean_and_quants(births_results, sim_years)

        return birth_data

    b_births = get_births(baseline_results_folder)
    i_births = get_births(intervention_results_folder)

    if show_and_store_graphs:
        analysis_utility_functions.basic_comparison_graph(
            sim_years, b_births, i_births, 'Births (mean)', 'Mean number of Births for scenarios',
            plot_destination_folder, 'births')

    # -------------------------------------------------Completed pregnancies...----------------------------------------
    def get_completed_pregnancies(comps_df, total_births_per_year, results_folder):
        ectopic_mean_numbers_per_year = analysis_utility_functions.get_mean_and_quants_from_str_df(
            comps_df['pregnancy_supervisor'], 'ectopic_unruptured', sim_years)[0]

        ia_mean_numbers_per_year = analysis_utility_functions.get_mean_and_quants_from_str_df(
            comps_df['pregnancy_supervisor'], 'induced_abortion', sim_years)[0]

        sa_mean_numbers_per_year = analysis_utility_functions.get_mean_and_quants_from_str_df(
            comps_df['pregnancy_supervisor'], 'spontaneous_abortion', sim_years)[0]

        an_stillbirth_results = extract_results(
            results_folder,
            module="tlo.methods.pregnancy_supervisor",
            key="antenatal_stillbirth",
            custom_generate_series=(
                lambda df: df.assign(year=df['date'].dt.year).groupby(['year'])['year'].count()
            ),
        )
        an_still_birth_data = analysis_utility_functions.get_mean_and_quants(an_stillbirth_results, sim_years)

        total_completed_pregnancies_per_year = [a + b + c + d + e for a, b, c, d, e in zip(total_births_per_year,
                                                                                           ectopic_mean_numbers_per_year,
                                                                                           ia_mean_numbers_per_year,
                                                                                           sa_mean_numbers_per_year,
                                                                                           an_still_birth_data[0])]

        return total_completed_pregnancies_per_year

    comp_preg_baseline = get_completed_pregnancies(b_comps_dfs, b_births[0], baseline_results_folder)
    comp_preg_intervention = get_completed_pregnancies(i_comps_dfs, i_births[0], intervention_results_folder)

    # ========================================== INTERVENTION COVERAGE... =============================================

    # 2.) Facility delivery
    # Total FDR per year (denominator - total births)
    def get_facility_delivery(results_folder, total_births_per_year):
        deliver_setting_results = extract_results(
                results_folder,
                module="tlo.methods.labour",
                key="delivery_setting_and_mode",
                custom_generate_series=(
                    lambda df_: df_.assign(year=df_['date'].dt.year).groupby(
                        ['year', 'facility_type'])['mother'].count()),
                do_scaling=False
            )

        hb_data = analysis_utility_functions.get_mean_and_quants_from_str_df(
            deliver_setting_results, 'home_birth', sim_years)
        home_birth_rate = [(x / y) * 100 for x, y in zip(hb_data[0], total_births_per_year)]

        hc_data = analysis_utility_functions.get_mean_and_quants_from_str_df(
            deliver_setting_results, 'hospital', sim_years)
        health_centre_rate = [(x / y) * 100 for x, y in zip(hc_data[0], total_births_per_year)]
        health_centre_lq = [(x / y) * 100 for x, y in zip(hc_data[1], total_births_per_year)]
        health_centre_uq = [(x / y) * 100 for x, y in zip(hc_data[2], total_births_per_year)]

        hp_data = analysis_utility_functions.get_mean_and_quants_from_str_df(
            deliver_setting_results, 'health_centre', sim_years)
        hospital_rate = [(x / y) * 100 for x, y in zip(hp_data[0], total_births_per_year)]
        hospital_lq = [(x / y) * 100 for x, y in zip(hp_data[1], total_births_per_year)]
        hospital_uq = [(x / y) * 100 for x, y in zip(hp_data[2], total_births_per_year)]

        total_fd_rate = [x + y for x, y in zip(health_centre_rate, hospital_rate)]
        fd_lqs = [x + y for x, y in zip(health_centre_lq, hospital_lq)]
        fd_uqs = [x + y for x, y in zip(health_centre_uq, hospital_uq)]

        return [total_fd_rate, fd_lqs, fd_uqs]


    b_fd = get_facility_delivery(baseline_results_folder, b_births[0])
    i_fd = get_facility_delivery(intervention_results_folder, i_births[0])

    if show_and_store_graphs:
        analysis_utility_functions.basic_comparison_graph(
            sim_years, b_fd, i_fd, '% of total births', 'Proportion of Women Delivering in a Health Facility per Year',
            plot_destination_folder, 'sba_prop_facility_deliv')

    # 3.) Postnatal Care
    def get_pnc(results_folder, total_births_per_year):
        pnc_results_maternal = extract_results(
            results_folder,
            module="tlo.methods.postnatal_supervisor",
            key="total_mat_pnc_visits",
            custom_generate_series=(
                lambda df_: df_.assign(year=df_['date'].dt.year).groupby(['year', 'visits'])['mother'].count()),
            do_scaling=False
        )

        pnc_results_newborn = extract_results(
            results_folder,
            module="tlo.methods.postnatal_supervisor",
            key="total_neo_pnc_visits",
            custom_generate_series=(
                lambda df_: df_.assign(year=df_['date'].dt.year).groupby(['year', 'visits'])['child'].count()),
            do_scaling=False
        )

        pnc_0_means = list()
        pnc_0_lqs = list()
        pnc_0_uqs = list()
        pnc_0_means_neo = list()
        pnc_0_lqs_neo = list()
        pnc_0_uqs_neo = list()

        for year in sim_years:
            pnc_0_means.append(pnc_results_maternal.loc[year, 0].mean())
            pnc_0_lqs.append(pnc_results_maternal.loc[year, 0].quantile(0.025))
            pnc_0_uqs.append(pnc_results_maternal.loc[year, 0].quantile(0.925))

            pnc_0_means_neo.append(pnc_results_newborn.loc[year, 0].mean())
            pnc_0_lqs_neo.append(pnc_results_newborn.loc[year, 0].quantile(0.025))
            pnc_0_uqs_neo.append(pnc_results_newborn.loc[year, 0].quantile(0.925))


        pnc_1_plus_rate_mat = [100 - ((x / y) * 100) for x, y in zip(pnc_0_means, total_births_per_year)]
        pnc_mat_lqs = [100 - ((x / y) * 100) for x, y in zip(pnc_0_lqs, total_births_per_year)]
        pnc_mat_uqs = [100 - ((x / y) * 100) for x, y in zip(pnc_0_uqs, total_births_per_year)]

        pnc1_plus_rate_neo = [100 - ((x / y) * 100) for x, y in zip(pnc_0_means_neo, total_births_per_year)]
        pnc_neo_lqs = [100 - ((x / y) * 100) for x, y in zip(pnc_0_lqs_neo, total_births_per_year)]
        pnc_neo_uqs = [100 - ((x / y) * 100) for x, y in zip(pnc_0_uqs_neo, total_births_per_year)]

        return [[pnc_1_plus_rate_mat, pnc_mat_lqs, pnc_mat_uqs],
                [pnc1_plus_rate_neo, pnc_neo_lqs, pnc_neo_uqs]]


    b_pnc_data = get_pnc(baseline_results_folder, b_births[0])
    b_pnc_m = b_pnc_data[0]
    b_pnc_n =b_pnc_data [1]

    i_pnc_data = get_pnc(intervention_results_folder, i_births[0])
    i_pnc_m = i_pnc_data[0]
    i_pnc_n = i_pnc_data[1]

    if show_and_store_graphs:
        analysis_utility_functions.basic_comparison_graph(
            sim_years, b_pnc_data[0], i_pnc_data[0], '% of total births',
            'Proportion of Women post-delivery attending PNC per year',
            plot_destination_folder, 'pnc_mat')

        analysis_utility_functions.basic_comparison_graph(
            sim_years, b_pnc_data[1], i_pnc_data[1], '% of total births',
            'Proportion of Women post-delivery attending PNC per year',
            plot_destination_folder, 'pnc_neo')

    # ========================================== COMPLICATION/DISEASE RATES.... =======================================
    # ---------------------------------------- Twinning Rate... -------------------------------------------------------
    # % Twin births/Total Births per year
    def get_twin_data(results_folder, total_births_per_year):
        twins_results = extract_results(
            results_folder,
            module="tlo.methods.newborn_outcomes",
            key="twin_birth",
            custom_generate_series=(
                lambda df: df.assign(year=df['date'].dt.year).groupby(['year'])['year'].count()
            ),
        )
        twin_data = analysis_utility_functions.get_mean_and_quants(twins_results, sim_years)

        total_deliveries = [x - y for x, y in zip(total_births_per_year, twin_data[0])]
        final_twining_rate = [(x / y) * 100 for x, y in zip(twin_data[0], total_deliveries)]
        lq_rate = [(x / y) * 100 for x, y in zip(twin_data[1], total_deliveries)]
        uq_rate = [(x / y) * 100 for x, y in zip(twin_data[2], total_deliveries)]

        return [final_twining_rate, lq_rate, uq_rate]

    b_twins = get_twin_data(baseline_results_folder, b_births[0])
    l_twins = get_twin_data(intervention_results_folder, i_births[0])

    if show_and_store_graphs:
        analysis_utility_functions.basic_comparison_graph(
            sim_years, b_twins, l_twins, 'Rate per 100 pregnancies', 'Yearly trends for Twin Births',
            plot_destination_folder, 'twin_rate')

    # ---------------------------------------- Early Pregnancy Loss... ----------------------------------------------
    # Ectopics
    b_ectopic_data = analysis_utility_functions.get_comp_mean_and_rate(
        'ectopic_unruptured', b_preg[0], b_comps_dfs['pregnancy_supervisor'], 1000, sim_years)

    i_ectopic_data = analysis_utility_functions.get_comp_mean_and_rate(
        'ectopic_unruptured', i_preg[0], i_comps_dfs['pregnancy_supervisor'], 1000, sim_years)

    # Spontaneous Abortions....
    b_sa_data = analysis_utility_functions.get_comp_mean_and_rate(
        'spontaneous_abortion', comp_preg_baseline, b_comps_dfs['pregnancy_supervisor'], 1000, sim_years)

    i_sa_data = analysis_utility_functions.get_comp_mean_and_rate(
        'spontaneous_abortion', comp_preg_intervention, i_comps_dfs['pregnancy_supervisor'], 1000, sim_years)

    # Induced Abortions...
    b_ia_data = analysis_utility_functions.get_comp_mean_and_rate(
        'induced_abortion', comp_preg_baseline, b_comps_dfs['pregnancy_supervisor'], 1000, sim_years)

    i_ia_data = analysis_utility_functions.get_comp_mean_and_rate(
        'induced_abortion', comp_preg_intervention, i_comps_dfs['pregnancy_supervisor'], 1000, sim_years)

    if show_and_store_graphs:
        analysis_utility_functions.basic_comparison_graph(
            sim_years, b_ectopic_data, i_ectopic_data, 'Rate per 100 pregnancies',
            'Yearly trends for Ectopic Pregnancy',
            plot_destination_folder, 'ectopic_rate')

        analysis_utility_functions.basic_comparison_graph(
            sim_years, b_sa_data, i_sa_data, 'Rate per 1000 completed pregnancies', 'Yearly rate of Miscarriage',
            plot_destination_folder, 'miscarriage_rate')

        analysis_utility_functions.basic_comparison_graph(
            sim_years, b_ia_data, i_ia_data, 'Rate per 1000 completed pregnancies', 'Yearly rate of Induced Abortion',
            plot_destination_folder, 'abortion_rate')

    # --------------------------------------------------- Syphilis Rate... --------------------------------------------
    b_syphilis_data = analysis_utility_functions.get_comp_mean_and_rate(
        'syphilis', comp_preg_baseline, b_comps_dfs['pregnancy_supervisor'], 1000, sim_years)
    i_syphilis_data = analysis_utility_functions.get_comp_mean_and_rate(
        'syphilis', comp_preg_intervention,  i_comps_dfs['pregnancy_supervisor'], 1000, sim_years)

    if show_and_store_graphs:
        analysis_utility_functions.basic_comparison_graph(
            sim_years, b_syphilis_data, i_syphilis_data, 'Rate per 1000 completed pregnancies', 'Yearly rate of Syphilis',
            plot_destination_folder, 'syphilis_rate')

    # ------------------------------------------------ Gestational Diabetes... ----------------------------------------
    b_gdm_data = analysis_utility_functions.get_comp_mean_and_rate(
        'gest_diab', comp_preg_baseline, b_comps_dfs['pregnancy_supervisor'], 1000, sim_years)
    i_gdm_data = analysis_utility_functions.get_comp_mean_and_rate(
        'gest_diab', comp_preg_intervention, i_comps_dfs['pregnancy_supervisor'], 1000, sim_years)

    if show_and_store_graphs:
        analysis_utility_functions.basic_comparison_graph(
            sim_years, b_gdm_data, i_gdm_data, 'Rate per 1000 completed pregnancies', 'Yearly rate of Gestational Diabetes',
            plot_destination_folder, 'gest_diab_rate')

    # ------------------------------------------------ PROM... --------------------------------------------------------
    b_prom_data = analysis_utility_functions.get_comp_mean_and_rate(
        'PROM', b_births[0], b_comps_dfs['pregnancy_supervisor'], 1000, sim_years)
    i_prom_data = analysis_utility_functions.get_comp_mean_and_rate(
        'PROM', i_births[0], i_comps_dfs['pregnancy_supervisor'], 1000, sim_years)

    if show_and_store_graphs:
        analysis_utility_functions.basic_comparison_graph(
            sim_years, b_prom_data, i_prom_data, 'Rate per 1000 completed pregnancies', 'Yearly rate of PROM',
            plot_destination_folder, 'prom_rate')

    # ---------------------------------------------- Anaemia... --------------------------------------------------------
    # Total prevalence of Anaemia at birth (total cases of anaemia at birth/ total births per year) and by severity

    def get_anaemia_output_at_birth(results_folder, total_births_per_year):
        anaemia_results = extract_results(
            results_folder,
            module="tlo.methods.pregnancy_supervisor",
            key="anaemia_on_birth",
            custom_generate_series=(
                lambda df: df.assign(year=df['date'].dt.year).groupby(['year', 'anaemia_status'])['year'].count()
            ),
        )

        no_anaemia_data = analysis_utility_functions.get_mean_and_quants_from_str_df(anaemia_results, 'none', sim_years)

        prevalence_of_anaemia_per_year = [100 - ((x / y) * 100) for x, y in zip(no_anaemia_data[0],
                                                                                total_births_per_year)]
        no_anaemia_lqs = [100 - ((x / y) * 100) for x, y in zip(no_anaemia_data[1], total_births_per_year)]
        no_anaemia_uqs = [100 - ((x / y) * 100) for x, y in zip(no_anaemia_data[2], total_births_per_year)]

        return [prevalence_of_anaemia_per_year, no_anaemia_lqs, no_anaemia_uqs]

    b_anaemia_b = get_anaemia_output_at_birth(baseline_results_folder, b_births[0])
    i_anaemia_b = get_anaemia_output_at_birth(intervention_results_folder, i_births[0])

    def get_anaemia_output_at_delivery(results_folder, total_births_per_year):
        pnc_anaemia = extract_results(
            results_folder,
            module="tlo.methods.postnatal_supervisor",
            key="total_mat_pnc_visits",
            custom_generate_series=(
                lambda df_: df_.assign(year=df_['date'].dt.year).groupby(['year', 'anaemia'])['mother'].count()),
            do_scaling=False
        )

        no_anaemia_data = analysis_utility_functions.get_mean_and_quants_from_str_df(pnc_anaemia, 'none', sim_years)
        prevalence_of_anaemia_per_year = [100 - ((x / y) * 100) for x, y in zip(no_anaemia_data[0],
                                                                                total_births_per_year)]
        no_anaemia_lqs = [100 - ((x / y) * 100) for x, y in zip(no_anaemia_data[1], total_births_per_year)]
        no_anaemia_uqs = [100 - ((x / y) * 100) for x, y in zip(no_anaemia_data[2], total_births_per_year)]

        return [prevalence_of_anaemia_per_year, no_anaemia_lqs, no_anaemia_uqs]

    b_anaemia_d = get_anaemia_output_at_delivery(baseline_results_folder, b_births[0])
    i_anaemia_d = get_anaemia_output_at_delivery(intervention_results_folder, i_births[0])

    if show_and_store_graphs:
        analysis_utility_functions.basic_comparison_graph(
            sim_years, b_anaemia_b, i_anaemia_b, 'Prevalence at birth',
            'Yearly prevalence of Anaemia (all severity) at birth', plot_destination_folder, 'anaemia_prev_birth')
        analysis_utility_functions.basic_comparison_graph(
            sim_years, b_anaemia_d, i_anaemia_d, 'Prevalence at Delivery',
            'Yearly prevalence of Anaemia (all severity) at delivery', plot_destination_folder, 'anaemia_prev_delivery')

    # ------------------------------------------- Hypertensive disorders ---------------------------------------------
    def get_htn_disorders_outputs(comps_df, total_births_per_year):

        output = dict()
        output['gh'] = analysis_utility_functions.get_comp_mean_and_rate_across_multiple_dataframes(
            'mild_gest_htn', total_births_per_year, 1000, [comps_df['pregnancy_supervisor'],
                                                           comps_df['postnatal_supervisor']], sim_years)

        output['sgh'] = analysis_utility_functions.get_comp_mean_and_rate_across_multiple_dataframes(
            'severe_gest_htn', total_births_per_year, 1000, [comps_df['pregnancy_supervisor'], comps_df['labour'],
                                                             comps_df['postnatal_supervisor']], sim_years)

        output['mpe'] = analysis_utility_functions.get_comp_mean_and_rate_across_multiple_dataframes(
            'mild_pre_eclamp', total_births_per_year, 1000, [comps_df['pregnancy_supervisor'],
                                                             comps_df['postnatal_supervisor']], sim_years)

        output['spe'] = analysis_utility_functions.get_comp_mean_and_rate_across_multiple_dataframes(
            'severe_pre_eclamp', total_births_per_year, 1000, [comps_df['pregnancy_supervisor'], comps_df['labour'],
                                                               comps_df['postnatal_supervisor']], sim_years)

        output['ec'] = analysis_utility_functions.get_comp_mean_and_rate_across_multiple_dataframes(
            'eclampsia', total_births_per_year, 1000, [comps_df['pregnancy_supervisor'], comps_df['labour'],
                                                       comps_df['postnatal_supervisor']], sim_years)
        return output

    b_htn_disorders = get_htn_disorders_outputs(b_comps_dfs, b_births[0])
    i_htn_disorders = get_htn_disorders_outputs(i_comps_dfs, i_births[0])

    if show_and_store_graphs:
        analysis_utility_functions.basic_comparison_graph(
            sim_years, b_htn_disorders['gh'], i_htn_disorders['gh'], 'Rate per 1000 births',
            'Rate of Gestational Hypertension per Year', plot_destination_folder, 'gest_htn_rate')

        analysis_utility_functions.basic_comparison_graph(
            sim_years, b_htn_disorders['sgh'], i_htn_disorders['sgh'], 'Rate per 1000 births',
            'Rate of Severe Gestational Hypertension per Year', plot_destination_folder, 'severe_gest_htn_rate')

        analysis_utility_functions.basic_comparison_graph(
            sim_years, b_htn_disorders['mpe'], i_htn_disorders['mpe'], 'Rate per 1000 births',
            'Rate of Mild pre-eclampsia per Year', plot_destination_folder, 'mild_pre_eclampsia_rate')

        analysis_utility_functions.basic_comparison_graph(
            sim_years, b_htn_disorders['spe'], i_htn_disorders['spe'], 'Rate per 1000 birth',
            'Rate of Severe pre-eclampsia per Year', plot_destination_folder, 'severe_pre_eclampsia_rate')

        analysis_utility_functions.basic_comparison_graph(
            sim_years, b_htn_disorders['ec'], i_htn_disorders['ec'], 'Rate per 1000 births',
            'Rate of Eclampsia per Year', plot_destination_folder, 'eclampsia_rate')

    #  ---------------------------------------------Placenta praevia... ------------------------------------------------
    b_pp_data = analysis_utility_functions.get_comp_mean_and_rate(
        'placenta_praevia', b_preg[0], b_comps_dfs['pregnancy_supervisor'], 1000, sim_years)
    i_pp_data = analysis_utility_functions.get_comp_mean_and_rate(
        'placenta_praevia', i_preg[0], i_comps_dfs['pregnancy_supervisor'], 1000, sim_years)

    if show_and_store_graphs:
        analysis_utility_functions.basic_comparison_graph(
            sim_years, b_pp_data, i_pp_data, 'Rate per 1000 pregnancies',
            'Rate of Placenta Praevia per Year', plot_destination_folder, 'praevia_rate')

    #  ---------------------------------------------Placental abruption... --------------------------------------------
    b_pa_data = analysis_utility_functions.get_comp_mean_and_rate_across_multiple_dataframes(
        'placental_abruption', b_births[0], 1000, [b_comps_dfs['pregnancy_supervisor'], b_comps_dfs['labour']],
        sim_years)
    i_pa_data = analysis_utility_functions.get_comp_mean_and_rate_across_multiple_dataframes(
        'placental_abruption', i_births[0], 1000, [i_comps_dfs['pregnancy_supervisor'], i_comps_dfs['labour']],
        sim_years)

    if show_and_store_graphs:
        analysis_utility_functions.basic_comparison_graph(
            sim_years, b_pa_data, i_pa_data, 'Rate per 1000 births',
            'Rate of Placental Abruption per Year', plot_destination_folder, 'abruption_rate')

    # --------------------------------------------- Antepartum Haemorrhage... -----------------------------------------
    # Rate of APH/total births (antenatal and labour)

    def get_aph_data(comps_df, total_births_per_year):
        mm_aph_data = analysis_utility_functions.get_comp_mean_and_rate_across_multiple_dataframes(
            'mild_mod_antepartum_haemorrhage', total_births_per_year, 1000,
            [comps_df['pregnancy_supervisor'], comps_df['labour']], sim_years)

        s_aph_data =analysis_utility_functions. get_comp_mean_and_rate_across_multiple_dataframes(
            'severe_antepartum_haemorrhage', total_births_per_year, 1000,
            [b_comps_dfs['pregnancy_supervisor'], b_comps_dfs['labour']], sim_years)

        total_aph_rates = [x + y for x, y in zip(mm_aph_data[0], s_aph_data[0])]
        aph_lqs = [x + y for x, y in zip(mm_aph_data[1], s_aph_data[1])]
        aph_uqs = [x + y for x, y in zip(mm_aph_data[2], s_aph_data[2])]

        return [total_aph_rates, aph_lqs, aph_uqs]

    b_aph_data = get_aph_data(b_comps_dfs, b_births[0])
    i_aph_data = get_aph_data(i_comps_dfs, i_births[0])

    if show_and_store_graphs:
        analysis_utility_functions.basic_comparison_graph(
            sim_years, b_aph_data, i_aph_data, 'Rate per 1000 births',
            'Rate of Antepartum Haemorrhage per Year', plot_destination_folder, 'aph_rate')

    # --------------------------------------------- Preterm birth ... ------------------------------------------------
    def get_ptl_data(total_births_per_year, comps_df):
        early_ptl_data = analysis_utility_functions.get_comp_mean_and_rate(
            'early_preterm_labour', total_births_per_year, comps_df['labour'], 100, sim_years)
        late_ptl_data = analysis_utility_functions.get_comp_mean_and_rate(
            'late_preterm_labour', total_births_per_year, comps_df['labour'], 100, sim_years)


        total_ptl_rates = [x + y for x, y in zip(early_ptl_data[0], late_ptl_data[0])]
        ptl_lqs = [x + y for x, y in zip(early_ptl_data[1], late_ptl_data[1])]
        ltl_uqs = [x + y for x, y in zip(early_ptl_data[2], late_ptl_data[2])]

        return [total_ptl_rates, ptl_lqs, ltl_uqs]

    b_ptl_data = get_ptl_data(b_births[0], b_comps_dfs)
    i_ptl_data = get_ptl_data(i_births[0], i_comps_dfs)

    # --------------------------------------------- Post term birth ... -----------------------------------------------
    b_potl_data = analysis_utility_functions.get_comp_mean_and_rate(
        'post_term_labour', b_births[0], b_comps_dfs['labour'], 100, sim_years)
    i_potl_data = analysis_utility_functions.get_comp_mean_and_rate(
        'post_term_labour', i_births[1], i_comps_dfs['labour'], 100, sim_years)

    if show_and_store_graphs:
        analysis_utility_functions.basic_comparison_graph(
            sim_years, b_ptl_data, i_ptl_data, 'Proportion of total births',
            'Preterm birth rate', plot_destination_folder, 'ptb_rate')

        analysis_utility_functions.basic_comparison_graph(
            sim_years, b_potl_data, i_potl_data, 'Proportion of total births',
            'Post term birth rate', plot_destination_folder, 'potl_rate')

    # todo plot early and late seperated

    # ------------------------------------------- Antenatal Stillbirth ... --------------------------------------------
    def get_an_stillbirth(results_folder, total_births_per_year):
        an_stillbirth_results = extract_results(
            results_folder,
            module="tlo.methods.pregnancy_supervisor",
            key="antenatal_stillbirth",
            custom_generate_series=(
                lambda df: df.assign(year=df['date'].dt.year).groupby(['year'])['year'].count()
            ),
        )
        an_still_birth_data = analysis_utility_functions.get_mean_and_quants(an_stillbirth_results, sim_years)

        an_sbr_per_year = [(x / y) * 1000 for x, y in zip(an_still_birth_data[0], total_births_per_year)]
        an_sbr_lqs = [(x / y) * 1000 for x, y in zip(an_still_birth_data[1], total_births_per_year)]
        an_sbr_uqs = [(x / y) * 1000 for x, y in zip(an_still_birth_data[2], total_births_per_year)]

        return [an_sbr_per_year, an_sbr_lqs, an_sbr_uqs]

    b_an_sbr = get_an_stillbirth(baseline_results_folder, b_births[0])
    i_an_sbr = get_an_stillbirth(intervention_results_folder, i_births[0])

    if show_and_store_graphs:
        analysis_utility_functions.basic_comparison_graph(
            sim_years, b_an_sbr, i_an_sbr, 'Rate per 1000 births',
            'Antenatal Stillbirth Rate per Year', plot_destination_folder, 'sbr_an')

    # ------------------------------------------------- Birth weight... ----------------------------------------------
    def get_neonatal_comp_dfs(results_folder):
        nb_comp_dfs = dict()
        nb_comp_dfs['newborn_outcomes'] = extract_results(
                results_folder,
                module="tlo.methods.newborn_outcomes",
                key="newborn_complication",
                custom_generate_series=(
                    lambda df_: df_.assign(year=df_['date'].dt.year).groupby(['year', 'type'])['newborn'].count()),
                do_scaling=False
            )

        nb_comp_dfs['newborn_postnatal'] = extract_results(
                results_folder,
                module="tlo.methods.postnatal_supervisor",
                key="newborn_complication",
                custom_generate_series=(
                    lambda df_: df_.assign(year=df_['date'].dt.year).groupby(['year', 'type'])['newborn'].count()),
                do_scaling=False
            )

        return nb_comp_dfs

    b_neonatal_comp_dfs = get_neonatal_comp_dfs(baseline_results_folder)
    i_neonatal_comp_dfs = get_neonatal_comp_dfs(intervention_results_folder)

    b_lbw_data = analysis_utility_functions.get_comp_mean_and_rate(
        'low_birth_weight', b_births[0], b_neonatal_comp_dfs['newborn_outcomes'], 100, sim_years)
    i_lbw_data = analysis_utility_functions.get_comp_mean_and_rate(
        'low_birth_weight', i_births[0], i_neonatal_comp_dfs['newborn_outcomes'], 100, sim_years)

    b_macro_data = analysis_utility_functions.get_comp_mean_and_rate(
        'macrosomia', b_births[0], b_neonatal_comp_dfs['newborn_outcomes'], 100, sim_years)
    i_macro_data = analysis_utility_functions.get_comp_mean_and_rate(
        'macrosomia', i_births[0], i_neonatal_comp_dfs['newborn_outcomes'], 100, sim_years)

    b_sga_data = analysis_utility_functions.get_comp_mean_and_rate(
        'small_for_gestational_age', b_births[0], b_neonatal_comp_dfs['newborn_outcomes'], 100, sim_years)
    i_sga_data = analysis_utility_functions.get_comp_mean_and_rate(
        'small_for_gestational_age', i_births[0], i_neonatal_comp_dfs['newborn_outcomes'], 100, sim_years)

    if show_and_store_graphs:
        analysis_utility_functions.basic_comparison_graph(
            sim_years, b_lbw_data, i_lbw_data, 'Proportion of total births',
            'Yearly Prevalence of Low Birth Weight', plot_destination_folder, 'neo_lbw_prev')

        analysis_utility_functions.basic_comparison_graph(
            sim_years, b_macro_data, i_macro_data, 'Proportion of total births',
            'Yearly Prevalence of Macrosomia', plot_destination_folder, 'neo_macrosomia_prev')

        analysis_utility_functions.basic_comparison_graph(
            sim_years, b_sga_data, i_sga_data, 'Proportion of total births',
            'Yearly Prevalence of Small for Gestational Age', plot_destination_folder, 'neo_sga_prev')

    # --------------------------------------------- Obstructed Labour... ---------------------------------------------
    b_ol_data = analysis_utility_functions.get_comp_mean_and_rate(
        'obstructed_labour', b_births[0], b_comps_dfs['labour'], 1000, sim_years)
    i_ol_data = analysis_utility_functions.get_comp_mean_and_rate(
        'obstructed_labour', i_births[0], i_comps_dfs['labour'], 1000, sim_years)

    if show_and_store_graphs:
        analysis_utility_functions.basic_comparison_graph(
            sim_years, b_ol_data, i_ol_data, 'Rate per 1000 births',
            'Obstructed Labour Rate per Year', plot_destination_folder, 'ol_rate')

    # --------------------------------------------- Uterine rupture... ------------------------------------------------
    b_ur_data = analysis_utility_functions.get_comp_mean_and_rate(
        'uterine_rupture', b_births[0], b_comps_dfs['labour'], 1000, sim_years)
    i_ur_data = analysis_utility_functions.get_comp_mean_and_rate(
        'uterine_rupture', i_births[0], i_comps_dfs['labour'], 1000, sim_years)

    if show_and_store_graphs:
        analysis_utility_functions.basic_comparison_graph(
            sim_years, b_ur_data, i_ur_data, 'Rate per 1000 births',
            'Rate of Uterine Rupture per Year', plot_destination_folder, 'ur_rate')

    # ---------------------------Caesarean Section Rate & Assisted Vaginal Delivery Rate... ---------------------------
    def get_delivery_data(results_folder,total_births_per_year):
        delivery_mode = extract_results(
            results_folder,
            module="tlo.methods.labour",
            key="delivery_setting_and_mode",
            custom_generate_series=(
                lambda df_: df_.assign(year=df_['date'].dt.year).groupby(['year', 'mode'])['mother'].count()),
            do_scaling=False
        )

        cs_data = analysis_utility_functions.get_comp_mean_and_rate(
            'caesarean_section', total_births_per_year, delivery_mode, 100, sim_years)
        avd_data = analysis_utility_functions.get_comp_mean_and_rate(
            'instrumental', total_births_per_year, delivery_mode, 100, sim_years)

        return [cs_data, avd_data]

    b_delivery_data = get_delivery_data(baseline_results_folder, b_births[0])
    i_delivery_data = get_delivery_data(intervention_results_folder, i_births[0])

    if show_and_store_graphs:
        analysis_utility_functions.basic_comparison_graph(
            sim_years, b_delivery_data[0], i_delivery_data[0], 'Proportion of total births',
            'Caesarean Section Rate per Year', plot_destination_folder, 'caesarean_section_rate')

        analysis_utility_functions.basic_comparison_graph(
            sim_years, b_delivery_data[1], i_delivery_data[1], 'Proportion of total births',
            'Assisted Vaginal Delivery Rate per Year', plot_destination_folder, 'avd_rate')

    # ------------------------------------------ Maternal Sepsis Rate... ----------------------------------------------
    def get_total_sepsis_rates(total_births_per_year, comps_df):
        an_sep_data = analysis_utility_functions.get_comp_mean_and_rate(
            'clinical_chorioamnionitis', total_births_per_year, comps_df['pregnancy_supervisor'], 1000, sim_years)
        la_sep_data = analysis_utility_functions.get_comp_mean_and_rate(
            'sepsis', total_births_per_year, comps_df['labour'], 1000, sim_years)

        pn_la_sep_data = analysis_utility_functions.get_comp_mean_and_rate(
            'sepsis_postnatal', total_births_per_year, comps_df['labour'], 1000, sim_years)
        pn_sep_data = analysis_utility_functions.get_comp_mean_and_rate(
            'sepsis', total_births_per_year, comps_df['postnatal_supervisor'], 1000, sim_years)

        complete_pn_sep_data = [x + y for x, y in zip(pn_la_sep_data[0], pn_sep_data[0])]
        complete_pn_sep_lq = [x + y for x, y in zip(pn_la_sep_data[1], pn_sep_data[1])]
        complete_pn_sep_up = [x + y for x, y in zip(pn_la_sep_data[2], pn_sep_data[2])]

        total_sep_rates = [x + y + z for x, y, z in zip(an_sep_data[0], la_sep_data[0], complete_pn_sep_data)]
        sep_lq = [x + y + z for x, y, z in zip(an_sep_data[1], la_sep_data[1], complete_pn_sep_lq)]
        sep_uq = [x + y + z for x, y, z in zip(an_sep_data[2], la_sep_data[2], complete_pn_sep_up)]

        return [total_sep_rates, sep_lq, sep_uq]

    b_sep_data = get_total_sepsis_rates(b_births[0], b_comps_dfs)
    i_sep_data = get_total_sepsis_rates(i_births[0], i_comps_dfs)

    if show_and_store_graphs:
        analysis_utility_functions.basic_comparison_graph(
            sim_years, b_sep_data, i_sep_data, 'Rate per 1000 births',
            'Rate of Maternal Sepsis per Year', plot_destination_folder, 'sepsis_rate')

    # ----------------------------------------- Postpartum Haemorrhage... ---------------------------------------------
    def get_pph_data(total_births_per_year, comps_df):
        la_pph_data = analysis_utility_functions.get_comp_mean_and_rate(
            'primary_postpartum_haemorrhage', total_births_per_year, comps_df['labour'], 1000, sim_years)
        pn_pph_data = analysis_utility_functions.get_comp_mean_and_rate(
            'secondary_postpartum_haemorrhage', total_births_per_year, comps_df['postnatal_supervisor'],
            1000, sim_years)

        total_pph_rates = [x + y for x, y in zip(la_pph_data[0], pn_pph_data[0])]
        pph_lq = [x + y for x, y in zip(la_pph_data[1], pn_pph_data[1])]
        pph_uq = [x + y for x, y in zip(la_pph_data[2], pn_pph_data[2])]

        return [total_pph_rates, pph_lq, pph_uq]

    b_pph_data = get_pph_data(b_births[0], b_comps_dfs)
    i_pph_data = get_pph_data(i_births[0], i_comps_dfs)

    if show_and_store_graphs:
        analysis_utility_functions.basic_comparison_graph(
            sim_years, b_pph_data, i_pph_data, 'Rate per 1000 births',
            'Rate of Postpartum Haemorrhage per Year', plot_destination_folder, 'pph_rate')

    # ------------------------------------------- Intrapartum Stillbirth ... ------------------------------------------
    def get_ip_stillbirths(results_folder, total_births_per_year):
        ip_stillbirth_results = extract_results(
            results_folder,
            module="tlo.methods.labour",
            key="intrapartum_stillbirth",
            custom_generate_series=(
                lambda df: df.assign(year=df['date'].dt.year).groupby(['year'])['year'].count()
            ),
        )

        ip_still_birth_data = analysis_utility_functions.get_mean_and_quants(ip_stillbirth_results, sim_years)
        ip_sbr_per_year = [(x/y) * 1000 for x, y in zip(ip_still_birth_data[0], total_births_per_year)]
        ip_sbr_lqs = [(x/y) * 1000 for x, y in zip(ip_still_birth_data[1], total_births_per_year)]
        ip_sbr_uqs = [(x/y) * 1000 for x, y in zip(ip_still_birth_data[2], total_births_per_year)]

        return [ip_sbr_per_year, ip_sbr_lqs, ip_sbr_uqs]

    b_ip_sbr = get_ip_stillbirths(baseline_results_folder, b_births[0])
    i_ip_sbr = get_ip_stillbirths(intervention_results_folder, i_births[0])

    if show_and_store_graphs:
        analysis_utility_functions.basic_comparison_graph(
            sim_years, b_ip_sbr, i_ip_sbr, 'Rate per 1000 births',
            'Intrapartum Stillbirth Rate per Year', plot_destination_folder, 'sbr_ip')

    # ==================================================== NEWBORN OUTCOMES ===========================================
    #  ------------------------------------------- Neonatal sepsis (labour & postnatal) -------------------------------
    def get_neonatal_sepsis(total_births_per_year, nb_outcomes_df, nb_outcomes_pn_df):
        early_ns_data = analysis_utility_functions.get_comp_mean_and_rate(
            'early_onset_sepsis', total_births_per_year, nb_outcomes_df, 1000, sim_years)
        early_ns_pn = analysis_utility_functions.get_comp_mean_and_rate(
            'early_onset_sepsis', total_births_per_year, nb_outcomes_pn_df, 1000, sim_years)
        late_ns_data = analysis_utility_functions.get_comp_mean_and_rate(
            'late_onset_sepsis', total_births_per_year, nb_outcomes_pn_df, 1000, sim_years)

        total_ns_rates = [x + y + z for x, y, z in zip(early_ns_data[0], early_ns_pn[0], late_ns_data[0])]
        ns_lqs = [x + y + z for x, y, z in zip(early_ns_data[1], early_ns_pn[1], late_ns_data[1])]
        ns_uqs = [x + y + z for x, y, z in zip(early_ns_data[2], early_ns_pn[2], late_ns_data[2])]

        return [total_ns_rates, ns_lqs, ns_uqs]

    b_n_sepsis = get_neonatal_sepsis(b_births[0], b_neonatal_comp_dfs['newborn_outcomes'],
                                     b_neonatal_comp_dfs['newborn_postnatal'])
    i_n_sepsis = get_neonatal_sepsis(i_births[0], i_neonatal_comp_dfs['newborn_outcomes'],
                                     i_neonatal_comp_dfs['newborn_postnatal'])

    if show_and_store_graphs:
        analysis_utility_functions.basic_comparison_graph(
            sim_years, b_n_sepsis, i_n_sepsis, 'Rate per 1000 births',
            'Rate of Neonatal Sepsis per year', plot_destination_folder, 'neo_sepsis_rate')

    #  ------------------------------------------- Neonatal encephalopathy --------------------------------------------
    def get_neonatal_encephalopathy(total_births_per_year, nb_outcomes_df):
        mild_data = analysis_utility_functions.get_comp_mean_and_rate(
            'mild_enceph', total_births_per_year, nb_outcomes_df, 1000, sim_years)
        mod_data = analysis_utility_functions.get_comp_mean_and_rate(
            'moderate_enceph', total_births_per_year, nb_outcomes_df, 1000, sim_years)
        sev_data = analysis_utility_functions.get_comp_mean_and_rate(
            'severe_enceph', total_births_per_year, nb_outcomes_df, 1000, sim_years)

        total_enceph_rates = [x + y + z for x, y, z in zip(mild_data[0], mod_data[0], sev_data[0])]
        enceph_lq = [x + y + z for x, y, z in zip(mild_data[1], mod_data[1], sev_data[1])]
        enceph_uq = [x + y + z for x, y, z in zip(mild_data[2], mod_data[2], sev_data[2])]

        return [total_enceph_rates, enceph_lq, enceph_uq]

    b_enceph = get_neonatal_encephalopathy(b_births[0], b_neonatal_comp_dfs['newborn_outcomes'])
    i_enceph = get_neonatal_encephalopathy(i_births[0], i_neonatal_comp_dfs['newborn_outcomes'])

    if show_and_store_graphs:
        analysis_utility_functions.basic_comparison_graph(
            sim_years, b_enceph, i_enceph, 'Rate per 1000 births',
            'Rate of Neonatal Encephalopathy per year', plot_destination_folder, 'neo_enceph_rate')

    # ----------------------------------------- Respiratory Depression -------------------------------------------------
    b_rd_data = analysis_utility_functions.get_comp_mean_and_rate(
        'not_breathing_at_birth', b_births[0], b_neonatal_comp_dfs['newborn_outcomes'], 1000, sim_years)
    i_rd_data = analysis_utility_functions.get_comp_mean_and_rate(
        'not_breathing_at_birth', i_births[0], i_neonatal_comp_dfs['newborn_outcomes'], 1000, sim_years)

    if show_and_store_graphs:
        analysis_utility_functions.basic_comparison_graph(
            sim_years, b_rd_data, i_rd_data, 'Rate per 1000 births',
            'Rate of Neonatal Respiratory Depression per year', plot_destination_folder, 'neo_resp_depression_rate')

    # ----------------------------------------- Respiratory Distress Syndrome -----------------------------------------
    def get_rds(la_comps, nb_outcomes_df):
        ept = analysis_utility_functions.get_mean_and_quants_from_str_df(la_comps, 'early_preterm_labour', sim_years)[0]
        # todo: should be live births
        lpt = analysis_utility_functions.get_mean_and_quants_from_str_df(la_comps, 'late_preterm_labour', sim_years)[0]
        total_ptbs = [x + y for x, y in zip(ept, lpt)]

        rds_data = analysis_utility_functions.get_comp_mean_and_rate(
            'respiratory_distress_syndrome', total_ptbs, nb_outcomes_df, 1000, sim_years)

        return rds_data

    b_rds = get_rds(b_comps_dfs['labour'], b_neonatal_comp_dfs['newborn_outcomes'])
    i_rds = get_rds(i_comps_dfs['labour'], i_neonatal_comp_dfs['newborn_outcomes'])

    if show_and_store_graphs:
        analysis_utility_functions.basic_comparison_graph(
            sim_years, b_rds, i_rds, 'Rate per 1000 preterm births',
            'Rate of Preterm Respiratory Distress Syndrome per year', plot_destination_folder, 'neo_rds_rate')

    # ===================================== COMPARING COMPLICATION LEVEL MMR ==========================================
    b_death_results = extract_results(
        baseline_results_folder,
        module="tlo.methods.demography",
        key="death",
        custom_generate_series=(
            lambda df: df.assign(year=df['date'].dt.year).groupby(['year', 'cause'])['year'].count()
        ),
    )

    i_death_results = extract_results(
        intervention_results_folder,
        module="tlo.methods.demography",
        key="death",
        custom_generate_series=(
            lambda df: df.assign(year=df['date'].dt.year).groupby(['year', 'cause'])['year'].count()
        ),
    )

    simplified_causes = ['ectopic_pregnancy', 'abortion', 'severe_pre_eclampsia', 'sepsis', 'uterine_rupture',
                         'postpartum_haemorrhage',  'antepartum_haemorrhage']

    for cause in simplified_causes:
        if (cause == 'ectopic_pregnancy') or (cause == 'antepartum_haemorrhage') or (cause == 'uterine_rupture'):
            b_deaths = analysis_utility_functions.get_mean_and_quants_from_str_df(b_death_results, cause, sim_years)[0]
            i_deaths = analysis_utility_functions.get_mean_and_quants_from_str_df(b_death_results, cause, sim_years)[0]

        elif cause == 'abortion':
            def get_ab_mmr(death_results):
                ia_deaths = analysis_utility_functions.get_mean_and_quants_from_str_df(
                    death_results, 'induced_abortion', sim_years)[0]
                sa_deaths = analysis_utility_functions.get_mean_and_quants_from_str_df(
                    death_results, 'spontaneous_abortion', sim_years)[0]
                deaths = [x + y for x, y in zip(ia_deaths, sa_deaths)]
                return deaths
            b_deaths = get_ab_mmr(b_death_results)
            i_deaths = get_ab_mmr(i_death_results)

        elif cause == 'severe_pre_eclampsia':
            def get_htn_mmr(death_results):
                spe_deaths = analysis_utility_functions.get_mean_and_quants_from_str_df(
                    death_results, 'severe_pre_eclampsia', sim_years)[0]
                ec_deaths = analysis_utility_functions.get_mean_and_quants_from_str_df(
                    death_results, 'eclampsia', sim_years)[0]
                sgh_deaths = analysis_utility_functions.get_mean_and_quants_from_str_df(
                    death_results, 'severe_gestational_hypertension', sim_years)[0]
                deaths = [x + y + z for x, y, z in zip(spe_deaths, ec_deaths, sgh_deaths)]
                return deaths
            b_deaths = get_htn_mmr(b_death_results)
            i_deaths = get_htn_mmr(i_death_results)

        elif cause == 'postpartum_haemorrhage':
            def get_pph_mmr(death_results):
                p_deaths = analysis_utility_functions.get_mean_and_quants_from_str_df(
                    death_results, 'postpartum_haemorrhage', sim_years)[0]
                s_deaths = analysis_utility_functions.get_mean_and_quants_from_str_df(
                    death_results, 'secondary_postpartum_haemorrhage', sim_years)[0]
                deaths = [x + y for x, y in zip(p_deaths, s_deaths)]
                return deaths
            b_deaths = get_pph_mmr(b_death_results)
            i_deaths = get_pph_mmr(i_death_results)

        elif cause == 'sepsis':
            def get_sep_mmr(death_results):
                a_deaths = analysis_utility_functions.get_mean_and_quants_from_str_df(
                    death_results, 'antenatal_sepsis', sim_years)[0]
                i_deaths = analysis_utility_functions.get_mean_and_quants_from_str_df(
                    death_results, 'intrapartum_sepsis', sim_years)[0]
                p_deaths = analysis_utility_functions.get_mean_and_quants_from_str_df(
                    death_results, 'postpartum_sepsis', sim_years)[0]
                deaths = [x + y + z for x, y, z in zip(a_deaths, i_deaths, p_deaths)]
                return deaths
            b_deaths = get_sep_mmr(b_death_results)
            i_deaths = get_sep_mmr(i_death_results)

        b_mmr = [(x / y) * 100000 for x, y in zip(b_deaths, b_births[0])]
        i_mmr = [(x / y) * 100000 for x, y in zip(i_deaths, i_births[0])]

        if show_and_store_graphs:
            plt.plot(sim_years, b_mmr, 'o-g', label="Baseline", color='deepskyblue')
            plt.plot(sim_years, i_mmr, 'o-g', label="Intervention", color='darkseagreen')
            plt.xlabel('Year')
            plt.ylabel('Deaths per 100,000 births')
            plt.title(f'Maternal Mortality Ratio per Year for {cause} by Scenario')
            plt.legend()
            plt.savefig(f'{plot_destination_folder}/mmr/mmr_{cause}.png')
            plt.show()

    # ===================================== COMPARING COMPLICATION LEVEL NMR ========================================
    simplified_causes_neo = ['prematurity', 'encephalopathy', 'neonatal_sepsis', 'neonatal_respiratory_depression']

    for cause in simplified_causes_neo:
        if (cause == 'encephalopathy') or (cause == 'neonatal_respiratory_depression'):
            b_deaths = analysis_utility_functions.get_mean_and_quants_from_str_df(b_death_results, cause, sim_years)[0]
            i_deaths = analysis_utility_functions.get_mean_and_quants_from_str_df(i_death_results, cause, sim_years)[0]

        elif cause == 'neonatal_sepsis':
            def get_neo_sep_deaths(death_results):
                early1 = analysis_utility_functions.get_mean_and_quants_from_str_df(
                    death_results, 'early_onset_neonatal_sepsis', sim_years)[0]
                early2 = analysis_utility_functions.get_mean_and_quants_from_str_df(
                    death_results, 'early_onset_sepsis', sim_years)[0]
                late = analysis_utility_functions.get_mean_and_quants_from_str_df(
                    death_results, 'late_onset_sepsis', sim_years)[0]
                deaths = [x + y + z for x, y, z in zip(early1, early2, late)]
                return deaths

            b_deaths = get_neo_sep_deaths(b_death_results)
            i_deaths = get_neo_sep_deaths(i_death_results)

        elif cause == 'prematurity':
            def get_pt_deaths(death_results):
                rds_deaths = analysis_utility_functions.get_mean_and_quants_from_str_df(
                    death_results, 'respiratory_distress_syndrome', sim_years)[0]
                other_deaths = analysis_utility_functions.get_mean_and_quants_from_str_df(
                    death_results, 'preterm_other', sim_years)[0]
                deaths = [x + y for x, y in zip(rds_deaths, other_deaths)]
                return deaths

            b_deaths = get_pt_deaths(b_death_results)
            i_deaths = get_pt_deaths(i_death_results)

        b_nmr = [(x / y) * 1000 for x, y in zip(b_deaths, b_births[0])]
        i_nmr = [(x / y) * 1000 for x, y in zip(i_deaths, i_births[0])]

        if show_and_store_graphs:
            plt.plot(sim_years, b_nmr, 'o-g', label="Baseline", color='deepskyblue')
            plt.plot(sim_years, i_nmr, 'o-g', label="Intervention", color='darkseagreen')
            plt.xlabel('Year')
            plt.ylabel('Deaths per 1000 births')
            plt.title(f'Neonatal Mortality Ratio per Year for {cause} by Scenario')
            plt.legend()
            plt.savefig(f'{plot_destination_folder}/nmr/nmr_{cause}.png')
            plt.show()
