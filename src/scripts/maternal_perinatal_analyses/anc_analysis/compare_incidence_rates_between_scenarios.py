from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from tlo.analysis.utils import (
    extract_results,
    get_scenario_outputs,
)

baseline_scenario_filename = 'baseline_anc_scenario.py'
intervention_scenario_filename = 'and_qual.py'

outputspath = Path('./outputs/sejjj49@ucl.ac.uk/')
graph_location = 'analysis_output_graphs_and_qual-2022-02-17T150358Z/comp_rates'
rfp = Path('./resources')

baseline_results_folder = get_scenario_outputs(baseline_scenario_filename, outputspath)[-1]
intervention_results_folder = get_scenario_outputs(intervention_scenario_filename, outputspath)[-1]

sim_years = [2020, 2021, 2022, 2023, 2024, 2025]


# ============================================HELPER FUNCTIONS... =====================================================
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


#  COMPLICATION DATA FRAMES....
b_comps_dfs = get_modules_maternal_complication_dataframes(baseline_results_folder)
i_comps_dfs = get_modules_maternal_complication_dataframes(intervention_results_folder)


def get_mean_and_quants(df):
    year_means = list()
    lower_quantiles = list()
    upper_quantiles = list()

    for year in sim_years:
        if year in df.index:
            year_means.append(df.loc[year].mean())
            lower_quantiles.append(df.loc[year].quantile(0.025))
            upper_quantiles.append(df.loc[year].quantile(0.925))
        else:
            year_means.append(0)
            lower_quantiles.append(0)
            lower_quantiles.append(0)

    return [year_means, lower_quantiles, upper_quantiles]


def get_mean_and_quants_from_str_df(df, complication):
    yearly_mean_number = list()
    yearly_lq = list()
    yearly_uq = list()
    for year in sim_years:
        if complication in df.loc[year].index:
            yearly_mean_number.append(df.loc[year, complication].mean())
            yearly_lq.append(df.loc[year, complication].quantile(0.025))
            yearly_uq.append(df.loc[year, complication].quantile(0.925))
        else:
            yearly_mean_number.append(0)
            yearly_lq.append(0)
            yearly_uq.append(0)

    return [yearly_mean_number, yearly_lq, yearly_uq]


def get_comp_mean_and_rate(complication, denominator_list, df, rate):
    yearly_means = get_mean_and_quants_from_str_df(df, complication)[0]
    yearly_lq = get_mean_and_quants_from_str_df(df, complication)[1]
    yearly_uq = get_mean_and_quants_from_str_df(df, complication)[2]

    yearly_mean_rate = [(x / y) * rate for x, y in zip(yearly_means, denominator_list)]
    yearly_lq_rate = [(x / y) * rate for x, y in zip(yearly_lq, denominator_list)]
    yearly_uq_rate = [(x / y) * rate for x, y in zip(yearly_uq, denominator_list)]

    return [yearly_mean_rate, yearly_lq_rate, yearly_uq_rate]


def get_comp_mean_and_rate_across_multiple_dataframes(complication, denominators, rate, dataframes):

    def get_list_of_rates_and_quants(df):
        rates_per_year = list()
        lq_per_year = list()
        uq_per_year = list()
        for year, denominator in zip(sim_years, denominators):
            if year in df.index:
                if complication in df.loc[year].index:
                    rates = (df.loc[year, complication].mean() / denominator) * rate
                    lq = (df.loc[year, complication].quantile(0.025) / denominator) * rate
                    uq = (df.loc[year, complication].quantile(0.925) / denominator) * rate
                    rates_per_year.append(rates)
                    lq_per_year.append(lq)
                    uq_per_year.append(uq)

                else:
                    rates_per_year.append(0)
                    lq_per_year.append(0)
                    uq_per_year.append(0)
            else:
                rates_per_year.append(0)
                lq_per_year.append(0)
                uq_per_year.append(0)

        return [rates_per_year, lq_per_year, uq_per_year]

    if len(dataframes) == 2:
        df_1_data = get_list_of_rates_and_quants(dataframes[0])
        df_2_data = get_list_of_rates_and_quants(dataframes[1])

        total_rates = [x + y for x, y in zip(df_1_data[0], df_2_data[0])]
        total_lq = [x + y for x, y in zip(df_1_data[1], df_2_data[1])]
        total_uq = [x + y for x, y in zip(df_1_data[2], df_2_data[2])]

    else:
        df_1_data = get_list_of_rates_and_quants(dataframes[0])
        df_2_data = get_list_of_rates_and_quants(dataframes[1])
        df_3_data = get_list_of_rates_and_quants(dataframes[2])

        total_rates = [x + y + z for x, y, z in zip(df_1_data[0], df_2_data[0], df_3_data[0])]
        total_lq = [x + y + z for x, y, z in zip(df_1_data[1], df_2_data[1], df_3_data[1])]
        total_uq = [x + y + z for x, y, z in zip(df_1_data[2], df_2_data[2], df_3_data[2])]

    return [total_rates, total_lq, total_uq]


def simple_line_chart(model_rate, target_rate, x_title, y_title, title, file_name):
    plt.plot(sim_years, model_rate, 'o-g', label="Model", color='deepskyblue')
    plt.plot(sim_years, target_rate,  'o-g', label="Target rate", color='darkseagreen')
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.title(title)
    plt.legend()
    plt.savefig(f'./outputs/sejjj49@ucl.ac.uk/{graph_location}/{file_name}.png')
    plt.show()


def simple_bar_chart(model_rates, x_title, y_title, title, file_name):
    bars = sim_years
    x_pos = np.arange(len(bars))
    plt.bar(x_pos, model_rates, label="Model", color='thistle')
    plt.xticks(x_pos, bars, rotation=90)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.title(title)
    plt.legend()
    plt.savefig(f'./outputs/sejjj49@ucl.ac.uk/{graph_location}/{file_name}.png')
    plt.show()


def line_graph_with_ci_and_target_rate(b_data, i_data, x_label, y_label, title, file_name):
    fig, ax = plt.subplots()
    ax.plot(sim_years, b_data[0], 'o-g', label="Baseline", color='deepskyblue')
    ax.fill_between(sim_years, b_data[1], b_data[2], color='b', alpha=.1, label="UI (2.5-92.5)")

    ax.plot(sim_years, i_data[0], 'o-g', label="Intervention", color='forestgreen')
    ax.fill_between(sim_years, i_data[1], i_data[2], color='g', alpha=.1, label="UI (2.5-92.5)")

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.savefig(f'./outputs/sejjj49@ucl.ac.uk/{graph_location}/{file_name}.png')
    plt.show()


# ============================================  DENOMINATORS... ======================================================
# ---------------------------------------------Total_pregnancies...---------------------------------------------------

def get_pregnancies(results_folder):
    pregnancy_poll_results = extract_results(
        results_folder,
        module="tlo.methods.contraception",
        key="pregnancy",
        custom_generate_series=(
            lambda df: df.assign(year=pd.to_datetime(df['date']).dt.year).groupby(['year'])['year'].count()
        ))

    mean_pp_pregs = get_mean_and_quants(pregnancy_poll_results)[0]
    lq_pp = get_mean_and_quants(pregnancy_poll_results)[1]
    uq_pp = get_mean_and_quants(pregnancy_poll_results)[2]
    return [mean_pp_pregs, lq_pp, uq_pp]

b_preg = get_pregnancies(baseline_results_folder)
i_preg = get_pregnancies(intervention_results_folder)

fig, ax = plt.subplots()
ax.plot(sim_years, b_preg[0], label='Baseline')
ax.fill_between(sim_years, b_preg[1], b_preg[2], color='b', alpha=.1, label="UI (2.5-92.5)")
ax.plot(sim_years, i_preg[0], label='Intervention')
ax.fill_between(sim_years, i_preg[1], i_preg[2], color='g', alpha=.1, label="UI (2.5-92.5)")
plt.xlabel('Year')
plt.ylabel('Pregnancies (mean)')
plt.title('Mean number of pregnancies for scenarios')
plt.legend()
plt.savefig(f'./outputs/sejjj49@ucl.ac.uk/{graph_location}/pregnancies.png')
plt.show()

# -----------------------------------------------------Total births...------------------------------------------------
def get_births(results_folder):
    births_results = extract_results(
        results_folder,
        module="tlo.methods.demography",
        key="on_birth",
        custom_generate_series=(
            lambda df: df.assign(year=df['date'].dt.year).groupby(['year'])['year'].count()
        ),
    )

    total_births_per_year = get_mean_and_quants(births_results)[0]
    lq_bi = get_mean_and_quants(births_results)[1]
    uq_bi = get_mean_and_quants(births_results)[2]
    return [total_births_per_year, lq_bi, uq_bi]

b_births = get_births(baseline_results_folder)
i_births = get_births(intervention_results_folder)

fig, ax = plt.subplots()
ax.plot(sim_years, b_births[0])
ax.fill_between(sim_years, b_births[1], b_births[2], color='b', alpha=.1, label="UI (2.5-92.5)")
ax.plot(sim_years, i_births[0])
ax.fill_between(sim_years, i_births[1], i_births[2], color='g', alpha=.1, label="UI (2.5-92.5)")
plt.xlabel('Year')
plt.ylabel('Births (mean)')
plt.title('Mean number of Births per Year by Scenario')
plt.legend()
plt.savefig(f'./outputs/sejjj49@ucl.ac.uk/{graph_location}/births.png')
plt.show()


# -------------------------------------------------Completed pregnancies...-------------------------------------------
def get_completed_pregnancies(comps_df, total_births_per_year, results_folder):
    ectopic_mean_numbers_per_year = get_mean_and_quants_from_str_df(comps_df['pregnancy_supervisor'],
                                                                    'ectopic_unruptured')[0]
    ia_mean_numbers_per_year = get_mean_and_quants_from_str_df(comps_df['pregnancy_supervisor'],
                                                               'induced_abortion')[0]
    sa_mean_numbers_per_year = get_mean_and_quants_from_str_df(comps_df['pregnancy_supervisor'],
                                                               'spontaneous_abortion')[0]

    an_stillbirth_results = extract_results(
        results_folder,
        module="tlo.methods.pregnancy_supervisor",
        key="antenatal_stillbirth",
        custom_generate_series=(
            lambda df: df.assign(year=df['date'].dt.year).groupby(['year'])['year'].count()
        ),
    )
    an_still_birth_data = get_mean_and_quants(an_stillbirth_results)

    total_completed_pregnancies_per_year = [a + b + c + d + e for a, b, c, d, e in zip(total_births_per_year,
                                                                                       ectopic_mean_numbers_per_year,
                                                                                       ia_mean_numbers_per_year,
                                                                                       sa_mean_numbers_per_year,
                                                                                       an_still_birth_data[0])]

    return total_completed_pregnancies_per_year

comp_preg_baseline = get_completed_pregnancies(b_comps_dfs, b_births[0], baseline_results_folder)
comp_preg_intervention = get_completed_pregnancies(i_comps_dfs, i_births[0], intervention_results_folder)

# ========================================== INTERVENTION COVERAGE... =================================================

# 2.) Facility delivery
# Total FDR per year (denominator - total births)
def get_facility_delivery(results_folder, total_births_per_year):
    deliver_setting_results = extract_results(
            results_folder,
            module="tlo.methods.labour",
            key="delivery_setting_and_mode",
            custom_generate_series=(
                lambda df_: df_.assign(year=df_['date'].dt.year).groupby(['year', 'facility_type'])['mother'].count()),
            do_scaling=False
        )

    hb_data = get_mean_and_quants_from_str_df(deliver_setting_results, 'home_birth')
    home_birth_rate = [(x / y) * 100 for x, y in zip(hb_data[0], total_births_per_year)]

    hc_data = get_mean_and_quants_from_str_df(deliver_setting_results, 'hospital')
    health_centre_rate = [(x / y) * 100 for x, y in zip(hc_data[0], total_births_per_year)]
    health_centre_lq = [(x / y) * 100 for x, y in zip(hc_data[1], total_births_per_year)]
    health_centre_uq = [(x / y) * 100 for x, y in zip(hc_data[2], total_births_per_year)]

    hp_data = get_mean_and_quants_from_str_df(deliver_setting_results, 'health_centre')
    hospital_rate = [(x / y) * 100 for x, y in zip(hp_data[0], total_births_per_year)]
    hospital_lq = [(x / y) * 100 for x, y in zip(hp_data[1], total_births_per_year)]
    hospital_uq = [(x / y) * 100 for x, y in zip(hp_data[2], total_births_per_year)]

    total_fd_rate = [x + y for x, y in zip(health_centre_rate, hospital_rate)]
    fd_lqs = [x + y for x, y in zip(health_centre_lq, hospital_lq)]
    fd_uqs = [x + y for x, y in zip(health_centre_uq, hospital_uq)]

    return [total_fd_rate, fd_lqs, fd_uqs]


b_fd = get_facility_delivery(baseline_results_folder, b_births[0])
i_fd = get_facility_delivery(intervention_results_folder, i_births[0])

line_graph_with_ci_and_target_rate(b_fd, i_fd, 'Year', '% of total births',
                                   'Proportion of Women Delivering in a Health Facility per Year',
                                   'sba_prop_facility_deliv')


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

line_graph_with_ci_and_target_rate(b_pnc_data[0], i_pnc_data[0], 'Year', '% of total births',
                                   'Proportion of Women post-delivery attending PNC per year', 'pnc_mat')

line_graph_with_ci_and_target_rate(b_pnc_data[1], i_pnc_data[1], 'Year',
                                   '% of total births', 'Proportion of Neonates per year attending PNC',
                                   'pnc_neo')


# ========================================== COMPLICATION/DISEASE RATES.... ===========================================
# ---------------------------------------- Twinning Rate... -----------------------------------------------------------
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

    mean_twin_births = get_mean_and_quants(twins_results)[0]
    total_deliveries = [x - y for x, y in zip(total_births_per_year, mean_twin_births)]
    final_twining_rate = [(x / y) * 100 for x, y in zip(mean_twin_births, total_deliveries)]
    lq_rate = [(x / y) * 100 for x, y in zip(get_mean_and_quants(twins_results)[1], total_deliveries)]
    uq_rate = [(x / y) * 100 for x, y in zip(get_mean_and_quants(twins_results)[2], total_deliveries)]

    return [final_twining_rate, lq_rate, uq_rate]

b_twins = get_twin_data(baseline_results_folder, b_births[0])
l_twins = get_twin_data(intervention_results_folder, i_births[0])

line_graph_with_ci_and_target_rate(b_twins, l_twins, 'Year', 'Rate per 100 pregnancies',
                                   'Yearly trends for Twin Births', 'twin_rate')


# ---------------------------------------- Early Pregnancy Loss... ----------------------------------------------------

b_ectopic_data = get_comp_mean_and_rate('ectopic_unruptured', b_preg[0], b_comps_dfs['pregnancy_supervisor'], 1000)
i_ectopic_data = get_comp_mean_and_rate('ectopic_unruptured', i_preg[0], i_comps_dfs['pregnancy_supervisor'], 1000)

line_graph_with_ci_and_target_rate(b_ectopic_data, i_ectopic_data, 'Year',
                                   'Rate per 100 pregnancies', 'Yearly trends for Ectopic Pregnancy', 'ectopic_rate')

# Spontaneous Abortions....

b_sa_data = get_comp_mean_and_rate('spontaneous_abortion', comp_preg_baseline, b_comps_dfs['pregnancy_supervisor'],
                                   1000)
i_sa_data = get_comp_mean_and_rate('spontaneous_abortion', comp_preg_intervention, i_comps_dfs['pregnancy_supervisor'],
                                   1000)

line_graph_with_ci_and_target_rate(b_sa_data, i_sa_data, 'Year',
                                   'Rate per 1000 completed pregnancies', 'Yearly rate of Miscarriage',
                                   'miscarriage_rate')

# Induced Abortions...
b_ia_data = get_comp_mean_and_rate('induced_abortion', comp_preg_baseline, b_comps_dfs['pregnancy_supervisor'],
                                   1000)
i_ia_data = get_comp_mean_and_rate('induced_abortion', comp_preg_intervention, i_comps_dfs['pregnancy_supervisor'],
                                   1000)

line_graph_with_ci_and_target_rate(b_ia_data, i_ia_data, 'Year',
                                   'Rate per 1000 completed pregnancies', 'Yearly rate of Induced Abortion',
                                   'abortion_rate')

# --------------------------------------------------- Syphilis Rate... ------------------------------------------------
b_syphilis_data = get_comp_mean_and_rate('syphilis', comp_preg_baseline, b_comps_dfs['pregnancy_supervisor'], 1000)
i_syphilis_data = get_comp_mean_and_rate('syphilis', comp_preg_intervention,  i_comps_dfs['pregnancy_supervisor'], 1000)

line_graph_with_ci_and_target_rate(b_syphilis_data, i_syphilis_data, 'Year',
                                   'Rate per 1000 completed pregnancies', 'Yearly rate of Syphilis',
                                   'syphilis_rate')

# ------------------------------------------------ Gestational Diabetes... -------------------------------------------
b_gdm_data = get_comp_mean_and_rate('gest_diab', comp_preg_baseline, b_comps_dfs['pregnancy_supervisor'], 1000)
i_gdm_data = get_comp_mean_and_rate('gest_diab', comp_preg_intervention, i_comps_dfs['pregnancy_supervisor'], 1000)

line_graph_with_ci_and_target_rate(b_gdm_data, i_gdm_data, 'Year',
                                   'Rate per 1000 completed pregnancies', 'Yearly rate of Gestational Diabetes',
                                   'gest_diab_rate')

# ------------------------------------------------ PROM... -----------------------------------------------------------
b_prom_data = get_comp_mean_and_rate('PROM', b_births[0], b_comps_dfs['pregnancy_supervisor'], 1000)
i_prom_data = get_comp_mean_and_rate('PROM', i_births[0], i_comps_dfs['pregnancy_supervisor'], 1000)

line_graph_with_ci_and_target_rate(b_prom_data, i_prom_data, 'Year',
                                   'Rate per 1000 completed pregnancies', 'Yearly rate of Gestational Diabetes',
                                   'gest_diab_rate')

# ---------------------------------------------- Anaemia... ----------------------------------------------------------
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
    no_anaemia_data = get_mean_and_quants_from_str_df(anaemia_results, 'none')
    prevalence_of_anaemia_per_year = [100 - ((x / y) * 100) for x, y in zip(no_anaemia_data[0], total_births_per_year)]
    no_anaemia_lqs = [100 - ((x / y) * 100) for x, y in zip(no_anaemia_data[1], total_births_per_year)]
    no_anaemia_uqs = [100 - ((x / y) * 100) for x, y in zip(no_anaemia_data[2], total_births_per_year)]

    return [prevalence_of_anaemia_per_year, no_anaemia_lqs, no_anaemia_uqs]

b_anaemia_b = get_anaemia_output_at_birth(baseline_results_folder, b_births[0])
i_anaemia_b = get_anaemia_output_at_birth(intervention_results_folder, i_births[0])

line_graph_with_ci_and_target_rate(b_anaemia_b, i_anaemia_b, 'Year', 'Prevalence at birth',
                                       'Yearly prevalence of Anaemia (all severity) at birth',
                                       'anaemia_prev_birth')


def get_anaemia_output_at_delivery(results_folder, total_births_per_year):
    pnc_anaemia = extract_results(
        results_folder,
        module="tlo.methods.postnatal_supervisor",
        key="total_mat_pnc_visits",
        custom_generate_series=(
            lambda df_: df_.assign(year=df_['date'].dt.year).groupby(['year', 'anaemia'])['mother'].count()),
        do_scaling=False
    )

    no_anaemia_data = get_mean_and_quants_from_str_df(pnc_anaemia, 'none')
    prevalence_of_anaemia_per_year = [100 - ((x / y) * 100) for x, y in zip(no_anaemia_data[0], total_births_per_year)]
    no_anaemia_lqs = [100 - ((x / y) * 100) for x, y in zip(no_anaemia_data[1], total_births_per_year)]
    no_anaemia_uqs = [100 - ((x / y) * 100) for x, y in zip(no_anaemia_data[2], total_births_per_year)]

    return [prevalence_of_anaemia_per_year, no_anaemia_lqs, no_anaemia_uqs]


b_anaemia_d = get_anaemia_output_at_delivery(baseline_results_folder, b_births[0])
i_anaemia_d = get_anaemia_output_at_delivery(intervention_results_folder, i_births[0])

line_graph_with_ci_and_target_rate(b_anaemia_d, i_anaemia_d, 'Year', 'Prevalence at birth',
                                   'Yearly prevalence of Anaemia (all severity) at delivery',
                                   'anaemia_prev_delivery')


# ------------------------------------------- Hypertensive disorders -------------------------------------------------
def get_htn_disorders_outputs(comps_df, total_births_per_year):

    output = dict()
    output['gh'] = get_comp_mean_and_rate_across_multiple_dataframes(
        'mild_gest_htn', total_births_per_year, 1000, [comps_df['pregnancy_supervisor'],
                                                       comps_df['postnatal_supervisor']])

    output['sgh'] = get_comp_mean_and_rate_across_multiple_dataframes(
        'severe_gest_htn', total_births_per_year, 1000, [comps_df['pregnancy_supervisor'], comps_df['labour'],
                                                         comps_df['postnatal_supervisor']])

    output['mpe'] = get_comp_mean_and_rate_across_multiple_dataframes(
        'mild_pre_eclamp', total_births_per_year, 1000, [comps_df['pregnancy_supervisor'],
                                                         comps_df['postnatal_supervisor']])

    output['spe'] = get_comp_mean_and_rate_across_multiple_dataframes(
        'severe_pre_eclamp', total_births_per_year, 1000, [comps_df['pregnancy_supervisor'], comps_df['labour'],
                                                           comps_df['postnatal_supervisor']])

    output['ec'] = get_comp_mean_and_rate_across_multiple_dataframes(
        'eclampsia', total_births_per_year, 1000, [comps_df['pregnancy_supervisor'], comps_df['labour'],
                                                   comps_df['postnatal_supervisor']])
    return output


b_htn_disorders = get_htn_disorders_outputs(b_comps_dfs, b_births[0])
i_htn_disorders = get_htn_disorders_outputs(i_comps_dfs, i_births[0])


line_graph_with_ci_and_target_rate(b_htn_disorders['gh'], i_htn_disorders['gh'], 'Year', 'Rate per 1000 births',
                                   'Rate of Gestational Hypertension per Year', 'gest_htn_rate')
line_graph_with_ci_and_target_rate(b_htn_disorders['sgh'], i_htn_disorders['sgh'], 'Year',
                                   'Rate per 1000 births', 'Rate of Severe Gestational Hypertension per Year',
                                   'severe_gest_htn_rate')
line_graph_with_ci_and_target_rate(b_htn_disorders['mpe'], i_htn_disorders['mpe'], 'Year',
                                   'Rate per 1000 births', 'Rate of Mild pre-eclampsia per Year',
                                   'mild_pre_eclampsia_rate')
line_graph_with_ci_and_target_rate(b_htn_disorders['spe'], i_htn_disorders['spe'], 'Year',
                                   'Rate per 1000 births', 'Rate of Severe pre-eclampsia per Year',
                                   'severe_pre_eclampsia_rate')
line_graph_with_ci_and_target_rate(b_htn_disorders['ec'], i_htn_disorders['ec'], 'Year',
                                   'Rate per 1000 births',
                                   'Rate of Eclampsia per Year', 'eclampsia_rate')


#  ---------------------------------------------Placenta praevia... -------------------------------------------------
b_pp_data = get_comp_mean_and_rate('placenta_praevia', b_preg[0], b_comps_dfs['pregnancy_supervisor'], 1000)
i_pp_data = get_comp_mean_and_rate('placenta_praevia', i_preg[0], i_comps_dfs['pregnancy_supervisor'], 1000)


line_graph_with_ci_and_target_rate(b_pp_data, i_pp_data,'Year', 'Rate per 1000 pregnancies',
                                   'Rate of Placenta Praevia per Year', 'praevia_rate')

#  ---------------------------------------------Placental abruption... -------------------------------------------------
b_pa_data = get_comp_mean_and_rate_across_multiple_dataframes('placental_abruption', b_births[0], 1000,
                                                              [b_comps_dfs['pregnancy_supervisor'],
                                                              b_comps_dfs['labour']])
i_pa_data = get_comp_mean_and_rate_across_multiple_dataframes('placental_abruption', i_births[0], 1000,
                                                              [i_comps_dfs['pregnancy_supervisor'],
                                                              i_comps_dfs['labour']])

line_graph_with_ci_and_target_rate(b_pa_data, i_pa_data, 'Year', 'Rate per 1000 births',
                                   'Rate of Placental Abruption per Year', 'abruption_rate')

# --------------------------------------------- Antepartum Haemorrhage... ---------------------------------------------
# Rate of APH/total births (antenatal and labour)

def get_aph_data(comps_df, total_births_per_year):
    mm_aph_data = get_comp_mean_and_rate_across_multiple_dataframes(
        'mild_mod_antepartum_haemorrhage', total_births_per_year, 1000,
        [comps_df['pregnancy_supervisor'], comps_df['labour']])

    s_aph_data = get_comp_mean_and_rate_across_multiple_dataframes(
        'severe_antepartum_haemorrhage', total_births_per_year, 1000,
        [b_comps_dfs['pregnancy_supervisor'], b_comps_dfs['labour']])

    total_aph_rates = [x + y for x, y in zip(mm_aph_data[0], s_aph_data[0])]
    aph_lqs = [x + y for x, y in zip(mm_aph_data[1], s_aph_data[1])]
    aph_uqs = [x + y for x, y in zip(mm_aph_data[2], s_aph_data[2])]

    return [total_aph_rates, aph_lqs, aph_uqs]

b_aph_data = get_aph_data(b_comps_dfs, b_births[0])
i_aph_data = get_aph_data(i_comps_dfs, i_births[0])

line_graph_with_ci_and_target_rate(b_aph_data, i_aph_data, 'Year', 'Rate per 1000 births',
                                   'Rate of Antepartum Haemorrhage per Year', 'aph_rate')


# --------------------------------------------- Preterm birth ... ------------------------------------------------
def get_ptl_data(total_births_per_year, comps_df):
    early_ptl_data = get_comp_mean_and_rate('early_preterm_labour', total_births_per_year, comps_df['labour'], 100)
    late_ptl_data = get_comp_mean_and_rate('late_preterm_labour', total_births_per_year, comps_df['labour'], 100)


    total_ptl_rates = [x + y for x, y in zip(early_ptl_data[0], late_ptl_data[0])]
    ptl_lqs = [x + y for x, y in zip(early_ptl_data[1], late_ptl_data[1])]
    ltl_uqs = [x + y for x, y in zip(early_ptl_data[2], late_ptl_data[2])]

    return [total_ptl_rates, ptl_lqs, ltl_uqs]

b_ptl_data = get_ptl_data(b_births[0], b_comps_dfs)
i_ptl_data = get_ptl_data(i_births[0], i_comps_dfs)

line_graph_with_ci_and_target_rate(b_ptl_data, i_ptl_data, 'Year', 'Proportion of total births',
                                   'Preterm birth rate', 'ptb_rate')


# todo plot early and late seperated

# --------------------------------------------- Post term birth ... -----------------------------------------------
b_potl_data = get_comp_mean_and_rate('post_term_labour', b_births[0], b_comps_dfs['labour'], 100)
i_potl_data = get_comp_mean_and_rate('post_term_labour', i_births[1], i_comps_dfs['labour'], 100)

line_graph_with_ci_and_target_rate(b_potl_data, i_potl_data, 'Year', 'Proportion of total births',
                                   'Post term birth rate', 'potl_rate')

# ------------------------------------------- Antenatal Stillbirth ... -----------------------------------------------
def get_an_stillbirth(results_folder, total_births_per_year):
    an_stillbirth_results = extract_results(
        results_folder,
        module="tlo.methods.pregnancy_supervisor",
        key="antenatal_stillbirth",
        custom_generate_series=(
            lambda df: df.assign(year=df['date'].dt.year).groupby(['year'])['year'].count()
        ),
    )
    an_still_birth_data = get_mean_and_quants(an_stillbirth_results)

    an_sbr_per_year = [(x / y) * 1000 for x, y in zip(an_still_birth_data[0], total_births_per_year)]
    an_sbr_lqs = [(x / y) * 1000 for x, y in zip(an_still_birth_data[1], total_births_per_year)]
    an_sbr_uqs = [(x / y) * 1000 for x, y in zip(an_still_birth_data[2], total_births_per_year)]

    return [an_sbr_per_year, an_sbr_lqs, an_sbr_uqs]

b_an_sbr = get_an_stillbirth(baseline_results_folder, b_births[0])
i_an_sbr = get_an_stillbirth(intervention_results_folder, i_births[0])

line_graph_with_ci_and_target_rate(b_an_sbr, i_an_sbr,'Year','Rate per 1000 births',
                                   'Antenatal Stillbirth Rate per Year', 'sbr_an')


# ------------------------------------------------- Birth weight... --------------------------------------------------
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


b_lbw_data = get_comp_mean_and_rate('low_birth_weight', b_births[0], b_neonatal_comp_dfs['newborn_outcomes'], 100)
i_lbw_data = get_comp_mean_and_rate('low_birth_weight', i_births[0], i_neonatal_comp_dfs['newborn_outcomes'], 100)

line_graph_with_ci_and_target_rate(b_lbw_data, i_lbw_data,'Year',
                                   'Proportion of total births', 'Yearly Prevalence of Low Birth Weight',
                                   'neo_lbw_prev')

b_macro_data = get_comp_mean_and_rate('macrosomia', b_births[0], b_neonatal_comp_dfs['newborn_outcomes'], 100)
i_macro_data = get_comp_mean_and_rate('macrosomia', i_births[0], i_neonatal_comp_dfs['newborn_outcomes'], 100)

line_graph_with_ci_and_target_rate(b_macro_data, i_macro_data, 'Year', 'Proportion of total births',
                                   'Yearly Prevalence of Macrosomia', 'neo_macrosomia_prev')

b_sga_data = get_comp_mean_and_rate('small_for_gestational_age', b_births[0],
                                    b_neonatal_comp_dfs['newborn_outcomes'], 100)
i_sga_data = get_comp_mean_and_rate('small_for_gestational_age', i_births[0],
                                    i_neonatal_comp_dfs['newborn_outcomes'], 100)

line_graph_with_ci_and_target_rate(b_sga_data, i_sga_data, 'Year',
                                   'Proportion of total births', 'Yearly Prevalence of Small for Gestational Age',
                                   'neo_sga_prev')

# --------------------------------------------- Obstructed Labour... --------------------------------------------------
b_ol_data = get_comp_mean_and_rate('obstructed_labour', b_births[0], b_comps_dfs['labour'], 1000)
i_ol_data = get_comp_mean_and_rate('obstructed_labour', i_births[0], i_comps_dfs['labour'], 1000)

line_graph_with_ci_and_target_rate(b_ol_data, i_ol_data, 'Year', 'Rate per 1000 births',
                                   'Obstructed Labour Rate per Year', 'ol_rate')


# --------------------------------------------- Uterine rupture... ---------------------------------------------------
b_ur_data = get_comp_mean_and_rate('uterine_rupture', b_births[0], b_comps_dfs['labour'], 1000)
i_ur_data = get_comp_mean_and_rate('uterine_rupture', i_births[0], i_comps_dfs['labour'], 1000)

line_graph_with_ci_and_target_rate(b_ur_data, i_ur_data, 'Year', 'Rate per 1000 births',
                                   'Rate of Uterine Rupture per Year', 'ur_rate')

# ---------------------------Caesarean Section Rate & Assisted Vaginal Delivery Rate... ------------------------------
def get_delivery_data(results_folder,total_births_per_year):
    delivery_mode = extract_results(
        results_folder,
        module="tlo.methods.labour",
        key="delivery_setting_and_mode",
        custom_generate_series=(
            lambda df_: df_.assign(year=df_['date'].dt.year).groupby(['year', 'mode'])['mother'].count()),
        do_scaling=False
    )

    cs_data = get_comp_mean_and_rate('caesarean_section', total_births_per_year, delivery_mode, 100)
    avd_data = get_comp_mean_and_rate('instrumental', total_births_per_year, delivery_mode, 100)

    return [cs_data, avd_data]

b_delivery_data = get_delivery_data(baseline_results_folder, b_births[0])
i_delivery_data = get_delivery_data(intervention_results_folder, i_births[0])


line_graph_with_ci_and_target_rate(b_delivery_data[0], i_delivery_data[0], 'Year',
                                   'Proportion of total births', 'Caesarean Section Rate per Year',
                                   'caesarean_section_rate')

line_graph_with_ci_and_target_rate(b_delivery_data[1], i_delivery_data[1], 'Year',
                                   'Proportion of total births', 'Assisted Vaginal Delivery Rate per Year', 'avd_rate')


# ------------------------------------------ Maternal Sepsis Rate... --------------------------------------------------
def get_total_sepsis_rates(total_births_per_year, comps_df):
    an_sep_data = get_comp_mean_and_rate('clinical_chorioamnionitis', total_births_per_year,
                                         comps_df['pregnancy_supervisor'], 1000)
    la_sep_data = get_comp_mean_and_rate('sepsis', total_births_per_year, comps_df['labour'], 1000)

    pn_la_sep_data = get_comp_mean_and_rate('sepsis_postnatal', total_births_per_year, comps_df['labour'], 1000)
    pn_sep_data = get_comp_mean_and_rate('sepsis', total_births_per_year, comps_df['postnatal_supervisor'], 1000)

    complete_pn_sep_data = [x + y for x, y in zip(pn_la_sep_data[0], pn_sep_data[0])]
    complete_pn_sep_lq = [x + y for x, y in zip(pn_la_sep_data[1], pn_sep_data[1])]
    complete_pn_sep_up = [x + y for x, y in zip(pn_la_sep_data[2], pn_sep_data[2])]

    total_sep_rates = [x + y + z for x, y, z in zip(an_sep_data[0], la_sep_data[0], complete_pn_sep_data)]
    sep_lq = [x + y + z for x, y, z in zip(an_sep_data[1], la_sep_data[1], complete_pn_sep_lq)]
    sep_uq = [x + y + z for x, y, z in zip(an_sep_data[2], la_sep_data[2], complete_pn_sep_up)]

    return [total_sep_rates, sep_lq, sep_uq]

b_sep_data = get_total_sepsis_rates(b_births[0], b_comps_dfs)
i_sep_data = get_total_sepsis_rates(i_births[0], i_comps_dfs)


line_graph_with_ci_and_target_rate(b_sep_data, i_sep_data, 'Year',
                                   'Rate per 1000 births', 'Rate of Maternal Sepsis per Year', 'sepsis_rate')

# ----------------------------------------- Postpartum Haemorrhage... -------------------------------------------------
def get_pph_data(total_births_per_year, comps_df):
    la_pph_data = get_comp_mean_and_rate('primary_postpartum_haemorrhage', total_births_per_year,
                                         comps_df['labour'], 1000)
    pn_pph_data = get_comp_mean_and_rate('secondary_postpartum_haemorrhage', total_births_per_year,
                                         comps_df['postnatal_supervisor'], 1000)

    total_pph_rates = [x + y for x, y in zip(la_pph_data[0], pn_pph_data[0])]
    pph_lq = [x + y for x, y in zip(la_pph_data[1], pn_pph_data[1])]
    pph_uq = [x + y for x, y in zip(la_pph_data[2], pn_pph_data[2])]

    return [total_pph_rates, pph_lq, pph_uq]

b_pph_data = get_pph_data(b_births[0], b_comps_dfs)
i_pph_data = get_pph_data(i_births[0], i_comps_dfs)

line_graph_with_ci_and_target_rate(b_pph_data, i_pph_data, 'Year', 'Rate per 1000 births',
                                   'Rate of Postpartum Haemorrhage per Year', 'pph_rate')

# ------------------------------------------- Intrapartum Stillbirth ... -----------------------------------------------
def get_ip_stillbirths(results_folder, total_births_per_year):
    ip_stillbirth_results = extract_results(
        results_folder,
        module="tlo.methods.labour",
        key="intrapartum_stillbirth",
        custom_generate_series=(
            lambda df: df.assign(year=df['date'].dt.year).groupby(['year'])['year'].count()
        ),
    )

    ip_still_birth_data = get_mean_and_quants(ip_stillbirth_results)
    ip_sbr_per_year = [(x/y) * 1000 for x, y in zip(ip_still_birth_data[0], total_births_per_year)]
    ip_sbr_lqs = [(x/y) * 1000 for x, y in zip(ip_still_birth_data[1], total_births_per_year)]
    ip_sbr_uqs = [(x/y) * 1000 for x, y in zip(ip_still_birth_data[2], total_births_per_year)]

    return [ip_sbr_per_year, ip_sbr_lqs, ip_sbr_uqs]

b_ip_sbr = get_ip_stillbirths(baseline_results_folder, b_births[0])
i_ip_sbr = get_ip_stillbirths(intervention_results_folder, i_births[0])


line_graph_with_ci_and_target_rate(b_ip_sbr, i_ip_sbr, 'Year',
                                   'Rate per 1000 births', 'Intrapartum Stillbirth Rate per Year', 'sbr_ip')


# ==================================================== NEWBORN OUTCOMES ===============================================
#  ------------------------------------------- Neonatal sepsis (labour & postnatal) -----------------------------------
def get_neonatal_sepsis(total_births_per_year, nb_outcomes_df, nb_outcomes_pn_df):
    early_ns_data = get_comp_mean_and_rate('early_onset_sepsis', total_births_per_year, nb_outcomes_df, 1000)
    early_ns_pn = get_comp_mean_and_rate('early_onset_sepsis', total_births_per_year, nb_outcomes_pn_df, 1000)
    late_ns_data = get_comp_mean_and_rate('late_onset_sepsis', total_births_per_year, nb_outcomes_pn_df, 1000)



    total_ns_rates = [x + y + z for x, y, z in zip(early_ns_data[0], early_ns_pn[0], late_ns_data[0])]
    ns_lqs = [x + y + z for x, y, z in zip(early_ns_data[1], early_ns_pn[1], late_ns_data[1])]
    ns_uqs = [x + y + z for x, y, z in zip(early_ns_data[2], early_ns_pn[2], late_ns_data[2])]

    return [total_ns_rates, ns_lqs, ns_uqs]

b_n_sepsis = get_neonatal_sepsis(b_births[0], b_neonatal_comp_dfs['newborn_outcomes'],
                                 b_neonatal_comp_dfs['newborn_postnatal'])
i_n_sepsis = get_neonatal_sepsis(i_births[0], i_neonatal_comp_dfs['newborn_outcomes'],
                                 i_neonatal_comp_dfs['newborn_postnatal'])

line_graph_with_ci_and_target_rate(b_n_sepsis, i_n_sepsis, 'Year', 'Rate per 1000 births',
                                   'Rate of Neonatal Sepsis per year', 'neo_sepsis_rate')


#  ------------------------------------------- Neonatal encephalopathy -----------------------------------------------

def get_neonatal_encephalopathy(total_births_per_year, nb_outcomes_df):
    mild_data = get_comp_mean_and_rate('mild_enceph', total_births_per_year, nb_outcomes_df, 1000)
    mod_data = get_comp_mean_and_rate('moderate_enceph', total_births_per_year, nb_outcomes_df, 1000)
    sev_data = get_comp_mean_and_rate('severe_enceph', total_births_per_year, nb_outcomes_df, 1000)

    total_enceph_rates = [x + y + z for x, y, z in zip(mild_data[0], mod_data[0], sev_data[0])]
    enceph_lq = [x + y + z for x, y, z in zip(mild_data[1], mod_data[1], sev_data[1])]
    enceph_uq = [x + y + z for x, y, z in zip(mild_data[2], mod_data[2], sev_data[2])]

    return [total_enceph_rates, enceph_lq, enceph_uq]

b_enceph = get_neonatal_encephalopathy(b_births[0], b_neonatal_comp_dfs['newborn_outcomes'])
i_enceph = get_neonatal_encephalopathy(i_births[0], i_neonatal_comp_dfs['newborn_outcomes'])

line_graph_with_ci_and_target_rate(b_enceph, i_enceph, 'Year', 'Rate per 1000 births',
                                   'Rate of Neonatal Encephalopathy per year',
                                   'neo_enceph_rate')

# ----------------------------------------- Respiratory Depression ---------------------------------------------------
b_rd_data = get_comp_mean_and_rate('not_breathing_at_birth', b_births[0], b_neonatal_comp_dfs['newborn_outcomes'], 1000)
i_rd_data = get_comp_mean_and_rate('not_breathing_at_birth', i_births[0], i_neonatal_comp_dfs['newborn_outcomes'], 1000)

line_graph_with_ci_and_target_rate(b_rd_data, i_rd_data, 'Year', 'Rate per 1000 births',
                                   'Rate of Neonatal Respiratory Depression per year', 'neo_resp_depression_rate')

# ----------------------------------------- Respiratory Distress Syndrome --------------------------------------------
def get_rds(la_comps, nb_outcomes_df):
    ept = get_mean_and_quants_from_str_df(la_comps, 'early_preterm_labour')[0]  # todo: should be live births
    lpt = get_mean_and_quants_from_str_df(la_comps, 'late_preterm_labour')[0]
    total_ptbs = [x + y for x, y in zip(ept, lpt)]

    rds_data = get_comp_mean_and_rate('respiratory_distress_syndrome', total_ptbs, nb_outcomes_df, 1000)

    return rds_data

b_rds = get_rds(b_comps_dfs['labour'], b_neonatal_comp_dfs['newborn_outcomes'])
i_rds = get_rds(i_comps_dfs['labour'], i_neonatal_comp_dfs['newborn_outcomes'])


line_graph_with_ci_and_target_rate(b_rds, i_rds, 'Year', 'Rate per 1000 preterm births',
                                   'Rate of Preterm Respiratory Distress Syndrome per year', 'neo_rds_rate')


# ===================================== COMPARING COMPLICATION LEVEL MMR =============================================
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
        b_deaths = get_mean_and_quants_from_str_df(b_death_results, cause)[0]
        i_deaths = get_mean_and_quants_from_str_df(b_death_results, cause)[0]

    elif cause == 'abortion':
        def get_ab_mmr(death_results):
            ia_deaths = get_mean_and_quants_from_str_df(death_results, 'induced_abortion')[0]
            sa_deaths = get_mean_and_quants_from_str_df(death_results, 'spontaneous_abortion')[0]
            deaths = [x + y for x, y in zip(ia_deaths, sa_deaths)]
            return deaths
        b_deaths = get_ab_mmr(b_death_results)
        i_deaths = get_ab_mmr(i_death_results)

    elif cause == 'severe_pre_eclampsia':
        def get_htn_mmr(death_results):
            spe_deaths = get_mean_and_quants_from_str_df(death_results, 'severe_pre_eclampsia')[0]
            ec_deaths = get_mean_and_quants_from_str_df(death_results, 'eclampsia')[0]
            sgh_deaths = get_mean_and_quants_from_str_df(death_results, 'severe_gestational_hypertension')[0]
            deaths = [x + y + z for x, y, z in zip(spe_deaths, ec_deaths, sgh_deaths)]
            return deaths
        b_deaths = get_htn_mmr(b_death_results)
        i_deaths = get_htn_mmr(i_death_results)

    elif cause == 'postpartum_haemorrhage':
        def get_pph_mmr(death_results):
            p_deaths = get_mean_and_quants_from_str_df(death_results, 'postpartum_haemorrhage')[0]
            s_deaths = get_mean_and_quants_from_str_df(death_results, 'secondary_postpartum_haemorrhage')[0]
            deaths = [x + y for x, y in zip(p_deaths, s_deaths)]
            return deaths
        b_deaths = get_pph_mmr(b_death_results)
        i_deaths = get_pph_mmr(i_death_results)

    elif cause == 'sepsis':
        def get_sep_mmr(death_results):
            a_deaths = get_mean_and_quants_from_str_df(death_results, 'antenatal_sepsis')[0]
            i_deaths = get_mean_and_quants_from_str_df(death_results, 'intrapartum_sepsis')[0]
            p_deaths = get_mean_and_quants_from_str_df(death_results, 'postpartum_sepsis')[0]
            deaths = [x + y + z for x, y, z in zip(a_deaths, i_deaths, p_deaths)]
            return deaths
        b_deaths = get_sep_mmr(b_death_results)
        i_deaths = get_sep_mmr(i_death_results)

    b_mmr = [(x / y) * 100000 for x, y in zip(b_deaths, b_births[0])]
    i_mmr = [(x / y) * 100000 for x, y in zip(i_deaths, i_births[0])]

    plt.plot(sim_years, b_mmr, 'o-g', label="Baseline", color='deepskyblue')
    plt.plot(sim_years, i_mmr, 'o-g', label="Intervention", color='darkseagreen')
    plt.xlabel('Year')
    plt.ylabel('Deaths per 100,000 births')
    plt.title(f'Maternal Mortality Ratio per Year for {cause} by Scenario')
    plt.legend()
    plt.savefig(f'./outputs/sejjj49@ucl.ac.uk/{graph_location}/mmr/mmr_{cause}.png')
    plt.show()

# ===================================== COMPARING COMPLICATION LEVEL NMR =============================================
simplified_causes_neo = ['prematurity', 'encephalopathy', 'neonatal_sepsis', 'neonatal_respiratory_depression']


for cause in simplified_causes_neo:
    if (cause == 'encephalopathy') or (cause == 'neonatal_respiratory_depression'):
        b_deaths = get_mean_and_quants_from_str_df(b_death_results, cause)[0]
        i_deaths = get_mean_and_quants_from_str_df(i_death_results, cause)[0]

    elif cause == 'neonatal_sepsis':
        def get_neo_sep_deaths(death_results):
            early1 = get_mean_and_quants_from_str_df(death_results, 'early_onset_neonatal_sepsis')[0]
            early2 = get_mean_and_quants_from_str_df(death_results, 'early_onset_sepsis')[0]
            late = get_mean_and_quants_from_str_df(death_results, 'late_onset_sepsis')[0]
            deaths = [x + y + z for x, y, z in zip(early1, early2, late)]
            return deaths

        b_deaths = get_neo_sep_deaths(b_death_results)
        i_deaths = get_neo_sep_deaths(i_death_results)

    elif cause == 'prematurity':
        def get_pt_deaths(death_results):
            rds_deaths = get_mean_and_quants_from_str_df(death_results, 'respiratory_distress_syndrome')[0]
            other_deaths = get_mean_and_quants_from_str_df(death_results, 'preterm_other')[0]
            deaths = [x + y for x, y in zip(rds_deaths, other_deaths)]
            return deaths

        b_deaths = get_pt_deaths(b_death_results)
        i_deaths = get_pt_deaths(i_death_results)

    b_nmr = [(x / y) * 1000 for x, y in zip(b_deaths, b_births[0])]
    i_nmr = [(x / y) * 1000 for x, y in zip(i_deaths, i_births[0])]

    plt.plot(sim_years, b_nmr, 'o-g', label="Baseline", color='deepskyblue')
    plt.plot(sim_years, i_nmr, 'o-g', label="Intervention", color='darkseagreen')
    plt.xlabel('Year')
    plt.ylabel('Deaths per 1000 births')
    plt.title(f'Neonatal Mortality Ratio per Year for {cause} by Scenario')
    plt.legend()
    plt.savefig(f'./outputs/sejjj49@ucl.ac.uk/{graph_location}/nmr/nmr_{cause}.png')
    plt.show()




