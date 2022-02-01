from pathlib import Path
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from tlo.analysis.utils import (
    extract_results,
    get_scenario_outputs, create_pickles_locally
)
from scripts.maternal_perinatal_analyses import analysis_utility_functions
# from tlo.methods.demography import get_scaling_factor


# %% Declare the name of the file that specified the scenarios used in this run.
baseline_scenario_filename = 'baseline_anc_scenario.py'
intervention_scenario_filename = 'increased_anc_scenario.py'

# %% Declare usual paths:
outputspath = Path('./outputs/sejjj49@ucl.ac.uk/')
graph_location = 'output_graphs_60k_increased_anc_scenario-2022-01-31T134117Z'
rfp = Path('./resources')

# Find results folder (most recent run generated using that scenario_filename)
baseline_results_folder = get_scenario_outputs(baseline_scenario_filename, outputspath)[-1]
intervention_results_folder = get_scenario_outputs(intervention_scenario_filename, outputspath)[-1]

# create_pickles_locally(baseline_results_folder)  # if not created via batch
# create_pickles_locally(intervention_results_folder)  # if not created via batch

sim_years = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027,
             2028, 2029, 2030]
intervention_years = [2020, 2021, 2022, 2023, 2024, 2025]


def get_mean_and_quants_from_str_df(df, complication):
    yearly_mean_number = list()
    yearly_lq = list()
    yearly_uq = list()
    for year in intervention_years:
        if complication in df.loc[year].index:
            yearly_mean_number.append(df.loc[year, complication].mean())
            yearly_lq.append(df.loc[year, complication].quantile(0.025))
            yearly_uq.append(df.loc[year, complication].quantile(0.925))
        else:
            yearly_mean_number.append(0)
            yearly_lq.append(0)
            yearly_uq.append(0)

    return [yearly_mean_number, yearly_lq, yearly_uq]


def get_mean_and_quants(df, sim_years):
    year_means = list()
    lower_quantiles = list()
    upper_quantiles = list()

    for year in sim_years:
        if year in df.index:
            year_means.append(df.loc[year].mean(axis=1).iloc[0])
            lower_quantiles.append(df.loc[year].iloc[0].quantile(0.025))
            upper_quantiles.append(df.loc[year].iloc[0].quantile(0.925))
        else:
            year_means.append(0)
            lower_quantiles.append(0)
            lower_quantiles.append(0)

    return [year_means, lower_quantiles, upper_quantiles]


# GET BIRTHS...
def get_total_births_per_year(folder):
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


baseline_births = get_total_births_per_year(baseline_results_folder)
intervention_births = get_total_births_per_year(intervention_results_folder)


# ===============================================CHECK INTERVENTION ===================================================
def get_anc_4_coverage(folder):
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
                mean = x.mean()
                lq = x.quantile(0.025)
                uq = x.quantile(0.925)
                anc_count_df.at[row, year] = [mean, lq, uq]
            else:
                anc_count_df.at[row, year] = [0, 0, 0]

    yearly_anc4_rates = list()
    anc4_lqs = list()
    anc4_uqs = list()

    for year in intervention_years:
        anc_total = 0
        four_or_more_visits = 0

        for row in anc_count_df[year]:
            anc_total += row[0]

        four_or_more_visits_slice = anc_count_df.loc[anc_count_df.index > 3]
        f_lqs = 0
        f_uqs = 0
        for row in four_or_more_visits_slice[year]:
            four_or_more_visits += row[0]
            f_lqs += row[1]
            f_uqs += row[2]

        yearly_anc4_rates.append((four_or_more_visits / anc_total) * 100)
        anc4_lqs.append((f_lqs / anc_total) * 100)
        anc4_uqs.append((f_uqs / anc_total) * 100)

    return[yearly_anc4_rates, anc4_lqs, anc4_uqs]


baseline_anc4_coverage = get_anc_4_coverage(baseline_results_folder)
intervention_anc4_coverage = get_anc_4_coverage(intervention_results_folder)

fig, ax = plt.subplots()
ax.plot(intervention_years, baseline_anc4_coverage[0], label="Baseline (mean)", color='deepskyblue')
ax.fill_between(intervention_years, baseline_anc4_coverage[1], baseline_anc4_coverage[2], color='b', alpha=.1,
                label="UI (2.5-92.5)")

ax.plot(intervention_years, intervention_anc4_coverage[0], label="Intervention (mean)", color='olivedrab')
ax.fill_between(intervention_years, intervention_anc4_coverage[1], intervention_anc4_coverage[2], color='g', alpha=.1,
                label="UI (2.5-92.5)")

plt.xlabel('Year')
plt.ylabel('Coverage of ANC4')
plt.title('ANC4 coverage in baseline and intervention scenarios from 2020')
plt.legend()
plt.savefig(f'./outputs/sejjj49@ucl.ac.uk/{graph_location}/anc4_intervention_coverage.png')
plt.show()


#  ------------------------------------------ MATERNAL AND NEONATAL DEATH ---------------------------------------------
def get_yearly_death_data(folder, births):
    death_results_labels = extract_results(
        folder,
        module="tlo.methods.demography",
        key="death",
        custom_generate_series=(
            lambda df: df.assign(year=df['date'].dt.year).groupby(['year', 'label'])['year'].count()),
        do_scaling=True)

    mmr = analysis_utility_functions.get_comp_mean_and_rate('Maternal Disorders', births, death_results_labels, 100000,
                                                            intervention_years)
    nmr = analysis_utility_functions.get_comp_mean_and_rate('Neonatal Disorders', births, death_results_labels, 1000,
                                                            intervention_years)
    crude_m_deaths = get_mean_and_quants_from_str_df(death_results_labels, 'Maternal Disorders')
    crude_n_deaths = get_mean_and_quants_from_str_df(death_results_labels, 'Neonatal Disorders')

    return {'mmr': mmr,
            'nmr': nmr,
            'crude_m_deaths': crude_m_deaths,
            'crude_n_deaths': crude_n_deaths}


baseline_death_data = get_yearly_death_data(baseline_results_folder, baseline_births)
intervention_death_data = get_yearly_death_data(intervention_results_folder, intervention_births)


def get_mmr_nmr_graphs(bdata, idata, group):
    fig, ax = plt.subplots()
    ax.plot(intervention_years, bdata[0], label="Baseline (mean)", color='deepskyblue')
    ax.fill_between(intervention_years, bdata[1], bdata[2], color='b', alpha=.1)
    ax.plot(intervention_years, idata[0], label="Intervention (mean)", color='olivedrab')
    ax.fill_between(intervention_years, idata[1], idata[2], color='g', alpha=.1)
    # ax.set(ylim=(0, 800))
    plt.xlabel('Year')
    if group == 'Neonatal':
        plt.ylabel("Deaths per 1000 live births")
    else:
        plt.ylabel("Deaths per 100,000 live births")
    plt.title(f'{group} Mortality Ratio per Year at Baseline and Under Intervention')
    plt.legend()
    plt.savefig(f'./outputs/sejjj49@ucl.ac.uk/{graph_location}/{group}_mr_int.png')
    plt.show()


get_mmr_nmr_graphs(baseline_death_data['mmr'], intervention_death_data['mmr'], 'Maternal')
get_mmr_nmr_graphs(baseline_death_data['nmr'], intervention_death_data['nmr'], 'Neonatal')


def get_crude_death_graphs_graphs(b_deaths, i_deaths, group):
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
    plt.savefig(f'./outputs/sejjj49@ucl.ac.uk/{graph_location}/{group}_crude_deaths_comparison.png')
    plt.show()


get_crude_death_graphs_graphs(baseline_death_data['crude_m_deaths'], intervention_death_data['crude_m_deaths'],
                              'Maternal')
get_crude_death_graphs_graphs(baseline_death_data['crude_n_deaths'], intervention_death_data['crude_n_deaths'],
                              'Neonatal')


# STILLBIRTH # todo: happy with scaling?
def get_yearly_sbr_data(folder, births):
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

    an_still_birth_data = analysis_utility_functions.get_mean_and_quants(an_stillbirth_results, intervention_years)
    ip_still_birth_data = analysis_utility_functions.get_mean_and_quants(ip_stillbirth_results, intervention_years)

    crude_sb = [x + y for x, y in zip(an_still_birth_data[0], ip_still_birth_data[0])]
    crude_sb_lqs = [x + y for x, y in zip(an_still_birth_data[1], ip_still_birth_data[1])]
    crude_sb_uqs = [x + y for x, y in zip(an_still_birth_data[2], ip_still_birth_data[2])]

    total_sbr = [((x + y) / z) * 1000 for x, y, z in zip(an_still_birth_data[0], ip_still_birth_data[0], births)]
    total_lqs = [((x + y) / z) * 1000 for x, y, z in zip(an_still_birth_data[1], ip_still_birth_data[1], births)]
    total_uqs = [((x + y) / z) * 1000 for x, y, z in zip(an_still_birth_data[2], ip_still_birth_data[2], births)]

    return {'sbr': [total_sbr, total_lqs, total_uqs],
            'crude_sb': [crude_sb, crude_sb_lqs, crude_sb_uqs]}

baseline_sb_data = get_yearly_sbr_data(baseline_results_folder, baseline_births)
intervention_sb_data = get_yearly_sbr_data(intervention_results_folder, intervention_births)

fig, ax = plt.subplots()
ax.plot(intervention_years, baseline_sb_data['sbr'][0], label="Baseline (mean)", color='deepskyblue')
ax.fill_between(intervention_years, baseline_sb_data['sbr'][1], baseline_sb_data['sbr'][2], color='b', alpha=.1)
ax.plot(intervention_years, intervention_sb_data['sbr'][0], label="Intervention (mean)", color='olivedrab')
ax.fill_between(intervention_years, intervention_sb_data['sbr'][1], intervention_sb_data['sbr'][2], color='g', alpha=.1)
plt.xlabel('Year')
plt.ylabel("Stillbirths per 1000 live births")
plt.title('Stillbirth Rate per Year at Baseline and Under Intervention')
plt.legend()
plt.savefig(f'./outputs/sejjj49@ucl.ac.uk/{graph_location}/sbr_int.png')
plt.show()

b_sb_ci = [(x - y) / 2 for x, y in zip(baseline_sb_data['crude_sb'][2], baseline_sb_data['crude_sb'][1])]
i_sb_ci = [(x - y) / 2 for x, y in zip(intervention_sb_data['crude_sb'][2], intervention_sb_data['crude_sb'][1])]

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
plt.savefig(f'./outputs/sejjj49@ucl.ac.uk/{graph_location}/crude_stillbirths_comparison.png')
plt.show()


# =================================================== DALYS ==========================================================
def get_dalys_from_scenario(results_folder):
    dalys_stacked = extract_results(
        results_folder,
        module="tlo.methods.healthburden",
        key="dalys_stacked",
        custom_generate_series=(
            lambda df: df.drop(
                columns='date').groupby(['year']).sum().stack()),
        do_scaling=True)

    def extract_dalys_tlo_model(group):
        stacked_dalys = list()
        stacked_dalys_lq = list()
        stacked_dalys_uq = list()

        for year in intervention_years:
            if year in dalys_stacked.index:
                stacked_dalys.append(dalys_stacked.loc[year, f'{group} Disorders'].mean())
                stacked_dalys_lq.append(dalys_stacked.loc[year, f'{group} Disorders'].quantile(0.025))
                stacked_dalys_uq.append(dalys_stacked.loc[year, f'{group} Disorders'].quantile(0.925))

        return [stacked_dalys, stacked_dalys_lq, stacked_dalys_uq]


    maternal_dalys = extract_dalys_tlo_model('Maternal')
    neonatal_dalys = extract_dalys_tlo_model('Neonatal')

    return [maternal_dalys, neonatal_dalys]

baseline_dalys = get_dalys_from_scenario(baseline_results_folder)
baseline_maternal_dalys = baseline_dalys[0]
baseline_neonatal_dalys = baseline_dalys[1]

intervention_dalys = get_dalys_from_scenario(intervention_results_folder)
intervention_maternal_dalys = intervention_dalys[0]
intervention_neonatal_dalys = intervention_dalys[1]


def get_daly_graphs(group, bl_dalys, int_dalys):
    fig, ax = plt.subplots()
    ax.plot(intervention_years, bl_dalys[0], label=f"{group} Baseline DALYs", color='deepskyblue')
    ax.fill_between(intervention_years, bl_dalys[1], bl_dalys[2], color='b', alpha=.1)

    ax.plot(intervention_years, int_dalys[0], label=f"{group} Intervention DALYs", color='olivedrab')
    ax.fill_between(intervention_years, int_dalys[1], int_dalys[2], color='g', alpha=.1)
    plt.xlabel('Year')
    plt.ylabel("Disability Adjusted Life Years (stacked)")
    plt.title(f'Total DALYs per Year Attributable to {group} disorders')
    plt.legend()
    plt.savefig(f'./outputs/sejjj49@ucl.ac.uk/{graph_location}/{group}_dalys_stacked.png')
    plt.show()

get_daly_graphs('Maternal', baseline_maternal_dalys, intervention_maternal_dalys)
get_daly_graphs('Neonatal', baseline_neonatal_dalys, intervention_neonatal_dalys)


# =============================================  COSTS/HEALTH SYSTEM =================================================
# 1.) Healthcare worker time cost
# 2.) Consumables cost
# 3.) Approximated total cost
# 4.) Approximated total cost per pregnancy
# Cost per daly averted?


"""
# HEALTH CARE WORKER TIME
# todo: not sure if we can extract this yet from master....
capacity = extract_results(
        baseline_results_folder,
        module="tlo.methods.healthsystem",
        key="Capacity",
        custom_generate_series=(
            lambda df: df.assign(year=df['date'].dt.year).groupby(['year'])['Frac_Time_Used_Overall'].mean()
        ),
    )

# cap = output['tlo.methods.healthsystem']['Capacity'].copy()
# cap["date"] = pd.to_datetime(cap["date"])
# cap = cap.set_index('date')
#
# frac_time_used = cap['Frac_Time_Used_Overall']
# cap = cap.drop(columns=['Frac_Time_Used_Overall'])
#
# # Plot Fraction of total time of health-care-workers being used
# frac_time_used.plot()
# plt.title("Fraction of total health-care worker time being used")
# plt.xlabel("Date")
# #plt.savefig(make_file_name('HSI_Frac_time_used'))
# plt.show()
#
# # %% Breakdowns by HSI:
# hsi = output['tlo.methods.healthsystem']['HSI_Event'].copy()
# hsi["date"] = pd.to_datetime(hsi["date"])
# hsi["month"] = hsi["date"].dt.month
# # Reduce TREATMENT_ID to the originating module
# hsi["Module"] = hsi["TREATMENT_ID"].str.split('_').apply(lambda x: x[0])
#
# # Plot the HSI that are taking place, by month, in a a particular year
# year = 2012
# evs = hsi.loc[hsi.date.dt.year == year]\
#     .groupby(by=['month', 'Module'])\
#     .size().reset_index().rename(columns={0: 'count'})\
#     .pivot_table(index='month', columns='Module', values='count', fill_value=0)
# evs *= scaling_factor
#
# evs.plot.bar(stacked=True)
# plt.title(f"HSI by Module, per Month (year {year})")
# plt.ylabel('Total per month')
# #plt.savefig(make_file_name('HSI_per_module_per_month'))
# plt.show()
#
# # Plot the breakdown of all HSI, over all the years
# evs = hsi.groupby(by=['Module'])\
#     .size().rename(columns={0: 'count'}) * scaling_factor
# evs.plot.pie()
# plt.title(f"HSI by Module")
# #plt.savefig(make_file_name('HSI_per_module'))
# plt.show()




# =================================================== CONSUMABLE COST =================================================



draws = [0, 1, 2, 3, 4]
resourcefilepath = Path("./resources/healthsystem/consumables/")
consumables_df = pd.read_csv(Path(resourcefilepath) / 'ResourceFile_Consumables.csv')

# TODO: this should be scaled to the correct population size?
# todo: also so slow...


def get_cons_cost_per_year(results_folder):
    # Create df that replicates the 'extracted' df
    total_cost_per_draw_per_year = pd.DataFrame(columns=[draws], index=[intervention_years])

    # Loop over each draw
    for draw in draws:
        # Load df, add year column and select only ANC interventions
        draw_df = load_pickled_dataframes(results_folder, draw=draw)

        cons = draw_df['tlo.methods.healthsystem']['Consumables']
        cons['year'] = cons['date'].dt.year
        total_anc_cons = cons.loc[cons.TREATMENT_ID.str.contains('AntenatalCare')]
        anc_cons = total_anc_cons.loc[total_anc_cons.year >= intervention_years[0]]

        cons_df_for_this_draw = pd.DataFrame(index=[intervention_years])

        # Loop over each year
        for year in intervention_years:
            # Select the year of interest
            year_df = anc_cons.loc[anc_cons.year == year]

            # For each row (hsi) in that year we unpack the dictionary
            for row in year_df.index:
                cons_dict = year_df.at[row, 'Item_Available']
                cons_dict = eval(cons_dict)

                # todo: check this works where there are muliple dicts
                # For each dictionary
                for k, v in cons_dict.items():
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
            total_cost_per_draw_per_year.at[index, draw] = cons_df_for_this_draw.loc[index].sum()

    final_cost_data = get_mean_and_quants(total_cost_per_draw_per_year, intervention_years)
    return final_cost_data


#baseline_cost_data = get_cons_cost_per_year(baseline_results_folder)
#intervention_cost_data = get_cons_cost_per_year(intervention_results_folder)

#fig, ax = plt.subplots()
#ax.plot(intervention_years, baseline_cost_data[0], label="Baseline (mean)", color='deepskyblue')
#ax.fill_between(intervention_years, baseline_cost_data[1], baseline_cost_data[2], color='b', alpha=.1)
#ax.plot(intervention_years, intervention_cost_data[0], label="Intervention (mean)", color='olivedrab')
#ax.fill_between(intervention_years, intervention_cost_data[1], intervention_cost_data[2], color='g', alpha=.1)
#plt.xlabel('Year')
#plt.ylabel("Total Cost (USD)")
#plt.title('Total Cost Attributable To Antenatal Care Per Year (in USD) (unscaled)')
#plt.legend()
# plt.savefig(f'./outputs/sejjj49@ucl.ac.uk/{graph_location}/COST.png')
#plt.show()

# (DALYS)



# ======================================================= ANC 4 ======================================================

# SCENARIO ONE: BASELINE ANC 4 COVERAGE VERSES 90% (UHC) EPMM COVERAGE (CONSUMABLES CONSTRAINED, SQUEEZE CONSTRAINED)

# MATERNAL DEATH
# STILLBIRTH
# NEONATAL DEATH
# HEALTH CARE WORKER TIME
# CONSUMABLES
# (DALYS)

# SCENARIO TWO: BASELINE ANC 4 COVERAGE VERSES 90% (UHC) EPMM COVERAGE (NO CONSUMABLES, QUALITY OR SQUEEZE CONTRAINTS
# IN THE COMPARATOR)

# MATERNAL DEATH
# STILLBIRTH
# NEONATAL DEATH
# HEALTH CARE WORKER TIME
# CONSUMABLES
# (DALYS)

# ======================================================= ANC 8 ======================================================

# SCENARIO ONE: BASELINE ANC 8 COVERAGE VERSES 50% COVERAGE (CONSUMABLES CONSTRAINED, SQUEEZE CONSTRAINED)

# MATERNAL DEATH
# STILLBIRTH
# NEONATAL DEATH
# HEALTH CARE WORKER TIME
# CONSUMABLES
# (DALYS)

# SCENARIO TWO: BASELINE ANC 8 COVERAGE VERSES 50% COVERAGE (NO CONSUMABLES, QUALITY OR SQUEEZE CONTRAINTS
# IN THE COMPARATOR)

# MATERNAL DEATH
# STILLBIRTH
# NEONATAL DEATH
# HEALTH CARE WORKER TIME
# CONSUMABLES
# (DALYS)

# SCENARIO THREE: BASELINE ANC 8 COVERAGE VERSES 90% (UHC) EPMM COVERAGE( CONSUMABLES CONSTRAINED, SQUEEZE CONSTRAINED)

# MATERNAL DEATH
# STILLBIRTH
# NEONATAL DEATH
# HEALTH CARE WORKER TIME
# CONSUMABLES
# (DALYS)

# SCENARIO FOUR: BASELINE ANC 8 COVERAGE VERSES 90% (UHC) EPMM COVERAGE (NO CONSUMABLES, QUALITY OR SQUEEZE CONTRAINTS
# IN THE COMPARATOR)

# MATERNAL DEATH
# STILLBIRTH
# NEONATAL DEATH
# HEALTH CARE WORKER TIME
# CONSUMABLES
# (DALYS)
"""
