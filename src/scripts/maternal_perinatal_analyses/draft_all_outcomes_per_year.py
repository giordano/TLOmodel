from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from tlo.analysis.utils import (
    extract_results, extract_str_results,
    get_scenario_outputs, create_pickles_locally, summarize
)

# %% Declare the name of the file that specified the scenarios used in this run.
scenario_filename = 'calibration_run_all_modules.py'  # <-- update this to look at other results

# %% Declare usual paths:
outputspath = Path('./outputs/sejjj49@ucl.ac.uk/')
graph_location = 'output graphs 11_08_21 50k_10yr10r_(calibration_run_all_modules-2021-08-10T163833Z)'
rfp = Path('./resources')

# Find results folder (most recent run generated using that scenario_filename)
results_folder = get_scenario_outputs(scenario_filename, outputspath)[-1]
#create_pickles_locally(results_folder)  # if not created via batch


# Enter the years the simulation has ran for here?
sim_years = [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]
# todo: replace with something more clever at some point

# ============================================HELPER FUNCTIONS... =====================================================

def get_modules_maternal_complication_dataframes(module):
    complications_df = extract_results(
        results_folder,
        module=f"tlo.methods.{module}",
        key="maternal_complication",
        custom_generate_series=(
            lambda df_: df_.assign(year=df_['date'].dt.year).groupby(['year', 'type'])['person'].count()),
        do_scaling=False
    )

    return complications_df

#  COMPLICATION DATA FRAMES....
an_comps = get_modules_maternal_complication_dataframes('pregnancy_supervisor')
la_comps = get_modules_maternal_complication_dataframes('labour')
pn_comps = get_modules_maternal_complication_dataframes('postnatal_supervisor')

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
    plt.plot(sim_years, model_rate, 'o-g', label="Model rate", color='steelblue')
    plt.plot(sim_years, target_rate, 'o-g', label="Target rate", color='darkseagreen')
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.title(title)
    plt.legend()
    ##plt.savefig(f'./outputs/sejjj49@ucl.ac.uk/{graph_location}/{file_name}.png')
    plt.show()


def simple_bar_chart(model_rates, x_title, y_title, title, file_name):
    bars = sim_years
    x_pos = np.arange(len(bars))
    plt.bar(x_pos, model_rates)
    plt.xticks(x_pos, bars)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.title(title)
    ##plt.savefig(f'./outputs/sejjj49@ucl.ac.uk/{graph_location}/{file_name}.png')
    plt.show()


def line_graph_with_ci_and_target_rate(mean_list, lq_list, uq_list, target_rate, x_label, y_label, title, file_name):
    fig, ax = plt.subplots()
    ax.plot(sim_years, mean_list)
    ax.fill_between(sim_years, lq_list, uq_list, color='b', alpha=.1)
    plt.plot(sim_years, target_rate, 'o-g', label="Target rate", color='darkseagreen')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    ##plt.savefig(f'./outputs/sejjj49@ucl.ac.uk/{graph_location}/{file_name}.png')
    plt.show()


# ============================================  DENOMINATORS... ======================================================
# ---------------------------------------------Total_pregnancies...---------------------------------------------------
pregnancy_poll_results = extract_results(
    results_folder,
    module="tlo.methods.contraception",
    key="pregnant_at_age",
    custom_generate_series=(
        lambda df: df.assign(year=pd.to_datetime(df['date']).dt.year).groupby(['year'])['year'].count()
    ))

contraception_failure = extract_results(
    results_folder,
    module="tlo.methods.contraception",
    key="fail_contraception",
    custom_generate_series=(
        lambda df: df.assign(year=pd.to_datetime(df['date']).dt.year).groupby(['year'])['year'].count()
    ))

mean_pp_pregs = get_mean_and_quants(pregnancy_poll_results)[0]
mean_cf_pregs = get_mean_and_quants(contraception_failure)[0]
total_pregnancies_per_year = [x + y for x, y in zip(mean_pp_pregs, mean_cf_pregs)]

lq_pp = get_mean_and_quants(pregnancy_poll_results)[1]
lq_cf = get_mean_and_quants(contraception_failure)[1]
uq_pp = get_mean_and_quants(pregnancy_poll_results)[2]
uq_cf = get_mean_and_quants(contraception_failure)[2]
total_lq = [x + y for x, y in zip(lq_pp, lq_cf)]
total_uq = [x + y for x, y in zip(uq_pp, uq_cf)]


fig, ax = plt.subplots()
ax.plot(sim_years, total_pregnancies_per_year)
ax.fill_between(sim_years, total_lq, total_uq, color='b', alpha=.1)
plt.xlabel('Year')
plt.ylabel('Pregnancies (mean)')
plt.title('Mean number of pregnancies')
##plt.savefig(f'./outputs/sejjj49@ucl.ac.uk/{graph_location}/pregnancies.png')
plt.show()

# -----------------------------------------------------Total births...------------------------------------------------
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

fig, ax = plt.subplots()
ax.plot(sim_years, total_births_per_year)
ax.fill_between(sim_years, lq_bi, uq_bi, color='b', alpha=.1)
plt.xlabel('Year')
plt.ylabel('Births (mean)')
plt.title('Mean number of Births per Year')
##plt.savefig(f'./outputs/sejjj49@ucl.ac.uk/{graph_location}/births.png')
plt.show()

# todo: some testing looking at live births vs total births...


# -------------------------------------------------Completed pregnancies...-------------------------------------------
ectopic_mean_numbers_per_year = get_mean_and_quants_from_str_df(an_comps, 'ectopic_unruptured')[0]
ia_mean_numbers_per_year = get_mean_and_quants_from_str_df(an_comps, 'induced_abortion')[0]
sa_mean_numbers_per_year = get_mean_and_quants_from_str_df(an_comps, 'spontaneous_abortion')[0]

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

# ========================================== INTERVENTION COVERAGE... =================================================
# 1.) Antenatal Care... # TODO: THIS COULD CERTAINLY BE SIMPLIFIED
# Mean proportion of women (across draws) who have given birth that have attended ANC1, ANC4+ and ANC8+ per year...
results = extract_results(
    results_folder,
    module="tlo.methods.care_of_women_during_pregnancy",
    key="anc_count_on_birth",
    custom_generate_series=(
        lambda df_: df_.assign(year=df_['date'].dt.year).groupby(['year', 'total_anc'])['person_id'].count()),
    do_scaling=False
)

anc_count_df = pd.DataFrame(columns=sim_years, index=[0, 1, 2, 3, 4, 5, 6, 7, 8])

# get yearly outputs
for year in sim_years:
    for row in anc_count_df.index:
        x = results.loc[year, row]
        mean = x.mean()
        lq = x.quantile(0.025)
        uq = x.quantile(0.925)
        anc_count_df.at[row, year] = [mean, lq, uq]

yearly_anc1_rates = list()
anc1_lqs = list()
anc1_uqs = list()
yearly_anc4_rates = list()
anc4_lqs = list()
anc4_uqs = list()
yearly_anc8_rates = list()
anc8_lqs = list()
anc8_uqs = list()


for year in sim_years:
    anc_total = 0
    four_or_more_visits = 0
    eight_or_more_visits = 0

    for row in anc_count_df[year]:
        anc_total += row[0]

    yearly_anc1_rates.append(100 - ((anc_count_df.at[0, year][0] / anc_total) * 100))
    anc1_lqs.append(100 - ((anc_count_df.at[0, year][1] / anc_total) * 100))
    anc1_uqs.append(100 - ((anc_count_df.at[0, year][2] / anc_total) * 100))

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

    eight_or_more_visits_slice = anc_count_df.loc[anc_count_df.index > 7]
    for row in eight_or_more_visits_slice[year]:
        eight_or_more_visits += row[0]

    yearly_anc8_rates.append((eight_or_more_visits / anc_total) * 100)

target_rate_an = list()
for year in sim_years:
    target_rate_an.append(95)

target_rate_anc4 = list()
for year in sim_years:
    if year < 2015:
        target_rate_anc4.append(46)
    else:
        target_rate_anc4.append(51)

line_graph_with_ci_and_target_rate(yearly_anc1_rates, anc1_lqs, anc1_uqs, target_rate_an,'Year', '% of total births',
                                   'Proportion of women attending >= 1 ANC contact per year', 'anc_prop_anc1')
line_graph_with_ci_and_target_rate(yearly_anc4_rates, anc4_lqs, anc4_uqs, target_rate_anc4, 'Year', '% total births',
                                   'Proportion of women attending >= 4 ANC contact per year', 'anc_prop_anc4')

#simple_line_chart(yearly_anc1_rates, target_rate_an, 'Year', '% of total births',
#                  'Proportion of women attending > 1 ANC contact per year', 'anc_prop_anc1')
#simple_line_chart(yearly_anc4_rates, target_rate_anc4, 'Year', '% total births',
#                  'Proportion of women attending >= 4 ANC contact per year', 'anc_prop_anc4')

plt.plot(sim_years, yearly_anc1_rates, 'o-g', label="anc1", color='palevioletred')
plt.plot(sim_years, yearly_anc4_rates, 'o-g',  label="anc4+", color='crimson')
plt.plot(sim_years, yearly_anc8_rates, 'o-g', label="anc8+", color='pink')
plt.xlabel('Year')
plt.ylabel('% total births')
plt.title('Proportion of women attending ANC1, AN4, ANC8 ')
plt.legend()
##plt.savefig(f'./outputs/sejjj49@ucl.ac.uk/{graph_location}/anc_coverage.png')
plt.show()

anc_count_df = anc_count_df.drop([0])
for year in sim_years:
    total_per_year = 0
    for row in anc_count_df[year]:
        total_per_year += row[0]

    for index in anc_count_df.index:
        anc_count_df.at[index, year] = (anc_count_df.at[index, year][0]/total_per_year) * 100

labels = sim_years
width = 0.35       # the width of the bars: can also be len(x) sequence
fig, ax = plt.subplots()
ax.bar(labels, anc_count_df.loc[8], width, label=8, bottom=anc_count_df.loc[1] + anc_count_df.loc[2] +
                                                           anc_count_df.loc[3] + anc_count_df.loc[4] +
                                                           anc_count_df.loc[5] + anc_count_df.loc[6] +
                                                           anc_count_df.loc[7])
ax.bar(labels, anc_count_df.loc[7], width, label=7, bottom=anc_count_df.loc[1] + anc_count_df.loc[2] +
                                                           anc_count_df.loc[3] + anc_count_df.loc[4] +
                                                           anc_count_df.loc[5] + anc_count_df.loc[6])
ax.bar(labels, anc_count_df.loc[6], width, label=6, bottom=anc_count_df.loc[1] + anc_count_df.loc[2] +
                                                           anc_count_df.loc[3] + anc_count_df.loc[4] +
                                                           anc_count_df.loc[5])
ax.bar(labels, anc_count_df.loc[5], width, label=5, bottom=anc_count_df.loc[1] + anc_count_df.loc[2] +
                                                           anc_count_df.loc[3] + anc_count_df.loc[4] )
ax.bar(labels, anc_count_df.loc[4], width, label=4, bottom=anc_count_df.loc[1] + anc_count_df.loc[2] +
                                                           anc_count_df.loc[3])
ax.bar(labels, anc_count_df.loc[3], width, label=3, bottom=anc_count_df.loc[1] + anc_count_df.loc[2])
ax.bar(labels, anc_count_df.loc[2], width, label=2, bottom=anc_count_df.loc[1])
ax.bar(labels, anc_count_df.loc[1], width, label=1,)
#ax.bar(labels, anc_count_df.loc[0], width, label=0)

ax.set_ylabel('% of total yearly visits')
ax.set_title('Number of ANC visits at birth per year')
ax.legend()
#plt.savefig(f'./outputs/sejjj49@ucl.ac.uk/{graph_location}/anc_total_visits.png')
plt.show()


# Mean proportion of women who attended at least one ANC visit that attended at < 4, 4-5, 6-7 and > 8 months
# gestation...

anc_ga_first_visit = extract_results(
    results_folder,
    module="tlo.methods.care_of_women_during_pregnancy",
    key="anc_count_on_birth",
    custom_generate_series=(
        lambda df_: df_.assign(year=df_['date'].dt.year).groupby(['year', 'total_anc', 'ga_anc_one'])['person_id'].count()),
    do_scaling=False
)

anc_before_four_months = list()
early_anc_4 = list()
anc_before_four_five_months = list()
anc_before_six_seven_months = list()
anc_before_eight_plus_months = list()

def get_means(df):
    mean_list = 0
    for index in df.index:
        mean_list += df.loc[index].mean()
    return mean_list

for year in sim_years:
    year_df = anc_ga_first_visit.loc[year]
    total_women_that_year = 0
    for index in year_df.index:
        total_women_that_year += year_df.loc[index].mean()

    year_series = anc_ga_first_visit.loc[year]
    anc1 = get_means(year_series.loc[0, 0:len(year_series.columns)])

    total_women_anc = total_women_that_year - anc1

    early_anc = year_series.loc[(slice(1, 8), slice(0, 13)), 0:len(year_series.columns)]
    early_anc4 = year_series.loc[(slice(4, 8), slice(0, 17)), 0:len(year_series.columns)]
    four_to_five = year_series.loc[(slice(1, 8), slice(14, 22)), 0:len(year_series.columns)]
    six_to_seven = year_series.loc[(slice(1, 8), slice(23, 31)), 0:len(year_series.columns)]
    eight_plus = year_series.loc[(slice(1, 8), slice(32, 50)), 0:len(year_series.columns)]

    sum_means_early = get_means(early_anc)
    sum_means_early_anc4 = get_means(early_anc4)
    sum_four = get_means(four_to_five)
    sum_six = get_means(six_to_seven)
    sum_eight = get_means(eight_plus)

    early_anc_4.append((sum_means_early_anc4 / total_women_that_year) * 100)
    anc_before_four_months.append((sum_means_early/total_women_anc) * 100)
    anc_before_four_five_months.append((sum_four/total_women_anc) * 100)
    anc_before_six_seven_months.append((sum_six/total_women_anc) * 100)
    anc_before_eight_plus_months.append((sum_eight/total_women_anc) * 100)

labels = sim_years
width = 0.35       # the width of the bars: can also be len(x) sequence
fig, ax = plt.subplots()
ax.bar(labels, anc_before_eight_plus_months, width, label='>8m',
       bottom=[x + y + z for x, y, z in zip(anc_before_four_months, anc_before_four_five_months,
                                            anc_before_six_seven_months)])
ax.bar(labels, anc_before_six_seven_months, width, label='6-7m',
       bottom=[x + y for x, y in zip(anc_before_four_months, anc_before_four_five_months)])
ax.bar(labels, anc_before_four_five_months, width, label='4-5m',
       bottom=anc_before_four_months)
ax.bar(labels, anc_before_four_months, width, label='<4m')
ax.set_ylabel('% of ANC1 visits by gestational age')
ax.set_title('Gestational age at first ANC visit by Year')
ax.legend()
##plt.savefig(f'./outputs/sejjj49@ucl.ac.uk/{graph_location}/anc_ga_first_visit_update.png')
plt.show()

target_rate_eanc4 = list()
for year in sim_years:
    if year < 2015:
        target_rate_eanc4.append(24.5)
    else:
        target_rate_eanc4.append(36.7)

simple_line_chart(early_anc_4, target_rate_eanc4, 'Year', '% total deliveries',
                  'Proportion of women attending attending ANC4+ with first visit early', 'anc_prop_early_anc4')


# TODO: quartiles, median month ANC1
# todo: target rates

# 2.) Facility delivery
# Total FDR per year (denominator - total births)
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

target_rate_fd = list()
for year in sim_years:
    if year < 2015:
        target_rate_fd.append(73)
    else:
        target_rate_fd.append(91)

line_graph_with_ci_and_target_rate(total_fd_rate, fd_lqs, fd_uqs, target_rate_fd, 'Year', '% of total births',
                                   'Proportion of Women Delivering in a Health Facility per Year',
                                   'sba_prop_facility_deliv')

labels = sim_years
width = 0.35       # the width of the bars: can also be len(x) sequence
fig, ax = plt.subplots()

ax.bar(labels, hospital_rate, width, label='Hospital Birth',
       bottom=[x + y for x, y in zip(home_birth_rate, health_centre_rate)])
ax.bar(labels, health_centre_rate, width, label='Health Centre Birth',
       bottom=home_birth_rate)
ax.bar(labels, home_birth_rate, width, label='Home Birth')
ax.set_ylabel('% of Births by Location')
ax.set_title('Proportion of Total Births by Location of Delivery')
ax.legend()
##plt.savefig(f'./outputs/sejjj49@ucl.ac.uk/{graph_location}/sba_delivery_location.png')
plt.show()


# 3.) Postnatal Care
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

maternal_target = list()
newborn_target = list()
for year in sim_years:
    newborn_target.append(60)
    if year < 2015:
        maternal_target.append(50)
    else:
        maternal_target.append(48)

line_graph_with_ci_and_target_rate(pnc_1_plus_rate_mat, pnc_mat_lqs, pnc_mat_uqs, maternal_target, 'Year',
                                   '% of total births',
                                   'Proportion of Women post-delivery attending PNC per year',
                                   'pnc_mat')
line_graph_with_ci_and_target_rate(pnc1_plus_rate_neo, pnc_neo_lqs, pnc_neo_uqs, newborn_target, 'Year',
                                   '% of total births', 'Proportion of Neonates per year attending PNC',
                                   'pnc_neo')

plt.plot(sim_years, pnc_1_plus_rate_mat, 'o-g', label="Maternal Model", color='darkturquoise')
plt.plot(sim_years, pnc1_plus_rate_neo, 'o-g', label="Neonatal Model", color='olivedrab')
plt.plot(sim_years, maternal_target, 'o-g', label="Maternal Target", color='powderblue')
plt.plot(sim_years, newborn_target, 'o-g', label="Neonatal Target", color='palegreen')
plt.xlabel('Year')
plt.ylabel('Proportion of total births')
plt.title('Yearly trends for PNC1 attendance')
plt.legend()
##plt.savefig(f'./outputs/sejjj49@ucl.ac.uk/{graph_location}/pnc_pnc1.png')
plt.show()


def get_early_late_pnc_split(module, target, file_name):
    pnc = extract_results(
        results_folder,
        module=f"tlo.methods.{module}",
        key="postnatal_check",
        custom_generate_series=(
            lambda df: df.assign(year=df['date'].dt.year).groupby(['year', 'timing',
                                                                   'visit_number'])['person_id'].count()
        ),
    )
    early_rate = list()
    late_rate = list()

    for year in sim_years:
        total = pnc.loc[year, 'early', 0].mean() + pnc.loc[year, 'late', 0].mean() + pnc.loc[year, 'none', 0].mean()
        early = (pnc.loc[year, 'early', 0].mean() / total) * 100
        late = ((pnc.loc[year, 'late', 0].mean() + pnc.loc[year, 'none', 0].mean()) / total) * 100
        early_rate.append(early)
        late_rate.append(late)

    labels = sim_years
    width = 0.35       # the width of the bars: can also be len(x) sequence
    fig, ax = plt.subplots()
    ax.bar(labels, early_rate, width, label='< 48hrs')
    ax.bar(labels, late_rate, width, bottom=early_rate, label='>48hrs')
    ax.set_ylabel('% of PNC 1 visits')
    ax.set_title(f'Proportion of {target} PNC1 Visits Occuring pre/post 48hrs Postnatal')
    ax.legend()
    ##plt.savefig(f'./outputs/sejjj49@ucl.ac.uk/{graph_location}/{file_name}.png')
    plt.show()


get_early_late_pnc_split('labour', 'Maternal', 'pnc_maternal_early')
get_early_late_pnc_split('newborn_outcomes', 'Neonatal', 'pnc_neonatal_early')

# ========================================== COMPLICATION/DISEASE RATES.... ===========================================
# ---------------------------------------- Twinning Rate... -----------------------------------------------------------
# % Twin births/Total Births per year
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
lq_rate = [(x / y) * 100 for x, y in zip( get_mean_and_quants(twins_results)[1], total_deliveries)]
uq_rate = [(x / y) * 100 for x, y in zip( get_mean_and_quants(twins_results)[2], total_deliveries)]

target_rate_twins = list()
for year in sim_years:
    target_rate_twins.append(4)

line_graph_with_ci_and_target_rate(final_twining_rate, lq_rate, uq_rate, target_rate_twins, 'Year',
                                   'Rate per 100 pregnancies', 'Yearly trends for Twin Births', 'twin_rate')


# ---------------------------------------- Early Pregnancy Loss... ----------------------------------------------------
# Ectopic pregnancies/Total pregnancies
ectopic_data = get_comp_mean_and_rate('ectopic_unruptured', total_pregnancies_per_year, an_comps, 1000)

target_rate = list()
for year in sim_years:
    if year < 2015:
        target_rate.append(4.9)
    else:
        target_rate.append(3.6)

line_graph_with_ci_and_target_rate(ectopic_data[0], ectopic_data[1], ectopic_data[2], target_rate, 'Year',
                                   'Rate per 100 pregnancies', 'Yearly trends for Ectopic Pregnancy', 'ectopic_rate')

# Ruptured ectopic pregnancies / Total Pregnancies
mean_unrup_ectopics = get_mean_and_quants_from_str_df(an_comps, 'ectopic_unruptured')[0]
proportion_of_ectopics_that_rupture_per_year = get_comp_mean_and_rate('ectopic_ruptured', mean_unrup_ectopics, an_comps,
                                                                      100)[0]
proportion_of_ectopics_that_rupture_per_year = [100 if i > 100 else i for i in
                                                proportion_of_ectopics_that_rupture_per_year]

target_rate_rup = list()
for year in sim_years:
    target_rate_rup.append(92)

simple_line_chart(proportion_of_ectopics_that_rupture_per_year, target_rate_rup, 'Year',
                  'Proportion of all Ectopic Cases',
                  'Proportion of Ectopic Pregnancies Leading to Rupture per year', 'ectopic_rupture_prop')

# Spontaneous Abortions....
spotaneous_abortion_data = get_comp_mean_and_rate('spontaneous_abortion',
                                                  total_completed_pregnancies_per_year, an_comps, 1000)

target_rate_sa = list()
for year in sim_years:
    target_rate_sa.append(189)

line_graph_with_ci_and_target_rate(spotaneous_abortion_data[0], spotaneous_abortion_data[1],
                                   spotaneous_abortion_data[2], target_rate_sa, 'Year',
                                   'Rate per 1000 completed pregnancies', 'Yearly rate of Miscarriage',
                                   'miscarriage_rate')

# Complicated SA / Total SA
mean_complicated_sa = get_mean_and_quants_from_str_df(an_comps, 'spontaneous_abortion')[0]
proportion_of_complicated_sa_per_year = get_comp_mean_and_rate('complicated_spontaneous_abortion',
                                                               mean_complicated_sa, an_comps, 100)[0]

simple_bar_chart(proportion_of_complicated_sa_per_year, 'Year', '% of Total Miscarriages',
                 'Proportion of miscarriages leading to complications', 'miscarriage_prop_complicated')

# Induced Abortions...
id_data = get_comp_mean_and_rate('induced_abortion', total_completed_pregnancies_per_year, an_comps, 1000)

target_rate_ia = list()
for year in sim_years:
    if year < 2015:
        target_rate_ia.append(86)
    else:
        target_rate_ia.append(159)

line_graph_with_ci_and_target_rate(id_data[0], id_data[1],
                                   id_data[2], target_rate_ia, 'Year',
                                   'Rate per 1000 completed pregnancies', 'Yearly rate of Abortion',
                                   'abortion_rate')

# Complicated IA / Total IA
mean_complicated_ia = get_mean_and_quants_from_str_df(an_comps, 'induced_abortion')[0]
proportion_of_complicated_ia_per_year = get_comp_mean_and_rate('complicated_induced_abortion',
                                                               mean_complicated_ia, an_comps, 100)[0]
simple_bar_chart(proportion_of_complicated_ia_per_year, 'Year', '% of Total Abortions',
                 'Proportion of Abortions leading to complications', 'abortion_prop_complicated')

# --------------------------------------------------- Syphilis Rate... ------------------------------------------------
syphilis_data = get_comp_mean_and_rate('syphilis', total_completed_pregnancies_per_year, an_comps,
                                                   1000)
target_rate_syph = list()
for year in sim_years:
    target_rate_syph.append(20)

line_graph_with_ci_and_target_rate(syphilis_data[0], syphilis_data[1],
                                   syphilis_data[2], target_rate_syph, 'Year',
                                   'Rate per 1000 completed pregnancies', 'Yearly rate of Syphilis',
                                   'syphilis_rate')

# ------------------------------------------------ Gestational Diabetes... -------------------------------------------
gdm_data = get_comp_mean_and_rate('gest_diab', total_completed_pregnancies_per_year, an_comps,
                                              1000)
target_rate_gdm = list()
for year in sim_years:
    target_rate_gdm.append(16)

line_graph_with_ci_and_target_rate(gdm_data[0], gdm_data[1],
                                   gdm_data[2], target_rate_gdm, 'Year',
                                   'Rate per 1000 completed pregnancies', 'Yearly rate of Gestational Diabetes',
                                   'gest_diab_rate')


# ------------------------------------------------ PROM... -----------------------------------------------------------
prom_data = get_comp_mean_and_rate('PROM', total_births_per_year, an_comps, 1000)
target_rate_prom = list()
for year in sim_years:
    target_rate_prom.append(27)

line_graph_with_ci_and_target_rate(prom_data[0], prom_data[1],
                                   prom_data[2], target_rate_prom, 'Year',
                                   'Rate per 1000 births', 'Yearly rate of PROM',
                                   'prom_rate')

# ---------------------------------------------- Anaemia... ----------------------------------------------------------
# Total prevalence of Anaemia at birth (total cases of anaemia at birth/ total births per year) and by severity
anaemia_results = extract_results(
    results_folder,
    module="tlo.methods.pregnancy_supervisor",
    key="anaemia_on_birth",
    custom_generate_series=(
        lambda df: df.assign(year=df['date'].dt.year).groupby(['year', 'anaemia_status'])['year'].count()
    ),
)

pnc_anaemia = extract_results(
    results_folder,
    module="tlo.methods.postnatal_supervisor",
    key="total_mat_pnc_visits",
    custom_generate_series=(
        lambda df_: df_.assign(year=df_['date'].dt.year).groupby(['year', 'anaemia'])['mother'].count()),
    do_scaling=False
)

def get_anaemia_graphs(df, timing):
    no_anaemia_data = get_mean_and_quants_from_str_df(df, 'none')
    prevalence_of_anaemia_per_year = [100 - ((x/y) * 100) for x, y in zip(no_anaemia_data[0], total_births_per_year)]
    no_anaemia_lqs = [100 - ((x/y) * 100) for x, y in zip(no_anaemia_data[1], total_births_per_year)]
    no_anaemia_uqs = [100 - ((x/y) * 100) for x, y in zip(no_anaemia_data[2], total_births_per_year)]

    target_rate_an = list()
    for year in sim_years:
        if year < 2015:
            target_rate_an.append(37.5)
        else:
            target_rate_an.append(45.1)

    line_graph_with_ci_and_target_rate(prevalence_of_anaemia_per_year, no_anaemia_lqs,
                                       no_anaemia_uqs, target_rate_an, 'Year', 'Prevalence at birth',
                      f'Yearly prevalence of Anaemia (all severity) at {timing}', f'anaemia_prev_{timing}')


    # todo: should maybe be total postnatal women still alive as opposed to births as will inflate
    mild_anaemia_at_birth = get_mean_and_quants_from_str_df(anaemia_results, 'mild')[0]
    prevalence_of_mild_anaemia_per_year = [(x/y) * 100 for x, y in zip(mild_anaemia_at_birth, total_births_per_year)]

    moderate_anaemia_at_birth = get_mean_and_quants_from_str_df(anaemia_results, 'moderate')[0]
    prevalence_of_mod_anaemia_per_year = [(x/y) * 100 for x, y in zip(moderate_anaemia_at_birth, total_births_per_year)]

    severe_anaemia_at_birth = get_mean_and_quants_from_str_df(anaemia_results, 'severe')[0]
    prevalence_of_sev_anaemia_per_year = [(x/y) * 100 for x, y in zip(severe_anaemia_at_birth, total_births_per_year)]

    plt.plot(sim_years, prevalence_of_mild_anaemia_per_year, label="mild")
    plt.plot(sim_years, prevalence_of_mod_anaemia_per_year, label="moderate")
    plt.plot(sim_years, prevalence_of_sev_anaemia_per_year, label="severe")
    plt.xlabel('Year')
    plt.ylabel(f'Prevalence at {timing}')
    plt.title(f'Yearly trends for prevalence of anaemia by severity at {timing}')
    plt.legend()
    ##plt.savefig(f'./outputs/sejjj49@ucl.ac.uk/{graph_location}/anaemia_by_severity_{timing}.png')
    plt.show()


get_anaemia_graphs(anaemia_results, 'delivery')
get_anaemia_graphs(pnc_anaemia, 'postnatal')

# ------------------------------------------- Hypertensive disorders -------------------------------------------------
gh_data = get_comp_mean_and_rate_across_multiple_dataframes('mild_gest_htn', total_births_per_year, 1000,
                                                                        [an_comps, pn_comps])
sgh_data = get_comp_mean_and_rate_across_multiple_dataframes('severe_gest_htn', total_births_per_year, 1000,
                                                                         [an_comps, la_comps, pn_comps])

mpe_data = get_comp_mean_and_rate_across_multiple_dataframes('mild_pre_eclamp', total_births_per_year, 1000,
                                                                         [an_comps, pn_comps])
spe_data = get_comp_mean_and_rate_across_multiple_dataframes('severe_pre_eclamp', total_births_per_year,
                                                                         1000, [an_comps, la_comps, pn_comps])
ec_data = get_comp_mean_and_rate_across_multiple_dataframes('eclampsia', total_births_per_year, 1000,
                                                                        [an_comps, la_comps, pn_comps])
target_rate_gh = list()
target_rate_sgh = list()
target_rate_mpe = list()
target_rate_spe = list()
target_rate_ec = list()

for year in sim_years:
    target_rate_gh.append(36.8)
    target_rate_sgh.append(8.1)
    target_rate_mpe.append(44)
    target_rate_spe.append(22)
    target_rate_ec.append(10)

line_graph_with_ci_and_target_rate(gh_data[0], gh_data[1], gh_data[2], target_rate_gh, 'Year', 'Rate per 1000 births',
                                   'Rate of Gestational Hypertension per Year', 'gest_htn_rate')

line_graph_with_ci_and_target_rate(sgh_data[0], sgh_data[1], sgh_data[2], target_rate_sgh, 'Year',
                                   'Rate per 1000 births','Rate of Severe Gestational Hypertension per Year',
                                   'severe_gest_htn_rate')

line_graph_with_ci_and_target_rate(mpe_data[0], mpe_data[1], mpe_data[2], target_rate_mpe, 'Year',
                                   'Rate per 1000 births', 'Rate of Mild pre-eclampsia per Year',
                                   'mild_pre_eclampsia_rate')

line_graph_with_ci_and_target_rate(spe_data[0], spe_data[1], spe_data[2], target_rate_spe, 'Year',
                                   'Rate per 1000 births', 'Rate of Severe pre-eclampsia per Year',
                                   'severe_pre_eclampsia_rate')

line_graph_with_ci_and_target_rate(ec_data[0], ec_data[1], ec_data[2], target_rate_ec, 'Year',
                                   'Rate per 1000 births',
                                   'Rate of Eclampsia per Year', 'eclampsia_rate')

#  ---------------------------------------------Placenta praevia... -------------------------------------------------
pp_data = get_comp_mean_and_rate('placenta_praevia', total_pregnancies_per_year, an_comps, 1000)
target_rate_pp = list()
for year in sim_years:
    target_rate_pp.append(5.67)

line_graph_with_ci_and_target_rate(pp_data[0], pp_data[1], pp_data[2], target_rate_pp, 'Year',
                                   'Rate per 1000 pregnancies', 'Rate of Placenta Praevia per Year', 'praevia_rate')

#  ---------------------------------------------Placental abruption... -------------------------------------------------
pa_data = get_comp_mean_and_rate_across_multiple_dataframes('placental_abruption', total_births_per_year, 1000,
                                                                               [an_comps, la_comps])
target_rate_pa = list()
for year in sim_years:
    target_rate_pa.append(3)


line_graph_with_ci_and_target_rate(pa_data[0], pa_data[1], pa_data[2], target_rate_pa, 'Year', 'Rate per 1000 births',
                  'Rate of Placental Abruption per Year', 'abruption_rate')

# --------------------------------------------- Antepartum Haemorrhage... ---------------------------------------------
# Rate of APH/total births (antenatal and labour)
mm_aph_data = get_comp_mean_and_rate_across_multiple_dataframes('mild_mod_antepartum_haemorrhage',
                                                                             total_births_per_year, 1000,
                                                                             [an_comps, la_comps])

s_aph_data = get_comp_mean_and_rate_across_multiple_dataframes('severe_antepartum_haemorrhage',
                                                                           total_births_per_year, 1000,
                                                                           [an_comps, la_comps])

total_aph_rates = [x + y for x, y in zip(mm_aph_data[0], s_aph_data[0])]
aph_lqs = [x + y for x, y in zip(mm_aph_data[1], s_aph_data[1])]
aph_uqs = [x + y for x, y in zip(mm_aph_data[2], s_aph_data[2])]

target_rate_aph = list()
for year in sim_years:
    target_rate_aph.append(6.4)
line_graph_with_ci_and_target_rate(total_aph_rates, aph_lqs, aph_uqs, target_rate_aph, 'Year', 'Rate per 1000 births',
                  'Rate of Antepartum Haemorrhage per Year', 'aph_rate')

# --------------------------------------------- Preterm birth ... ------------------------------------------------
early_ptl_data = get_comp_mean_and_rate('early_preterm_labour', total_births_per_year, la_comps, 100)
late_ptl_data = get_comp_mean_and_rate('late_preterm_labour', total_births_per_year, la_comps, 100)

target_rate_ptl = list()
for year in sim_years:
    target_rate_ptl.append(20)

total_ptl_rates = [x + y for x, y in zip(early_ptl_data[0], late_ptl_data[0])]
ptl_lqs = [x + y for x, y in zip(early_ptl_data[1], late_ptl_data[1])]
ltl_uqs = [x + y for x, y in zip(early_ptl_data[2], late_ptl_data[2])]

line_graph_with_ci_and_target_rate(total_ptl_rates, ptl_lqs, ltl_uqs, target_rate_ptl, 'Year',
                                   'Proportion of total births', 'Preterm birth rate', 'ptb_rate')


# todo plot early and late seperated

# --------------------------------------------- Post term birth ... -----------------------------------------------
potl_data = get_comp_mean_and_rate('post_term_labour', total_births_per_year, la_comps, 100)
target_rate_potl = list()
for year in sim_years:
    target_rate_potl.append(3.2)

line_graph_with_ci_and_target_rate(potl_data[0], potl_data[1], potl_data[2], target_rate_potl, 'Year',
                                   'Proportion of total births', 'Post term birth rate', 'potl_rate')

# ------------------------------------------- Antenatal Stillbirth ... -----------------------------------------------
an_sbr_per_year = [(x/y) * 1000 for x, y in zip(an_still_birth_data[0], total_births_per_year)]
an_sbr_lqs = [(x/y) * 1000 for x, y in zip(an_still_birth_data[1], total_births_per_year)]
an_sbr_uqs = [(x/y) * 1000 for x, y in zip(an_still_birth_data[2], total_births_per_year)]

target_rate_an_sb = list()
for year in sim_years:
    target_rate_an_sb.append(10)

line_graph_with_ci_and_target_rate(an_sbr_per_year, an_sbr_lqs, an_sbr_uqs, target_rate_an_sb, 'Year',
                                   'Rate per 1000 births',
                  'Antenatal Stillbirth Rate per Year', 'sbr_an')

# ------------------------------------------------- Birth weight... --------------------------------------------------
nb_outcomes_df = extract_results(
        results_folder,
        module="tlo.methods.newborn_outcomes",
        key="newborn_complication",
        custom_generate_series=(
            lambda df_: df_.assign(year=df_['date'].dt.year).groupby(['year', 'type'])['newborn'].count()),
        do_scaling=False
    )

nb_outcomes_pn_df = extract_results(
        results_folder,
        module="tlo.methods.postnatal_supervisor",
        key="newborn_complication",
        custom_generate_series=(
            lambda df_: df_.assign(year=df_['date'].dt.year).groupby(['year', 'type'])['newborn'].count()),
        do_scaling=False
    )

lbw_data = get_comp_mean_and_rate('low_birth_weight', total_births_per_year, nb_outcomes_df, 100)
macro_data = get_comp_mean_and_rate('macrosomia', total_births_per_year, nb_outcomes_df, 100)
sga_data = get_comp_mean_and_rate('small_for_gestational_age', total_births_per_year, nb_outcomes_df, 100)

lbw_tr=list()
macro_tr = list()
dummy_tr = list()
for year in sim_years:
    dummy_tr.append(0)
    macro_tr.append(5.13)
    lbw_tr.append(12)

line_graph_with_ci_and_target_rate(lbw_data[0], lbw_data[1], lbw_data[2], lbw_tr, 'Year',
                                   'Proportion of total births', 'Yearly Prevalence of Low Birth Weight',
                                   'neo_lbw_prev')
line_graph_with_ci_and_target_rate(macro_data[0], macro_data[1], macro_data[2], macro_tr,'Year',
                                   'Proportion of total births', 'Yearly Prevalence of Macrosomia',
                                   'neo_macrosomia_prev')

line_graph_with_ci_and_target_rate(sga_data[0], sga_data[1], sga_data[2], dummy_tr, 'Year',
                                   'Proportion of total births', 'Yearly Prevalence of Small for Gestational Age',
                                   'neo_sga_prev')

# todo: check with Ines r.e. SGA and the impact on her modules
# todo: check rates/denominators

# --------------------------------------------- Obstructed Labour... --------------------------------------------------
ol_data = get_comp_mean_and_rate('obstructed_labour', total_births_per_year, la_comps, 1000)
target_rate_ol = list()
for year in sim_years:
    if year < 2015:
        target_rate_ol.append(17)
    else:
        target_rate_ol.append(31)

line_graph_with_ci_and_target_rate(ol_data[0], ol_data[1], ol_data[2], target_rate_ol, 'Year', 'Rate per 1000 births',
                                   'Obstructed Labour Rate per Year', 'ol_rate')

# --------------------------------------------- Uterine rupture... ---------------------------------------------------
ur_data = get_comp_mean_and_rate('uterine_rupture', total_births_per_year, la_comps, 1000)
target_rate_ur = list()
for year in sim_years:
    target_rate_ur.append(1.15)

line_graph_with_ci_and_target_rate(ur_data[0], ur_data[1], ur_data[2], target_rate_ur, 'Year', 'Rate per 1000 births',
                                   'Rate of Uterine Rupture per Year', 'ur_rate')

# ---------------------------Caesarean Section Rate & Assisted Vaginal Delivery Rate... ------------------------------
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

target_rate_cs = list()
target_rate_avd = list()
for year in sim_years:
    target_rate_cs.append(6.2)
    target_rate_avd.append(1)

line_graph_with_ci_and_target_rate(cs_data[0], cs_data[1], cs_data[2], target_rate_cs, 'Year', 'Proportion of total births',
                  'Caesarean Section Rate per Year', 'caesarean_section_rate')
line_graph_with_ci_and_target_rate(avd_data[0], avd_data[1], avd_data[2], target_rate_avd, 'Year',
                                   'Proportion of total births',
                  'Assisted Vaginal Delivery Rate per Year', 'avd_rate')

# ------------------------------------------ Maternal Sepsis Rate... --------------------------------------------------
an_sep_data = get_comp_mean_and_rate('clinical_chorioamnionitis', total_births_per_year, an_comps, 1000)
la_sep_data = get_comp_mean_and_rate('sepsis', total_births_per_year, la_comps, 1000)
pn_sep_data = get_comp_mean_and_rate('sepsis', total_births_per_year, pn_comps, 1000)

total_sep_rates = [x + y + z for x, y, z in zip(an_sep_data[0], la_sep_data[0], pn_sep_data[0])]
sep_lq = [x + y + z for x, y, z in zip(an_sep_data[1], la_sep_data[1], pn_sep_data[1])]
sep_uq = [x + y + z for x, y, z in zip(an_sep_data[2], la_sep_data[2], pn_sep_data[2])]

target_rate_sep = list()
for year in sim_years:
    if year < 2015:
        target_rate_sep.append(4.7)
    else:
        target_rate_sep.append(1.89)

line_graph_with_ci_and_target_rate(total_sep_rates, sep_lq, sep_uq, target_rate_sep, 'Year',
                                   'Rate per 1000 births', 'Rate of Maternal Sepsis per Year', 'sepsis_rate')

# ----------------------------------------- Postpartum Haemorrhage... -------------------------------------------------
la_pph_data = get_comp_mean_and_rate('primary_postpartum_haemorrhage', total_births_per_year, la_comps, 1000)
pn_pph_data = get_comp_mean_and_rate('secondary_postpartum_haemorrhage', total_births_per_year, pn_comps, 1000)

total_pph_rates = [x + y for x, y in zip(la_pph_data[0], pn_pph_data[0])]
pph_lq = [x + y for x, y in zip(la_pph_data[1], pn_pph_data[1])]
pph_uq = [x + y for x, y in zip(la_pph_data[2], pn_pph_data[2])]

target_rate_pph = list()
for year in sim_years:
    if year < 2015:
        target_rate_pph.append(16)
    else:
        target_rate_pph.append(14.6)

line_graph_with_ci_and_target_rate(total_pph_rates, pph_lq, pph_uq, target_rate_pph, 'Year', 'Rate per 1000 births',
                  'Rate of Postpartum Haemorrhage per Year', 'pph_rate')

# ------------------------------------------- Intrapartum Stillbirth ... -----------------------------------------------
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

target_rate_ip_sb = list()
for year in sim_years:
    target_rate_ip_sb.append(10)

line_graph_with_ci_and_target_rate(ip_sbr_per_year, ip_sbr_lqs, ip_sbr_uqs, target_rate_ip_sb, 'Year',
                                   'Rate per 1000 births', 'Intrapartum Stillbirth Rate per Year', 'sbr_ip')

total_sbr = [x + y for x, y in zip(an_sbr_per_year, ip_sbr_per_year)]
total_lqs = [x + y for x, y in zip(an_sbr_lqs, ip_sbr_lqs)]
total_uqs = [x + y for x, y in zip(an_sbr_uqs, ip_sbr_uqs)]

target_rate_sbr = list()
for year in sim_years:
    if year < 2015:
        target_rate_sbr.append(20)
    else:
        target_rate_sbr.append(16.3)

line_graph_with_ci_and_target_rate(total_sbr, total_lqs, total_uqs, target_rate_sbr, 'Year', 'Rate per 1000 births',
                  'Total Stillbirth Rate per Year', 'sbr_total')

# ----------------------------------------- Fistula... -------------------------------------------------
vv_fis_data = get_comp_mean_and_rate('vesicovaginal_fistula', total_births_per_year, pn_comps, 1000)
rv_fis_data = get_comp_mean_and_rate('rectovaginal_fistula', total_births_per_year, pn_comps, 1000)

total_fistula_rates = [x + y for x, y in zip(vv_fis_data[0], rv_fis_data[0])]
fis_lqs = [x + y for x, y in zip(vv_fis_data[1], rv_fis_data[1])]
fis_uqs = [x + y for x, y in zip(vv_fis_data[2], rv_fis_data[2])]

target_rate_fistula = list()
for year in sim_years:
    target_rate_fistula.append(6)

line_graph_with_ci_and_target_rate(total_fistula_rates, fis_lqs, fis_uqs, target_rate_fistula, 'Year',
                                   'Rate per 1000 births',
                  'Rate of Obstetric Fistula per Year', 'fistula_rate')

# ----------------------------------------- Direct Maternal Death... -------------------------------------------------
# TODO: not sure about the uncertainty here!
death_results = extract_results(
    results_folder,
    module="tlo.methods.demography",
    key="death",
    custom_generate_series=(
        lambda df: df.assign(year=df['date'].dt.year).groupby(['year', 'cause'])['year'].count()
    ),
)

direct_causes = ['ectopic_pregnancy', 'spontaneous_abortion', 'induced_abortion',
                 'severe_gestational_hypertension', 'severe_pre_eclampsia', 'eclampsia', 'antenatal_sepsis',
                 'uterine_rupture', 'intrapartum_sepsis', 'postpartum_sepsis', 'postpartum_haemorrhage',
                 'secondary_postpartum_haemorrhage', 'antepartum_haemorrhage']


#list_of_proportions_dicts = list()
proportions_dicts = dict()
total_deaths_per_year = list()
sum_lqs = list()
sum_uqs = list()

for year in sim_years:
    yearly_mean_number = list()
    yearly_lq =list()
    yearly_uq = list()
    causes = dict()

    for complication in direct_causes:
        if complication in death_results.loc[year].index:
            if complication == 'severe_gestational_hypertension': # todo:remove
                yearly_mean_number.append(0)
                yearly_lq.append(0)
                yearly_uq.append(0)
            else:
                mean = death_results.loc[year, complication].mean()
                yearly_mean_number.append(mean)
                yearly_lq.append(death_results.loc[year, complication].quantile(0.025))
                yearly_uq.append(death_results.loc[year, complication].quantile(0.925))
                causes.update({f'{complication}': mean})

            # causes.update({f'{complication}_{year}': mean})
        else:
            yearly_mean_number.append(0)
            yearly_lq.append(0)
            yearly_uq.append(0)

    total_deaths_this_year = sum(yearly_mean_number)
    total_lq = sum(yearly_lq)
    total_uq = sum(yearly_uq)
    total_deaths_per_year.append(total_deaths_this_year)
    sum_lqs.append(total_lq)
    sum_uqs.append(total_uq)

    for complication in causes:
        causes[complication] = (causes[complication] / total_deaths_this_year) * 100
    new_dict = {year: causes}
    proportions_dicts.update(new_dict)


direct_mmr_per_year = [(x/y) * 100000 for x, y in zip(total_deaths_per_year, total_births_per_year)]
lq_mmr = [(x/y) * 100000 for x, y in zip(sum_lqs, total_births_per_year)]
uq_mmr = [(x/y) * 100000 for x, y in zip(sum_uqs, total_births_per_year)]

target_rate_mmr = list()
for year in sim_years:
    if year < 2015:
        target_rate_mmr.append(675)
    else:
        target_rate_mmr.append(439)

line_graph_with_ci_and_target_rate(direct_mmr_per_year, lq_mmr, uq_mmr, target_rate_mmr, 'Year',
                                   'Rate per 100,000 births',
                  'Maternal Mortality Rate per Year', 'mmr')

props_df = pd.DataFrame(data=proportions_dicts)
labels = list(props_df.index)
values = list()
for index in props_df.index:
    values.append(props_df.loc[index].mean())
sizes = values
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title(f'Proportion of total maternal deaths by cause (mean) 2011-2020')
##plt.savefig(f'./outputs/sejjj49@ucl.ac.uk/{graph_location}/mat_death_by_cause_total.png')
plt.show()


death_results_labels = extract_results(
    results_folder,
    module="tlo.methods.demography",
    key="death",
    custom_generate_series=(
        lambda df: df.assign(year=df['date'].dt.year).groupby(['year', 'label'])['year'].count()
    ),
)
mm = get_comp_mean_and_rate('Maternal Disorders', total_births_per_year, death_results_labels, 100000)

line_graph_with_ci_and_target_rate(mm[0], mm[1], mm[2], target_rate_mmr, 'Year',
                                   'Rate per 100,000 births', 'Maternal Mortality Rate per Year (using labels)', 'mmr')

# ==================================================== NEWBORN OUTCOMES ===============================================
#  ------------------------------------------- Neonatal sepsis (labour & postnatal) -----------------------------------
early_ns_data = get_comp_mean_and_rate('early_onset_sepsis', total_births_per_year, nb_outcomes_df, 1000)
late_ns_data = get_comp_mean_and_rate('late_onset_sepsis', total_births_per_year, nb_outcomes_pn_df, 1000)

target_rate_ns = list()
for year in sim_years:
    target_rate_ns.append(39.3)

total_ns_rates = [x + y for x, y in zip(early_ns_data[0], late_ns_data[0])]
ns_lqs = [x + y for x, y in zip(early_ns_data[1], late_ns_data[1])]
ns_uqs = [x + y for x, y in zip(early_ns_data[2], late_ns_data[2])]

line_graph_with_ci_and_target_rate(total_ns_rates, ns_lqs, ns_uqs, target_rate_ns,  'Year', 'Rate per 1000 births',
                  'Rate of Neonatal Sepsis per year', 'neo_sepsis_rate')

# TODO: more analysis on rate by timing?

#  ------------------------------------------- Neonatal encephalopathy -----------------------------------------------
mild_data = get_comp_mean_and_rate('mild_enceph', total_births_per_year, nb_outcomes_df, 1000)
mod_data = get_comp_mean_and_rate('moderate_enceph', total_births_per_year, nb_outcomes_df, 1000)
sev_data = get_comp_mean_and_rate('severe_enceph', total_births_per_year, nb_outcomes_df, 1000)

total_enceph_rates = [x + y + z for x, y, z in zip(mild_data[0], mod_data[0], sev_data[0])]
enceph_lq = [x + y + z for x, y, z in zip(mild_data[1], mod_data[1], sev_data[1])]
enceph_uq = [x + y + z for x, y, z in zip(mild_data[2], mod_data[2], sev_data[2])]


target_rate_enceph = list() # todo: replace
for year in sim_years:
    target_rate_enceph.append(19)

line_graph_with_ci_and_target_rate(total_enceph_rates, enceph_lq, enceph_uq, target_rate_enceph,   'Year',
                                   'Rate per 1000 births',
                  'Rate of Neonatal Encephalopathy per year', 'neo_enceph_rate')

# ----------------------------------------- Respiratory Depression ---------------------------------------------------
rd_data = get_comp_mean_and_rate('not_breathing_at_birth', total_births_per_year, nb_outcomes_df, 1000)

target_rate_rd = list() # todo: replace
for year in sim_years:
    target_rate_rd.append(0)

line_graph_with_ci_and_target_rate(rd_data[0], rd_data[1], rd_data[2], target_rate_rd, 'Year', 'Rate per 1000 births',
                  'Rate of Neonatal Respiratory Depression per year', 'neo_resp_depression_rate')

# ----------------------------------------- Respiratory Distress Syndrome --------------------------------------------
ept = get_mean_and_quants_from_str_df(la_comps, 'early_preterm_labour')[0]
lpt = get_mean_and_quants_from_str_df(la_comps, 'late_preterm_labour')[0]
total_ptbs = [x + y for x, y in zip(ept, lpt)]

rds_data = get_comp_mean_and_rate('respiratory_distress_syndrome', total_ptbs, nb_outcomes_df, 1000)

target_rate_rds = list()  # todo: replace
for year in sim_years:
    target_rate_rds.append(350)
line_graph_with_ci_and_target_rate(rds_data[0], rds_data[1], rds_data[2], target_rate_rds, 'Year',
                                   'Rate per 1000 preterm births',
                                   'Rate of Preterm Respiratory Distress Syndrome per year', 'neo_rds_rate')


# ----------------------------------------- Congenital Anomalies ------------------------------------------------------
rate_of_ca = get_comp_mean_and_rate('congenital_heart_anomaly', total_births_per_year, nb_outcomes_df, 1000)[0]
rate_of_laa = get_comp_mean_and_rate('limb_or_musculoskeletal_anomaly', total_births_per_year, nb_outcomes_df, 1000)[0]
rate_of_ua = get_comp_mean_and_rate('urogenital_anomaly', total_births_per_year, nb_outcomes_df, 1000)[0]
rate_of_da = get_comp_mean_and_rate('digestive_anomaly', total_births_per_year, nb_outcomes_df, 1000)[0]
rate_of_oa = get_comp_mean_and_rate('other_anomaly', total_births_per_year, nb_outcomes_df, 1000)[0]

plt.plot(sim_years, rate_of_ca, label="heart")
plt.plot(sim_years, rate_of_laa, label="limb/musc.")
plt.plot(sim_years, rate_of_ua, label="urogenital")
plt.plot(sim_years, rate_of_da, label="digestive")
plt.plot(sim_years, rate_of_oa, label="other")

plt.xlabel('Year')
plt.ylabel('Rate per 1000 births')
plt.title('Yearly trends for Congenital Birth Anomalies')
plt.legend()
##plt.savefig(f'./outputs/sejjj49@ucl.ac.uk/{graph_location}/neo_rate_of_cong_anom.png')
plt.show()


# Breastfeeding
# todo

# =================================================== Neonatal Death ==================================================

direct_neonatal_causes = ['early_onset_neonatal_sepsis', 'late_onset_sepsis', 'encephalopathy', 'preterm_other',
                          'respiratory_distress_syndrome', 'neonatal_respiratory_depression',
                          'congenital_heart_anomaly', 'limb_or_musculoskeletal_anomaly', 'urogenital_anomaly',
                          'digestive_anomaly', 'other_anomaly']

list_of_proportions_dicts_nb = list()
total_deaths_per_year_nb = list()

for year in sim_years:
    yearly_mean_number = list()
    causes = dict()

    for complication in direct_neonatal_causes:
        if complication in death_results.loc[year].index:
            mean = death_results.loc[year, complication].mean()
            yearly_mean_number.append(mean)
            causes.update({f'{complication}_{year}': mean})
        else:
            yearly_mean_number.append(0)

    total_deaths_this_year = sum(yearly_mean_number)
    total_deaths_per_year_nb.append(total_deaths_this_year)

    for complication in causes:
        causes[complication] = (causes[complication] / total_deaths_this_year) * 100

    list_of_proportions_dicts_nb.append(causes)


direct_nmr_per_year = [(x/y) * 1000 for x, y in zip(total_deaths_per_year_nb, total_births_per_year)]
target_rate_nmr = list()
for year in sim_years:
    if year < 2015:
        target_rate_nmr.append(25)
    else:
        target_rate_nmr.append(22)

nm = get_comp_mean_and_rate('Neonatal Disorders', total_births_per_year, death_results_labels, 1000)

line_graph_with_ci_and_target_rate(nm[0], nm[1], nm[2], target_rate_nmr, 'Year',
                                   'Rate per 1000 births', 'Neonatal Mortality Rate per Year (using labels)', 'nmr')

# todo: force colours for each complication in each year to be the same
for year, dictionary in zip(sim_years, list_of_proportions_dicts_nb):
    labels = list(dictionary.keys())
    sizes = list(dictionary.values())
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title(f'Proportion of total neonatal deaths by cause in {year} ')
    ##plt.savefig(f'./outputs/sejjj49@ucl.ac.uk/{graph_location}/neo_death_by_cause_{year}.png')
    plt.show()






