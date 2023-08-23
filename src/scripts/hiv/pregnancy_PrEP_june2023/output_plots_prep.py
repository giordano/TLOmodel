import datetime
import pickle
from pathlib import Path

# import lacroix
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
#import statsmodels.api as sm
import numpy as np
import pandas as pd
import os
from tlo.analysis.utils import (
    extract_results,
    get_scenario_outputs,
    compare_number_of_deaths,
    extract_params,
    get_scenario_info,
    load_pickled_dataframes,
    summarize,
)
from tlo import Date


plt.style.use('seaborn-darkgrid')

# Set the working directory
# os.chdir('/Users/wenjiazhang/Documents/MSc_HDA/Summer/TLOmodel/')

resourcefilepath = Path("./resources")
outputpath = Path("./outputs")  # folder for convenience of storing outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")


# %% Function to make standard plot to compare model and data
def make_plot(model=None, data_mid=None, data_low=None, data_high=None, title_str=None):
    assert model is not None
    assert title_str is not None

    # Make plot
    fig, ax = plt.subplots()
    ax.plot(model.index, model.values, "-", color="r")

    if data_mid is not None:
        ax.plot(data_mid.index, data_mid.values, "-")
    if (data_low is not None) and (data_high is not None):
        ax.fill_between(data_low.index, data_low, data_high, alpha=0.2)
    plt.title(title_str)
    plt.legend(["Model", "Data"])
    plt.gca().set_ylim(bottom=0)
    plt.savefig(
        outputpath / (title_str.replace(" ", "_") + datestamp + ".pdf"), format="pdf"
    )
    # plt.show()

# ---------------------------------------------------------------------- #
# %%: DATA
# ---------------------------------------------------------------------- #
start_date = 2010
end_date = 2020

# HIV resourcefile
xls = pd.ExcelFile(resourcefilepath / "ResourceFile_HIV.xlsx")

# HIV UNAIDS data
data_hiv_unaids = pd.read_excel(xls, sheet_name="unaids_infections_art2021")
data_hiv_unaids.index = pd.to_datetime(data_hiv_unaids["year"], format="%Y")
data_hiv_unaids = data_hiv_unaids.drop(columns=["year"])

# HIV UNAIDS data
data_hiv_unaids_deaths = pd.read_excel(xls, sheet_name="unaids_mortality_dalys2021")
data_hiv_unaids_deaths.index = pd.to_datetime(
    data_hiv_unaids_deaths["year"], format="%Y"
)
data_hiv_unaids_deaths = data_hiv_unaids_deaths.drop(columns=["year"])

# AIDSinfo (UNAIDS)
data_hiv_aidsinfo = pd.read_excel(xls, sheet_name="children0_14_prev_AIDSinfo")
data_hiv_aidsinfo.index = pd.to_datetime(data_hiv_aidsinfo["year"], format="%Y")
data_hiv_aidsinfo = data_hiv_aidsinfo.drop(columns=["year"])

# unaids program performance
data_hiv_program = pd.read_excel(xls, sheet_name="unaids_program_perf")
data_hiv_program.index = pd.to_datetime(data_hiv_program["year"], format="%Y")
data_hiv_program = data_hiv_program.drop(columns=["year"])

# MPHIA HIV data - age-structured
data_hiv_mphia_inc = pd.read_excel(xls, sheet_name="MPHIA_incidence2015")
data_hiv_mphia_inc_estimate = data_hiv_mphia_inc.loc[
    (data_hiv_mphia_inc.age == "15-49"), "total_percent_annual_incidence"
].values[0]
data_hiv_mphia_inc_lower = data_hiv_mphia_inc.loc[
    (data_hiv_mphia_inc.age == "15-49"), "total_percent_annual_incidence_lower"
].values[0]
data_hiv_mphia_inc_upper = data_hiv_mphia_inc.loc[
    (data_hiv_mphia_inc.age == "15-49"), "total_percent_annual_incidence_upper"
].values[0]
data_hiv_mphia_inc_yerr = [
    abs(data_hiv_mphia_inc_lower - data_hiv_mphia_inc_estimate),
    abs(data_hiv_mphia_inc_upper - data_hiv_mphia_inc_estimate),
]

data_hiv_mphia_prev = pd.read_excel(xls, sheet_name="MPHIA_prevalence_art2015")

# DHS HIV data
data_hiv_dhs_prev = pd.read_excel(xls, sheet_name="DHS_prevalence")

# MoH HIV testing data
data_hiv_moh_tests = pd.read_excel(xls, sheet_name="MoH_numbers_tests")
data_hiv_moh_tests.index = pd.to_datetime(data_hiv_moh_tests["year"], format="%Y")
data_hiv_moh_tests = data_hiv_moh_tests.drop(columns=["year"])

# MoH HIV ART data
# todo this is quarterly
data_hiv_moh_art = pd.read_excel(xls, sheet_name="MoH_number_art")

# ---------------------------------------------------------------------- #
# %%: OUTPUTS
# ---------------------------------------------------------------------- #

# load the results
with open(outputpath / "default_run.pickle", "rb") as f:
    output = pickle.load(f)

# person-years all ages (irrespective of HIV status)
py_ = output["tlo.methods.demography"]["person_years"]
years = pd.to_datetime(py_["date"]).dt.year
py = pd.Series(dtype="int64", index=years)
for year in years:
    tot_py = (
        (py_.loc[pd.to_datetime(py_["date"]).dt.year == year]["M"]).apply(pd.Series)
        + (py_.loc[pd.to_datetime(py_["date"]).dt.year == year]["F"]).apply(pd.Series)
    ).transpose()
    py[year] = tot_py.sum().values[0]

py.index = pd.to_datetime(years, format="%Y")

# ----------------------------- HIV -------------------------------------- #

prev_and_inc_over_time = output["tlo.methods.hiv"][
    "summary_inc_and_prev_for_adults_and_children_and_fsw"
]
prev_and_inc_over_time = prev_and_inc_over_time.set_index("date")

# HIV - prevalence among in adults aged 15-49
title_str = "HIV Prevalence in Adults Aged 15-49 (%)"
make_plot(
    title_str=title_str,
    model=prev_and_inc_over_time["hiv_prev_adult_1549"] * 100,
    data_mid=data_hiv_unaids["prevalence_age15_49"] * 100,
    data_low=data_hiv_unaids["prevalence_age15_49_lower"] * 100,
    data_high=data_hiv_unaids["prevalence_age15_49_upper"] * 100,
)

# MPHIA
plt.plot(
    prev_and_inc_over_time.index[2],
    data_hiv_mphia_prev.loc[
        data_hiv_mphia_prev.age == "Total 15-49", "total percent hiv positive"
    ].values[0],
    "gx",
)

# DHS
x_values = [prev_and_inc_over_time.index[0], prev_and_inc_over_time.index[2]]
y_values = data_hiv_dhs_prev.loc[
    (data_hiv_dhs_prev.Year >= 2010), "HIV prevalence among general population 15-49"
]
y_lower = abs(
    y_values
    - (
        data_hiv_dhs_prev.loc[
            (data_hiv_dhs_prev.Year >= 2010),
            "HIV prevalence among general population 15-49 lower",
        ]
    )
)
y_upper = abs(
    y_values
    - (
        data_hiv_dhs_prev.loc[
            (data_hiv_dhs_prev.Year >= 2010),
            "HIV prevalence among general population 15-49 upper",
        ]
    )
)
plt.errorbar(x_values, y_values, yerr=[y_lower, y_upper], fmt="o")
plt.ylim((0, 15))
plt.xlabel("Year")
plt.ylabel("Prevalence (%)")

# handles for legend
red_line = mlines.Line2D([], [], color="C3", markersize=15, label="TLO")
blue_line = mlines.Line2D([], [], color="C0", markersize=15, label="UNAIDS")
green_cross = mlines.Line2D(
    [], [], linewidth=0, color="g", marker="x", markersize=7, label="MPHIA"
)
orange_ci = mlines.Line2D([], [], color="C1", marker=".", markersize=15, label="DHS")
plt.legend(handles=[red_line, blue_line, green_cross, orange_ci])
plt.savefig(
    outputpath / (title_str.replace(" ", "_") + datestamp + ".pdf"), format="pdf"
)
plt.show()


# ---------------------------------------------------------------------- #

# HIV Incidence 15-49
title_str = "HIV Incidence in Adults (15-49) (per 100 pyar)"
make_plot(
    title_str=title_str,
    model=prev_and_inc_over_time["hiv_adult_inc_1549"] * 100,
    data_mid=data_hiv_unaids["incidence_per1000_age15_49"] / 10,
    data_low=data_hiv_unaids["incidence_per1000_age15_49_lower"] / 10,
    data_high=data_hiv_unaids["incidence_per1000_age15_49_upper"] / 10,
)

# MPHIA
plt.errorbar(
    prev_and_inc_over_time.index[2],
    data_hiv_mphia_inc_estimate,
    yerr=[[data_hiv_mphia_inc_yerr[0]], [data_hiv_mphia_inc_yerr[1]]],
    fmt="o",
)

# handles for legend
red_line = mlines.Line2D([], [], color="C3", markersize=15, label="TLO")
blue_line = mlines.Line2D([], [], color="C0", markersize=15, label="UNAIDS")
orange_ci = mlines.Line2D([], [], color="C1", marker=".", markersize=15, label="MPHIA")
plt.legend(handles=[red_line, blue_line, orange_ci])
# plt.savefig(
#     outputpath / (title_str.replace(" ", "_") + datestamp + ".pdf"), format="pdf"
# )
plt.show()

# ---------------------------------------------------------------------- #

# HIV Prevalence Children
title_str = "HIV Prevalence in Children (0-14) (%)"
make_plot(
    title_str=title_str,
    model=prev_and_inc_over_time["hiv_prev_child"] * 100,
    data_mid=data_hiv_aidsinfo["prevalence_0_14"] * 100,
    data_low=data_hiv_aidsinfo["prevalence_0_14_lower"] * 100,
    data_high=data_hiv_aidsinfo["prevalence_0_14_upper"] * 100,
)
# MPHIA
plt.plot(
    prev_and_inc_over_time.index[2],
    data_hiv_mphia_prev.loc[
        data_hiv_mphia_prev.age == "Total 0-14", "total percent hiv positive"
    ].values[0],
    "gx",
)

# handles for legend
red_line = mlines.Line2D([], [], color="C3", markersize=15, label="TLO")
blue_line = mlines.Line2D([], [], color="C0", markersize=15, label="UNAIDS")
green_cross = mlines.Line2D(
    [], [], linewidth=0, color="g", marker="x", markersize=7, label="MPHIA"
)
plt.legend(handles=[red_line, blue_line, green_cross])
plt.savefig(
    outputpath / (title_str.replace(" ", "_") + datestamp + ".pdf"), format="pdf"
)
plt.show()


# ---------------------------------------------------------------------- #

# HIV Incidence Children
title_str = "HIV Incidence in Children (0-14) per 100 py"
make_plot(
    title_str=title_str,
    model=prev_and_inc_over_time["hiv_child_inc"] * 100,
    data_mid=data_hiv_aidsinfo["incidence0_14_per100py"],
    data_low=data_hiv_aidsinfo["incidence0_14_per100py_lower"],
    data_high=data_hiv_aidsinfo["incidence0_14_per100py_upper"],
)
# plt.savefig(
#     outputpath / (title_str.replace(" ", "_") + datestamp + ".pdf"), format="pdf"
# )
plt.show()

# ---------------------------------------------------------------------- #

# HIV prevalence among female sex workers:

make_plot(
    title_str="HIV Prevalence among Female Sex Workers (%)",
    model=prev_and_inc_over_time["hiv_prev_fsw"] * 100,
)
plt.show()

# HIV prevalence among female aged 15 above
make_plot(
    title_str="HIV Prevalence among Females above 15+ (%)",
    model=prev_and_inc_over_time["female_prev_15plus"] * 100,
)
plt.show()

# ------------------------PrEP intervention ------------------------------#
# ----------------------- ANC visits
anc_visit = output["tlo.methods.care_of_women_during_pregnancy"]["anc_proportion_on_birth"]
anc_visit = anc_visit.set_index("date")

# Proportion of ANC visits
make_plot(
    title_str="Proportion of Pregnant Women Attending >=1 ANC visits",
    model=anc_visit["proportion_attended_at_least_one_anc"],
)
plt.show()

# -----------------------PrEP
cov_over_time = output["tlo.methods.hiv"]["hiv_program_coverage"]
cov_over_time = cov_over_time.set_index("date")

# PrEP among FSW
make_plot(
    title_str="Proportion of FSW That Are On PrEP",
    model=cov_over_time["prop_fsw_on_prep"],
)
plt.show()

# PrEP among pregnant women
cov_over_time_prep = output["tlo.methods.hiv"]["prep_status_logging"]
cov_over_time_prep = cov_over_time_prep.set_index("date")

make_plot(
    title_str="Proportion of Pregnant Women That Are On PrEP",
    model=cov_over_time_prep["prop_pregnant_women_on_prep"],
)
plt.show()

# PrEP among breastfeeding women
make_plot(
    title_str="Proportion of Breastfeeding Women That Are On PrEP",
    model=cov_over_time_prep["prop_breastfeeding_women_on_prep"],
)
plt.show()

make_plot(
    title_str="Proportion of Females That Are On PrEP",
    model=cov_over_time_prep["total_females_on_prep"],
)
plt.show()

# ---------------------------------------------------------------------- #
# %%: DEATHS
# ---------------------------------------------------------------------- #

# deaths
deaths = output["tlo.methods.demography"]["death"].copy()  # outputs individual deaths
deaths = deaths.set_index("date")

# AIDS DEATHS
# limit to deaths among aged 15+, include HIV/TB deaths
keep = (deaths.age >= 15) & (
    (deaths.cause == "AIDS_TB") | (deaths.cause == "AIDS_non_TB")
)
deaths_AIDS = deaths.loc[keep].copy()
deaths_AIDS["year"] = deaths_AIDS.index.year
tot_aids_deaths = deaths_AIDS.groupby(by=["year"]).size()
tot_aids_deaths.index = pd.to_datetime(tot_aids_deaths.index, format="%Y")

# aids mortality rates per 100k person-years
total_aids_deaths_rate_100kpy = (tot_aids_deaths / py) * 100000
#
# # ---------------------------------------------------------------------- #
#
# AIDS deaths (including HIV/TB deaths)
make_plot(
    title_str="Mortality to HIV-AIDS per 1000 capita, data=UNAIDS",
    model=total_aids_deaths_rate_100kpy,
    data_mid=data_hiv_unaids_deaths["AIDS_mortality_per_100k"],
    data_low=data_hiv_unaids_deaths["AIDS_mortality_per_100k_lower"],
    data_high=data_hiv_unaids_deaths["AIDS_mortality_per_100k_upper"],
)

plt.show()


# ---------------------------------------------------------------------- #
# %%: DALYS
# ---------------------------------------------------------------------- #

# download all files (and get most recent [-1])
results1 = get_scenario_outputs("scenario1.py", outputspath)[-1]
results2 = get_scenario_outputs("scenario2.py", outputspath)[-1]
results3 = get_scenario_outputs("scenario3.py", outputspath)[-1]
results4 = get_scenario_outputs("scenario4.py", outputspath)[-1]

TARGET_PERIOD = (Date(2023, 1, 1), Date(2036, 1, 1))


def num_dalys_by_cause(_df):
    """Return total number of DALYS (Stacked) (total by age-group within the TARGET_PERIOD)"""
    return _df \
        .loc[_df.year.between(*[i.year for i in TARGET_PERIOD])] \
        .drop(columns=['date', 'sex', 'age_range', 'year']) \
        .sum()


def return_daly_summary(results_folder):
    dalys = extract_results(
        results_folder,
        module='tlo.methods.healthburden',
        key='dalys_stacked',
        custom_generate_series=num_dalys_by_cause,
        do_scaling=True
    )
    dalys.columns = dalys.columns.get_level_values(0)
    # combine two labels for non-AIDS TB (this now fixed in latest code)
    dalys.loc['AIDS'] = dalys.loc['HIV'] + dalys.loc['HIV/AIDS']
    out = pd.DataFrame()
    out['median'] = dalys.median(axis=1).round(decimals=-3).astype(int)
    out['lower'] = dalys.quantile(q=0.025, axis=1).round(decimals=-3).astype(int)
    out['upper'] = dalys.quantile(q=0.975, axis=1).round(decimals=-3).astype(int)

    return out


def return_daly_summary2(results_folder):
    dalys = extract_results(
        results_folder,
        module='tlo.methods.healthburden',
        key='dalys_stacked',
        custom_generate_series=num_dalys_by_cause,
        do_scaling=True
    )
    dalys.columns = dalys.columns.get_level_values(0)
    # combine two labels for non-AIDS TB (this now fixed in latest code)
    # dalys.loc['TB (non-AIDS)'] = dalys.loc['TB (non-AIDS)'] + dalys.loc['non_AIDS_TB']
    # dalys.drop(['non_AIDS_TB'], inplace=True)
    out = pd.DataFrame()
    out['median'] = dalys.median(axis=1).round(decimals=-3).astype(int)
    out['lower'] = dalys.quantile(q=0.025, axis=1).round(decimals=-3).astype(int)
    out['upper'] = dalys.quantile(q=0.975, axis=1).round(decimals=-3).astype(int)

    return out


dalys1 = return_daly_summary(results1)
dalys2 = return_daly_summary(results2)
dalys3 = return_daly_summary2(results3)
dalys4 = return_daly_summary2(results4)

dalys1.loc['Column_Total'] = dalys1.sum(numeric_only=True, axis=0)
dalys2.loc['Column_Total'] = dalys2.sum(numeric_only=True, axis=0)
dalys3.loc['Column_Total'] = dalys3.sum(numeric_only=True, axis=0)
dalys4.loc['Column_Total'] = dalys4.sum(numeric_only=True, axis=0)

# create full table for export
daly_table = pd.DataFrame()
daly_table['scenario1'] = dalys0['median'].astype(str) + \
                          " (" + dalys0['lower'].astype(str) + " - " + \
                          dalys0['upper'].astype(str) + ")"
daly_table['scenario2'] = dalys1['median'].astype(str) + \
                          " (" + dalys1['lower'].astype(str) + " - " + \
                          dalys1['upper'].astype(str) + ")"
daly_table['scenario3'] = dalys2['median'].astype(str) + \
                          " (" + dalys2['lower'].astype(str) + " - " + \
                          dalys2['upper'].astype(str) + ")"
daly_table['scenario4'] = dalys2['median'].astype(str) + \
                          " (" + dalys2['lower'].astype(str) + " - " + \
                          dalys2['upper'].astype(str) + ")"
daly_table.to_csv(outputspath / "daly_summary.csv")

# extract dalys averted by each scenario relative to scenario 1 (the standard)
# comparison should be run-by-run
full_dalys1 = extract_results(
    results1,
    module='tlo.methods.healthburden',
    key='dalys_stacked',
    custom_generate_series=num_dalys_by_cause,
    do_scaling=True
)
full_dalys1.loc['AIDS'] = full_dalys1.loc['HIV'] + full_dalys1.loc['HIV/AIDS']
full_dalys1.loc['Column_Total'] = full_dalys1.sum(numeric_only=True, axis=0)

full_dalys2 = extract_results(
    results2,
    module='tlo.methods.healthburden',
    key='dalys_stacked',
    custom_generate_series=num_dalys_by_cause,
    do_scaling=True
)
full_dalys2.loc['AIDS'] = full_dalys2.loc['HIV'] + full_dalys2.loc['HIV/AIDS']
full_dalys2.loc['Column_Total'] = full_dalys2.sum(numeric_only=True, axis=0)

full_dalys3 = extract_results(
    results3,
    module='tlo.methods.healthburden',
    key='dalys_stacked',
    custom_generate_series=num_dalys_by_cause,
    do_scaling=True
)
full_dalys3.loc['AIDS'] = full_dalys3.loc['HIV'] + full_dalys3.loc['HIV/AIDS']
full_dalys3.loc['Column_Total'] = full_dalys3.sum(numeric_only=True, axis=0)

full_dalys4 = extract_results(
    results4,
    module='tlo.methods.healthburden',
    key='dalys_stacked',
    custom_generate_series=num_dalys_by_cause,
    do_scaling=True
)
full_dalys4.loc['AIDS'] = full_dalys4.loc['HIV'] + full_dalys4.loc['HIV/AIDS']
full_dalys4.loc['Column_Total'] = full_dalys4.sum(numeric_only=True, axis=0)

writer = pd.ExcelWriter(r"outputs/wenjia.zhang22@imperial.ac.uk/full_dalys.xlsx")
full_dalys1.to_excel(writer, sheet_name='sc1')
full_dalys2.to_excel(writer, sheet_name='sc2')
full_dalys3.to_excel(writer, sheet_name='sc3')
full_dalys4.to_excel(writer, sheet_name='sc4')
writer.save()

# DALYs averted: baseline - scenario
# positive value will be DALYs averted due to interventions
# negative value will be higher DALYs reported, therefore increased health burden
sc2_sc1 = full_dalys1.subtract(full_dalys2, fill_value=0)
sc2_sc1_median = sc2_sc1.median(axis=1)
sc2_sc1_lower = sc2_sc1.quantile(q=0.025, axis=1)
sc2_sc1_upper = sc2_sc1.quantile(q=0.975, axis=1)

sc3_sc1 = full_dalys1.subtract(full_dalys3, fill_value=0)
sc3_sc1_median = sc3_sc1.median(axis=1)
sc3_sc1_lower = sc3_sc1.quantile(q=0.025, axis=1)
sc3_sc1_upper = sc3_sc1.quantile(q=0.975, axis=1)

sc4_sc1 = full_dalys1.subtract(full_dalys4, fill_value=0)
sc4_sc1_median = sc4_sc1.median(axis=1)
sc4_sc1_lower = sc4_sc1.quantile(q=0.025, axis=1)
sc4_sc1_upper = sc4_sc1.quantile(q=0.975, axis=1)

# create full table for export
daly_averted_table = pd.DataFrame()
daly_averted_table['cause'] = sc1_sc0_median.index
daly_averted_table['scenario2_med'] = [int(round(x, -3)) for x in sc2_sc1_median]
daly_averted_table['scenario2_low'] = [int(round(x, -3)) for x in sc2_sc1_lower]
daly_averted_table['scenario2_upp'] = [int(round(x, -3)) for x in sc2_sc1_upper]
daly_averted_table['scenario3_med'] = [int(round(x, -3)) for x in sc3_sc1_median]
daly_averted_table['scenario3_low'] = [int(round(x, -3)) for x in sc3_sc1_lower]
daly_averted_table['scenario3_upp'] = [int(round(x, -3)) for x in sc3_sc1_upper]
daly_averted_table['scenario4_med'] = [int(round(x, -3)) for x in sc4_sc1_median]
daly_averted_table['scenario4_low'] = [int(round(x, -3)) for x in sc4_sc1_lower]
daly_averted_table['scenario4_upp'] = [int(round(x, -3)) for x in sc4_sc1_upper]

daly_averted_table.to_csv(outputspath / "daly_averted_summary.csv")

# this is now unconstrained scenario first!!
aids_dalys_diff = [sc2_sc1_median['AIDS'],
                   sc3_sc1_median['AIDS'],
                   sc4_sc1_median['AIDS']]


# ggplot - for DALYS
plt.style.use('ggplot')

aids_colour = "#8949ab"
hiv_colour = "#ed7e7a"
total_colour = "#eede77"

# present DALYs in millions
million = 1000000
aids_dalys_diff = [x / million for x in aids_dalys_diff]
hiv_dalys_diff = [x / million for x in hiv_dalys_diff]
total_dalys_diff = [x / million for x in total_dalys_diff]


fig, ax1 = plt.subplots(nrows=1, ncols=1,
                                             figsize=(5, 4))
fig.suptitle('')

# DALYs
labels = ['Unconstrained scale-up', 'Constrained scale-up']
x = np.arange(len(labels))  # the label locations
width = 0.2  # the width of the bars

rects1 = ax1.bar(x - width, aids_dalys_diff, width, label='AIDS', color=aids_colour)
rects2 = ax1.bar(x, tb_dalys_diff, width, label='HIV', color=tb_colour)
rects3 = ax1.bar(x + width, total_dalys_diff, width, label='Total', color=total_colour)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax1.set_ylabel('DALYs averted, millions')
ax1.set_title('')
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.legend(["AIDS", "HIV", "Total"], frameon=False)

fig.tight_layout()
fig.savefig(outputspath / "DALYS.png")

plt.show()
