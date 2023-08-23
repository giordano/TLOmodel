"""This file uses the results of the batch file to make some summary statistics.
The results of the batchrun were put into the 'outputspath' results_folder
function weighted_mean_for_data_comparison can be used to select which parameter sets to use
make plots for top 5 parameter sets just to make sure they are looking ok
"""

import datetime
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

from tlo.analysis.utils import (
    compare_number_of_deaths,
    extract_params,
    extract_results,
    get_scenario_info,
    get_scenario_outputs,
    load_pickled_dataframes,
    summarize,
)

resourcefilepath = Path("./resources")
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

outputspath = Path("./outputs/t.mangal@imperial.ac.uk")

# %% read in data files for plots
# load all the data for calibration

# TB WHO data
xls_tb = pd.ExcelFile(resourcefilepath / "ResourceFile_TB.xlsx")

data_tb_who = pd.read_excel(xls_tb, sheet_name="WHO_activeTB2023")
data_tb_who = data_tb_who.loc[
    (data_tb_who.year >= 2010)
]  # include only years post-2010
data_tb_who.index = data_tb_who["year"]
data_tb_who = data_tb_who.drop(columns=["year"])

# TB latent data (Houben & Dodd 2016)
data_tb_latent = pd.read_excel(xls_tb, sheet_name="latent_TB2014_summary")
data_tb_latent_all_ages = data_tb_latent.loc[data_tb_latent.Age_group == "0_80"]
data_tb_latent_estimate = data_tb_latent_all_ages.proportion_latent_TB.values[0]
data_tb_latent_lower = abs(
    data_tb_latent_all_ages.proportion_latent_TB_lower.values[0]
    - data_tb_latent_estimate
)
data_tb_latent_upper = abs(
    data_tb_latent_all_ages.proportion_latent_TB_upper.values[0]
    - data_tb_latent_estimate
)
data_tb_latent_yerr = [data_tb_latent_lower, data_tb_latent_upper]

# TB deaths WHO
deaths_2010_2014 = data_tb_who.loc["2010-01-01":"2014-01-01"]
deaths_2015_2019 = data_tb_who.loc["2015-01-01":"2019-01-01"]

deaths_2010_2014_average = deaths_2010_2014.loc[:, "num_deaths_tb_nonHiv"].values.mean()
deaths_2010_2014_average_low = deaths_2010_2014.loc[:, "num_deaths_tb_nonHiv_low"].values.mean()
deaths_2010_2014_average_high = deaths_2010_2014.loc[:, "num_deaths_tb_nonHiv_high"].values.mean()

deaths_2015_2019_average = deaths_2015_2019.loc[:, "num_deaths_tb_nonHiv"].values.mean()
deaths_2015_2019_average_low = deaths_2015_2019.loc[:, "num_deaths_tb_nonHiv_low"].values.mean()
deaths_2015_2019_average_high = deaths_2015_2019.loc[:, "num_deaths_tb_nonHiv_high"].values.mean()

# TB treatment coverage
data_tb_ntp = pd.read_excel(xls_tb, sheet_name="NTP2019")
data_tb_ntp.index = data_tb_ntp["year"]
data_tb_ntp = data_tb_ntp.drop(columns=["year"])

# HIV resourcefile
xls = pd.ExcelFile(resourcefilepath / "ResourceFile_HIV.xlsx")

# HIV UNAIDS data
data_hiv_unaids = pd.read_excel(xls, sheet_name="unaids_infections_art2021")
data_hiv_unaids.index = data_hiv_unaids["year"]
data_hiv_unaids = data_hiv_unaids.drop(columns=["year"])

# HIV UNAIDS data
data_hiv_unaids_deaths = pd.read_excel(xls, sheet_name="unaids_mortality_dalys2021")
data_hiv_unaids_deaths.index = data_hiv_unaids_deaths["year"]
data_hiv_unaids_deaths = data_hiv_unaids_deaths.drop(columns=["year"])

# AIDSinfo (UNAIDS)
data_hiv_aidsinfo = pd.read_excel(xls, sheet_name="children0_14_prev_AIDSinfo")
data_hiv_aidsinfo.index = data_hiv_aidsinfo["year"]
data_hiv_aidsinfo = data_hiv_aidsinfo.drop(columns=["year"])

# unaids program performance
data_hiv_program = pd.read_excel(xls, sheet_name="unaids_program_perf")
data_hiv_program.index = data_hiv_program["year"]
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
data_hiv_moh_tests.index = data_hiv_moh_tests["year"]
data_hiv_moh_tests = data_hiv_moh_tests.drop(columns=["year"])

# MoH HIV ART data
# todo this is quarterly
data_hiv_moh_art = pd.read_excel(xls, sheet_name="MoH_number_art")

# %% Analyse results of runs

# 0) Find results_folder associated with a given batch_file (and get most recent [-1])
results_folder = get_scenario_outputs("batch_test_runs.py", outputspath)[-1]

# Declare path for output graphs from this script
make_graph_file_name = lambda stub: results_folder / f"{stub}.png"  # noqa: E731

# look at one log (so can decide what to extract)
log = load_pickled_dataframes(results_folder)

# get basic information about the results
info = get_scenario_info(results_folder)

# 1) Extract the parameters that have varied over the set of simulations
params = extract_params(results_folder)

# choose which draw to summarise / visualise
draw = 0

# %% extract results
# Load and format model results (with year as integer):

# ---------------------------------- HIV ---------------------------------- #
model_hiv_adult_prev = summarize(
    extract_results(
        results_folder,
        module="tlo.methods.hiv",
        key="summary_inc_and_prev_for_adults_and_children_and_fsw",
        column="hiv_prev_adult_15plus",
        index="date",
        do_scaling=False,
    ),
    collapse_columns=True,
)
model_hiv_adult_prev.index = model_hiv_adult_prev.index.year

model_hiv_adult_inc = summarize(
    extract_results(
        results_folder,
        module="tlo.methods.hiv",
        key="summary_inc_and_prev_for_adults_and_children_and_fsw",
        column="hiv_adult_inc_1549",
        index="date",
        do_scaling=False,
    ),
    collapse_columns=True,
)
model_hiv_adult_inc.index = model_hiv_adult_inc.index.year

model_hiv_child_prev = summarize(
    extract_results(
        results_folder,
        module="tlo.methods.hiv",
        key="summary_inc_and_prev_for_adults_and_children_and_fsw",
        column="hiv_prev_child",
        index="date",
        do_scaling=False,
    ),
    collapse_columns=True,
)
model_hiv_child_prev.index = model_hiv_child_prev.index.year

model_hiv_child_inc = summarize(
    extract_results(
        results_folder,
        module="tlo.methods.hiv",
        key="summary_inc_and_prev_for_adults_and_children_and_fsw",
        column="hiv_child_inc",
        index="date",
        do_scaling=False,
    ),
    collapse_columns=True,
)
model_hiv_child_inc.index = model_hiv_child_inc.index.year

model_hiv_fsw_prev = summarize(
    extract_results(
        results_folder,
        module="tlo.methods.hiv",
        key="summary_inc_and_prev_for_adults_and_children_and_fsw",
        column="hiv_prev_fsw",
        index="date",
        do_scaling=False,
    ),
    collapse_columns=True,
)
model_hiv_fsw_prev.index = model_hiv_fsw_prev.index.year


# ---------------------------------- PERSON-YEARS ---------------------------------- #

# function to extract person-years by year
# call this for each run and then take the mean to use as denominator for mortality / incidence etc.
def get_person_years(draw, run):
    log = load_pickled_dataframes(results_folder, draw, run)

    py_ = log["tlo.methods.demography"]["person_years"]
    years = pd.to_datetime(py_["date"]).dt.year
    py = pd.Series(dtype="int64", index=years)
    for year in years:
        tot_py = (
            (py_.loc[pd.to_datetime(py_["date"]).dt.year == year]["M"]).apply(pd.Series) +
            (py_.loc[pd.to_datetime(py_["date"]).dt.year == year]["F"]).apply(pd.Series)
        ).transpose()
        py[year] = tot_py.sum().values[0]

    py.index = pd.to_datetime(years, format="%Y")

    return py


# for draw 0, get py for all runs
number_runs = info["runs_per_draw"]
py_summary = pd.DataFrame(data=None, columns=range(0, number_runs))

# draw number (default = 0) is specified above
for run in range(0, number_runs):
    py_summary.iloc[:, run] = get_person_years(draw, run)

py_summary["mean"] = py_summary.mean(axis=1)

# ---------------------------------- TB ---------------------------------- #

model_tb_inc = summarize(
    extract_results(
        results_folder,
        module="tlo.methods.tb",
        key="tb_incidence",
        column="num_new_active_tb",
        index="date",
        do_scaling=False,
    ),
    collapse_columns=True,
)
model_tb_inc.index = model_tb_inc.index.year
activeTB_inc_rate = (model_tb_inc.divide(py_summary["mean"].values[1:10], axis=0)) * 100000

model_tb_mdr = summarize(
    extract_results(
        results_folder,
        module="tlo.methods.tb",
        key="tb_mdr",
        column="tbPropActiveCasesMdr",
        index="date",
        do_scaling=False,
    ),
    collapse_columns=True,
)

model_tb_mdr.index = model_tb_mdr.index.year

model_tb_hiv_prop = summarize(
    extract_results(
        results_folder,
        module="tlo.methods.tb",
        key="tb_incidence",
        column="prop_active_tb_in_plhiv",
        index="date",
        do_scaling=False,
    ),
    collapse_columns=True,
)

model_tb_hiv_prop.index = model_tb_hiv_prop.index.year

# ---------------------------------- DEATHS ---------------------------------- #

results_deaths = extract_results(
    results_folder,
    module="tlo.methods.demography",
    key="death",
    custom_generate_series=(
        lambda df: df.assign(year=df["date"].dt.year).groupby(
            ["year", "cause"])["person_id"].count()
    ),
    do_scaling=False,
)

results_deaths = results_deaths.reset_index()

# summarise across runs
aids_non_tb_deaths_table = results_deaths.loc[results_deaths.cause == "AIDS_non_TB"]
aids_tb_deaths_table = results_deaths.loc[results_deaths.cause == "AIDS_TB"]
tb_deaths_table = results_deaths.loc[results_deaths.cause == "TB"]

# ------------ summarise deaths producing df for each draw

# AIDS deaths
aids_deaths = {}  # dict of df

for draw in info["number_of_draws"]:
    draw = draw

    # rename dataframe
    name = "model_deaths_AIDS_draw" + str(draw)
    # select cause of death
    tmp = results_deaths.loc[
        (results_deaths.cause == "AIDS_TB") | (results_deaths.cause == "AIDS_non_TB")
        ]

    # select draw - drop columns where draw != 0, but keep year and cause
    tmp2 = tmp.loc[
           :, ("draw" == draw)
           ].copy()  # selects only columns for draw=0 (removes year/cause)
    # join year and cause back to df - needed for groupby
    frames = [tmp["year"], tmp["cause"], tmp2]
    tmp3 = pd.concat(frames, axis=1)

    # create new column names, dependent on number of runs in draw
    base_columns = ["year", "cause"]
    run_columns = ["run" + str(x) for x in range(0, info["runs_per_draw"])]
    base_columns.extend(run_columns)
    tmp3.columns = base_columns
    tmp3 = tmp3.set_index("year")

    # sum rows for each year (2 entries = 2 causes of death)
    # for each run need to combine deaths in each year, may have different numbers of runs
    aids_deaths[name] = pd.DataFrame(tmp3.groupby(["year"]).sum())

    # double check all columns are float64 or quantile argument will fail
    cols = [
        col
        for col in aids_deaths[name].columns
        if aids_deaths[name][col].dtype == "float64"
    ]
    aids_deaths[name]["median"] = (
        aids_deaths[name][cols].astype(float).quantile(0.5, axis=1)
    )
    aids_deaths[name]["lower"] = (
        aids_deaths[name][cols].astype(float).quantile(0.025, axis=1)
    )
    aids_deaths[name]["upper"] = (
        aids_deaths[name][cols].astype(float).quantile(0.975, axis=1)
    )

    # AIDS mortality rates per 100k person-years
    aids_deaths[name]["aids_deaths_rate_100kpy"] = (
            aids_deaths[name]["median"].values / py_summary["mean"].values) * 100000

    aids_deaths[name]["aids_deaths_rate_100kpy_lower"] = (
        aids_deaths[name]["lower"].values / py_summary["mean"].values) * 100000

    aids_deaths[name]["aids_deaths_rate_100kpy_upper"] = (
        aids_deaths[name]["upper"].values / py_summary["mean"].values) * 100000


# HIV/TB deaths
aids_tb_deaths = {}  # dict of df

for draw in info["number_of_draws"]:
    draw = draw

    # rename dataframe
    name = "model_deaths_AIDS_TB_draw" + str(draw)
    # select cause of death
    tmp = results_deaths.loc[
        (results_deaths.cause == "AIDS_TB")
    ]
    # select draw - drop columns where draw != 0, but keep year and cause
    tmp2 = tmp.loc[
           :, ("draw" == draw)
           ].copy()  # selects only columns for draw=0 (removes year/cause)
    # join year and cause back to df - needed for groupby
    frames = [tmp["year"], tmp["cause"], tmp2]
    tmp3 = pd.concat(frames, axis=1)

    # create new column names, dependent on number of runs in draw
    base_columns = ["year", "cause"]
    run_columns = ["run" + str(x) for x in range(0, info["runs_per_draw"])]
    base_columns.extend(run_columns)
    tmp3.columns = base_columns
    tmp3 = tmp3.set_index("year")

    # sum rows for each year (2 entries)
    # for each run need to combine deaths in each year, may have different numbers of runs
    aids_tb_deaths[name] = pd.DataFrame(tmp3.groupby(["year"]).sum())

    # double check all columns are float64 or quantile argument will fail
    cols = [
        col
        for col in aids_tb_deaths[name].columns
        if aids_tb_deaths[name][col].dtype == "float64"
    ]
    aids_tb_deaths[name]["median"] = (
        aids_tb_deaths[name][cols].astype(float).quantile(0.5, axis=1)
    )
    aids_tb_deaths[name]["lower"] = (
        aids_tb_deaths[name][cols].astype(float).quantile(0.025, axis=1)
    )
    aids_tb_deaths[name]["upper"] = (
        aids_tb_deaths[name][cols].astype(float).quantile(0.975, axis=1)
    )

    # AIDS_TB mortality rates per 100k person-years
    aids_tb_deaths[name]["aids_TB_deaths_rate_100kpy"] = (
            aids_tb_deaths[name]["median"].values / py_summary["mean"].values) * 100000

    aids_tb_deaths[name]["aids_TB_deaths_rate_100kpy_lower"] = (
        aids_tb_deaths[name]["lower"].values / py_summary["mean"].values) * 100000

    aids_tb_deaths[name]["aids_TB_deaths_rate_100kpy_upper"] = (
        aids_tb_deaths[name]["upper"].values / py_summary["mean"].values) * 100000


# TB deaths excluding HIV
tb_deaths = {}  # dict of df

for draw in info["number_of_draws"]:
    draw = draw

    # rename dataframe
    name = "model_deaths_TB_draw" + str(draw)
    # select cause of death
    tmp = results_deaths.loc[
        (results_deaths.cause == "TB")
    ]
    # select draw - drop columns where draw != 0, but keep year and cause
    tmp2 = tmp.loc[
           :, ("draw" == draw)
           ].copy()  # selects only columns for draw=0 (removes year/cause)
    # join year and cause back to df - needed for groupby
    frames = [tmp["year"], tmp["cause"], tmp2]
    tmp3 = pd.concat(frames, axis=1)

    # create new column names, dependent on number of runs in draw
    base_columns = ["year", "cause"]
    run_columns = ["run" + str(x) for x in range(0, info["runs_per_draw"])]
    base_columns.extend(run_columns)
    tmp3.columns = base_columns
    tmp3 = tmp3.set_index("year")

    # sum rows for each year (2 entries)
    # for each run need to combine deaths in each year, may have different numbers of runs
    tb_deaths[name] = pd.DataFrame(tmp3.groupby(["year"]).sum())

    # double check all columns are float64 or quantile argument will fail
    cols = [
        col
        for col in tb_deaths[name].columns
        if tb_deaths[name][col].dtype == "float64"
    ]
    tb_deaths[name]["median"] = (
        tb_deaths[name][cols].astype(float).quantile(0.5, axis=1)
    )
    tb_deaths[name]["lower"] = (
        tb_deaths[name][cols].astype(float).quantile(0.025, axis=1)
    )
    tb_deaths[name]["upper"] = (
        tb_deaths[name][cols].astype(float).quantile(0.975, axis=1)
    )

    # AIDS_TB mortality rates per 100k person-years
    tb_deaths[name]["TB_death_rate_100kpy"] = (
            tb_deaths[name]["median"].values / py_summary["mean"].values) * 100000

    tb_deaths[name]["TB_death_rate_100kpy_lower"] = (
        tb_deaths[name]["lower"].values / py_summary["mean"].values) * 100000

    tb_deaths[name]["TB_death_rate_100kpy_upper"] = (
        tb_deaths[name]["upper"].values / py_summary["mean"].values) * 100000


# ---------------------------------- PROGRAM COVERAGE ---------------------------------- #

# TB treatment coverage
model_tb_tx = summarize(
    extract_results(
        results_folder,
        module="tlo.methods.tb",
        key="tb_treatment",
        column="tbTreatmentCoverage",
        index="date",
        do_scaling=False,
    ),
    collapse_columns=True,
)

model_tb_tx.index = model_tb_tx.index.year

# HIV treatment coverage
model_hiv_tx = summarize(
    extract_results(
        results_folder,
        module="tlo.methods.hiv",
        key="hiv_program_coverage",
        column="art_coverage_adult",
        index="date",
        do_scaling=False,
    ),
    collapse_columns=True,
)

model_hiv_tx.index = model_hiv_tx.index.year


# %% Function to make standard plot to compare model and data
def make_plot(
    model=None,
    model_low=None,
    model_high=None,
    data_name=None,
    data_mid=None,
    data_low=None,
    data_high=None,
    xlab=None,
    ylab=None,
    title_str=None,
):
    assert model is not None
    assert title_str is not None

    # Make plot
    fig, ax = plt.subplots()
    ax.plot(model.index, model.values, "-", color="C3")
    if (model_low is not None) and (model_high is not None):
        ax.fill_between(model_low.index, model_low, model_high, color="C3", alpha=0.2)

    if data_mid is not None:
        ax.plot(data_mid.index, data_mid.values, "-", color="C0")
    if (data_low is not None) and (data_high is not None):
        ax.fill_between(data_low.index, data_low, data_high, color="C0", alpha=0.2)

    if xlab is not None:
        ax.set_xlabel(xlab)

    if ylab is not None:
        ax.set_xlabel(ylab)

    plt.title(title_str)
    plt.legend(["TLO", data_name])
    # plt.gca().set_ylim(bottom=0)
    # plt.savefig(outputspath / (title_str.replace(" ", "_") + datestamp + ".pdf"), format='pdf')


# %% make plots

# HIV - prevalence among in adults aged 15-49

make_plot(
    title_str="HIV Prevalence in Adults Aged 15-49 (%)",
    model=model_hiv_adult_prev["mean"] * 100,
    model_low=model_hiv_adult_prev["lower"] * 100,
    model_high=model_hiv_adult_prev["upper"] * 100,
    data_name="UNAIDS",
    data_mid=data_hiv_unaids["prevalence_age15_49"] * 100,
    data_low=data_hiv_unaids["prevalence_age15_49_lower"] * 100,
    data_high=data_hiv_unaids["prevalence_age15_49_upper"] * 100,
)

# data: MPHIA
plt.plot(
    model_hiv_adult_prev.index[6],
    data_hiv_mphia_prev.loc[
        data_hiv_mphia_prev.age == "Total 15-49", "total percent hiv positive"
    ].values[0],
    "gx",
)

# data: DHS
x_values = [model_hiv_adult_prev.index[0], model_hiv_adult_prev.index[5]]
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
plt.errorbar(x_values, y_values, yerr=[y_lower, y_upper], fmt="ko")

plt.ylim((0, 15))
plt.xlabel = ("Year",)
plt.ylabel = "HIV prevalence (%)"

# handles for legend
red_line = mlines.Line2D([], [], color="C3", markersize=15, label="TLO")
blue_line = mlines.Line2D([], [], color="C0", markersize=15, label="UNAIDS")
green_cross = mlines.Line2D(
    [], [], linewidth=0, color="g", marker="x", markersize=7, label="MPHIA"
)
orange_ci = mlines.Line2D([], [], color="black", marker=".", markersize=15, label="DHS")
plt.legend(handles=[red_line, blue_line, green_cross, orange_ci])
# plt.savefig(make_graph_file_name("HIV_Prevalence_in_Adults"))

plt.show()

# ---------------------------------------------------------------------- #

# HIV Incidence in adults aged 15-49 per 100 population
make_plot(
    title_str="HIV Incidence in Adults Aged 15-49 per 100 population",
    model=model_hiv_adult_inc["mean"] * 100,
    model_low=model_hiv_adult_inc["lower"] * 100,
    model_high=model_hiv_adult_inc["upper"] * 100,
    data_name="UNAIDS",
    data_mid=data_hiv_unaids["incidence_per1000_age15_49"] / 10,
    data_low=data_hiv_unaids["incidence_per1000_age15_49_lower"] / 10,
    data_high=data_hiv_unaids["incidence_per1000_age15_49_upper"] / 10,
)

plt.xlabel = ("Year",)
plt.ylabel = "HIV incidence per 1000 population"

# MPHIA
plt.errorbar(
    model_hiv_adult_inc.index[6],
    data_hiv_mphia_inc_estimate,
    yerr=[[data_hiv_mphia_inc_yerr[0]], [data_hiv_mphia_inc_yerr[1]]],
    fmt="gx",
)

plt.ylim(0, 1.0)
plt.xlim(2010, 2020)
#
# handles for legend
red_line = mlines.Line2D([], [], color="C3", markersize=15, label="TLO")
blue_line = mlines.Line2D([], [], color="C0", markersize=15, label="UNAIDS")
orange_ci = mlines.Line2D(
    [], [], color="green", marker="x", markersize=8, label="MPHIA"
)
plt.legend(handles=[red_line, blue_line, orange_ci])

# plt.savefig(make_graph_file_name("HIV_Incidence_in_Adults"))

plt.show()

# ---------------------------------------------------------------------- #

# HIV Prevalence Children
make_plot(
    title_str="HIV Prevalence in Children 0-14 (%)",
    model=model_hiv_child_prev["mean"] * 100,
    model_low=model_hiv_child_prev["lower"] * 100,
    model_high=model_hiv_child_prev["upper"] * 100,
    data_name="UNAIDS",
    data_mid=data_hiv_aidsinfo["prevalence_0_14"] * 100,
    data_low=data_hiv_aidsinfo["prevalence_0_14_lower"] * 100,
    data_high=data_hiv_aidsinfo["prevalence_0_14_upper"] * 100,
    xlab="Year",
    ylab="HIV prevalence (%)",
)

# MPHIA
plt.plot(
    model_hiv_child_prev.index[6],
    data_hiv_mphia_prev.loc[
        data_hiv_mphia_prev.age == "Total 0-14", "total percent hiv positive"
    ].values[0],
    "gx",
)

plt.xlim = (2010, 2020)
plt.ylim = (0, 5)

# handles for legend
red_line = mlines.Line2D([], [], color="C3", markersize=15, label="TLO")
blue_line = mlines.Line2D([], [], color="C0", markersize=15, label="UNAIDS")
green_cross = mlines.Line2D(
    [], [], linewidth=0, color="g", marker="x", markersize=7, label="MPHIA"
)
plt.legend(handles=[red_line, blue_line, green_cross])
# plt.savefig(make_graph_file_name("HIV_Prevalence_in_Children"))

plt.show()

# ---------------------------------------------------------------------- #

# HIV Incidence Children
make_plot(
    title_str="HIV Incidence in Children (0-14) (per 100 pyar)",
    model=model_hiv_child_inc["mean"] * 100,
    model_low=model_hiv_child_inc["lower"] * 100,
    model_high=model_hiv_child_inc["upper"] * 100,
    data_mid=data_hiv_aidsinfo["incidence0_14_per100py"],
    data_low=data_hiv_aidsinfo["incidence0_14_per100py_lower"],
    data_high=data_hiv_aidsinfo["incidence0_14_per100py_upper"],
)
# plt.savefig(make_graph_file_name("HIV_Incidence_in_Children"))

plt.show()

# ---------------------------------------------------------------------- #

# HIV prevalence among female sex workers:
make_plot(
    title_str="HIV Prevalence among Female Sex Workers (%)",
    model=model_hiv_fsw_prev["mean"] * 100,
    model_low=model_hiv_fsw_prev["lower"] * 100,
    model_high=model_hiv_fsw_prev["upper"] * 100,
)
# plt.savefig(make_graph_file_name("HIV_Prevalence_FSW"))

plt.show()

# ----------------------------- TB -------------------------------------- #

# Active TB incidence per 100,000 person-years - annual outputs

make_plot(
    title_str="Active TB Incidence (per 100k person-years)",
    model=activeTB_inc_rate["mean"],
    model_low=activeTB_inc_rate["lower"],
    model_high=activeTB_inc_rate["upper"],
    data_name="WHO_TB",
    data_mid=data_tb_who["incidence_per_100k"],
    data_low=data_tb_who["incidence_per_100k_low"],
    data_high=data_tb_who["incidence_per_100k_high"],
)

# plt.savefig(make_graph_file_name("TB_Incidence"))

plt.show()

# ---------------------------------------------------------------------- #

# proportion TB cases that are MDR

make_plot(
    title_str="Proportion of active TB cases that are MDR",
    model=model_tb_mdr["mean"],
    model_low=model_tb_mdr["lower"],
    model_high=model_tb_mdr["upper"],
)
# data from ResourceFile_TB sheet WHO_mdrTB2017
plt.errorbar(model_tb_mdr.index[7], 0.0075, yerr=[[0.0059], [0.0105]], fmt="o")
plt.legend(["TLO", "", "WHO"])
# plt.ylim((0, 15))
# plt.savefig(make_graph_file_name("Proportion_TB_Cases_MDR"))
plt.show()

# ---------------------------------------------------------------------- #

# proportion TB cases that are HIV+
# expect around 60% falling to 50% by 2017

make_plot(
    title_str="Proportion of active cases that are HIV+",
    model=model_tb_hiv_prop["mean"],
    model_low=model_tb_hiv_prop["lower"],
    model_high=model_tb_hiv_prop["upper"],
)
# plt.savefig(make_graph_file_name("Proportion_TB_Cases_MDR"))

plt.show()

# ---------------------------------------------------------------------- #
#
# # AIDS deaths (including HIV/TB deaths)
# make_plot(
#     title_str="Mortality to HIV-AIDS per 100,000 capita",
#     model=total_aids_deaths_rate_100kpy,
#     model_low=total_aids_deaths_rate_100kpy_lower,
#     model_high=total_aids_deaths_rate_100kpy_upper,
#     data_name="UNAIDS",
#     data_mid=data_hiv_unaids_deaths["AIDS_mortality_per_100k"],
#     data_low=data_hiv_unaids_deaths["AIDS_mortality_per_100k_lower"],
#     data_high=data_hiv_unaids_deaths["AIDS_mortality_per_100k_upper"],
# )
# plt.savefig(make_graph_file_name("AIDS_mortality"))
#
# plt.show()

# # ---------------------------------------------------------------------- #
#
# # AIDS/TB deaths
# make_plot(
#     title_str="Mortality to HIV-AIDS-TB per 100,000 capita",
#     model=total_aids_TB_deaths_rate_100kpy,
#     model_low=total_aids_TB_deaths_rate_100kpy_lower,
#     model_high=total_aids_TB_deaths_rate_100kpy_upper,
#     data_name="WHO",
#     data_mid=data_tb_who["mortality_tb_hiv_per_100k"],
#     data_low=data_tb_who["mortality_tb_hiv_per_100k_low"],
#     data_high=data_tb_who["mortality_tb_hiv_per_100k_high"],
# )
# plt.savefig(make_graph_file_name("AIDS_TB_mortality"))
#
# plt.show()
#
# # ---------------------------------------------------------------------- #
#
# # TB deaths (excluding HIV/TB deaths)
# make_plot(
#     title_str="TB mortality rate per 100,000 population",
#     model=tot_tb_non_hiv_deaths_rate_100kpy,
#     model_low=tot_tb_non_hiv_deaths_rate_100kpy_lower,
#     model_high=tot_tb_non_hiv_deaths_rate_100kpy_upper,
#     data_name="WHO",
#     data_mid=data_tb_who["mortality_tb_excl_hiv_per_100k"],
#     data_low=data_tb_who["mortality_tb_excl_hiv_per_100k_low"],
#     data_high=data_tb_who["mortality_tb_excl_hiv_per_100k_high"],
# )
# plt.savefig(make_graph_file_name("TB_mortality"))
#
# plt.show()

# ---------------------------------------------------------------------- #

# TB treatment coverage
make_plot(
    title_str="TB treatment coverage",
    model=model_tb_tx["mean"] * 100,
    model_low=model_tb_tx["lower"] * 100,
    model_high=model_tb_tx["upper"] * 100,
    data_name="NTP",
    data_mid=data_tb_ntp["treatment_coverage"],
)
# plt.savefig(make_graph_file_name("TB_treatment_coverage"))

plt.show()

# ---------------------------------------------------------------------- #

# HIV treatment coverage
make_plot(
    title_str="HIV treatment coverage",
    model=model_hiv_tx["mean"] * 100,
    model_low=model_hiv_tx["lower"] * 100,
    model_high=model_hiv_tx["upper"] * 100,
    data_name="UNAIDS",
    data_mid=data_hiv_unaids["ART_coverage_all_HIV_adults"],
    data_low=data_hiv_unaids["ART_coverage_all_HIV_adults_lower"],
    data_high=data_hiv_unaids["ART_coverage_all_HIV_adults_upper"],
)

# plt.savefig(make_graph_file_name("HIV_treatment_coverage"))

plt.show()

# ---------------------------------------------------------------------- #
# %%: DEATHS - GBD COMPARISON
# ---------------------------------------------------------------------- #
# get numbers of deaths from model runs
results_deaths = extract_results(
    results_folder,
    module="tlo.methods.demography",
    key="death",
    custom_generate_series=(
        lambda df: df.assign(year=df["date"].dt.year).groupby(
            ["year", "cause"])["person_id"].count()
    ),
    do_scaling=True,
)

results_deaths = results_deaths.reset_index()

# results_deaths.columns.get_level_values(1)
# Index(['', '', 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2], dtype='object', name='run')
#
# results_deaths.columns.get_level_values(0)  # this is higher level
# Index(['year', 'cause', 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3], dtype='object', name='draw')

# AIDS deaths
# select cause of death
tmp = results_deaths.loc[
    (results_deaths.cause == "AIDS_TB") | (results_deaths.cause == "AIDS_non_TB")
    ]
# select draw - drop columns where draw != 0, but keep year and cause
tmp2 = tmp.loc[
       :, ("draw" == draw)
       ].copy()  # selects only columns for draw=0 (removes year/cause)
# join year and cause back to df - needed for groupby
frames = [tmp["year"], tmp["cause"], tmp2]
tmp3 = pd.concat(frames, axis=1)

# create new column names, dependent on number of runs in draw
base_columns = ["year", "cause"]
run_columns = ["run" + str(x) for x in range(0, info["runs_per_draw"])]
base_columns.extend(run_columns)
tmp3.columns = base_columns
tmp3 = tmp3.set_index("year")

# sum rows for each year (2 entries)
# for each run need to combine deaths in each year, may have different numbers of runs
model_deaths_AIDS = pd.DataFrame(tmp3.groupby(["year"]).sum())

# double check all columns are float64 or quantile argument will fail
model_2010_median = model_deaths_AIDS.iloc[2].quantile(0.5)
model_2015_median = model_deaths_AIDS.iloc[5].quantile(0.5)
model_2010_low = model_deaths_AIDS.iloc[2].quantile(0.025)
model_2015_low = model_deaths_AIDS.iloc[5].quantile(0.025)
model_2010_high = model_deaths_AIDS.iloc[2].quantile(0.975)
model_2015_high = model_deaths_AIDS.iloc[5].quantile(0.975)

# get GBD estimates from any log_filepath
outputpath = Path("./outputs")  # folder for convenience of storing outputs
list_of_paths = outputpath.glob('*.log')  # gets latest log file
latest_path = max(list_of_paths, key=lambda p: p.stat().st_ctime)
death_compare = compare_number_of_deaths(latest_path, resourcefilepath)


# include all ages and both sexes
deaths2010 = death_compare.loc[("2010-2014", slice(None), slice(None), "AIDS")].sum()
deaths2015 = death_compare.loc[("2015-2019", slice(None), slice(None), "AIDS")].sum()

# include all ages and both sexes
deaths2010_TB = death_compare.loc[("2010-2014", slice(None), slice(None), "TB (non-AIDS)")].sum()
deaths2015_TB = death_compare.loc[("2015-2019", slice(None), slice(None), "TB (non-AIDS)")].sum()

x_vals = [1, 2, 3, 4]
labels = ["2010-2014", "2010-2014", "2015-2019", "2015-2019"]
col = ["mediumblue", "mediumseagreen", "mediumblue", "mediumseagreen"]
# handles for legend
blue_patch = mpatches.Patch(color="mediumblue", label="GBD")
green_patch = mpatches.Patch(color="mediumseagreen", label="TLO")

# plot AIDS deaths
y_vals = [
    deaths2010["GBD_mean"],
    model_2010_median,
    deaths2015["GBD_mean"],
    model_2015_median,
]
y_lower = [
    abs(deaths2010["GBD_lower"] - deaths2010["GBD_mean"]),
    abs(model_2010_low - model_2010_median),
    abs(deaths2015["GBD_lower"] - deaths2015["GBD_mean"]),
    abs(model_2015_low - model_2015_median),
]
y_upper = [
    abs(deaths2010["GBD_upper"] - deaths2010["GBD_mean"]),
    abs(model_2010_high - model_2010_median),
    abs(deaths2015["GBD_upper"] - deaths2015["GBD_mean"]),
    abs(model_2015_high - model_2015_median),
]
plt.bar(x_vals, y_vals, color=col)
plt.errorbar(
    x_vals, y_vals,
    yerr=[y_lower, y_upper],
    ls="none",
    marker="o",
    markeredgecolor="red",
    markerfacecolor="red",
    ecolor="red",
)
plt.xticks(ticks=x_vals, labels=labels)
plt.title("Deaths per year due to AIDS")
plt.legend(handles=[blue_patch, green_patch])
plt.tight_layout()
# plt.savefig(make_graph_file_name("AIDS_deaths_with_GBD"))
plt.show()

# -------------------------------------------------------------------------------------

# TB deaths
# select cause of death
tmp = results_deaths.loc[(results_deaths.cause == "TB")]
# select draw - drop columns where draw != 0, but keep year and cause
tmp2 = tmp.loc[
       :, ("draw" == draw)
       ].copy()  # selects only columns for draw=0 (removes year/cause)
# join year and cause back to df - needed for groupby
frames = [tmp["year"], tmp["cause"], tmp2]
tmp3 = pd.concat(frames, axis=1)

# create new column names, dependent on number of runs in draw
base_columns = ["year", "cause"]
run_columns = ["run" + str(x) for x in range(0, info["runs_per_draw"])]
base_columns.extend(run_columns)
tmp3.columns = base_columns
tmp3 = tmp3.set_index("year")

# sum rows for each year (2 entries)
# for each run need to combine deaths in each year, may have different numbers of runs
model_deaths_TB = pd.DataFrame(tmp3.groupby(["year"]).sum())

# double check all columns are float64 or quantile argument will fail
model_2010_median = model_deaths_TB.iloc[2].quantile(0.5)
model_2015_median = model_deaths_TB.iloc[5].quantile(0.5)
model_2010_low = model_deaths_TB.iloc[2].quantile(0.025)
model_2015_low = model_deaths_TB.iloc[5].quantile(0.025)
model_2010_high = model_deaths_TB.iloc[2].quantile(0.975)
model_2015_high = model_deaths_TB.iloc[5].quantile(0.975)

x_vals = [1, 2, 3, 4, 5, 6]
labels = ["2010-2014", "2010-2014", "2010-2014", "2015-2019", "2015-2019", "2015-2019"]
col = ["mediumblue", "mediumseagreen", "red", "mediumblue", "mediumseagreen", "red"]
# handles for legend
blue_patch = mpatches.Patch(color="mediumblue", label="GBD")
green_patch = mpatches.Patch(color="mediumseagreen", label="WHO")
red_patch = mpatches.Patch(color="red", label="TLO")


# plot TB deaths
y_vals = [
    deaths2010_TB["GBD_mean"],
    deaths_2010_2014_average,
    model_2010_median,
    deaths2015_TB["GBD_mean"],
    deaths_2015_2019_average,
    model_2015_median,
]
y_lower = [
    abs(deaths2010_TB["GBD_lower"] - deaths2010_TB["GBD_mean"]),
    deaths_2010_2014_average_low,
    abs(model_2010_low - model_2010_median),
    abs(deaths2015_TB["GBD_lower"] - deaths2015_TB["GBD_mean"]),
    deaths_2015_2019_average_low,
    abs(model_2015_low - model_2015_median),
]
y_upper = [
    abs(deaths2010_TB["GBD_upper"] - deaths2010_TB["GBD_mean"]),
    deaths_2010_2014_average_high,
    abs(model_2010_high - model_2010_median),
    abs(deaths2015_TB["GBD_upper"] - deaths2015_TB["GBD_mean"]),
    deaths_2015_2019_average_high,
    abs(model_2015_high - model_2015_median),
]

plt.bar(x_vals, y_vals, color=col)
plt.errorbar(
    x_vals, y_vals,
    yerr=[y_lower, y_upper],
    ls="none",
    marker="o",
    markeredgecolor="lightskyblue",
    markerfacecolor="lightskyblue",
    ecolor="lightskyblue",
)
plt.xticks(ticks=x_vals, labels=labels)
plt.title("Deaths per year due to TB")
plt.legend(handles=[blue_patch, green_patch, red_patch])
plt.tight_layout()
# plt.savefig(make_graph_file_name("TB_deaths_with_GBD"))
plt.show()


# download all files (and get most recent [-1])
results0 = get_scenario_outputs("scenario0.py", outputspath)[-1]
results1 = get_scenario_outputs("scenario1.py", outputspath)[-1]
results2 = get_scenario_outputs("scenario2.py", outputspath)[-1]

# hcw time
hcw_time = pd.read_csv(resourcefilepath / "healthsystem/human_resources/definitions/ResourceFile_Appt_Time_Table.csv")

# colour scheme
berry = lacroix.colorList('CranRaspberry')  # ['#F2B9B8', '#DF7878', '#E40035', '#009A90', '#0054A4', '#001563']
baseline_colour = berry[5]  # '#001563'
sc1_colour = berry[3]  # '#009A90'
sc2_colour = berry[2]  # '#E40035'
# %% ---------------------------------- Fraction HCW time-------------------------------------

# fraction of HCW time
# output difference in fraction HCW time from baseline for each scenario
def summarise_frac_hcws(results_folder):
    capacity0 = extract_results(
        results0,
        module="tlo.methods.healthsystem.summary",
        key="Capacity",
        column="average_Frac_Time_Used_Overall",
    )

    capacity = extract_results(
        results_folder,
        module="tlo.methods.healthsystem.summary",
        key="Capacity",
        column="average_Frac_Time_Used_Overall",
    )

    tmp = (capacity.subtract(capacity0) / capacity0) * 100

    hcw = pd.DataFrame(index=capacity.index, columns=["median", "lower", "upper"])
    hcw["median"] = tmp.median(axis=1)
    hcw["lower"] = tmp.quantile(q=0.025, axis=1)
    hcw["upper"] = tmp.quantile(q=0.975, axis=1)

    return hcw


hcw1 = summarise_frac_hcws(results1)
hcw2 = summarise_frac_hcws(results2)

# %%:  ---------------------------------- DALYS ---------------------------------- #
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
    dalys.loc['TB (non-AIDS)'] = dalys.loc['TB (non-AIDS)'] + dalys.loc['non_AIDS_TB']
    dalys.drop(['non_AIDS_TB'], inplace=True)
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


dalys0 = return_daly_summary(results0)
dalys1 = return_daly_summary(results1)
dalys2 = return_daly_summary2(results2)

dalys0.loc['Column_Total'] = dalys0.sum(numeric_only=True, axis=0)
dalys1.loc['Column_Total'] = dalys1.sum(numeric_only=True, axis=0)
dalys2.loc['Column_Total'] = dalys2.sum(numeric_only=True, axis=0)

# create full table for export
daly_table = pd.DataFrame()
daly_table['scenario0'] = dalys0['median'].astype(str) + \
                          " (" + dalys0['lower'].astype(str) + " - " + \
                          dalys0['upper'].astype(str) + ")"
daly_table['scenario1'] = dalys1['median'].astype(str) + \
                          " (" + dalys1['lower'].astype(str) + " - " + \
                          dalys1['upper'].astype(str) + ")"
daly_table['scenario2'] = dalys2['median'].astype(str) + \
                          " (" + dalys2['lower'].astype(str) + " - " + \
                          dalys2['upper'].astype(str) + ")"

daly_table.to_csv(outputspath / "daly_summary.csv")
# extract dalys averted by each scenario relative to scenario 0
# comparison should be run-by-run
full_dalys0 = extract_results(
    results0,
    module='tlo.methods.healthburden',
    key='dalys_stacked',
    custom_generate_series=num_dalys_by_cause,
    do_scaling=True
)
full_dalys0.loc['TB (non-AIDS)'] = full_dalys0.loc['TB (non-AIDS)'] + full_dalys0.loc['non_AIDS_TB']
full_dalys0.drop(['non_AIDS_TB'], inplace=True)
full_dalys0.loc['Column_Total'] = full_dalys0.sum(numeric_only=True, axis=0)

full_dalys1 = extract_results(
    results1,
    module='tlo.methods.healthburden',
    key='dalys_stacked',
    custom_generate_series=num_dalys_by_cause,
    do_scaling=True
)
full_dalys1.loc['TB (non-AIDS)'] = full_dalys1.loc['TB (non-AIDS)'] + full_dalys1.loc['non_AIDS_TB']
full_dalys1.drop(['non_AIDS_TB'], inplace=True)
full_dalys1.loc['Column_Total'] = full_dalys1.sum(numeric_only=True, axis=0)

full_dalys2 = extract_results(
    results2,
    module='tlo.methods.healthburden',
    key='dalys_stacked',
    custom_generate_series=num_dalys_by_cause,
    do_scaling=True
)
full_dalys2.loc['Column_Total'] = full_dalys2.sum(numeric_only=True, axis=0)

writer = pd.ExcelWriter(r"outputs/t.mangal@imperial.ac.uk/full_dalys.xlsx")
full_dalys0.to_excel(writer, sheet_name='sc0')
full_dalys1.to_excel(writer, sheet_name='sc1')
full_dalys2.to_excel(writer, sheet_name='sc2')
writer.save()

# DALYs averted: baseline - scenario
# positive value will be DALYs averted due to interventions
# negative value will be higher DALYs reported, therefore increased health burden
sc1_sc0 = full_dalys0.subtract(full_dalys1, fill_value=0)
sc1_sc0_median = sc1_sc0.median(axis=1)
sc1_sc0_lower = sc1_sc0.quantile(q=0.025, axis=1)
sc1_sc0_upper = sc1_sc0.quantile(q=0.975, axis=1)

sc2_sc0 = full_dalys0.subtract(full_dalys2, fill_value=0)
sc2_sc0_median = sc2_sc0.median(axis=1)
sc2_sc0_lower = sc2_sc0.quantile(q=0.025, axis=1)
sc2_sc0_upper = sc2_sc0.quantile(q=0.975, axis=1)

# create full table for export
daly_averted_table = pd.DataFrame()
daly_averted_table['cause'] = sc1_sc0_median.index
daly_averted_table['scenario1_med'] = [int(round(x, -3)) for x in sc1_sc0_median]
daly_averted_table['scenario1_low'] = [int(round(x, -3)) for x in sc1_sc0_lower]
daly_averted_table['scenario1_upp'] = [int(round(x, -3)) for x in sc1_sc0_upper]
daly_averted_table['scenario2_med'] = [int(round(x, -3)) for x in sc2_sc0_median]
daly_averted_table['scenario2_low'] = [int(round(x, -3)) for x in sc2_sc0_lower]
daly_averted_table['scenario2_upp'] = [int(round(x, -3)) for x in sc2_sc0_upper]

daly_averted_table.to_csv(outputspath / "daly_averted_summary.csv")

# this is now unconstrained scenario first!!
aids_dalys_diff = [sc2_sc0_median['AIDS'],
                   sc1_sc0_median['AIDS']]
tb_dalys_diff = [sc2_sc0_median['TB (non-AIDS)'],
                 sc1_sc0_median['TB (non-AIDS)']]
total_dalys_diff = [sc2_sc0_median['Column_Total'],
                    sc1_sc0_median['Column_Total']]

# -------------------------- plots ---------------------------- #
# plt.style.use('ggplot')
#
# aids_colour = "#8949ab"
# tb_colour = "#ed7e7a"
# total_colour = "#eede77"
#
# years = list((range(2010, 2036, 1)))
# years_num = pd.Series(years)
## fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2,
#                                figsize=(14, 6))
# # constrained_layout=True)
# fig.suptitle('')
#
# # HCW time
# # labels = ['Baseline', 'Constrained scale-up', 'Unconstrained scale-up']
# # x = np.arange(len(labels))  # the label locations
# width = 0.2  # the width of the bars
#
# ax1.bar(years_num[13:26], hcw1["median"].loc[13:25], width, color=sc1_colour)
# ax1.bar(years_num[13:26] + width, hcw2["median"].loc[13:25], width, color=sc2_colour)
#
# ax1.set_ylabel("% difference HCW time", rotation=90, labelpad=15)
# # ax1.set_ylim([-0.5, 1.5])
#
# ax1.yaxis.set_label_position("left")
# ax1.legend(["Constrained scale-up", "Unconstrained scale-up"], frameon=False)
#
# # DALYs
# labels = ['Constrained scale-up', 'Unconstrained scale-up']
# x = np.arange(len(labels))  # the label locations
# width = 0.2  # the width of the bars
#
# rects1 = ax2.bar(x - width, aids_dalys_diff, width, label='AIDS', color=aids_colour)
# rects2 = ax2.bar(x, tb_dalys_diff, width, label='TB', color=tb_colour)
# rects3 = ax2.bar(x + width, total_dalys_diff, width, label='Total', color=total_colour)
#
# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax2.set_ylabel('DALYs')
# ax2.set_title('')
# ax2.set_xticks(x)
# ax2.set_xticklabels(labels)
# ax2.legend(["AIDS", "TB", "Total"], frameon=False)
#
# font = {'family': 'sans-serif',
#         'color': 'black',
#         'weight': 'bold',
#         'size': 11,
#         }
#
# ax1.text(-0.15, 1.05, 'A)', horizontalalignment='center',
#          verticalalignment='center', transform=ax1.transAxes, fontdict=font)
#
# ax2.text(-0.1, 1.05, 'B)', horizontalalignment='center',
#          verticalalignment='center', transform=ax2.transAxes, fontdict=font)
#
# fig.tight_layout()
# fig.savefig(outputspath / "HCW_DALYS.png")
#
# plt.show()


