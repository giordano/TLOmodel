"""This file uses the results of the batch file to make some summary statistics.
The results of the batchrun were put into the 'outputspath' results_folder
function weighted_mean_for_data_comparison can be used to select which parameter sets to use
make plots for top 5 parameter sets just to make sure they are looking ok
"""

import datetime
import pickle
from pathlib import Path
from tlo import Date
import os
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from tlo.analysis.utils import (
    compare_number_of_deaths,
    extract_params,
    extract_results,
    get_scenario_info,
    get_scenario_outputs,
    load_pickled_dataframes,
    summarize,
)
plt.style.use('seaborn-darkgrid')

# Set the working directory
os.chdir('/Users/wenjiazhang/Documents/MSc_HDA/Summer/TLOmodel/')
resourcefilepath = Path("./resources")
outputspath = Path("./outputs/wz2016@ic.ac.uk/")

datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# %% read in data files for plots
# load all the data for calibration

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
data_hiv_moh_art = pd.read_excel(xls, sheet_name="MoH_number_art")

# %% Analyse results of runs

# 0) Find results_folder associated with a given batch_file (and get most recent [-1])
results_folder = get_scenario_outputs("batch_prep_run-2023-08-18T231546Z.py", outputspath)[-1]

# Declare path for output graphs from this script
make_graph_file_name = lambda stub: results_folder / f"{stub}.png"  # noqa: E731

# look at one log (so can decide what to extract)
log = load_pickled_dataframes(results_folder)

# get basic information about the results -  to get basic information about the scenario - specifically the number of draws number_of_draws and number of runs per draw runs_per_draw as a dictionary.
info = get_scenario_info(results_folder)

# 1) Extract the parameters that have varied over the set of simulations
params = extract_params(results_folder)

# choose which draw to summarise / visualise
# draw = 1

# %% extract results
# Load and format model results (with year as integer):
def extract_total_deaths(results_folder):

    def extract_deaths_total(df: pd.DataFrame) -> pd.Series:
        return pd.Series({"Total": len(df)})

    return extract_results(
        results_folder,
        module="tlo.methods.demography",
        key="death",
        custom_generate_series=extract_deaths_total,
        do_scaling=True
    )
def plot_summarized_total_deaths(summarized_total_deaths):
    fig, ax = plt.subplots()

    draws = list(summarized_total_deaths.keys())
    means = [summarized_total_deaths[draw]['mean'] for draw in draws]
    lowers = [summarized_total_deaths[draw]['lower'] for draw in draws]
    uppers = [summarized_total_deaths[draw]['upper'] for draw in draws]

    ax.bar(
        draws,
        means,
        yerr=[np.array(means) - np.array(lowers), np.array(uppers) - np.array(means)],
        tick_label=draws
    )
    ax.set_ylabel("Total number of deaths")
    ax.set_xlabel("Draw Number")
    fig.tight_layout()
    return fig, ax

def compute_difference_in_deaths_across_runs(total_deaths, info):
    deaths_difference_by_run = [
        total_deaths[0][run_number]["Total"] - total_deaths[1][run_number]["Total"]
        for run_number in range(scenario_info["runs_per_draw"])
    ]
    return np.mean(deaths_difference_by_run)

def summarize(total_deaths, collapse_columns=False):
    summary = {}
    for draw, deaths_df in total_deaths.items():
        total_values = deaths_df.loc['Total'].tolist()

        summary[draw] = {
            'mean': np.mean(total_values),
            'lower': np.percentile(total_values, 5),
            'upper': np.percentile(total_values, 95)
        }
    return summary

def plot_summarized_total_deaths(summarized_total_deaths):
    fig, ax = plt.subplots()

    draws = list(summarized_total_deaths.keys())
    means = [summarized_total_deaths[draw]['mean'] for draw in draws]
    lowers = [summarized_total_deaths[draw]['lower'] for draw in draws]
    uppers = [summarized_total_deaths[draw]['upper'] for draw in draws]

    ax.bar(
        draws,
        means,
        yerr=[np.array(means) - np.array(lowers), np.array(uppers) - np.array(means)],
        tick_label=draws
    )
    ax.set_ylabel("Total number of deaths")
    ax.set_xlabel("Draw Number")
    fig.tight_layout()
    return fig, ax


# We first look at total deaths in the scenario runs for each draw
all_total_deaths = {draw: extract_total_deaths(results_folder) for draw in range(info['number_of_draws'])}

# Compute the summary statistics for each draw
summarized_deaths = summarize(all_total_deaths)

# Plot the summarized deaths for each draw
fig_1, ax_1 = plot_summarized_total_deaths(summarized_deaths)
plt.show()

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

model_hiv_female_15plus_prev = summarize(
    extract_results(
        results_folder,
        module="tlo.methods.hiv",
        key="summary_inc_and_prev_for_adults_and_children_and_fsw",
        column="female_prev_15plus",
        index="date",
        do_scaling=False,
    ),
    collapse_columns=True,
)
model_hiv_female_15plus_prev.index = model_hiv_female_15plus_prev.index.year

model_preg_women_prep = summarize(
    extract_results(
        results_folder,
        module="tlo.methods.hiv",
        key="prep_status_logging",
        column="prop_pregnant_women_on_prep",
        index="date",
        do_scaling=False,
    ),
    collapse_columns=True,
)
model_preg_women_prep.index = model_preg_women_prep.index.year

model_breastfeeding_women_prep = summarize(
    extract_results(
        results_folder,
        module="tlo.methods.hiv",
        key="prep_status_logging",
        column="prop_breastfeeding_women_on_prep",
        index="date",
        do_scaling=False,
    ),
    collapse_columns=True,
)
model_breastfeeding_women_prep.index = model_breastfeeding_women_prep.index.year

model_fsw_prep = summarize(
    extract_results(
        results_folder,
        module="tlo.methods.hiv",
        key="prep_status_logging",
        column="prop_fsw_on_prep",
        index="date",
        do_scaling=False,
    ),
    collapse_columns=True,
)
model_fsw_prep.index = model_fsw_prep.index.year

model_females_prep = summarize(
    extract_results(
        results_folder,
        module="tlo.methods.hiv",
        key="prep_status_logging",
        column="total_females_on_prep",
        index="date",
        do_scaling=False,
    ),
    collapse_columns=True,
)
model_females_prep.index = model_females_prep.index.year

model_females_prep = summarize(
    extract_results(
        results_folder,
        module="tlo.methods.hiv",
        key="prep_status_logging",
        column="total_females_on_prep",
        index="date",
        do_scaling=False,
    ),
    collapse_columns=True,
)
model_females_prep.index = model_females_prep.index.year

# model_ancvisit = summarize(
#    extract_results(
#     results_folder,
#       module="tlo.methods.care_of_women_during_pregnancy",
#       key="anc_proportion_on_birth",
#       column="proportion_attended_at_least_one_anc",
#        index="date",
#        do_scaling=False,
#    ),
#    collapse_columns=True,
# )
# model_ancvisit.index = model_ancvisit.index.year
# ---------------------------------- PERSON-YEARS ---------------------------------- #
# function to extract person-years by year
# call this for each run and then take the mean to use as denominator for mortality / incidence etc.
#def get_person_years(draw, run):
#    log = load_pickled_dataframes(results_folder, draw, run)

#    if "tlo.methods.demography" not in log:
#        print(f"Missing 'tlo.methods.demography' for draw: {draw}, run: {run}")
#        return pd.Series(dtype="int64")

#    if "person_years" not in log["tlo.methods.demography"]:
#        print(f"Missing 'person_years' inside 'tlo.methods.demography' for draw: {draw}, run: {run}")
#        return pd.Series(dtype="int64")

#    py_ = log["tlo.methods.demography"]["person_years"]
#    years = pd.to_datetime(py_["date"]).dt.year
#    py = pd.Series(dtype="int64", index=years)
#    for year in years:
#        tot_py = (
#            (py_.loc[pd.to_datetime(py_["date"]).dt.year == year]["M"]).apply(pd.Series) +
#            (py_.loc[pd.to_datetime(py_["date"]).dt.year == year]["F"]).apply(pd.Series)
#        ).transpose()
#        py[year] = tot_py.sum().values[0]

#    py.index = pd.to_datetime(years, format="%Y")

#    return py

# for draw 0, get py for all runs
#number_runs = info["runs_per_draw"]
#py_summary = pd.DataFrame(data=None, columns=range(0, number_runs))

# draw number (default = 0) is specified above
#for run in range(0, number_runs):
#    py_summary.iloc[:, run] = get_person_years(draw, run)

#py_summary["mean"] = py_summary.mean(axis=1)


def get_person_years(draw, run):
    log = load_pickled_dataframes(results_folder, draw, run)

    if "tlo.methods.demography" not in log:
        print(f"Missing 'tlo.methods.demography' for draw: {draw}, run: {run}")
        return None

    if "person_years" not in log["tlo.methods.demography"]:
        print(f"Missing 'person_years' inside 'tlo.methods.demography' for draw: {draw}, run: {run}")
        return None

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

number_runs = info["runs_per_draw"]
py_summary = pd.DataFrame()

for run in range(0, number_runs):
    py_run = get_person_years(draw, run)

    if py_run is not None:
        py_summary[run] = py_run

py_summary["mean"] = py_summary.mean(axis=1)

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

# ------------ summarise deaths producing df for each draw

# AIDS deaths
aids_deaths = {}  # dict of df

for draw in range(info["number_of_draws"]):
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

for draw in range(info["number_of_draws"]):
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


# ---------------------------------- PROGRAM COVERAGE ---------------------------------- #
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

    # Adjust x-ticks to show every half year
    #total_months = len(model.index)
    #ticks = np.arange(0, total_months, 6)  # every 6th month
    #ax.set_xticks(ticks)
    #ax.set_xticklabels(model.index[ticks], rotation=45)  # Using rotation for better visibility

    #plt.tight_layout()
    # plt.show()

# %% make plots

# HIV - prevalence among in adults aged 15-49
make_plot(
    title_str="HIV Prevalence in Adults Aged 15-49 (%)",
    model=model_hiv_adult_prev[draw,"mean"] * 100,
    model_low=model_hiv_adult_prev[draw,"lower"] * 100,
    model_high=model_hiv_adult_prev[draw,"upper"] * 100,
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
    model=model_hiv_adult_inc[(draw,"mean")] * 100,
    model_low=model_hiv_adult_inc[(draw,"lower")] * 100,
    model_high=model_hiv_adult_inc[(draw,"upper")] * 100,
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
    model=model_hiv_child_prev[(draw,"mean")] * 100,
    model_low=model_hiv_child_prev[(draw,"lower")] * 100,
    model_high=model_hiv_child_prev[(draw,"upper")] * 100,
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
    model=model_hiv_child_inc[(draw,"mean")] * 100,
    model_low=model_hiv_child_inc[(draw,"lower")] * 100,
    model_high=model_hiv_child_inc[(draw,"upper")] * 100,
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
    model=model_hiv_fsw_prev[(draw,"mean")] * 100,
    model_low=model_hiv_fsw_prev[(draw,"lower")] * 100,
    model_high=model_hiv_fsw_prev[(draw,"upper")] * 100,
)
#plt.savefig(make_graph_file_name("HIV_Prevalence_FSW"))
plt.show()


# ---------------------------------------------------------------------- #

# HIV prevalence among female aged 15 above
make_plot(
    title_str="HIV Prevalence among Females above 15+ (%)",
    model=model_hiv_female_15plus_prev[(draw,"mean")] * 100,
)
plt.show()

# ------------------------PrEP intervention ------------------------------#
# ----------------------- ANC visits
# make_plot(
#    title_str="Proportion of Pregnant Women Attending >=1 ANC visits",
#     model=model_ancvist[(draw,"mean")] * 100,
# )
# plt.show()
# -----------------------PrEP
# PrEP among FSW
make_plot(
    title_str="Proportion of Female Sex Workers That Are On PrEP(%)",
    model=model_fsw_prep[(draw,"mean")] * 100,
    model_low=model_fsw_prep[(draw,"lower")] * 100,
    model_high=model_fsw_prep[(draw,"upper")] * 100,
)
plt.show()

# PrEP among pregnant women
make_plot(
    title_str="Proportion of Pregnant Women That Are On PrEP(%)",
    model=model_preg_women_prep[(draw,"mean")] * 100,
    model_low=model_preg_women_prep[(draw,"lower")] * 100,
    model_high=model_preg_women_prep[(draw,"upper")] * 100,
)
plt.show()

# PrEP among breastfeeding women
make_plot(
    title_str="Proportion of Breastfeeding Women That Are On PrEP(%)",
    model=model_breastfeeding_women_prep[(draw,"mean")] * 100,
    model_low=model_breastfeeding_women_prep[(draw,"lower")] * 100,
    model_high=model_breastfeeding_women_prep[(draw,"upper")] * 100,
)
plt.show()

# Total Females on PrEP
make_plot(
    title_str="Proportion of Females That Are On PrEP(%)",
    model=model_females_prep[(draw,"mean")] * 100,
)
plt.show()

# ---------------------------------------------------------------------- #

# HIV treatment coverage
make_plot(
    title_str="HIV treatment coverage",
    model=model_hiv_tx[(draw,"mean")] * 100,
    model_low=model_hiv_tx[(draw,"mean")] * 100,
    model_high=model_hiv_tx[(draw,"mean")] * 100,
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

#  ---------------------------------- DALYS ---------------------------------- #
dalys_list = []

for idx, results_folder in enumerate(scenario_folders):
    if idx < 2:
        dalys = return_daly_summary(results_folder)
    else:
        dalys = return_daly_summary2(results_folder)

    dalys.loc['Column_Total'] = dalys.sum(numeric_only=True, axis=0)
    dalys_list.append(dalys)

daly_table = pd.DataFrame()

for idx, dalys in enumerate(dalys_list):
    daly_table[f'scenario{idx}'] = dalys['median'].astype(str) + \
                                  " (" + dalys['lower'].astype(str) + " - " + \
                                  dalys['upper'].astype(str) + ")"

writer = pd.ExcelWriter(outputspath / "full_dalys.xlsx")

for idx, results_folder in enumerate(scenario_folders):
    full_dalys = extract_results(
        results_folder,
        module='tlo.methods.healthburden',
        key='dalys_stacked',
        custom_generate_series=num_dalys_by_cause,
        do_scaling=True
    )
    if idx < 2:
        full_dalys.loc['AIDS)'] = full_dalys.loc['TB (non-AIDS)'] + full_dalys.loc['non_AIDS_TB']
        full_dalys.drop(['non_AIDS_TB'], inplace=True)
    full_dalys.loc['Column_Total'] = full_dalys.sum(numeric_only=True, axis=0)

    full_dalys.to_excel(writer, sheet_name=f'sc{idx}')

writer.save()



results0 = get_scenario_outputs("batch_prep_run-2023-08-18T231546Z.py", outputspath)[-1]
results1 = get_scenario_outputs("batch_prep_run-2023-08-18T231546Z.py", outputspath)[-1]
results2 = get_scenario_outputs("batch_prep_run-2023-08-18T231546Z.py", outputspath)[-1]
results3 = get_scenario_outputs("batch_prep_run-2023-08-18T231546Z.py", outputspath)[-1]

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
full_dalys0.loc['AIDS)'] = full_dalys0.loc['TB (non-AIDS)'] + full_dalys0.loc['non_AIDS_TB']
full_dalys0.drop(['non_AIDS_TB'], inplace=True)
full_dalys0.loc['Column_Total'] = full_dalys0.sum(numeric_only=True, axis=0)

full_dalys1 = extract_results(
    results1,
    module='tlo.methods.healthburden',
    key='dalys_stacked',
    custom_generate_series=num_dalys_by_cause,
    do_scaling=True
)
full_dalys1.loc['AIDS'] = full_dalys1.loc['AIDS']


full_dalys2 = extract_results(
    results2,
    module='tlo.methods.healthburden',
    key='dalys_stacked',
    custom_generate_series=num_dalys_by_cause,
    do_scaling=True
)
full_dalys2.loc['Column_Total'] = full_dalys2.sum(numeric_only=True, axis=0)

writer = pd.ExcelWriter(r"outputs/wz2016@ic.ac.uk/full_dalys.xlsx")
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


# -------------------------- plots ---------------------------- #
# plt.style.use('ggplot')
#
# aids_colour = "#8949ab"
# tb_colour = "#ed7e7a"
# total_colour = "#eede77"
#
# years = list((range(2010, 2036, 1)))
# years_num = pd.Series(years)
#
# fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2,
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

# -------------------------- DALYs only plot ---------------------------- #

plt.style.use('ggplot')

aids_colour = "#8949ab"

# present DALYs in millions
million = 1000000
aids_dalys_diff = [x / million for x in aids_dalys_diff]
total_dalys_diff = [x / million for x in total_dalys_diff]

fig, ax1 = plt.subplots(nrows=1, ncols=1,
                                             figsize=(5, 4))
fig.suptitle('')

# DALYs
labels = ['Unconstrained scale-up', 'Constrained scale-up']
x = np.arange(len(labels))  # the label locations
width = 0.2  # the width of the bars

rects1 = ax1.bar(x - width, aids_dalys_diff, width, label='AIDS', color=aids_colour)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax1.set_ylabel('DALYs averted, millions')
ax1.set_title('')
ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.legend(["AIDS"], frameon=False)

font = {'family': 'sans-serif',
        'color': 'black',
        'weight': 'bold',
        'size': 11,
        }

fig.tight_layout()
fig.savefig(outputspath / "DALYS.png")

plt.show()
