import datetime
import pickle
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from tlo.analysis.utils import extract_results, get_scenario_outputs
from tlo.analysis.utils import compare_number_of_deaths

plt.style.use('seaborn-darkgrid')

# Set the working directory
os.chdir('/Users/wenjiazhang/Documents/MSc_HDA/Summer/TLOmodel/')

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
    prev_and_inc_over_time.index[1],
    data_hiv_mphia_prev.loc[
        data_hiv_mphia_prev.age == "Total 15-49", "total percent hiv positive"
    ].values[0],
    "gx",
)

# DHS
x_values = [prev_and_inc_over_time.index[0], prev_and_inc_over_time.index[1]]
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
    prev_and_inc_over_time.index[1],
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
    prev_and_inc_over_time.index[1],
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

# ------------------------PrEP intervention -------------------------------- #
# PrEP
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
    title_str="Proportion of  Females That Are On PrEP",
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

