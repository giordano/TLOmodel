"""This file uses the results of the batch file to make some summary statistics.
The results of the batchrun were put into the 'outputs' results_folder
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.lines as mlines

import numpy as np
import pandas as pd
import datetime

from tlo.analysis.utils import (
    extract_params,
    extract_results,
    get_grid,
    get_scenario_info,
    get_scenario_outputs,
    load_pickled_dataframes,
    summarize,
)
resourcefilepath = Path("./resources")
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

outputspath = Path('./outputs/t.mangal@imperial.ac.uk')

# %% read in data files for plots
# load all the data for calibration

# TB WHO data
xls_tb = pd.ExcelFile(resourcefilepath / 'ResourceFile_TB.xlsx')

data_tb_who = pd.read_excel(xls_tb, sheet_name='WHO_activeTB2020')
data_tb_who = data_tb_who.loc[(data_tb_who.year >= 2010)]  # include only years post-2010
data_tb_who.index = pd.to_datetime(data_tb_who['year'], format='%Y')
data_tb_who = data_tb_who.drop(columns=['year'])

# TB latent data (Houben & Dodd 2016)
data_tb_latent = pd.read_excel(xls_tb, sheet_name='latent_TB2014_summary')
data_tb_latent_all_ages = data_tb_latent.loc[data_tb_latent.Age_group == '0_80']
data_tb_latent_estimate = data_tb_latent_all_ages.proportion_latent_TB.values[0]
data_tb_latent_lower = abs(data_tb_latent_all_ages.proportion_latent_TB_lower.values[0] - data_tb_latent_estimate)
data_tb_latent_upper = abs(data_tb_latent_all_ages.proportion_latent_TB_upper.values[0] - data_tb_latent_estimate)
data_tb_latent_yerr = [data_tb_latent_lower, data_tb_latent_upper]

# TB treatment coverage
data_tb_ntp = pd.read_excel(xls_tb, sheet_name='NTP2019')
data_tb_ntp.index = pd.to_datetime(data_tb_ntp['year'], format='%Y')
data_tb_ntp = data_tb_ntp.drop(columns=['year'])

# HIV resourcefile
xls = pd.ExcelFile(resourcefilepath / 'ResourceFile_HIV.xlsx')

# HIV UNAIDS data
data_hiv_unaids = pd.read_excel(xls, sheet_name='unaids_infections_art2021')
data_hiv_unaids.index = pd.to_datetime(data_hiv_unaids['year'], format='%Y')
data_hiv_unaids = data_hiv_unaids.drop(columns=['year'])

# HIV UNAIDS data
data_hiv_unaids_deaths = pd.read_excel(xls, sheet_name='unaids_mortality_dalys2021')
data_hiv_unaids_deaths.index = pd.to_datetime(data_hiv_unaids_deaths['year'], format='%Y')
data_hiv_unaids_deaths = data_hiv_unaids_deaths.drop(columns=['year'])

# AIDSinfo (UNAIDS)
data_hiv_aidsinfo = pd.read_excel(xls, sheet_name='children0_14_prev_AIDSinfo')
data_hiv_aidsinfo.index = pd.to_datetime(data_hiv_aidsinfo['year'], format='%Y')
data_hiv_aidsinfo = data_hiv_aidsinfo.drop(columns=['year'])

# unaids program performance
data_hiv_program = pd.read_excel(xls, sheet_name='unaids_program_perf')
data_hiv_program.index = pd.to_datetime(data_hiv_program['year'], format='%Y')
data_hiv_program = data_hiv_program.drop(columns=['year'])

# MPHIA HIV data - age-structured
data_hiv_mphia_inc = pd.read_excel(xls, sheet_name='MPHIA_incidence2015')
data_hiv_mphia_inc_estimate = data_hiv_mphia_inc.loc[
             (data_hiv_mphia_inc.age == "15-49"), "total_percent_annual_incidence"].values[0]
data_hiv_mphia_inc_lower = data_hiv_mphia_inc.loc[
             (data_hiv_mphia_inc.age == "15-49"), "total_percent_annual_incidence_lower"].values[0]
data_hiv_mphia_inc_upper = data_hiv_mphia_inc.loc[
             (data_hiv_mphia_inc.age == "15-49"), "total_percent_annual_incidence_upper"].values[0]
data_hiv_mphia_inc_yerr = [abs(data_hiv_mphia_inc_lower - data_hiv_mphia_inc_estimate),
                           abs(data_hiv_mphia_inc_upper - data_hiv_mphia_inc_estimate)]

data_hiv_mphia_prev = pd.read_excel(xls, sheet_name='MPHIA_prevalence_art2015')

# DHS HIV data
data_hiv_dhs_prev = pd.read_excel(xls, sheet_name='DHS_prevalence')

# MoH HIV testing data
data_hiv_moh_tests = pd.read_excel(xls, sheet_name='MoH_numbers_tests')
data_hiv_moh_tests.index = pd.to_datetime(data_hiv_moh_tests['year'], format='%Y')
data_hiv_moh_tests = data_hiv_moh_tests.drop(columns=['year'])

# MoH HIV ART data
# todo this is quarterly
data_hiv_moh_art = pd.read_excel(xls, sheet_name='MoH_number_art')


# %% Analyse results of runs

# 0) Find results_folder associated with a given batch_file (and get most recent [-1])
results_folder = get_scenario_outputs('calibration.py', outputspath)[-1]

# Declare path for output graphs from this script
make_graph_file_name = lambda stub: results_folder / f"{stub}.png"  # noqa: E731

# look at one log (so can decide what to extract)
log = load_pickled_dataframes(results_folder)

# get basic information about the results
info = get_scenario_info(results_folder)

# 1) Extract the parameters that have varied over the set of simulations
params = extract_params(results_folder)

# %% Function to make standard plot to compare model and data
def make_plot(
    model=None,
    model_low=None,
    model_high=None,
    data_name=None,
    data_mid=None,
    data_low=None,
    data_high=None,
    xlim=None,
    ylim=None,
    xlab=None,
    ylab=None,
    title_str=None
):
    assert model is not None
    assert title_str is not None

    # Make plot
    fig, ax = plt.subplots()
    ax.plot(model.index, model.values, '-', color='C3')
    if (model_low is not None) and (model_high is not None):
        ax.fill_between(model_low.index,
                        model_low,
                        model_high,
                        color='C3',
                        alpha=0.2)

    if data_mid is not None:
        ax.plot(data_mid.index, data_mid.values, '-', color='g')
    if (data_low is not None) and (data_high is not None):
        ax.fill_between(data_low.index,
                        data_low,
                        data_high,
                        color='g',
                        alpha=0.2)

    if xlim is not None:
        ax.set_xlim(xlim)

    if ylim is not None:
        ax.set_xlim(ylim)

    if xlab is not None:
        ax.set_xlabel(xlab)

    if ylab is not None:
        ax.set_xlabel(ylab)

    plt.title(title_str)
    plt.legend(['Model', data_name])
    plt.gca().set_ylim(bottom=0)
    plt.savefig(outputspath / (title_str.replace(" ", "_") + datestamp + ".pdf"), format='pdf')


# %% extract results
# Load and format model results (with year as integer):
model_hiv_adult_prev = summarize(extract_results(results_folder,
                                      module="tlo.methods.hiv",
                                      key="summary_inc_and_prev_for_adults_and_children_and_fsw",
                                      column="hiv_prev_adult_15plus",
                                      index="date",
                                      do_scaling=False
                                      ),
                      collapse_columns=True
                      )
model_hiv_adult_prev.index = model_hiv_adult_prev.index.year

model_hiv_adult_inc = summarize(extract_results(results_folder,
                                      module="tlo.methods.hiv",
                                      key="summary_inc_and_prev_for_adults_and_children_and_fsw",
                                      column="hiv_adult_inc_1549",
                                      index="date",
                                      do_scaling=False
                                      ),
                      collapse_columns=True
                      )
model_hiv_adult_inc.index = model_hiv_adult_inc.index.year

model_hiv_child_prev = summarize(extract_results(results_folder,
                                      module="tlo.methods.hiv",
                                      key="summary_inc_and_prev_for_adults_and_children_and_fsw",
                                      column="hiv_prev_child",
                                      index="date",
                                      do_scaling=False
                                      ),
                      collapse_columns=True
                      )
model_hiv_child_prev.index = model_hiv_child_prev.index.year




# %% make plots
# HIV - prevalence among in adults aged 15-49

make_plot(
    title_str="HIV Prevalence in Adults Aged 15-49 (%)",
    model=model_hiv_adult_prev['mean'] * 100,
    model_low=model_hiv_adult_prev['lower'] * 100,
    model_high=model_hiv_adult_prev['upper'] * 100,
    data_name='UNAIDS',
    data_mid=data_hiv_unaids['prevalence_age15_49'],
    data_low=data_hiv_unaids['prevalence_age15_49_lower'],
    data_high=data_hiv_unaids['prevalence_age15_49_upper'],
    xlim=[2010, 2020],
    ylim=[0, 15],
    xlab="Year",
    ylab="HIV prevalence (%)"
)

# data: MPHIA
plt.plot(model_hiv_adult_prev.index[6], data_hiv_mphia_prev.loc[
    data_hiv_mphia_prev.age == "Total 15-49", "total percent hiv positive"].values[0], 'gx')

# data: DHS
x_values = [model_hiv_adult_prev.index[0], model_hiv_adult_prev.index[5]]
y_values = data_hiv_dhs_prev.loc[
                (data_hiv_dhs_prev.Year >= 2010), "HIV prevalence among general population 15-49"]
y_lower = abs(y_values - (data_hiv_dhs_prev.loc[
                (data_hiv_dhs_prev.Year >= 2010), "HIV prevalence among general population 15-49 lower"]))
y_upper = abs(y_values - (data_hiv_dhs_prev.loc[
                (data_hiv_dhs_prev.Year >= 2010), "HIV prevalence among general population 15-49 upper"]))
plt.errorbar(x_values, y_values,
             yerr=[y_lower, y_upper], fmt='o')

# handles for legend
red_line = mlines.Line2D([], [], color='C3',
                         markersize=15, label='TLO')
blue_line = mlines.Line2D([], [], color='C0',
                          markersize=15, label='UNAIDS')
green_cross = mlines.Line2D([], [], linewidth=0, color='g', marker='x',
                          markersize=7, label='MPHIA')
orange_ci = mlines.Line2D([], [], color='C1', marker='.',
                          markersize=15, label='DHS')
plt.legend(handles=[red_line, blue_line, green_cross, orange_ci])

plt.show()


# ---------------------------------------------------------------------- #

# HIV Incidence 15-49 per 100 population
fig, ax = plt.subplots()
# model
ax.plot(model_hiv_adult_inc.index, model_hiv_adult_inc['mean'] * 100,
        label='TLO (mean)', color='C3')
ax.fill_between(model_hiv_adult_inc.index,
                model_hiv_adult_inc['lower'] * 100,
                model_hiv_adult_inc['upper'] * 100,
                color='C3',
                alpha=0.2,
                zorder=5
                )
# data: UNAIDS
ax.plot(model_hiv_adult_prev.index, data_hiv_unaids['incidence_per_1000'] / 10,
        label='UNAIDS', color='g')
ax.fill_between(data_hiv_unaids.index,
                data_hiv_unaids['incidence_per_1000_lower'] / 10,
                data_hiv_unaids['incidence_per_1000_upper'] / 10,
                color='g',
                alpha=0.2,
                zorder=5
                )
# MPHIA
plt.errorbar(model_hiv_adult_inc.index[6], data_hiv_mphia_inc_estimate,
             yerr=[[data_hiv_mphia_inc_yerr[0]], [data_hiv_mphia_inc_yerr[1]]], fmt='o')

ax.set_title("HIV Incidence in Adults Aged 15-49 per 100 population")
ax.set_xlabel("Year")
ax.set_ylabel("HIV incidence per 100 population")
ax.set_xlim(2010, 2020)
ax.set_ylim(0, 6)

# handles for legend
red_line = mlines.Line2D([], [], color='C3',
                         markersize=15, label='TLO')
blue_line = mlines.Line2D([], [], color='C0',
                          markersize=15, label='UNAIDS')
orange_ci = mlines.Line2D([], [], color='C1', marker='.',
                          markersize=15, label='MPHIA')
plt.legend(handles=[red_line, blue_line, orange_ci])

fig.tight_layout()
plt.savefig(make_graph_file_name("HIV_inc_adults"))
plt.show()


# ---------------------------------------------------------------------- #

# HIV Prevalence Children
# HIV Incidence 15-49
fig, ax = plt.subplots()
# model
ax.plot(model_hiv_child_prev.index, model_hiv_child_prev['mean'] * 100,
        label='TLO (mean)', color='C3')
ax.fill_between(model_hiv_child_prev.index,
                model_hiv_child_prev['lower'] * 100,
                model_hiv_child_prev['upper'] * 100,
                color='C3',
                alpha=0.2,
                zorder=5
                )
# data: UNAIDS
ax.plot(data_hiv_aidsinfo.index, data_hiv_aidsinfo['prevalence_0_14'] * 100,
        label='UNAIDS', color='g')
ax.fill_between(data_hiv_aidsinfo.index,
                data_hiv_aidsinfo['prevalence_0_14_lower'] * 100,
                data_hiv_aidsinfo['prevalence_0_14_upper'] * 100,
                color='g',
                alpha=0.2,
                zorder=5
                )
# MPHIA
plt.plot(model_hiv_child_prev.index[6], data_hiv_mphia_prev.loc[
    data_hiv_mphia_prev.age == "Total 0-14", "total percent hiv positive"].values[0], 'gx')

ax.set_title("HIV Prevalence in Children 0-14 (%)")
ax.set_xlabel("Year")
ax.set_ylabel("HIV prevalence (%)")
ax.set_xlim(2010, 2020)
ax.set_ylim(0, 5)

# handles for legend
red_line = mlines.Line2D([], [], color='C3',
                         markersize=15, label='TLO')
blue_line = mlines.Line2D([], [], color='C0',
                          markersize=15, label='UNAIDS')
green_cross = mlines.Line2D([], [], linewidth=0, color='g', marker='x',
                          markersize=7, label='MPHIA')
plt.legend(handles=[red_line, blue_line, green_cross])

fig.tight_layout()
plt.savefig(make_graph_file_name("HIV_prev_children"))
plt.show()

