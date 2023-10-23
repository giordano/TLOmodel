"""This file uses the results of the scenario runs to generate plots

*1 Epi outputs (incidence and mortality)

"""

import datetime
from pathlib import Path

# import lacroix
from matplotlib import pyplot as plt
import matplotlib.colors as colors
import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns

from tlo.analysis.utils import (
    compare_number_of_deaths,
    extract_params,
    extract_results,
    get_scenario_info,
    get_scenario_outputs,
    load_pickled_dataframes,
    summarize,
    make_age_grp_lookup,
    make_age_grp_types,
)

outputspath = Path("./outputs")

# Find results_folder associated with a given batch_file (and get most recent [-1])
results_folder = get_scenario_outputs("effect_of_treatment_packages_combined.py", outputspath)[-1]

# Declare path for output graphs from this script
make_graph_file_name = lambda stub: results_folder / f"{stub}.png"  # noqa: E731

# look at one log (so can decide what to extract)
log = load_pickled_dataframes(results_folder)

# get basic information about the results
scenario_info = get_scenario_info(results_folder)

# Extract the parameters that have varied over the set of simulations
params = extract_params(results_folder)

# colour scheme
# berry = lacroix.colorList('CranRaspberry')  # ['#F2B9B8', '#DF7878', '#E40035', '#009A90', '#0054A4', '#001563']

# -----------------------------------------------------------------------------------------
# %% Epi outputs
# -----------------------------------------------------------------------------------------


def summarize_median(results: pd.DataFrame) -> pd.DataFrame:
    """ edit existing utility function to return:
    median and 95% quantiles
    """

    summary = pd.DataFrame(
        columns=pd.MultiIndex.from_product(
            [
                results.columns.unique(level='draw'),
                ["median", "lower", "upper"]
            ],
            names=['draw', 'stat']),
        index=results.index
    )

    summary.loc[:, (slice(None), "median")] = results.groupby(axis=1, by='draw').quantile(0.5).values
    summary.loc[:, (slice(None), "lower")] = results.groupby(axis=1, by='draw').quantile(0.025).values
    summary.loc[:, (slice(None), "upper")] = results.groupby(axis=1, by='draw').quantile(0.975).values

    return summary


# ---------------------------------- HIV ---------------------------------- #

# HIV incidence
hiv_inc = summarize_median(
    extract_results(
        results_folder,
        module="tlo.methods.hiv",
        key="summary_inc_and_prev_for_adults_and_children_and_fsw",
        column="hiv_adult_inc_1549",
        index="date",
        do_scaling=False
    )
)

hiv_cases = summarize_median(
    extract_results(
        results_folder,
        module="tlo.methods.hiv",
        key="summary_inc_and_prev_for_adults_and_children_and_fsw",
        column="n_new_infections_adult_1549",
        index="date",
        do_scaling=False
    )
)

adult_pop = summarize_median(
    extract_results(
        results_folder,
        module="tlo.methods.hiv",
        key="summary_inc_and_prev_for_adults_and_children_and_fsw",
        column="pop_total",
        index="date",
        do_scaling=False
    )
)

adult_plhiv = summarize_median(
    extract_results(
        results_folder,
        module="tlo.methods.hiv",
        key="summary_inc_and_prev_for_adults_and_children_and_fsw",
        column="total_plhiv",
        index="date",
        do_scaling=False
    )
)

# ---------------------------------- PERSON-YEARS ---------------------------------- #
# for each scenario, return a df with the person-years logged in each draw/run
# to be used for calculating tb incidence or mortality rates


def get_person_years(_df):
    """ extract person-years for each draw/run
    sums across men and women
    will skip column if particular run has failed
    """
    years = pd.to_datetime(_df["date"]).dt.year
    py = pd.Series(dtype="int64", index=years)
    for year in years:
        tot_py = (
            (_df.loc[pd.to_datetime(_df["date"]).dt.year == year]["M"]).apply(pd.Series) +
            (_df.loc[pd.to_datetime(_df["date"]).dt.year == year]["F"]).apply(pd.Series)
        ).transpose()
        py[year] = tot_py.sum().values[0]

    py.index = pd.to_datetime(years, format="%Y")

    return py


py0 = extract_results(
    results_folder,
    module="tlo.methods.demography",
    key="person_years",
    custom_generate_series=get_person_years,
    do_scaling=False
)


# ---------------------------------- TB ---------------------------------- #
# number new active tb cases - convert to incidence rate
def tb_inc_func(results_folder):
    inc = extract_results(
        results_folder,
        module="tlo.methods.tb",
        key="tb_incidence",
        column="num_new_active_tb",
        index="date",
        do_scaling=False
    )

    inc.columns = inc.columns.get_level_values(0)

    # divide each run of tb incidence by py from that run
    # tb logger starts at 2011-01-01, demog starts at 2010-01-01
    # extract py log from 2011
    py = extract_results(
        results_folder,
        module="tlo.methods.demography",
        key="person_years",
        custom_generate_series=get_person_years,
        do_scaling=False
    )
    py.columns = py.columns.get_level_values(0)

    inc_per_py = inc / py
    # remove first row (2010) as nan
    inc_per_py = inc_per_py.iloc[1:]

    # Calculate the mean of each row for each column name
    summary = pd.DataFrame(
        columns=pd.MultiIndex.from_product(
            [
                inc.columns.unique(level='draw'),
                ["median", "lower", "upper"]
            ],
            names=['draw', 'stat']),
        index=inc.index
    )

    summary.loc[:, (slice(None), "median")] = inc_per_py.groupby(axis=1, by='draw').quantile(0.5).values
    summary.loc[:, (slice(None), "lower")] = inc_per_py.groupby(axis=1, by='draw').quantile(0.025).values
    summary.loc[:, (slice(None), "upper")] = inc_per_py.groupby(axis=1, by='draw').quantile(0.975).values

    return summary


tb_inc = tb_inc_func(results_folder)


# ---------------------------------- MALARIA ---------------------------------- #

# malaria incidence
# value is per 1000py
mal_inc = summarize_median(
    extract_results(
        results_folder,
        module="tlo.methods.malaria",
        key="incidence",
        column="inc_clin_counter",
        index="date",
        do_scaling=False
    )
)

# ---------------------------------- SMOOTH DATA ---------------------------------- #
#  create smoothed lines
num_interp = 24


def create_smoothed_lines(data_x, df_y):
    """
    pass a dataframe into function
    this will smooth the data according to the lowess parameters defined
    and return a df with smoothed data across the xvals array
    """

    xvals = np.linspace(start=data_x.min(), stop=data_x.max(), num=num_interp)
    smoothed_df = pd.DataFrame()

    names = ['median', 'lower', 'upper']
    repetitions = scenario_info['number_of_draws']

    # Create the column names
    column_names = [f'{name}_{i}' for i in range(repetitions) for name in names]

    # Collapse the multi-index columns
    collapsed_df = df_y.droplevel(0, axis=1)  # Drop the first level of the columns
    collapsed_df.columns = [str(i) for i in range(len(collapsed_df.columns))]

    # extract each column in turn to smooth
    for column_name in collapsed_df.columns:

        y = collapsed_df[column_name]
        lowess = sm.nonparametric.lowess(endog=y, exog=data_x, frac=0.45, xvals=xvals, it=0)
        smoothed_df[column_name] = lowess

    smoothed_df.columns = column_names

    return smoothed_df


data_x = hiv_inc.index.year  # 2011 onwards

# smooth outputs for plots
smoothed_hiv_inc = create_smoothed_lines(data_x, hiv_inc)
smoothed_tb_inc = create_smoothed_lines(data_x, tb_inc)
smoothed_mal_inc = create_smoothed_lines(data_x, mal_inc)


# ---------------------------------- PLOTS ---------------------------------- #
plt.style.use('default')  # to reset

# plt.style.use('ggplot')
plt.style.use('seaborn-bright')

font = {'family': 'sans-serif',
        'color': 'black',
        'weight': 'bold',
        'size': 11,
        }

# select mean values for plotting
mean_hiv_inc = hiv_inc.iloc[:, hiv_inc.columns.get_level_values(1) == 'mean']
mean_tb_inc = tb_inc.tail(tb_inc.shape[0]-1)  # remove 2010 as nan
mean_mal_inc = mal_inc.iloc[:, mal_inc.columns.get_level_values(1) == 'mean']

year = mean_hiv_inc.index.year

labels = year
fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3,
                                    constrained_layout=True,
                                    figsize=(8, 3))
fig.suptitle('')

ax1.plot(mean_hiv_inc)
ax1.set(title='HIV',
        ylabel='HIV Incidence, per capita')
plt.xticks(ticks=mean_hiv_inc.index, labels=labels)
ax1.tick_params(axis='x', rotation=70)

# TB incidence
ax2.plot(mean_tb_inc)
ax2.set(title='TB',
        ylabel='TB Incidence, per capita')
plt.xticks(ticks=mean_tb_inc.index, labels=labels)
ax2.tick_params(axis='x', rotation=70)

# Malaria incidence
ax3.plot(mean_mal_inc)
ax3.set(title='Malaria',
        ylabel='Malaria Incidence, per 1000')
plt.xticks(ticks=mean_mal_inc.index, labels=labels)
ax3.tick_params(axis='x', rotation=70)

plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left',
           labels=['baseline', '-hiv', '-tb', '-malaria', '-all3'],)

plt.show()
