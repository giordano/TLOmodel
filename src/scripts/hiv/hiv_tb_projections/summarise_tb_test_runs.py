"""This file uses the results of the batch file to make some summary statistics.
The results of the batchrun were put into the 'outputspath' results_folder

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

data_tb_who = pd.read_excel(xls_tb, sheet_name="WHO_activeTB2020")
data_tb_who = data_tb_who.loc[
    (data_tb_who.year >= 2010)
]  # include only years post-2010
data_tb_who.index = data_tb_who["year"]
data_tb_who = data_tb_who.drop(columns=["year"])


# %% Analyse results of runs

# 0) Find results_folder associated with a given batch_file (and get most recent [-1])
results_folder = get_scenario_outputs("tb_transmission_rate_batch_runs.py", outputspath)[-1]


# Declare path for output graphs from this script
make_graph_file_name = lambda stub: results_folder / f"{stub}.png"  # noqa: E731

# look at one log (so can decide what to extract)
log = load_pickled_dataframes(results_folder)

# get basic information about the results
info = get_scenario_info(results_folder)

# 1) Extract the parameters that have varied over the set of simulations
params = extract_params(results_folder)


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
draw=0
for run in range(0, number_runs):
    py_summary.iloc[:, run] = get_person_years(draw, run)

py_summary["mean"] = py_summary.mean(axis=1)

# ---------------------------------- TB ---------------------------------- #

tb_inc1 = extract_results(
        results_folder,
        module="tlo.methods.tb",
        key="tb_incidence",
        column="num_new_active_tb",
        index="date",
        do_scaling=False,
)

tb_inc1.columns = tb_inc1.columns.get_level_values(0)
active_tb_rate = (tb_inc1.divide(py_summary["mean"].values[0:30], axis=0)) * 100000

plt.plot(active_tb_rate)
plt.show()
