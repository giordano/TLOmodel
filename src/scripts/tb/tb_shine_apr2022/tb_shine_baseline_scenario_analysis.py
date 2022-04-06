from pathlib import Path

import datetime
import pandas as pd
import matplotlib.pyplot as plt

from tlo.analysis.utils import (
    extract_results,
    extract_params,
    get_scenario_info,
    get_scenario_outputs,
    load_pickled_dataframes,
    summarize,
)

outputspath = Path("./outputs/lmu17@ic.ac.uk")

# Find results_folder associated with a given batch_file (and get most recent [-1])
results_folder = get_scenario_outputs("tb_shine_baseline_scenario.py", outputspath)[-1]

# look at one log (so can decide what to extract)
log = load_pickled_dataframes(results_folder)

# get basic information about the results
info = get_scenario_info(results_folder)

# extract the parameters that have varied over the set of simulations
params = extract_params(results_folder)


# -------------------------------------------------------------------------------------- #
#                                     ANALYSIS PLOTS                                     #
# -------------------------------------------------------------------------------------- #

# -------------------------------- SET UP ANALYSIS PLOTS ------------------------------- #
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
    fic, ax = plt.subplots()
    ax.plot(model.index, model.values, "-", color="r")

    if (model_low is not None) and (model_high is not None):
        ax.fill_between(model_low.index, model_low, model_high, color="r", alpha=0.2)

    if data_mid is not None:
        ax.plot(data_mid.index, data_mid.values, "-", color="b")

    if (data_low is not None) and (data_high is not None):
        ax.fill_between(data_low.index, data_low, data_high, color="b", alpha=0.2)

    if xlab is not None:
        ax.set_xlabel(xlab)

    if ylab is not None:
        ax.set_xlabel(ylab)

    plt.ylabel(ylab)
    plt.xlabel(xlab)
    plt.title(title_str)
    plt.legend(["TLO Model", data_name])


# -------------------------------------- LOAD DATA ------------------------------------- #
resourcefilepath = Path("./resources")
datestamp = datetime.date.today().strftime("__%Y_%m_%d")
xls_tb = pd.ExcelFile(resourcefilepath / "ResourceFile_TB.xlsx")

data_who_tb_2020 = pd.read_excel(resourcefilepath / 'ResourceFile_TB.xlsx', sheet_name='WHO_activeTB2020')
data_who_tb_2020.index = data_who_tb_2020["year"]
data_who_tb_2020 = data_who_tb_2020.drop(columns=["year"])

data_NTP_2019 = pd.read_excel(resourcefilepath / 'ResourceFile_TB.xlsx', sheet_name='NTP2019')
data_NTP_2019.index = data_NTP_2019["year"]
data_NTP_2019 = data_NTP_2019.drop(columns=["year"])

# ------------------------------------ ANALYSIS PLOTS ---------------------------------- #
# (1) Number of New Active TB Infections

new_active_tb_children = summarize(
    extract_results(
        results_folder,
        module="tlo.methods.tb",
        key="tb_incidence",
        column="num_new_active_tb_child",
        index="date",
        do_scaling=False
    ),
    collapse_columns=True,
)

new_active_tb_children.index = new_active_tb_children.index.year

make_plot(
    title_str="Number of New Active TB Infections (0-16 years)",
    model=new_active_tb_children["mean"],
    model_low=new_active_tb_children["lower"],
    model_high=new_active_tb_children["upper"],
    data_name="WHO Estimates",
    data_mid=data_who_tb_2020['estimated_inc_number_children'],
    data_low=data_who_tb_2020['estimated_inc_number_children_low'],
    data_high=data_who_tb_2020['estimated_inc_number_children_high'],
    xlab="Year",
    ylab="Number of New Active TB Infections",
)

plt.show()

# (2) Number of Diagnosed TB Cases

new_diagnosed_tb_children = summarize(
    extract_results(
        results_folder,
        module="tlo.methods.tb",
        key="tb_treatment",
        column="tbNewDiagnosis",
        index="date",
        do_scaling=False
    ),
    collapse_columns=True,
)

new_diagnosed_tb_children.index = new_diagnosed_tb_children.index.year

make_plot(
    title_str="Number of Diagnosed TB Cases (0-16 years)",
    model=new_diagnosed_tb_children["mean"],
    model_low=new_diagnosed_tb_children["lower"],
    model_high=new_diagnosed_tb_children["upper"],
    data_name="WHO Estimates",
    data_mid=data_who_tb_2020['new_cases_014'],
    xlab="Year",
    ylab="Number of Diagnosed TB Cases",
)

#plt.plot(new_diagnosed_tb_children.index[6], data_NTP_2019.loc[
#    data_NTP_2019.index == "total_case_notification_children"].values[6], 'gx')
plt.show()

# (3) Number of Treated TB Cases

new_treated_tb_children = summarize(
    extract_results(
        results_folder,
        module="tlo.methods.tb",
        key="tb_treatment",
        column="tbNewTreatment",
        index="date",
        do_scaling=False
    ),
    collapse_columns=True,
)

new_treated_tb_children.index = new_treated_tb_children.index.year

make_plot(
    title_str="Number of Treated TB Cases (0-16 years)",
    model=new_treated_tb_children["mean"],
    model_low=new_treated_tb_children["lower"],
    model_high=new_treated_tb_children["upper"],
    #data_name="WHO Estimates",
    #data_mid=data_who_tb_2020['estimated_inc_number_children'],
    #data_low=data_who_tb_2020['estimated_inc_number_children_low'],
    #data_high=data_who_tb_2020['estimated_inc_number_children_high'],
    xlab="Year",
    ylab="Number of Treated TB Cases",
)

plt.show()
