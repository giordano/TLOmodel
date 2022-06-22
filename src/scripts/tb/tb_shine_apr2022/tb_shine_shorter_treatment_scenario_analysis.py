from pathlib import Path


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

outputspath = Path("./outputs/t.mangal@imperial.ac.uk")
rfp = Path('./resources')

# Find results_folder associated with a given batch_file (and get most recent [-1])
# todo change name
results_folder = get_scenario_outputs("tb_shine_shorter_treatment_scenario.py", outputspath)[-1]

# get basic information about the results
info = get_scenario_info(results_folder)

# extract the parameters that have varied over the set of simulations
params = extract_params(results_folder)


def make_plot(model=None, model_low=None, model_high=None):
    assert model is not None

    fic, ax = plt.subplots()
    ax.plot(model.index, model.values, "-", color="r")

    if (model_low is not None) and (model_high is not None):
        ax.fill_between(model_low.index, model_low, model_high, color="r", alpha=0.2)

# ------------------------------ EXTRACT HEALTHSYSTEM SUMMARY DATA  ------------------------------ #


def extract_healthsystem_summary(results_folder: Path, module: str, key: str, column: str) -> pd.DataFrame:
    info = get_scenario_info(results_folder)

    draw_run_dict = {}
    draw_dict = {}
    for draw in range(info['number_of_draws']):
        for run in range(info['runs_per_draw']):
            hs_summary = load_pickled_dataframes(results_folder, draw, run)[module][key].set_index('date').loc[:,
                         [column]]
            hs_summary = hs_summary[column].apply(pd.Series).fillna(0)

            draw_run_number = str(draw) + str(run)
            draw_run_dict[draw_run_number] = hs_summary

        draw_number = str(draw)
        draw_dict[draw_number] = pd.concat(draw_run_dict).groupby(level=1).mean()

    results = pd.concat(draw_dict, axis=1)

    return results


# ---------------------------------------- ANALYSIS PLOTS ---------------------------------------- #
# -----------------------------------------  CONSUMABLES ----------------------------------------- #
cons = extract_healthsystem_summary(
    results_folder,
    module="tlo.methods.healthsystem.summary",
    key="Consumables",
    column="Item_Available")

plt.plot(cons)
plt.title("Consumables Use Over Time")
plt.xlabel("Year")
plt.ylabel("Quantity")
plt.show()


cons_by_type = extract_results(
    results_folder,
    module="tlo.methods.healthsystem.summary",
    key="Consumables",
    column="Item_Available"
)

from collections import defaultdict
from tlo import Date


TARGET_PERIOD = (Date(2010, 1, 1), Date(2016, 1, 1))

def drop_outside_period(_df):
    """Return a dataframe which only includes for which the date is within the limits defined by TARGET_PERIOD"""
    return _df.drop(index=_df.index[~_df['date'].between(*TARGET_PERIOD)])


def get_counts_of_items_requested(_df):
    _df = drop_outside_period(_df)

    counts_of_available = defaultdict(int)
    counts_of_not_available = defaultdict(int)

    for _, row in _df.iterrows():
        for item, num in eval(row['Item_Available']).items():
            counts_of_available[item] += num
        for item, num in eval(row['Item_NotAvailable']).items():
            counts_of_not_available[item] += num

    return pd.concat(
        {'Available': pd.Series(counts_of_available), 'Not_Available': pd.Series(counts_of_not_available)},
        axis=1
    ).fillna(0).astype(int).stack()



cons_req = summarize(
    extract_results(
        results_folder,
        module='tlo.methods.healthsystem',
        key='Consumables',
        custom_generate_series=get_counts_of_items_requested,
        do_scaling=False
    ),
    only_mean=True,
    collapse_columns=True
)


hsi_by_type = extract_results(
    results_folder,
    module="tlo.methods.healthsystem",
    key="HSI_Event",
    custom_generate_series=(
        lambda df_: df_.assign(year=df_['date'].dt.year).groupby(['year', 'TREATMENT_ID'])['TREATMENT_ID'].count()),
    do_scaling=False
)

tb_inc = extract_results(
        results_folder,
        module="tlo.methods.tb",
        key="tb_incidence",
        column="num_new_active_tb_child",
        index="date",
        do_scaling=False,
)

tb_standard = extract_results(
        results_folder,
        module="tlo.methods.tb",
        key="tb_treatment_regimen",
        column="TBTxChild",
        index="date",
        do_scaling=False,
)

tb_shorter = extract_results(
        results_folder,
        module="tlo.methods.tb",
        key="tb_treatment_regimen",
        column="TBTxChildShorter",
        index="date",
        do_scaling=False,
)
