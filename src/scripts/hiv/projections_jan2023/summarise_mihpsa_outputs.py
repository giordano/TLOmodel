"""This file uses the results of the batch file to make some summary statistics.
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
# %% Analyse results of runs

# 0) Find results_folder associated with a given batch_file (and get most recent [-1])
results_folder = get_scenario_outputs("mihpsa_runs.py", outputspath)[-1]


# Collect results from each draw/run
def extract_outputs(results_folder: Path,
                    module: str,
                    key: str,
                    column: str) -> pd.DataFrame:
    # get number of draws and numbers of runs
    info = get_scenario_info(results_folder)

    df_pop = pd.DataFrame()
    df_prev = pd.DataFrame()
    df_inf = pd.DataFrame()
    df_dx = pd.DataFrame()
    df_tx = pd.DataFrame()
    df_vs = pd.DataFrame()
    df_out = pd.DataFrame()

    # 10 draws, 1 run
    run = 0
    for draw in range(info['number_of_draws']):
        # load the log file
        df: pd.DataFrame = load_pickled_dataframes(results_folder, draw, run, module)[module][key]
        # first column is mean

        tmp = pd.DataFrame(df[column].to_list(), columns=['pop', 'prev',
                                                          "inf", "dx",
                                                          "tx", "vs"])
        df_pop[draw] = tmp["pop"]
        df_prev[draw] = tmp["prev"]
        df_inf[draw] = tmp["inf"]
        df_dx[draw] = tmp["dx"]
        df_tx[draw] = tmp["tx"]
        df_vs[draw] = tmp["vs"]

    df_out["mean_pop"] = df_pop.mean(axis=1)
    df_out["mean_prev"] = df_prev.mean(axis=1)
    df_out["mean_inf"] = df_inf.mean(axis=1)
    df_out["mean_dx"] = df_dx.mean(axis=1)
    df_out["mean_tx"] = df_tx.mean(axis=1)
    df_out["mean_vs"] = df_vs.mean(axis=1)

    return df_out


baseline_all = extract_outputs(results_folder=results_folder,
                               module="tlo.methods.hiv",
                               key="hiv_baseline_outputs",
                               column="outputs_age15_64")

# extract pop size for M, F and Total
# take mean and store

# extract prevalence for M, F and Total
# extract new infections for M, F, and Total
# prop dx
# if dx, % on tx
# if tx, % VS


# baseline_outputs = log["tlo.methods.hiv"]["hiv_baseline_outputs"]
# detailed_outputs = log["tlo.methods.hiv"]["hiv_detailed_outputs"]
# deaths = log["tlo.methods.hiv"]["death"]
#


# write to excel
with pd.ExcelWriter(outputspath / ("MIHPSA_outputs2" + ".xlsx"), engine='openpyxl') as writer:
    baseline_outputs.to_excel(writer, sheet_name='Sheet1', index=False)
    detailed_outputs.to_excel(writer, sheet_name='Sheet2', index=False)
    deaths.to_excel(writer, sheet_name='Sheet3', index=False)
    writer.save()

detailed_outputs.to_csv(outputspath / ("MIHPSA_detailed_outputs" + ".csv"), index=None)
