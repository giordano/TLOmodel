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

# look at one log (so can decide what to extract)
log = load_pickled_dataframes(results_folder)
log2 = load_pickled_dataframes(results_folder, draw=0, run=1, name=None)
log3 = load_pickled_dataframes(results_folder, draw=0, run=2, name=None)

baseline_outputs = log["tlo.methods.hiv"]["hiv_baseline_outputs"]
detailed_outputs = log["tlo.methods.hiv"]["hiv_detailed_outputs"]
deaths = log["tlo.methods.hiv"]["death"]

# write to excel
with pd.ExcelWriter(outputspath / ("MIHPSA_outputs2" + ".xlsx"), engine='openpyxl') as writer:
    baseline_outputs.to_excel(writer, sheet_name='Sheet1', index=False)
    detailed_outputs.to_excel(writer, sheet_name='Sheet2', index=False)
    deaths.to_excel(writer, sheet_name='Sheet3', index=False)
    writer.save()

detailed_outputs.to_csv(outputspath / ("MIHPSA_detailed_outputs" + ".csv"), index=None)
