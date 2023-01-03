
import datetime
import math
import os
from pathlib import Path

# import lacroix
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import seaborn as sns
from matplotlib.gridspec import GridSpec

from tlo import Date
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

# download all files (and get most recent [-1])
results0 = get_scenario_outputs("scenario_0.py", outputspath)[-1]
results1 = get_scenario_outputs("scenario_1.py", outputspath)[-1]
results2 = get_scenario_outputs("scenario_2.py", outputspath)[-1]
results3 = get_scenario_outputs("scenario_3.py", outputspath)[-1]
results4 = get_scenario_outputs("scenario_4.py", outputspath)[-1]

# function to identify runs which don't end in fadeout




# ---------------------------------- TB ---------------------------------- #
# for full output
model_tb_inc_full = extract_results(
        results0,
        module="tlo.methods.tb",
        key="tb_incidence",
        column="num_new_active_tb",
        index="date",
        do_scaling=False,
    )


model_tb_inc = summarize_across_draws(
    extract_results(
        results0,
        module="tlo.methods.tb",
        key="tb_incidence",
        column="num_new_active_tb",
        index="date",
        do_scaling=False,
    ),
)
model_tb_inc.index = model_tb_inc.index.year
activeTB_inc_rate = model_tb_inc["median"] / 200000 * 100000

activeTB_inc_rate.index = model_tb_inc.index
activeTB_inc_rate_low = pd.Series(
    (model_tb_inc["lower"].values / py_mean.values[1:26]) * 100000
)
activeTB_inc_rate_low.index = model_tb_inc.index
activeTB_inc_rate_high = pd.Series(
    (model_tb_inc["upper"].values / py_mean.values[1:26]) * 100000
)
activeTB_inc_rate_high.index = model_tb_inc.index
