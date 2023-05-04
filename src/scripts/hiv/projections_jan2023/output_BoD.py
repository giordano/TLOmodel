
"""
Extracts DALYs and mortality from the TB module
 """
import argparse
import datetime
import pickle
from typing import Tuple
from pathlib import Path
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from tlo.analysis.utils import extract_results, make_age_grp_lookup, summarize, compare_number_of_deaths
from tlo import Date

resourcefilepath = Path("./resources")
outputpath = Path("./outputs")  # folder for convenience of storing outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

TARGET_PERIOD = (Date(2010, 1, 1), Date(2014, 12, 31))

def get_num_deaths(_df):
    """Return total number of Deaths (total within the TARGET_PERIOD)
    """
    return pd.Series(data=len(_df.loc[pd.to_datetime(_df.date).between(*TARGET_PERIOD)]))

def get_num_dalys(_df):
    """Return total number of DALYS (Stacked) by label (total within the TARGET_PERIOD)"""
    return pd.Series(
        data=_df
        .loc[_df.year.between(*[i.year for i in TARGET_PERIOD])]
        .drop(columns=['date', 'sex', 'age_range', 'year'])
        .sum().sum()
    )

    # Quantify the health gains associated with all interventions combined.


    num_deaths = extract_results(
        results_folder,
        module='tlo.methods.demography',
        key='death',
        custom_generate_series=get_num_deaths,
        do_scaling=True
    )

    num_dalys = extract_results(
        results_folder,
        module='tlo.methods.healthburden',
        key='dalys_stacked',
        custom_generate_series=get_num_dalys(),
        do_scaling=True
    )

    num_deaths_summarized = summarize(num_deaths).loc[0].unstack()
    num_dalys_summarized = summarize(num_dalys).loc[0].unstack()

# Plot for total number of DALYs from the scenario
    name_of_plot= f'Total DALYS, {target_period()}'
    fig, ax = do_bar_plot_with_ci(num_dalys_summarized / 1e6)
    ax.set_title(name_of_plot)
    ax.set_ylabel('DALYS (Millions)')
    fig.tight_layout()
    fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
    fig.show()
    plt.close(fig)

# plot of total number of deaths from the scenario
    name_of_plot= f'Total Deaths, {target_period()}'
    fig, ax = do_bar_plot_with_ci(num_deaths_summarized / 1e6)
    ax.set_title(name_of_plot)
    ax.set_ylabel('Deaths (Millions)')
    fig.tight_layout()
    fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
    fig.show()
    plt.close(fig)


