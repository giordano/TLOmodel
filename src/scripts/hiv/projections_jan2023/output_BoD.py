
"""
Extracts DALYs and mortality from the TB module
 """
import argparse
import datetime
import pickle
from pathlib import Path
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from tlo.analysis.utils import extract_results,  summarize
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

    def make_plot(_df, annotations=None):
        """Make a vertical bar plot for each row of _df, using the columns to identify the height of the bar and the
         extent of the error bar."""
        yerr = np.array([
            (_df['mean'] - _df['lower']).values,
            (_df['upper'] - _df['mean']).values,
        ])

        xticks = {(i + 0.5): k for i, k in enumerate(_df.index)}
        fig, ax = plt.subplots()
        ax.bar(
            xticks.keys(),
            _df['mean'].values,
            yerr=yerr,
            alpha=0.5,
            ecolor='black',
            capsize=10,
        )
        if annotations:
            for xpos, ypos, text in zip(xticks.keys(), _df['mean'].values, annotations):
                ax.text(xpos, ypos, text, horizontalalignment='center')
        ax.set_xticks(list(xticks.keys()))
        ax.set_xticklabels(list(xticks.values()), rotation=90)
        ax.grid(axis="y")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        fig.tight_layout()
        plt.savefig(
            outputpath / (name_of_plot.replace(" ", "_") + datestamp + ".pdf"), format="pdf"
        )
        return fig, ax

# Quantify the health gains associated with all interventions combined
    with open(outputpath / "default_run.pickle", "rb") as f:
     output = pickle.load(f)

     results_folder = Path("./output")

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
    fig, ax = make_plot(num_dalys_summarized / 1e6)
    ax.set_title(name_of_plot)
    ax.set_ylabel('DALYS (Millions)')
    fig.tight_layout()
    fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
    plt.show()

# plot of total number of deaths from the scenario
    name_of_plot= f'Total Deaths, {target_period()}'
    fig, ax = make_plot(num_deaths_summarized / 1e6)
    ax.set_title(name_of_plot)
    ax.set_ylabel('Deaths (Millions)')
    fig.tight_layout()
    fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
    plt.show()




