"""Produce plots to show the health impact (deaths, dalys) each the healthcare system (overall health impact) when
running under different MODES and POLICIES (scenario_impact_of_policy.py)"""

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from tlo import Date
from tlo.analysis.utils import extract_results, summarize


def apply(results_folder: Path, output_folder: Path, resourcefilepath: Path = None, ):
    """Produce standard set of plots describing the effect of each TREATMENT_ID.
    - We estimate the epidemiological impact as the EXTRA deaths that would occur if that treatment did not occur.
    - We estimate the draw on healthcare system resources as the FEWER appointments when that treatment does not occur.
    """

    TARGET_PERIOD = (Date(2010, 1, 1), Date(2010, 1, 1) + pd.DateOffset(years=5))

    # Definitions of general helper functions
    make_graph_file_name = lambda stub: output_folder / f"{stub.replace('*', '_star_')}.png"  # noqa: E731

    def target_period() -> str:
        """Returns the target period as a string of the form YYYY-YYYY"""
        return "-".join(str(t.year) for t in TARGET_PERIOD)

    def get_parameter_names_from_scenario_file() -> Tuple[str]:
        """Get the tuple of names of the scenarios from `Scenario` class used to create the results."""
        from scripts.healthsystem.impact_of_policy.scenario_impact_of_policy import (
            ImpactOfHealthSystemMode,
        )
        e = ImpactOfHealthSystemMode()
        return tuple(e._scenarios.keys())

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

    def set_param_names_as_column_index_level_0(_df):
        """Set the columns index (level 0) as the param_names."""
        ordered_param_names_no_prefix = {i: x for i, x in enumerate(param_names)}
        names_of_cols_level0 = [ordered_param_names_no_prefix.get(col) for col in _df.columns.levels[0]]
        assert len(names_of_cols_level0) == len(_df.columns.levels[0])
        _df.columns = _df.columns.set_levels(names_of_cols_level0, level=0)
        return _df

    def find_difference_relative_to_comparison(_ser: pd.Series,
                                               comparison: str,
                                               scaled: bool = False,
                                               drop_comparison: bool = True,
                                               ):
        """Find the difference in the values in a pd.Series with a multi-index, between the draws (level 0)
        within the runs (level 1), relative to where draw = `comparison`.
        The comparison is `X - COMPARISON`."""
        return _ser \
            .unstack(level=0) \
            .apply(lambda x: (x - x[comparison]) / (x[comparison] if scaled else 1.0), axis=1) \
            .drop(columns=([comparison] if drop_comparison else [])) \
            .stack()

    def do_bar_plot_with_ci(_df, annotations=None):
        """Make a vertical bar plot for each row of _df, using the columns to identify the height of the bar and the
         extent of the error bar."""
        yerr = np.array([
            (_df['mean'] - _df['lower']).values,
            (_df['upper'] - _df['mean']).values,
        ])
        colours = ("#CC99FF","#FF0000", "#FF6666","#FF8000","#660066","#994C00","#009900")

        xticks = {(i + 0.5): k for i, k in enumerate(_df.index)}

        fig, ax = plt.subplots()
        plt.axhline(y = 4.686262e+07/1e6, color = 'black', linestyle = '--')
        ax.bar(
            xticks.keys(),
            _df['mean'].values,
            yerr=yerr,
        #    alpha=0.5,
            ecolor='black',
            capsize=10,
            color=colours
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

        return fig, ax

    # %% Define parameter names
    param_names = get_parameter_names_from_scenario_file()
    print("Param names from scenario file", param_names)


    # %% Quantify the health gains associated with all interventions combined.

    # Absolute Number of Deaths and DALYs
    num_deaths = extract_results(
        results_folder,
        module='tlo.methods.demography',
        key='death',
        custom_generate_series=get_num_deaths,
        do_scaling=True
    ).pipe(set_param_names_as_column_index_level_0)

    num_dalys = extract_results(
        results_folder,
        module='tlo.methods.healthburden',
        key='dalys_stacked',
        custom_generate_series=get_num_dalys,
        do_scaling=True
    ).pipe(set_param_names_as_column_index_level_0)

    num_dalys_summarized = summarize(num_dalys).loc[0].unstack()
    print(num_dalys_summarized)

    all_cons = False

    if all_cons:
        num_dalys_summarized = num_dalys_summarized.drop(index=["No Healthcare System", "Unlimited Resources default cons", "Unlimited Resources all cons", "Unlimited Efficiency all cons", "Unlimited Efficiency default cons", "RMNCH default cons", "Random default cons", "Naive default cons", "Vertical Programmes default cons", "Clinically Vulnerable default cons", "EHP1_binary default cons", "EHP3_LPP_binary default cons"])
    else:
        num_dalys_summarized = num_dalys_summarized.drop(index=["No Healthcare System", "Unlimited Resources default cons", "Unlimited Resources all cons", "Unlimited Efficiency all cons", "Unlimited Efficiency default cons", "RMNCH all cons", "Random all cons", "Naive all cons", "Vertical Programmes all cons", "Clinically Vulnerable all cons", "EHP1_binary all cons", "EHP3_LPP_binary all cons"])

    if(all_cons):
        num_dalys_summarized = num_dalys_summarized.rename(index={'EHP1_binary all cons': 'EHP III'})
        num_dalys_summarized = num_dalys_summarized.rename(index={'EHP3_LPP_binary all cons': 'LCOA EHP'})
        num_dalys_summarized = num_dalys_summarized.rename(index={'Naive all cons': 'Naive'})
        num_dalys_summarized = num_dalys_summarized.rename(index={'Random all cons': 'Random'})
        num_dalys_summarized = num_dalys_summarized.rename(index={'RMNCH all cons': 'RMNCH'})
        num_dalys_summarized = num_dalys_summarized.rename(index={'Vertical Programmes all cons': 'Vertical Programmes'})
        num_dalys_summarized = num_dalys_summarized.rename(index={'Clinically Vulnerable all cons': 'Clinically Vulnerable'})

    else:
        num_dalys_summarized = num_dalys_summarized.rename(index={'EHP1_binary default cons': 'EHP III'})
        num_dalys_summarized = num_dalys_summarized.rename(index={'EHP3_LPP_binary default cons': 'LCOA EHP'})
        num_dalys_summarized = num_dalys_summarized.rename(index={'Naive default cons': 'Naive'})
        num_dalys_summarized = num_dalys_summarized.rename(index={'Random default cons': 'Random'})
        num_dalys_summarized = num_dalys_summarized.rename(index={'RMNCH default cons': 'RMNCH'})
        num_dalys_summarized = num_dalys_summarized.rename(index={'Vertical Programmes default cons': 'Vertical Programmes'})
        num_dalys_summarized = num_dalys_summarized.rename(index={'Clinically Vulnerable default cons': 'Clinically Vulnerable'})
    print(num_dalys_summarized)



    # TOTAL DALYS
    if all_cons:
        name_of_plot = f'Total DALYS unlimited consumables availability, {target_period()}'
    else:
        name_of_plot = f'Total DALYS status-quo consumables availability, {target_period()}'
    fig, ax = do_bar_plot_with_ci(num_dalys_summarized / 1e6)
    ax.set_title(name_of_plot)
    ax.set_ylabel('DALYS (Millions)')
    fig.tight_layout()
    plt.ylim(36,50)
    plt.yticks(np.arange(36, 52, 2))
    fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
    fig.show()
    plt.close(fig)



if __name__ == "__main__":
    rfp = Path('resources')

    parser = argparse.ArgumentParser(
        description="Produce plots to show the impact each set of treatments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-path",
        help=(
            "Directory to write outputs to. If not specified (set to None) outputs "
            "will be written to value of --results-path argument."
        ),
        type=Path,
        default=None,
        required=False,
    )
    parser.add_argument(
        "--resources-path",
        help="Directory containing resource files",
        type=Path,
        default=Path('resources'),
        required=False,
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        help=(
            "Directory containing results from running "
            "src/scripts/healthsystem/impact_of_policy/scenario_impact_of_policy.py "
        ),
        default=None,
        required=False
    )
    args = parser.parse_args()
    assert args.results_path is not None
    results_path = args.results_path

    output_path = results_path if args.output_path is None else args.output_path

    apply(
        results_folder=results_path,
        output_folder=output_path,
        resourcefilepath=args.resources_path
    )
