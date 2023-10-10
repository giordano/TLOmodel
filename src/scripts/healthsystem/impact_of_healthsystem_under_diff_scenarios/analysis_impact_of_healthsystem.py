"""Produce plots to show the impact each the healthcare system (overall health impact) when running under different
scenarios (scenario_impact_of_healthsystem.py)"""

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from tlo import Date
from tlo.analysis.utils import (
    extract_results,
    make_age_grp_lookup,
    summarize,
    CAUSE_OF_DEATH_OR_DALY_LABEL_TO_COLOR_MAP,
    order_of_cause_of_death_or_daly_label,
    get_color_cause_of_death_or_daly_label,
)


def apply(results_folder: Path, output_folder: Path, resourcefilepath: Path = None, dalys_averted_by_wealth_and_label=None):
    """Produce standard set of plots describing the effect of each TREATMENT_ID.
    - We estimate the epidemiological impact as the EXTRA deaths that would occur if that treatment did not occur.
    - We estimate the draw on healthcare system resources as the FEWER appointments when that treatment does not occur.
    """

    TARGET_PERIOD = (Date(2015, 1, 1), Date(2019, 12, 31))

    # Definitions of general helper functions
    make_graph_file_name = lambda stub: output_folder / f"{stub.replace('*', '_star_')}.png"  # noqa: E731

    _, age_grp_lookup = make_age_grp_lookup()

    def target_period() -> str:
        """Returns the target period as a string of the form YYYY-YYYY"""
        return "-".join(str(t.year) for t in TARGET_PERIOD)

    def get_parameter_names_from_scenario_file() -> Tuple[str]:
        """Get the tuple of names of the scenarios from `Scenario` class used to create the results."""
        from scripts.healthsystem.impact_of_healthsystem_under_diff_scenarios.scenario_impact_of_healthsystem import (
            ImpactOfHealthSystemAssumptions,
        )
        e = ImpactOfHealthSystemAssumptions()
        return tuple(e._scenarios.keys())

    def get_num_deaths(_df):
        """Return total number of Deaths (total within the TARGET_PERIOD)
        """
        return pd.Series(data=len(_df.loc[pd.to_datetime(_df.date).between(*TARGET_PERIOD)]))

    def get_num_dalys(_df):
        """Return total number of DALYS (Stacked) by label (total within the TARGET_PERIOD)
        """
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

        xticks = {(i + 0.5): k for i, k in enumerate(_df.index)}

        fig, ax = plt.subplots()
        ax.bar(
            xticks.keys(),
            _df['mean'].values,
            yerr=yerr,
            alpha=0.5,
            ecolor='black',
            capsize=10,
            label=xticks.values()
        )
        if annotations:
            for xpos, ypos, text in zip(xticks.keys(), _df['upper'].values, annotations):
                ax.text(xpos, ypos*1.05, text, horizontalalignment='center')
        ax.set_xticks(list(xticks.keys()))
        ax.set_xticklabels(list(xticks.values()), rotation=90)
        ax.grid(axis="y")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        fig.tight_layout()

        return fig, ax

    # %% Define parameter names
    param_names = get_parameter_names_from_scenario_file()

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

    # %% Charts of total numbers of deaths / DALYS
    num_dalys_summarized = summarize(num_dalys).loc[0].unstack().reindex(param_names)
    num_deaths_summarized = summarize(num_deaths).loc[0].unstack().reindex(param_names)

    # Update naming of scenarios
    scenario_renaming = {
        '+ Perfect Clinical Practice': '+ Perfect Healthcare System Function'
    }

    num_deaths_summarized.index = pd.Series(num_deaths_summarized.index).replace(scenario_renaming)
    num_dalys_summarized.index = pd.Series(num_dalys_summarized.index).replace(scenario_renaming)

    name_of_plot = f'Deaths, {target_period()}'
    fig, ax = do_bar_plot_with_ci(num_deaths_summarized / 1e6)
    ax.set_title(name_of_plot)
    ax.set_ylabel('(Millions)')
    fig.tight_layout()
    ax.axhline(num_deaths_summarized.loc['Status Quo', 'mean']/1e6, color='green', alpha=0.5)
    ax.containers[1][2].set_color('green')
    ax.containers[1][0].set_color('k')
    ax.containers[1][1].set_color('red')
    fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
    fig.show()
    plt.close(fig)

    name_of_plot = f'DALYs, {target_period()}'
    fig, ax = do_bar_plot_with_ci(num_dalys_summarized / 1e6)
    ax.set_title(name_of_plot)
    ax.set_ylabel('(Millions)')
    ax.axhline(num_dalys_summarized.loc['Status Quo', 'mean']/1e6, color='green', alpha=0.5)
    ax.containers[1][2].set_color('green')
    ax.containers[1][0].set_color('k')
    ax.containers[1][1].set_color('red')
    fig.tight_layout()
    fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
    fig.show()
    plt.close(fig)

    # %% Deaths and DALYS averted relative to Status Quo
    num_deaths_averted = summarize(
        -1.0 *
        pd.DataFrame(
            find_difference_relative_to_comparison(
                num_deaths.loc[0],
                comparison='Status Quo')
        ).T
    ).iloc[0].unstack().reindex(param_names).drop(['No Healthcare System', 'With Hard Constraints', 'Status Quo'])

    pc_deaths_averted = 100.0 * summarize(
        -1.0 *
        pd.DataFrame(
            find_difference_relative_to_comparison(
                num_deaths.loc[0],
                comparison='Status Quo',
                scaled=True)
        ).T
    ).iloc[0].unstack().reindex(param_names).drop(['No Healthcare System', 'With Hard Constraints', 'Status Quo'])

    num_dalys_averted = summarize(
        -1.0 *
        pd.DataFrame(
            find_difference_relative_to_comparison(
                num_dalys.loc[0],
                comparison='Status Quo')
        ).T
    ).iloc[0].unstack().reindex(param_names).drop(['No Healthcare System', 'With Hard Constraints', 'Status Quo'])

    pc_dalys_averted = 100.0 * summarize(
        -1.0 *
        pd.DataFrame(
            find_difference_relative_to_comparison(
                num_dalys.loc[0],
                comparison='Status Quo',
                scaled=True)
        ).T
    ).iloc[0].unstack().reindex(param_names).drop(['No Healthcare System', 'With Hard Constraints', 'Status Quo'])

    # DEATHS
    name_of_plot = f'Additional Deaths Averted vs Status Quo, {target_period()}'
    fig, ax = do_bar_plot_with_ci(
        num_deaths_averted.clip(lower=0.0),
        annotations=[
            f"{round(row['mean'], 1)} ({round(row['lower'], 1)}-{round(row['upper'], 1)}) %"
            for _, row in pc_deaths_averted.clip(lower=0.0).iterrows()
        ]
    )
    ax.set_title(name_of_plot)
    ax.set_ylabel('Additional Deaths Averted')
    fig.tight_layout()
    fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
    fig.show()
    plt.close(fig)

    # DALYS
    name_of_plot = f'Additional DALYs Averted vs Status Quo, {target_period()}'
    fig, ax = do_bar_plot_with_ci(
        (num_dalys_averted / 1e6).clip(lower=0.0),
        annotations=[
            f"{round(row['mean'], 1)} ({round(row['lower'], 1)}-{round(row['upper'], 1)}) %"
            for _, row in pc_dalys_averted.clip(lower=0.0).iterrows()
        ]
    )
    ax.set_title(name_of_plot)
    ax.set_ylim(0, 12)
    ax.set_ylabel('Additional DALYS Averted (Millions)')
    fig.tight_layout()
    fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_').replace(',', '')))
    fig.show()
    plt.close(fig)

    # Plot DALYS incurred wrt wealth under selected scenarios (No HealthSystem, SQ, Perfect Healthcare seeking)
    #  in order to understand the root causes of the observation that more DALYS are averted under SQ in higher
    #  wealth quantiles.

    def get_total_num_dalys_by_wealth_and_label(_df):
        """Return the total number of DALYS in the TARGET_PERIOD by wealth and cause label."""
        wealth_cats = {5: "0-19%", 4: "20-39%", 3: "40-59%", 2: "60-79%", 1: "80-100%"}

        return (
            _df.loc[_df["year"].between(*[d.year for d in TARGET_PERIOD])]
            .drop(columns=["date", "year"])
            .assign(
                li_wealth=lambda x: x["li_wealth"]
                .map(wealth_cats)
                .astype(pd.CategoricalDtype(wealth_cats.values(), ordered=True))
            )
            .melt(id_vars=["li_wealth"], var_name="label")
            .groupby(by=["li_wealth", "label"])["value"]
            .sum()
        )

    total_num_dalys_by_wealth_and_label = summarize(
        extract_results(
            results_folder,
            module="tlo.methods.healthburden",
            key="dalys_by_wealth_stacked_by_age_and_time",
            custom_generate_series=get_total_num_dalys_by_wealth_and_label,
            do_scaling=True,
        ),
    ).pipe(set_param_names_as_column_index_level_0)[
        ['No Healthcare System', 'Status Quo', 'Perfect Healthcare Seeking']
    ].loc[:, (slice(None), 'mean')].droplevel(axis=1, level=1)

    fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True)
    name_of_plot = f'DALYS Incurred by Wealth and Cause {target_period()}'
    for _ax, _scenario_name, in zip(ax, total_num_dalys_by_wealth_and_label.columns):
        format_to_plot = total_num_dalys_by_wealth_and_label[_scenario_name].unstack()
        format_to_plot = format_to_plot \
            .sort_index(axis=0) \
            .reindex(columns=CAUSE_OF_DEATH_OR_DALY_LABEL_TO_COLOR_MAP.keys(), fill_value=0.0) \
            .sort_index(axis=1, key=order_of_cause_of_death_or_daly_label)
        (
            format_to_plot / 1e6
        ).plot.bar(stacked=True,
                   ax=_ax,
                   color=[get_color_cause_of_death_or_daly_label(_label) for _label in format_to_plot.columns],
                   )
        _ax.axhline(0.0, color='black')
        _ax.set_title(f'{_scenario_name}')
        _ax.set_ylabel('Number of DALYs Averted (/1e6)')
        _ax.set_ylim(0, 20)
        _ax.set_xlabel('Wealth Percentile')
        _ax.grid()
        _ax.spines['top'].set_visible(False)
        _ax.spines['right'].set_visible(False)
        _ax.legend(ncol=3, fontsize=8, loc='upper right')
        _ax.legend().set_visible(False)
    fig.suptitle(name_of_plot)
    fig.tight_layout()
    fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_')))
    fig.show()
    plt.close(fig)

    # Normalised Total DALYS (where DALYS in the highest wealth class are 100)
    # todo - copy the color coding from the bar plots above
    tots = total_num_dalys_by_wealth_and_label.groupby(axis=0, level=0).sum()
    normalised_tots = tots.div(tots.loc['80-100%'])

    fig, ax = plt.subplots(nrows=1, ncols=2, sharey=False)
    name_of_plot = f'DALYS Incurred by Wealth {target_period()}'
    (tots / 1e6).plot(ax=ax[0])
    ax[0].set_ylabel('Total DALYS (/million)')
    ax[0].set_xlabel('Wealth Percentile')
    ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=90)
    ax[0].set_ylim(0, 15)
    ax[0].legend(fontsize=8)
    (normalised_tots * 100.0).plot(ax=ax[1])
    ax[1].set_ylabel('Normalised DALYS\n100 = Highest Wealth quantile')
    ax[1].set_xlabel('Wealth Percentile')
    ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=90)
    ax[1].axhline(100.0, color='k')
    ax[1].legend().set_visible(False)
    fig.suptitle(name_of_plot)
    fig.tight_layout()
    fig.savefig(make_graph_file_name(name_of_plot.replace(' ', '_')))
    fig.show()
    plt.close(fig)

    #todo: Which disease are over-represented in No Healthcare System scenario...?


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
            "Directory containing results from running src/scripts/healthsystem/"
            "impact_of_healthsystem_under_diff_scenarios/scenario_impact_of_healthsystem.py "
            "script."
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
