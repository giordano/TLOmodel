"""Analyse the results of scenario to test impact of Noxpert diagnosis."""
# python src/scripts/hiv/projetions_jan2023/analysis_impact_of_noxpert_diagnosis2.py --scenario-outputs-folder outputs/nic503@york.ac.uk --show-figures
import argparse
import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tlo.analysis.utils import (
    extract_params,
    extract_results,
    get_scenario_info,
    get_scenario_outputs,
    load_pickled_dataframes,
    summarize,
)

resourcefilepath = Path("./resources")
datestamp = datetime.date.today().strftime("__%Y_%m_%d")
outputspath = Path("./outputs/nic503@york.ac.uk")

def extract_total_deaths(results_folder):
    def extract_deaths_total(df: pd.DataFrame) -> pd.Series:
        return pd.Series({"Total": len(df)})

    sum_deaths = extract_results(
        results_folder,
        module="tlo.methods.demography",
        key="death",
        custom_generate_series=extract_deaths_total,
        do_scaling=True
    )
    if sum_deaths is not None:
        sum_deaths.to_excel(outputspath / "total_deaths.xlsx", index=True)
    else:
        print("Error: Unable to extract total deaths.")

def extract_total_dalys(results_folder):

    def extract_dalys_total(df: pd.DataFrame) -> pd.Series:
        return pd.Series({"Total": len(df)})

    sum_dalys= extract_results(
        results_folder,
        module="tlo.methods.healthburden",
        key="dalys_stacked",
        custom_generate_series=extract_dalys_total,
        do_scaling=True
    )
    sum_dalys.to_excel(outputspath/"total_dalys.xlsx", index=True)
def make_plot(summarized_total_deaths, param_strings):
    fig, ax = plt.subplots()
    number_of_draws = len(param_strings)
    statistic_values = {
        s: np.array(
            [summarized_total_deaths[(d, s)].values[0] for d in range(number_of_draws)]
        )
        for s in ["mean", "lower", "upper"]
    }
    ax.bar(
        param_strings,
        statistic_values["mean"],
        yerr=[
            statistic_values["mean"] - statistic_values["lower"],
            statistic_values["upper"] - statistic_values["mean"]
        ]
    )
    ax.set_ylabel("Total number of deaths")
    fig.tight_layout()
    return fig, ax
def compute_difference_in_deaths_across_runs(total_deaths, scenario_info):
    deaths_difference_by_run = [
        total_deaths[0][run_number]["total_deaths"] - total_deaths[1][run_number]["total_deaths"]
        for run_number in range(scenario_info["runs_per_draw"])
    ]
    return np.mean(deaths_difference_by_run)
    deaths_difference_by_run.to_excel(outputspath/"total_dalys.xlsx", index=True)


if __name__ == "__main__":

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        "Analyse scenario results for noXpert pathway"
    )
    parser.add_argument(
        "--scenario-outputs-folder",
        type=Path,
        required=True,
        help="Path to folder containing scenario outputs",
    )
    parser.add_argument(
        "--show-figures",
        action="store_true",
        help="Whether to interactively show generated Matplotlib figures",
    )
    parser.add_argument(
        "--save-figures",
        action="store_true",
        help="Whether to save generated Matplotlib figures to results folder",
    )
    args = parser.parse_args()

    # Find results_folder associated with a given batch_file and get most recent
    results_folder = get_scenario_outputs("scenario_impact_noXpert_diagnosis.py", outputspath)[-1]

    # Load log (useful for checking what can be extracted)
    log = load_pickled_dataframes(results_folder)

    # Get basic information about the results
    scenario_info = get_scenario_info(results_folder)

    # Get the parameters that have varied over the set of simulations
    params = extract_params(results_folder)

    # Create a list of strings summarizing the parameter values in the different draws
    param_strings = [f"{row.module_param}={row.value}" for _, row in params.iterrows()]
    # extracts deaths from runs
    total_deaths = extract_total_deaths(results_folder)

    # Compute and print the difference between the deaths across the scenario draws
    mean_deaths_difference_by_run = compute_difference_in_deaths_across_runs(
        total_deaths, scenario_info
    )
    print(f"Mean difference in total deaths = {mean_deaths_difference_by_run:.3g}")

    # Plot the total deaths across the two scenario draws as a bar plot with error bars
    fig_1, ax_1 = make_plot(summarize(total_deaths), param_strings)
    #fig_2, ax_1 = make_plot(summarize(sum_dalys), param_strings)

    # Show Matplotlib figure windows
    if args.show_figures:
        plt.show()

    if args.save_figures:
        fig_1.savefig(results_folder / "total_deaths_across_scenario_draws.pdf")

