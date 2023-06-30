"""Analyse scenarios for impact of TB-related development assistance for health."""
# python src/scripts/hiv/projections_jan2023/Analysis_TB_DAH_scenarios.py --scenario-outputs-folder outputs\nic503@york.ac.uk
import argparse
import datetime
from tlo import Date
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from tlo.analysis.utils import (
    extract_params,
    extract_results,
    get_scenario_info,
    get_scenario_outputs,
    load_pickled_dataframes,
    summarize,
)
resourcefilepath = Path("./resources")
# datestamp = datetime.date.today().strftime("__%Y_%m_%d")
outputspath = Path("./outputs/nic503@york.ac.uk")
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

# Get basic information about the results
results_folder = get_scenario_outputs("baseline_tb_services_scenario-2023-06-23T213051Z", outputspath)[-1]

# Look at one log (so can decide what to extract)
log = load_pickled_dataframes(results_folder)

# Get basic information about the results
info = get_scenario_info(results_folder)
#print("The scenario info as follows:")
#print(info)

# Check if runs_per_draw is 1
if info["runs_per_draw"] == 1:
    # Extract the parameters that have varied over the set of simulations
    params = extract_params(results_folder)
    #draw = 0

    def get_person_years(_df):
        """ extract person-years for each draw/run """
        years = pd.to_datetime(_df["date"]).dt.year
        py = pd.Series(dtype="int64", index=years)
        for year in years:
            tot_py = (
                (_df.loc[pd.to_datetime(_df["date"]).dt.year == year]["M"]).apply(pd.Series) +
                (_df.loc[pd.to_datetime(_df["date"]).dt.year == year]["F"]).apply(pd.Series)
            ).transpose()
            py[year] = tot_py.sum().values[0]
        py.index = pd.to_datetime(years, format="%Y")
        return py

    pyears = summarize(
        extract_results(
            results_folder,
            module="tlo.methods.demography",
            key="person_years",
            custom_generate_series=get_person_years,
            do_scaling=True
        )
    )

    # Prints person-years time to excel
    tb_pyears = pd.DataFrame(pyears)
    tb_pyears.to_excel(outputspath / "pyears_baseline_run_1.xlsx")

    # Number of TB deaths and mortality rate
    results_deaths = extract_results(
        results_folder,
        module="tlo.methods.demography",
        key="death",
        custom_generate_series=(
            lambda df: df.assign(year=df["date"].dt.year).groupby(["year", "cause"])["person_id"].count()
        ),
        do_scaling=True,
    )

    # Removes multi-index
    results_deaths = results_deaths.reset_index()
    print(results_deaths)

    tb_deaths = results_deaths.loc[results_deaths.cause.isin(["AIDS_non_TB", "AIDS_TB", "TB"])]
    AIDS_TB = results_deaths.loc[results_deaths.cause == "AIDS_non_TB"]
    AIDS_non_TB = results_deaths.loc[results_deaths.cause == "AIDS_TB"]
    TB = results_deaths.loc[results_deaths.cause == "TB"]

    combined_tb_table = pd.concat([AIDS_non_TB, AIDS_TB, TB])
    combined_tb_table.to_excel(outputspath / "combined_tb_table_baseline_run_1.xlsx")

    scaling_factor_key = log['tlo.methods.demography']['scaling_factor']
    print(f"Keys of log['tlo.methods.demography']: {log['tlo.methods.demography'].keys()}")
    print("Scaling Factor Key:", scaling_factor_key)

    def tb_mortality_rate(results_folder, pyears):
        tb_deaths = extract_results(
            results_folder,
            module="tlo.methods.demography",
            key="death",
            custom_generate_series=(
                lambda df: df.assign(year=df["date"].dt.year).groupby(["year", "cause"])["person_id"].count()
            ),
            do_scaling=True,
        )

        # Select only causes AIDS_TB, AIDS_non_TB, and TB
        tb_deaths1 = results_deaths.loc[(results_deaths.cause == "AIDS_TB") |
                                        (results_deaths.cause == "AIDS_non_TB") |
                                        (results_deaths.cause == "AIDS_non_TB")]
        # Group deaths by year
        tb_deaths2 = pd.DataFrame(tb_deaths1.groupby(["year"]).sum())
        tb_deaths2.to_excel(outputspath / "raw_mortality_baseline_run_1.xlsx")

        # Divide each draw/run by the respective person-years from that run
        # Need to reset index as they don't match exactly (date format)
        tb_deaths3 = tb_deaths2.reset_index(drop=True) / pyears.reset_index(drop=True)
        print("deaths3 are:", tb_deaths)

        tb_mortality_rate = {}  # empty dict
        tb_mortality_rate["year"] = tb_deaths2.index
        tb_mortality_rate["median"] = tb_deaths3.quantile(0.5, axis=1) * 100000
        tb_mortality_rate["lower"] = tb_deaths3.quantile(0.025, axis=1) * 100000
        tb_mortality_rate["upper"] = tb_deaths3.quantile(0.975, axis=1) * 100000
        return tb_mortality_rate

    # Call the function with appropriate arguments
    mortality_rates = tb_mortality_rate(results_folder, pyears)
    mortality_rates_summary = pd.DataFrame.from_dict(mortality_rates)

    # Print the resulting mortality rates
    mortality_rates_summary.to_excel(outputspath / "mortality_rates_summary_run_1.xlsx", index=False)
    print(mortality_rates_summary)

    # Print scaling factor to population level estimates
    print(f"The scaling factor is: {log['tlo.methods.demography']['scaling_factor']}")

    def get_tb_dalys(df_):
        # Get DALYs of TB
        years = df_['year'].value_counts().keys()
        dalys = pd.Series(dtype='float64', index=years)
        for year in years:
            tot_dalys = df_.drop(columns='date').groupby(['year']).sum().apply(pd.Series)
            dalys[year] = tot_dalys.loc[(year, ['TB (non-AIDS)', 'non_AIDS_TB'])].sum()
        dalys.sort_index()
        return dalys

    # Extract DALYs from model and scale
    tb_dalys_count = extract_results(
        results_folder,
        module="tlo.methods.healthburden",
        key="dalys",
        custom_generate_series=get_tb_dalys,
        do_scaling=True
    )

    # Get mean/upper/lower statistics
    dalys_summary = summarize(tb_dalys_count).sort_index()
    print("DALYs for TB are as follows:")
    print(dalys_summary)
    dalys_summary.to_excel(outputspath / "summarised_tb_dalys_baseline_run_1.xlsx")

    # Extracting secondary outcomes
    print(f"Keys of log['tlo.methods.tb']: {log['tlo.methods.tb'].keys()}")

    # Extracts PLHIV with TB
    tb_hiv_prop = summarize(
        extract_results(
            results_folder,
            module="tlo.methods.tb",
            key="tb_incidence",
            column="prop_active_tb_in_plhiv",
            index="date",
            do_scaling=False,
        ),
        collapse_columns=True,
    )

    tb_hiv_prop.index = tb_hiv_prop.index.year
    tb_hiv_prop_with_year = pd.DataFrame(tb_hiv_prop)
    tb_hiv_prop.to_excel(outputspath / "PLHIV_tb_baseline_run_1.xlsx")

    # MDR TB cases
    mdr_tb_cases = summarize(
        extract_results(
            results_folder,
            module="tlo.methods.tb",
            key="tb_mdr",
            column="tbPropActiveCasesMdr",
            index="date",
            do_scaling=False,
        ),
        collapse_columns=True,
    )
    mdr_tb_cases.index = mdr_tb_cases.index.year
    mdr_tb = pd.DataFrame(mdr_tb_cases)
    mdr_tb.to_excel(outputspath / "mdr_tb_baseline_run_1.xlsx")

    # TB treatment coverage
    tb_treatment = summarize(
        extract_results(
            results_folder,
            module="tlo.methods.tb",
            key="tb_treatment",
            column="tbTreatmentCoverage",
            index="date",
            do_scaling=False,
        ),
        collapse_columns=True,
    )
    tb_treatment_cov = pd.DataFrame(tb_treatment)
    tb_treatment_cov.to_excel(outputspath / "tb_treatment_coverage_baseline_run_1.xlsx")

    # TB Incidence
    tb_inc = summarize(
        extract_results(
            results_folder,
            module="tlo.methods.tb",
            key="tb_incidence",
            column="num_new_active_tb",
            index="date",
            do_scaling=False,
        ),
        collapse_columns=True,
    )
    tb_incidence = pd.DataFrame(tb_inc)
    tb_incidence.to_excel(outputspath / "tb_incidence_baseline_run_1.xlsx")

    tb_inc.index = tb_inc.index.year
    tb_inc = tb_inc.reset_index(drop=True)
    pyears = pyears.reset_index(drop=True)

    # Computing TB incidence rate
    activeTB_inc_rate = (tb_inc["mean"] / pyears["mean"]) * 100000
    activeTB_inc_rate = pd.DataFrame(activeTB_inc_rate)
    activeTB_inc_rate.to_excel(outputspath / "tb_incidence_rate_baseline_run_1.xlsx")

    # Calculate the rates
    activeTB_inc_rate = (tb_inc["mean"] / pyears["mean"]) * 100000
    upper_activeTB_inc_rate = (tb_inc["upper"] / pyears["upper"]) * 100000
    lower_activeTB_inc_rate = (tb_inc["lower"] / pyears["lower"]) * 100000

    # Create a new dataframe with the calculated values
    summarized_incidence = pd.DataFrame({
        "TB_inc_rate": activeTB_inc_rate,
        "upper": upper_activeTB_inc_rate,
        "lower": lower_activeTB_inc_rate
    })

    # summarized_incidence .to_excel(outputspath / "baseline_incidence_rate.xlsx")
    if __name__ == "__main__":

        parser = argparse.ArgumentParser(
            "Analyse scenario results for baseline scenario for TB diagnosis"
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