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

outputspath = Path("./outputs/lmu17@ic.ac.uk")
rfp = Path('./resources')

# Find results_folder associated with a given batch_file (and get most recent [-1])
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


# --------------------- AVERAGE WORKER TIME --------------------- #

average_worker_time = summarize(
    extract_results(
        results_folder,
        module="tlo.methods.healthsystem.summary",
        key="Capacity",
        column="average_Frac_Time_Used_Overall",
        index="date",
        do_scaling=False))

make_plot(model=average_worker_time[0]["mean"],
          model_low=average_worker_time[0]["lower"],
          model_high=average_worker_time[0]["upper"])
plt.plot(average_worker_time[1]["mean"])
plt.title('Fraction of Health Worker Time Used')
plt.xlabel('Year')
plt.ylabel('Fraction of Health Worker Time Used (Averaged per year)')
plt.legend(['Baseline Scenario', 'Shorter Treatment Scenario'])
plt.show()


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






