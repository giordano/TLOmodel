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

data_who_tb_2020 = pd.read_excel(rfp / 'ResourceFile_TB.xlsx', sheet_name='WHO_activeTB2020')
data_who_tb_2020.index = pd.to_datetime(data_who_tb_2020['year'], format='%Y')
data_who_tb_2020 = data_who_tb_2020.drop(columns=['year'])


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

# -----------------------------------------  TB INCIDENCE ----------------------------------------- #
new_active_tb = summarize(extract_results(
    results_folder,
    module="tlo.methods.tb",
    key="tb_shine_data",
    column="NewActiveTBCases",
    index="date",
    do_scaling=False))



fig, ax = plt.subplots()
plt.plot(data_who_tb_2020.index, data_who_tb_2020["estimated_inc_number_children"], label="WHO Estimates", color="b")
plt.fill_between(data_who_tb_2020["estimated_inc_number_children_low"].index,
                 data_who_tb_2020["estimated_inc_number_children_low"], data_who_tb_2020["estimated_inc_number_children_high"],
                 color="b", alpha=0.2)
plt.plot(new_active_tb.index, new_active_tb[0]["mean"], label="Scenario 0", color="r")
plt.fill_between(new_active_tb[0]["lower"].index, new_active_tb[0]["lower"], new_active_tb[0]["upper"], color="r", alpha=0.2)
plt.plot(new_active_tb.index, new_active_tb[1]["mean"], label="Scenario 4", color="g")
plt.fill_between(new_active_tb[1]["lower"].index, new_active_tb[1]["lower"], new_active_tb[1]["upper"], color="g", alpha=0.2)
plt.xlabel('Year')
plt.ylabel('Number of New Active TB Cases')
plt.title('New Active TB Cases (0 - 16 Years)')
plt.legend()
plt.show()

# -----------------------------------------  TB DIAGNOSIS ----------------------------------------- #

new_diagnosed_tb = summarize(extract_results(
    results_folder,
    module="tlo.methods.tb",
    key="tb_shine_data",
    column="NewDiagnosedTBCases",
    index="date",
    do_scaling=False))




fig, ax = plt.subplots()
plt.plot(data_who_tb_2020.index, data_who_tb_2020["new_cases_014"], label="WHO Estimates", color="b")
plt.plot(new_diagnosed_tb.index, new_diagnosed_tb[0]["mean"], label="Scenario 0", color="r")
plt.fill_between(new_diagnosed_tb[0]["lower"].index, new_diagnosed_tb[0]["lower"], new_diagnosed_tb[0]["upper"], color="r", alpha=0.2)
plt.plot(new_diagnosed_tb.index, new_diagnosed_tb[1]["mean"], label="Scenario 4", color="g")
plt.fill_between(new_diagnosed_tb[1]["lower"].index, new_diagnosed_tb[1]["lower"], new_diagnosed_tb[1]["upper"], color="g", alpha=0.2)
plt.ylim(bottom=600)
plt.xlabel('Year')
plt.ylabel('Number of New Diagnosed TB Cases')
plt.title('New Diagnosed TB Cases (0 - 16 Years)')
plt.legend()
plt.show()

# -----------------------------------------  TB TREATMENT ----------------------------------------- #

new_treated_tb = summarize(extract_results(
    results_folder,
    module="tlo.methods.tb",
    key="tb_shine_data",
    column="NewTreatedTBCases",
    index="date",
    do_scaling=False))



fig, ax = plt.subplots()
plt.plot(new_treated_tb.index, new_treated_tb[0]["mean"], label="Scenario 0", color="r")
plt.fill_between(new_treated_tb[0]["lower"].index, new_treated_tb[0]["lower"], new_treated_tb[0]["upper"], color="r", alpha=0.2)
plt.plot(new_treated_tb.index, new_treated_tb[1]["mean"], label="Scenario 4", color="g")
plt.fill_between(new_treated_tb[1]["lower"].index, new_treated_tb[1]["lower"], new_treated_tb[1]["upper"], color="g", alpha=0.2)
plt.ylim(bottom=300)
plt.xlabel('Year')
plt.ylabel('Number of New Treated TB Cases')
plt.title('New Treated TB Cases (0 - 16 Years)')
plt.legend()
plt.show()

# -----------------------------------------  CONSUMABLES - TREATMENT ----------------------------------------- #
cons_treatment = extract_healthsystem_summary(
    results_folder,
    module="tlo.methods.healthsystem.summary",
    key="Consumables",
    column="Item_Available")


fig, ax = plt.subplots()
plt.plot(cons_treatment.index, cons_treatment['0']['178'], label="Scenario 0: Child Tx", color="b")
plt.plot(cons_treatment.index, cons_treatment['0']['179'], label="Scenario 0: Child Re-tx", color="r")
plt.plot(cons_treatment.index, cons_treatment['1']['178'], label="Scenario 4: Child Tx", color="b", linestyle="dashed")
plt.plot(cons_treatment.index, cons_treatment['1']['179'], label="Scenario 4: Child Re-tx", color="r", linestyle="dashed")
plt.plot(cons_treatment.index, cons_treatment['1']['2675'], label="Scenario 4: Child Tx Shorter", color="g", linestyle="dashed")
plt.title("Treatment Use Over Time")
plt.xlabel("Year")
plt.ylabel("Quantity")
plt.legend()
plt.show()

# -----------------------------------------  CONSUMABLES - DIAGNOSTIC TESTS ----------------------------------------- #

fig, ax = plt.subplots()
plt.plot(cons_treatment.index, cons_treatment['0']['175'], label="Scenario 0: X-ray", color="b")
plt.plot(cons_treatment.index, cons_treatment['0']['184'], label="Scenario 0: Microscopy", color="r")
plt.plot(cons_treatment.index, cons_treatment['0']['187'], label="Scenario 0: Xpert", color="g")
plt.plot(cons_treatment.index, cons_treatment['1']['175'], label="Scenario 4: X-ray", color="b", linestyle="dashed")
plt.plot(cons_treatment.index, cons_treatment['1']['184'], label="Scenario 4: Microscopy", color="r", linestyle="dashed")
plt.plot(cons_treatment.index, cons_treatment['1']['187'], label="Scenario 4: Xpert", color="g", linestyle="dashed")
plt.title("Diagnostic Test Use Over Time")
plt.xlabel("Year")
plt.ylabel("Quantity")
plt.legend()
plt.show()







