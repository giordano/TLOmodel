from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tlo.analysis.utils import (
    extract_results,
    get_scenario_outputs,
    summarize,
)

outputspath = Path('./outputs/rmjlra2@ucl.ac.uk/')

results_folder = get_scenario_outputs('analysis_rti_determine_number_of_runs.py', outputspath)[- 1]

extracted_incidence_of_RTI = extract_results(results_folder,
                                             module="tlo.methods.rti",
                                             key="summary_1m",
                                             column="incidence of rti per 100,000",
                                             index="date"
                                             )
number_of_runs = extracted_incidence_of_RTI[0].columns
means = []
upper_q = []
lower_q = []
df = pd.DataFrame()
number_of_samples = 101

for i in number_of_runs:
    run_for_this_sample = []
    for n_sample in range(1, number_of_samples):
        n_runs = np.random.choice(number_of_runs, i + 1, replace=False)
        sample = extracted_incidence_of_RTI[0][n_runs]
        mean_each_run = sample.mean()

        sample_mean = mean_each_run.mean()
        run_for_this_sample.append(sample_mean)
    df[f"mean_for_{i + 1}_runs"] = run_for_this_sample

overall_mean = df['mean_for_20_runs'].mean()
scale_off_gbd = np.divide(954.2, overall_mean)
for col in df.columns:
    df[col] *= scale_off_gbd

gbd_inc_data = pd.read_csv("C:/Users/Robbie Manning Smith/Desktop/gbddata/all_countries_inc_data.csv")
gbd_inc_data = gbd_inc_data.loc[gbd_inc_data['location'] == 'Malawi']
gbd_inc_data = gbd_inc_data.loc[gbd_inc_data['measure'] == 'Incidence']
GBD_mean_inc = gbd_inc_data.val.mean()
gbd_inc_upper = gbd_inc_data.upper.mean()
gbd_inc_lower = gbd_inc_data.lower.mean()
for idx, col in enumerate(df.columns):
    plt.scatter([idx] * len(df[col]), df[col])
plt.xticks(number_of_runs, [f"{i + 1}" for i in number_of_runs])
plt.xlabel('N samples')
plt.ylabel('Mean incidence of RTI')
plt.fill_between(number_of_runs, gbd_inc_upper, gbd_inc_lower, alpha=0.5)
plt.savefig("C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/FinalPaperOutput/Number_of_run_justification.png",
            bbox_inches='tight')

