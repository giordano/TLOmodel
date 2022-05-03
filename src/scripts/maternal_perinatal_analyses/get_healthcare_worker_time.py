from pathlib import Path
import pandas as pd

import numpy as np
import time
from tlo.analysis.utils import load_pickled_dataframes, get_scenario_outputs, extract_results

start_time = time.time()

outputspath = './outputs/sejjj49@ucl.ac.uk/'

bl_scenario_filename = 'baseline_sba_scenario'
int_scenario_filename = 'bemonc'

baseline_results_folder = get_scenario_outputs(bl_scenario_filename, outputspath)[-1]
intervention_results_folder = get_scenario_outputs(int_scenario_filename, outputspath)[-1]

intervention_years = list(range(2020, 2031))
runs = list(range(0, 4))

resourcefilepath = Path("./resources/healthsystem/human_resources/definitions")
time_by_appt = pd.read_csv(Path(resourcefilepath) / 'ResourceFile_Appt_Time_Table.csv')


def get_hcw_time_per_year(results_folder, sf):

    # Create df that replicates the 'extracted' df
    total_time_per_draw_per_year = pd.DataFrame(columns=[runs], index=[intervention_years])

    def get_hcw_time_by_appt_fp_and_fl(hcw):
        time_dict = {}
        for fl in ['1a', '1b', '2']:
            fl_dict = {}
            for at in ['NormalDelivery', 'CompDelivery', 'Csection', 'MajorSurg', 'InpatientDays']:
                if fl not in time_by_appt.loc[(time_by_appt.Appt_Type_Code == at)]['Facility_Level'].values:
                    time = 0
                elif hcw not in time_by_appt.loc[(time_by_appt.Facility_Level == fl) &
                                                 (time_by_appt.Appt_Type_Code == at)]['Officer_Category'].values:
                    time = 0
                else:
                    time = float(time_by_appt.loc[
                                     (time_by_appt.Appt_Type_Code == at) & (time_by_appt.Officer_Category == hcw) &
                                     (time_by_appt.Facility_Level == fl), 'Time_Taken_Mins'].values)

                fl_dict.update({at: time})
            time_dict.update({fl: fl_dict})

        return time_dict

    nm_time = get_hcw_time_by_appt_fp_and_fl('Nursing_and_Midwifery')
    clin_time = get_hcw_time_by_appt_fp_and_fl('Clinical')

    # TODO: add CSection code to CEmONC HSI
    # TODO: pharmacy time?

    # Loop over each draw
    for run in runs:
        # Load df, add year column and select only ANC interventions
        run_df = load_pickled_dataframes(results_folder, draw=0, run=run)

        hsi = run_df['tlo.methods.healthsystem']['HSI_Event']
        hsi['year'] = hsi['date'].dt.year

        # Get HSIs
        emonc_hsi = hsi.loc[hsi.TREATMENT_ID.str.contains('SkilledBirthAttendanceDuringLabour') & hsi.did_run]
        cemonc_hsi = hsi.loc[hsi.TREATMENT_ID.str.contains('ComprehensiveEmergencyObstetricCare') & hsi.did_run]
        newborn_hsi = hsi.loc[hsi.TREATMENT_ID.str.contains('CareOfTheNewbornBySkilledAttendant') &
                              hsi.did_run]

        for year in intervention_years:
            # Get total number of anc1 visits (have a higher HCW time requirement)

            # todo: this wont work when the value of the dict changes...idk why it would though
            total_emonc_normal = len(
                emonc_hsi.loc[(emonc_hsi.year == year) &
                              (emonc_hsi.Number_By_Appt_Type_Code == {'NormalDelivery': 1})])

            total_emonc_comp = len(
                emonc_hsi.loc[(emonc_hsi.year == year) &
                              (emonc_hsi.Number_By_Appt_Type_Code == {'CompDelivery': 1})])

            assert total_emonc_normal + total_emonc_comp == len(emonc_hsi.loc[(emonc_hsi.year == year)])

            total_cemonc_surg = len(
                cemonc_hsi.loc[(cemonc_hsi.year == year) &
                               (cemonc_hsi.Number_By_Appt_Type_Code == {'MajorSurg': 1})])

            total_cemonc_other = len(
                cemonc_hsi.loc[(cemonc_hsi.year == year) &
                               (cemonc_hsi.Number_By_Appt_Type_Code == {'MajorSurg': 0})])

            assert total_cemonc_surg + total_cemonc_other == len(cemonc_hsi.loc[(cemonc_hsi.year == year)])

            total_newborn = len(newborn_hsi.loc[(newborn_hsi.year == year)])

            # TODO: THIS IS WRONG- WE NEED TO LOG FACILITY LEVEL OF HSIS
            total_bemonc_hcw_time = \
                (total_emonc_normal * (nm_time['1b']['NormalDelivery'] +
                                       clin_time['1b']['NormalDelivery'])) + \
                (total_emonc_comp * (nm_time['1b']['NormalDelivery'] +
                                     clin_time['1b']['NormalDelivery'])) + \
                (total_newborn * (nm_time['1b']['InpatientDays'] +
                                  clin_time['1b']['InpatientDays']))

            total_cemonc_time = \
                (total_cemonc_surg * (nm_time['1b']['MajorSurg'] +
                                      clin_time['1b']['MajorSurg'])) + \
                (total_cemonc_other * (nm_time['1b']['InpatientDays'] +
                                       clin_time['1b']['InpatientDays']))

            total_time_per_draw_per_year.loc[year, run] = (total_bemonc_hcw_time + total_cemonc_time / 60) * sf
            # returns time in hours

            # todo: better to output for cemonc/emonc HSIs

    # TODO: not sure on the maths here (percentiles vs quantiles)
    mean_time = [total_time_per_draw_per_year.loc[year].to_numpy().mean() for year in intervention_years]
    lq_time = [np.percentile(total_time_per_draw_per_year.loc[year].to_numpy(), 2.5) for year in intervention_years]
    uq_time = [np.percentile(total_time_per_draw_per_year.loc[year].to_numpy(), 92.5) for year in intervention_years]

    return [mean_time, lq_time, uq_time]

baseline_cost_data = get_hcw_time_per_year(baseline_results_folder, scaling_factors[k])
intervention_cost_data = get_hcw_time_per_year(intervention_results_folder, 'AntenatalCare')

end_time = time.time()
print("The time of execution of above program is :", end_time-start_time)
