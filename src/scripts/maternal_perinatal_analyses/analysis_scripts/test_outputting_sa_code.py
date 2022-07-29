import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from tlo.analysis.utils import extract_results, get_scenario_outputs, get_scenario_info, summarize, extract_params

plt.style.use('seaborn-darkgrid')


scenario_filename = 'pregnancy_only_scenario.py'
output_path = './outputs/sejjj49@ucl.ac.uk'
results_folder = get_scenario_outputs(scenario_filename, output_path)[-1]
comparison_draw = 66
params_of_interest = {'PregnancySupervisor': ['treatment_effect_modifier_all_delays',
                                              'treatment_effect_modifier_one_delay'],

                                   'Labour': ['prob_haemostatis_uterotonics',
                                              'prob_successful_manual_removal_placenta',
                                              'pph_treatment_effect_mrp_md',
                                              'success_rate_pph_surgery',
                                              'pph_treatment_effect_surg_md',
                                              'pph_treatment_effect_hyst_md',
                                              'pph_bt_treatment_effect_md',

                                              'sepsis_treatment_effect_md',

                                              'eclampsia_treatment_effect_severe_pe',
                                              'eclampsia_treatment_effect_md',
                                              'anti_htns_treatment_effect_md',

                                              'prob_hcw_avail_uterotonic',
                                              'prob_hcw_avail_man_r_placenta',
                                              'prob_hcw_avail_blood_tran',

                                              'prob_hcw_avail_iv_abx',

                                              'prob_hcw_avail_anticonvulsant',

                                              'treatment_effect_modifier_one_delay',
                                              'treatment_effect_modifier_all_delays',
                                              'mean_hcw_competence_hc',
                                              'mean_hcw_competence_hp']}


path = f'{output_path}/test_sensitivity_analysis_{results_folder.name}'
if not os.path.isdir(path):
    os.makedirs(path)

scenario_inf = get_scenario_info(results_folder)

br = extract_results(
            results_folder,
            module="tlo.methods.demography",
            key="on_birth",
            custom_generate_series=(
                lambda df: df.assign(year=df['date'].dt.year).groupby(['year'])['year'].count()),
            do_scaling=False
        )
births = summarize(br)

dr = extract_results(
            results_folder,
            module="tlo.methods.demography",
            key="death",
            custom_generate_series=(
                lambda df: df.assign(year=df['date'].dt.year).groupby(['year', 'label'])['year'].count()),
            do_scaling=False)
death_results_total = dr.fillna(0)
death_results = death_results_total.loc[2010]

draws_list = list(range(scenario_inf['number_of_draws']))
sum_dr = summarize(death_results)

# by cause
dcr = extract_results(
        results_folder,
        module="tlo.methods.demography",
        key="death",
        custom_generate_series=(
            lambda df: df.assign(year=df['date'].dt.year).groupby(['year', 'cause'])['year'].count().fillna(0)
        ),
    )
death_cause_total = dcr.fillna(0)
death_cause = death_cause_total.loc[2010]
sum_dc = summarize(death_cause)

d_by_c_dfs_dict = {}
for c in ['eclampsia', 'sepsis', 'pph']:
    if c == 'eclampsia':
        total = sum_dc.loc['severe_pre_eclampsia'] + sum_dc.loc['eclampsia']
    if c == 'sepsis':
        total = sum_dc.loc['antenatal_sepsis'] + sum_dc.loc['intrapartum_sepsis'] + sum_dc.loc['postpartum_sepsis']
    if c == 'pph':
        total = sum_dc.loc['postpartum_haemorrhage'] + sum_dc.loc['secondary_postpartum_haemorrhage']

    d_by_c_dfs_dict.update({c: total})


ansb = extract_results(
    results_folder,
    module="tlo.methods.pregnancy_supervisor",
    key="antenatal_stillbirth",
    custom_generate_series=(
            lambda df: df.assign(year=df['date'].dt.year).groupby(['year'])['year'].count()),
    do_scaling=False
     )
an_still_births = ansb.fillna(0)
sum_an_sb = summarize(an_still_births)

ipsb = extract_results(
    results_folder,
    module="tlo.methods.labour",
    key="intrapartum_stillbirth",
    custom_generate_series=(
                lambda df: df.assign(year=df['date'].dt.year).groupby(['year'])['year'].count()),
    do_scaling=False
    )
ip_still_births = ipsb.fillna(0)
sum_ip_sb = summarize(ip_still_births)

tot_sb = an_still_births + ip_still_births
sum_tot_sb = summarize(tot_sb)

# todo: 1 - massive graph?
# todo: 2 - per parameter graphs?


def output_comparison_graph_for_given_param_change(param, orig_val, target_m, cause_dict):

    param_df = extract_params(results_folder)
    p_row = param_df.loc[param_df['module_param'] == param]['value']
    p_vals = [p_row.loc[draw][0] for draw in p_row.index]
    s_pval = [str(val) for val in p_vals]
    s_pval.append(f'baseline ({orig_val})')

    draws = list(p_row.index)
    draws.append(comparison_draw)

    if not target_m == 'all':
        d_by_c = cause_dict[target_m]

        cd_vals = [[d_by_c.loc[(v, 'mean')] for v in draws],
                  [d_by_c.loc[(v, 'lower')]for v in draws],
                  [d_by_c.loc[(v, 'upper')] for v in draws]]

    d_vals = [[sum_dr.loc['Maternal Disorders', v]['mean'] for v in draws],
              [sum_dr.loc['Maternal Disorders', v]['lower'] for v in draws],
              [sum_dr.loc['Maternal Disorders', v]['upper'] for v in draws]]

    b_vals = [[float(births[(v, 'mean')].values) for v in draws],
              [float(births[(v, 'lower')].values) for v in draws],
              [float(births[(v, 'upper')].values) for v in draws]]

    def get_mmrs(deaths):
        mmr = [[(x / y) * 100_000 for x, y in zip(deaths[0], b_vals[0])],
               [(x / y) * 100_000 for x, y in zip(deaths[1], b_vals[1])],
               [(x / y) * 100_000 for x, y in zip(deaths[2], b_vals[2])]]
        ci = [(x - y) / 2 for x, y in zip(mmr[2], mmr[1])]
        return [mmr, ci]

    mmr_total_data = get_mmrs(d_vals)
    if not target_m == 'all':
        mmr_cause_data = get_mmrs(cd_vals)

    def plot_bars(mmr, ci, type, color):
        plt.bar(s_pval, mmr[0], color=color, width=0.6, yerr=ci)
        plt.xlabel("Values for given Parameter")
        plt.ylabel('MMR')
        plt.title(f'{type} MMR by Draw varying {param}')
        plt.savefig(f'{path}/{type}_mmr_{param.replace(":", "_")}.png')
        plt.show()

    plot_bars(mmr_total_data[0], mmr_total_data[1], 'Total', 'blue')
    if not target_m == 'all':
        plot_bars(mmr_cause_data[0], mmr_cause_data[1], 'Cause specific', 'pink')

    def get_pdiffs(mmr):
        p_diffs = list()
        for i in mmr[0]:
            p_diffs.append(100 - ((i / mmr[0][-1]) * 100))
        p_diffs.remove(p_diffs[-1])
        return p_diffs

    if not target_m == 'all':
        mmr_cause_pd = get_pdiffs(mmr_cause_data[0])
    mmr_total_pd = get_pdiffs(mmr_total_data[0])

    s_pval.remove(f'baseline ({orig_val})')

    def plot_pdiff(p_diffs, type, colour):
        plt.bar(s_pval, p_diffs, color=colour, width=0.6)
        plt.xlabel("Values for given Parameter")
        plt.ylabel('Percentage Difference')
        plt.title(f'Percent Diff from Baseline {type} MMR in {param}')
        plt.savefig(f'{path}/{type}_pdiff_{param.replace(":", "_")}.png')
        plt.show()

    if not target_m == 'all':
        plot_pdiff(mmr_cause_pd, 'Cause specific', 'red')
    plot_pdiff(mmr_total_pd, 'Total', 'green')

"""
    def plot_bc(mean_vals, ci, group):
        plt.bar(s_pval, mean_vals, color='blue', width=0.6, yerr=ci)
        plt.xlabel("Parameter Values")
        plt.ylabel('Number of Deaths')
        plt.title(f'Number of Deaths due to {group} by Draw varying {param}')
        plt.savefig(f'{path}/{group} per draw.png')
        plt.show()

    for group, denom in (['Maternal Disorders', 'Neonatal Disorders'], [100_000, 1000]):

        mean_vals = [sum_dr.loc[group, draw]['mean'] for draw in draws_list]
        mean_birth_vals = [float(births[(draw, 'mean')].values) for draw in draws_list]
        rate = [(x/y) * denom for x, y in zip(mean_vals, mean_birth_vals)]
        ci_crude = [(x - y) / 2 for x, y in zip([sum_dr.loc[group, draw]['upper'] for draw in draws_list],
                                          [sum_dr.loc[group, draw]['lower'] for draw in draws_list])]
        plot_bc(mean_vals, ci_crude, f'{group} (Crude)')


    isb_mv = [float(sum_ip_sb[(draw, 'mean')].values) for draw in draws_list]
    isb_ci = [(x - y) / 2 for x, y in zip([float(sum_ip_sb[(draw, 'upper')].values) for draw in draws_list],
                                          [float(sum_ip_sb[(draw, 'lower')].values) for draw in draws_list])]
    plot_bc(isb_mv, isb_ci, 'Intrapartum Stillbirths')

    asb_mv = [float(sum_an_sb[(draw, 'mean')].values) for draw in draws_list]
    asb_ci = [(x - y) / 2 for x, y in zip([float(sum_an_sb[(draw, 'upper')].values) for draw in draws_list],
                                          [float(sum_an_sb[(draw, 'lower')].values) for draw in draws_list])]
    plot_bc(asb_mv, asb_ci, 'Antenatal Stillbirths')

    tsb_mv = [float(sum_tot_sb[(draw, 'mean')].values) for draw in draws_list]
    tsb_ci = [(x - y) / 2 for x, y in zip([float(sum_tot_sb[(draw, 'upper')].values) for draw in draws_list],
                                          [float(sum_tot_sb[(draw, 'lower')].values) for draw in draws_list])]
    plot_bc(tsb_mv, tsb_ci, 'Total Stillbirths')
    """

p_dict = {'Labour:prob_haemostatis_uterotonics': [0.57, 'pph'],
          'Labour:pph_treatment_effect_mrp_md': [0.7, 'pph'],
          'Labour:pph_treatment_effect_surg_md': [0.25, 'pph'],
          'Labour:pph_treatment_effect_hyst_md': [0.25, 'pph'],
          'Labour:pph_bt_treatment_effect_md': [0.4, 'pph'],
          'Labour:sepsis_treatment_effect_md': [0.2, 'sepsis'],
          'Labour:eclampsia_treatment_effect_severe_pe': [0.41, 'eclampsia'],
          'Labour:eclampsia_treatment_effect_md': [0.4, 'eclampsia'],
          'Labour:anti_htns_treatment_effect_md': [0.5, 'eclampsia'],
          'Labour:prob_hcw_avail_uterotonic': [0.99, 'pph'],
          'Labour:prob_hcw_avail_man_r_placenta': [0.82, 'pph'],
          'Labour:prob_hcw_avail_blood_tran': [0.86, 'pph'],
          'Labour:prob_hcw_avail_iv_abx': [0.99, 'sepsis'],
          'Labour:prob_hcw_avail_anticonvulsant': [0.93, 'eclampsia'],
          'Labour:success_rate_pph_surgery': [0.79, 'pph'],
          'Labour:treatment_effect_modifier_one_delay': [0.75, 'all'],
          'Labour:treatment_effect_modifier_all_delays':[0.5, 'all'],
          'Labour:mean_hcw_competence_hc': [0.602, 'all'],
          'Labour:mean_hcw_competence_hp': [0.662, 'all'],
          'PregnancySupervisor:treatment_effect_modifier_one_delay': [0.75, 'all'],
          'PregnancySupervisor:treatment_effect_modifier_all_delays': [0.5, 'all']}

for k in p_dict:
    cause_dict = d_by_c_dfs_dict
    output_comparison_graph_for_given_param_change(k, p_dict[k][0], p_dict[k][1], cause_dict)
