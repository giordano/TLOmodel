import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from tlo.analysis.utils import extract_results, get_scenario_outputs, get_scenario_info, summarize, extract_params

plt.style.use('seaborn-darkgrid')


scenario_filename = 'pregnancy_only_scenario.py'
output_path = './outputs/sejjj49@ucl.ac.uk/'
results_folder = get_scenario_outputs(scenario_filename, output_path)[-1]

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

def output_comparison_graph_for_given_param_change(param):

    param_df = extract_params(results_folder)
    p_row = param_df.loc[param_df['module_param'] == param]['value']
    p_vals = [p_row.loc[draw][0] for draw in draws_list]
    s_pval = [str(val) for val in p_vals]

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


output_comparison_graph_for_given_param_change('Labour:prob_delay_one_two_fd')
