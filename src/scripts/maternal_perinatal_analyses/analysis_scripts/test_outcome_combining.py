import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from tlo.analysis.utils import extract_results, get_scenario_outputs

from ..analysis_scripts import analysis_utility_functions

plt.style.use('seaborn-darkgrid')


def test(scenario_file_dict, outputspath, intervention_years, service_of_interest):
    """

    """

    # Create dictionary containing the results folder for each scenario
    results_folders = {k: get_scenario_outputs(scenario_file_dict[k], outputspath)[-1] for k in scenario_file_dict}

    # Create folder to store graphs (if it hasnt already been created when ran previously)
    path = f'{outputspath}/test_analysis_{results_folders["Status Quo"].name}'
    folder = f'{path}/{service_of_interest}'

    if not os.path.isdir(path):
        os.makedirs(f'{outputspath}/test_analysis_{results_folders["Status Quo"].name}')

    if not os.path.isdir(folder):
        os.makedirs(f'{path}/{service_of_interest}')

    # Save the file path
    plot_destination_folder = folder

    def extract_deaths(folder, intervention_years):
        dr = extract_results(
            folder,
            module="tlo.methods.demography",
            key="death",
            custom_generate_series=(
                lambda df: df.assign(year=df['date'].dt.year).groupby(['year', 'label'])['year'].count()),
            do_scaling=False)
        death_results = dr.fillna(0)

        total_m_deaths = pd.Series(data=0, index=death_results.columns)
        total_n_deaths = pd.Series(data=0, index=death_results.columns)
        for year in intervention_years:
            m_deaths = death_results.loc[year, 'Maternal Disorders']
            n_deaths = death_results.loc[year, 'Neonatal Disorders']
            total_m_deaths += m_deaths
            total_n_deaths += n_deaths

        crude_m_deaths = [total_m_deaths.mean(), total_m_deaths.quantile(0.025), total_m_deaths.quantile(0.925)]
        crude_n_deaths = [total_n_deaths.mean(), total_n_deaths.quantile(0.025), total_n_deaths.quantile(0.925)]

        ansb = extract_results(
            folder,
            module="tlo.methods.pregnancy_supervisor",
            key="antenatal_stillbirth",
            custom_generate_series=(
                lambda df: df.assign(year=df['date'].dt.year).groupby(['year'])['year'].count()),
            do_scaling=False
        )
        an_still_births = ansb.fillna(0)

        ipsb = extract_results(
            folder,
            module="tlo.methods.labour",
            key="intrapartum_stillbirth",
            custom_generate_series=(
                lambda df: df.assign(year=df['date'].dt.year).groupby(['year'])['year'].count()),
            do_scaling=False
        )
        ip_still_births = ipsb.fillna(0)

        total_sb = pd.Series(data=0, index=an_still_births.columns)
        for year in intervention_years:
            ip_sbs = an_still_births.loc[year]
            an_sbs = ip_still_births.loc[year]
            total_sb += ip_sbs
            total_sb += an_sbs

        crude_m_deaths = [total_m_deaths.mean(), total_m_deaths.quantile(0.025), total_m_deaths.quantile(0.925)]
        crude_n_deaths = [total_n_deaths.mean(), total_n_deaths.quantile(0.025), total_n_deaths.quantile(0.925)]
        crude_sbs = [total_sb.mean(), total_sb.quantile(0.025), total_sb.quantile(0.925)]

        br = extract_results(
            folder,
            module="tlo.methods.demography",
            key="on_birth",
            custom_generate_series=(
                lambda df: df.assign(year=df['date'].dt.year).groupby(['year'])['year'].count()),
            do_scaling=False
        )
        births_results = br.fillna(0)
        total_births = pd.Series(data=0, index=births_results.columns)
        for year in intervention_years:
            births = births_results.loc[year]
            total_births += births

        mmr = [((crude_m_deaths[0]/total_births.mean()) * 100_000),
               ((crude_m_deaths[1]/total_births.quantile(0.025)) * 100_000),
               ((crude_m_deaths[2]/total_births.quantile(0.925)) * 100_000)]

        nmr = [((crude_n_deaths[0] / total_births.mean()) * 1000),
               ((crude_n_deaths[1] / total_births.quantile(0.025)) * 1000),
               ((crude_n_deaths[2] / total_births.quantile(0.925)) * 1000)]

        sbr = [((crude_sbs[0] / total_births.mean()) * 1000),
               ((crude_sbs[1] / total_births.quantile(0.025)) * 1000),
               ((crude_sbs[2] / total_births.quantile(0.925)) * 1000)]

        return {'crude_mat_deaths': crude_m_deaths,
                'crude_neo_deaths': crude_n_deaths,
                'crude_stillbirths': crude_sbs,
                'mmr': mmr,
                'nmr': nmr,
                'sbr': sbr}

    # Extract data from scenarios
    death = {k: extract_deaths(results_folders[k], intervention_years) for k in results_folders}

    for data, colour in zip(['crude_mat_deaths', 'crude_neo_deaths', 'crude_stillbirths', 'mmr', 'nmr', 'sbr'],
                            ['bisque', 'powderblue', 'mistyrose', 'thistle', 'bisque', 'powderblue']):
        labels = results_folders.keys()

        mean_vals = list()
        lq_vals = list()
        uq_vals = list()
        for k in death:
            mean_vals.append(death[k][data][0])
            lq_vals.append(death[k][data][1])
            uq_vals.append(death[k][data][2])

        fig = plt.figure(figsize=(10, 5))

        ci = [(x - y) / 2 for x, y in zip(uq_vals, lq_vals)]
        plt.bar(labels, mean_vals, color=colour, width=0.6, yerr=ci)
        plt.xlabel("Scenario")
        plt.ylabel(f'Mean {data}')
        plt.title(f'Mean {data} across Intervention Period (2020-2030) by Scenario')
        plt.savefig(f'{plot_destination_folder}/{data}.png')
        plt.show()

