import pandas as pd
import os
from matplotlib import pyplot as plt

from tlo.analysis.utils import (
    extract_results,
    get_scenario_outputs,
)

from scripts.maternal_perinatal_analyses import analysis_utility_functions


def met_need_and_contributing_factors_for_deaths(scenario_file_dict, outputspath, intervention_years):
    """
    """

    # Find results folder (most recent run generated using that scenario_filename)
    results_folders = {k: get_scenario_outputs(scenario_file_dict[k], outputspath)[-1] for k in scenario_file_dict}

    path = f'{outputspath}/met_need_{results_folders["Status Quo"].name}'
    if not os.path.isdir(path):
        os.makedirs(f'{outputspath}/met_need_{results_folders["Status Quo"].name}')

    plot_destination_folder = path

    # Get complication dataframes
    comp_dfs = {k: analysis_utility_functions.get_modules_maternal_complication_dataframes(results_folders[k]) for
                k in results_folders}

    treatments = ['pac', 'ep_case_mang', 'abx_an_sepsis', 'uterotonics', 'man_r_placenta', 'abx_pn_sepsis',
                  'ur_surg', 'mag_sulph_an_severe_pre_eclamp', 'mag_sulph_an_eclampsia',
                  'iv_htns_an_severe_pre_eclamp', 'iv_htns_an_severe_gest_htn', 'iv_htns_an_eclampsia',
                  'iv_htns_pn_severe_pre_eclamp', 'iv_htns_pn_severe_gest_htn', 'iv_htns_pn_eclampsia',
                  'mag_sulph_pn_severe_pre_eclamp', 'mag_sulph_pn_eclampsia']

    # ============================================ MET NEED ==========================================================
    def get_total_interventions_delivered(results_folder, interventions, intervention_years):

        intervention_results = extract_results(
            results_folder,
            module="tlo.methods.labour.detail",
            key="intervention",
            custom_generate_series=(
                lambda df: df.assign(year=df['date'].dt.year).groupby(['year', 'int'])['year'].count()),
            do_scaling=True
        )

        treatment_dict = dict()

        for treatment in interventions:

            treatment_dict.update({treatment: analysis_utility_functions.get_mean_and_quants_from_str_df(
                intervention_results, treatment, intervention_years)})

        return treatment_dict

    ints = {k: get_total_interventions_delivered(results_folders[k], treatments, intervention_years) for k in
            results_folders}

    # todo: (notes) Uterotonics were previously conditioned on them stopping bleeding not just being delivered (now
    #  been moved)
    # todo: (notes) Same issue with retained placenta

    def get_crude_complication_numbers(comp_dfs, intervention_years):
        crude_comps = dict()

        def sum_lists(list1, list2):
            mean = [x + y for x, y in zip(list1[0], list2[0])]
            lq = [x + y for x, y in zip(list1[1], list2[1])]
            uq = [x + y for x, y in zip(list1[2], list2[2])]

            return [mean, lq, uq]

        # Ectopic
        crude_comps.update({'ectopic': analysis_utility_functions.get_mean_and_quants_from_str_df(
                    comp_dfs['pregnancy_supervisor'], 'ectopic_unruptured', intervention_years)})

        # Complicated abortion
        incidence_compsa = analysis_utility_functions.get_mean_and_quants_from_str_df(
            comp_dfs['pregnancy_supervisor'], 'complicated_spontaneous_abortion', intervention_years)
        incidence_compia = analysis_utility_functions.get_mean_and_quants_from_str_df(
            comp_dfs['pregnancy_supervisor'], 'complicated_induced_abortion', intervention_years)
        crude_comps.update({'abortion': sum_lists(incidence_compia, incidence_compsa)})

        # Antenatal/Intrapartum Sepsis
        incidence_an_sep = analysis_utility_functions.get_mean_and_quants_from_str_df(
            comp_dfs['pregnancy_supervisor'], 'clinical_chorioamnionitis', intervention_years)
        incidence_la_sep = analysis_utility_functions.get_mean_and_quants_from_str_df(
            comp_dfs['labour'], 'sepsis', intervention_years)
        crude_comps.update({'an_ip_sepsis': sum_lists(incidence_an_sep, incidence_la_sep)})

        # Postpartum Sepsis
        incidence_pn_l_sep = analysis_utility_functions.get_mean_and_quants_from_str_df(
            comp_dfs['labour'], 'sepsis_postnatal', intervention_years)
        incidence_pn_p_sep = analysis_utility_functions.get_mean_and_quants_from_str_df(
            comp_dfs['postnatal_supervisor'], 'sepsis', intervention_years)
        crude_comps.update({'pp_sepsis': sum_lists(incidence_pn_l_sep, incidence_pn_p_sep)})

        # PPH - uterine atony
        incidence_ua_pph = analysis_utility_functions.get_mean_and_quants_from_str_df(
            comp_dfs['labour'], 'pph_uterine_atony', intervention_years)
        incidence_oth_pph = analysis_utility_functions.get_mean_and_quants_from_str_df(
            comp_dfs['labour'], 'pph_other', intervention_years)
        crude_comps.update({'pph_uterine_atony': sum_lists(incidence_ua_pph, incidence_oth_pph)})

        # PPH - retained placenta
        incidence_p_rp = analysis_utility_functions.get_mean_and_quants_from_str_df(
            comp_dfs['labour'], 'pph_retained_placenta', intervention_years)
        incidence_s_rp = analysis_utility_functions.get_mean_and_quants_from_str_df(
            comp_dfs['postnatal_supervisor'], 'secondary_postpartum_haemorrhage', intervention_years)
        crude_comps.update({'pph_retained_p': sum_lists(incidence_p_rp, incidence_s_rp)})

        # Uterine rupture
        crude_comps.update({'uterine_rupture': analysis_utility_functions.get_mean_and_quants_from_str_df(
            comp_dfs['labour'], 'uterine_rupture', intervention_years)})

        # Severe pre-eclampsia - antenatal
        incidence_a_spe = analysis_utility_functions.get_mean_and_quants_from_str_df(
            comp_dfs['pregnancy_supervisor'], 'severe_pre_eclamp', intervention_years)
        incidence_p_spe = analysis_utility_functions.get_mean_and_quants_from_str_df(
            comp_dfs['labour'], 'severe_pre_eclamp', intervention_years)
        crude_comps.update({'spe_an_la': sum_lists(incidence_a_spe, incidence_p_spe)})

        # Severe pre-eclampsia - postnatal
        crude_comps.update({'spe_pn': analysis_utility_functions.get_mean_and_quants_from_str_df(
            comp_dfs['postnatal_supervisor'], 'severe_pre_eclamp', intervention_years)})

        # Severe gestational hypertension - antenatal
        incidence_a_sgh = analysis_utility_functions.get_mean_and_quants_from_str_df(
            comp_dfs['pregnancy_supervisor'], 'severe_gest_htn', intervention_years)
        incidence_p_sgh = analysis_utility_functions.get_mean_and_quants_from_str_df(
            comp_dfs['labour'], 'severe_gest_htn', intervention_years)
        crude_comps.update({'sgh_an_la': sum_lists(incidence_a_sgh, incidence_p_sgh)})

        # Severe gestational hypertension - postnatal
        crude_comps.update({'sgh_pn': analysis_utility_functions.get_mean_and_quants_from_str_df(
            comp_dfs['postnatal_supervisor'], 'severe_gest_htn', intervention_years)})

        # Eclampsia - antenatal
        incidence_a_ec = analysis_utility_functions.get_mean_and_quants_from_str_df(
            comp_dfs['pregnancy_supervisor'], 'eclampsia', intervention_years)
        incidence_p_ec = analysis_utility_functions.get_mean_and_quants_from_str_df(
            comp_dfs['labour'], 'eclampsia', intervention_years)
        crude_comps.update({'ec_an_la': sum_lists(incidence_a_ec, incidence_p_ec)})

        # Eclampsia - postnatal
        crude_comps.update({'ec_pn': analysis_utility_functions.get_mean_and_quants_from_str_df(
            comp_dfs['postnatal_supervisor'], 'eclampsia', intervention_years)})

        return crude_comps

    comp_numbers = {k: get_crude_complication_numbers(comp_dfs[k], intervention_years) for k in results_folders}

    def get_met_need(ints, crude_comps):


        def update_met_need_dict(comp, treatment):
            if (0 in ints[treatment][0]) or (0 in crude_comps[comp][0]):
                mean_met_need = [0] * len(intervention_years)
            else:
                mean_met_need = [(x / y) * 100 for x, y in zip(ints[k][0], crude_comps[comp][0])]

            if (0 in ints[treatment][1]) or (0 in crude_comps[comp][1]):
                lq_mn = [0] * len(intervention_years)
            else:
                lq_mn = [(x / y) * 100 for x, y in zip(ints[treatment][1], crude_comps[comp][1])]

            if (0 in ints[treatment][2]) or (0 in crude_comps[comp][2]):
                uq_mn = [0] * len(intervention_years)
            else:
                uq_mn = [(x / y) * 100 for x, y in zip(ints[treatment][2], crude_comps[comp][2])]

            met_need_dict.update({treatment: [mean_met_need, lq_mn, uq_mn]})


        met_need_dict = dict()
        comp_and_treatment = {'ectopic': 'ep_case_mang',
                              'abortion': 'pac',
                              'an_ip_sepsis': 'abx_an_sepsis',
                              'uterine_rupture': 'ur_surg',
                              'pph_uterine_atony': 'uterotonics',
                              'pph_retained_p': 'man_r_placenta',
                              'pp_sepsis': 'abx_pn_sepsis',
                              'spe_an_la': ['mag_sulph_an_severe_pre_eclamp', 'iv_htns_an_severe_pre_eclamp'],
                              'spe_pn': ['iv_htns_pn_severe_pre_eclamp', 'mag_sulph_pn_severe_pre_eclamp'],
                              'sgh_an_la': 'iv_htns_an_severe_gest_htn',
                              'sgh_pn': 'iv_htns_pn_severe_gest_htn',
                              'ec_an_la': ['iv_htns_an_eclampsia', 'mag_sulph_an_eclampsia'],
                              'ec_pn': ['iv_htns_an_eclampsia', 'iv_htns_pn_eclampsia']}

        for k in comp_and_treatment:
            if isinstance(comp_and_treatment[k], list):
                update_met_need_dict(k, comp_and_treatment[k])
            else:
                for l in comp_and_treatment[k]:
                  update_met_need_dict(k, comp_and_treatment[k][l])


        # todo: check this....
        # todo: blood, CS, AVD, other surgeries

        return met_need_dict

    met_need = {k: get_met_need(ints[k], comp_numbers[k]) for k in ints}

    for t in treatments:
        fig, ax = plt.subplots()
        for k, colour in zip(met_need, ['deepskyblue', 'olivedrab', 'darksalmon', 'darkviolet']):
            ax.plot(intervention_years, met_need[k][t][0], label=k, color=colour)
            ax.fill_between(intervention_years, met_need[k][t][1], met_need[k][t][2], color=colour, alpha=.1)

        plt.ylabel('% of Cases Receiving Treatment')
        plt.xlabel('Year')
        plt.title(f'Met need for {t} Per Year by Scenario')
        plt.gca().set_ylim(bottom=0)
        plt.legend()
        plt.savefig(f'{plot_destination_folder}/{t}.png')
        plt.show()


# ===================================== CONTRIBUTION TO DEATH ========================================================
    factors = ['delay_one_two', 'delay_three', 'didnt_seek_care', 'cons_not_avail', 'comp_not_avail',
               'hcw_not_avail']

    def get_factors_impacting_death(results_folder, factors, intervention_years):
        total_deaths = extract_results(
            results_folder,
            module="tlo.methods.labour.detail",
            key="death_mni",
            custom_generate_series=(
                lambda df: df.assign(year=df['date'].dt.year).groupby(['year'])['year'].count()),
            do_scaling=True
        )

        deaths = analysis_utility_functions.get_mean_and_quants(total_deaths, intervention_years)

        factors_prop = dict()
        for factor in factors:

            factor_df = extract_results(
                results_folder,
                module="tlo.methods.labour.detail",
                key="death_mni",
                custom_generate_series=(
                    lambda df: df.loc[df[factor]].assign(
                        year=df['date'].dt.year).groupby(['year', factor])['year'].count()),
                do_scaling=True
            )

            year_means = list()
            lower_quantiles = list()
            upper_quantiles = list()

            # year_means = [factor_df.loc[year, True].mean() for year in intervention_years if year in factor_df.index]

            for year in intervention_years:
                if year in factor_df.index:
                    year_means.append(factor_df.loc[year, True].mean())
                    lower_quantiles.append(factor_df.loc[year, True].quantile(0.025))
                    upper_quantiles.append(factor_df.loc[year, True].quantile(0.925))
                else:
                    year_means.append(0)
                    lower_quantiles.append(0)
                    lower_quantiles.append(0)

            factor_data = [year_means, lower_quantiles, upper_quantiles]

            # factor_data = analysis_utility_functions.get_mean_and_quants(factor_df, intervention_years)

            if (0 in factor_data[0]) or (0 in deaths[0]):
                mean = [0] * len(intervention_years)
            else:
                mean = [(x / y) * 100 for x, y in zip(factor_data[0], deaths[0])]

            if (0 in factor_data[1]) or (0 in deaths[1]):
                lq = [0] * len(intervention_years)
            else:
                lq = [(x / y) * 100 for x, y in zip(factor_data[1], deaths[1])]

            if (0 in factor_data[2]) or (0 in deaths[2]):
                uq = [0] * len(intervention_years)
            else:
                uq = [(x / y) * 100 for x, y in zip(factor_data[2], deaths[2])]

            factors_prop.update({factor: [mean, lq, uq]})

        return factors_prop

    death_causes = {k: get_factors_impacting_death(results_folders[k], factors, intervention_years) for k in
                    results_folders}

    for f in factors:
        fig, ax = plt.subplots()
        for k, colour in zip(death_causes, ['deepskyblue', 'olivedrab', 'darksalmon', 'darkviolet']):
            ax.plot(intervention_years, death_causes[k][f][0], label=k, color=colour)
            ax.fill_between(intervention_years, death_causes[k][f][1], death_causes[k][f][2], color=colour, alpha=.1)

        plt.ylabel('% of total deaths')
        plt.xlabel('Year')
        plt.title(f'Proportion of Total Deaths in which {f} Per Year by Scenario')
        plt.gca().set_ylim(bottom=0)
        plt.legend()
        plt.savefig(f'{plot_destination_folder}/{f}_factor_in_death.png')
        plt.show()

    x ='y'
