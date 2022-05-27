from matplotlib import pyplot as plt
import numpy as np
from tlo.analysis.utils import extract_results
plt.style.use('seaborn-darkgrid')


# =========================================== FUNCTIONS TO EXTRACT RATES  ============================================
def get_mean_and_quants_from_str_df(df, complication, sim_years):
    yearly_mean_number = list()
    yearly_lq = list()
    yearly_uq = list()
    for year in sim_years:
        if complication in df.loc[year].index:
            yearly_mean_number.append(df.loc[year, complication].mean())
            yearly_lq.append(df.loc[year, complication].quantile(0.025))
            yearly_uq.append(df.loc[year, complication].quantile(0.925))
        else:
            yearly_mean_number.append(0)
            yearly_lq.append(0)
            yearly_uq.append(0)

    return [yearly_mean_number, yearly_lq, yearly_uq]


def get_comp_mean_and_rate(complication, denominator_list, df, rate, years):
    yearly_means = get_mean_and_quants_from_str_df(df, complication, years)[0]
    yearly_lq = get_mean_and_quants_from_str_df(df, complication, years)[1]
    yearly_uq = get_mean_and_quants_from_str_df(df, complication, years)[2]

    yearly_mean_rate = [(x / y) * rate for x, y in zip(yearly_means, denominator_list)]
    yearly_lq_rate = [(x / y) * rate for x, y in zip(yearly_lq, denominator_list)]
    yearly_uq_rate = [(x / y) * rate for x, y in zip(yearly_uq, denominator_list)]

    return [yearly_mean_rate, yearly_lq_rate, yearly_uq_rate]


def get_mean_and_quants(df, sim_years):
    year_means = list()
    lower_quantiles = list()
    upper_quantiles = list()

    for year in sim_years:
        if year in df.index:
            year_means.append(df.loc[year].mean())
            lower_quantiles.append(df.loc[year].quantile(0.025))
            upper_quantiles.append(df.loc[year].quantile(0.925))
        else:
            year_means.append(0)
            lower_quantiles.append(0)
            lower_quantiles.append(0)

    return [year_means, lower_quantiles, upper_quantiles]


def get_comp_mean_and_rate_across_multiple_dataframes(complication, denominators, rate, dataframes, sim_years):
    def get_list_of_rates_and_quants(df):
        rates_per_year = list()
        lq_per_year = list()
        uq_per_year = list()
        for year, denominator in zip(sim_years, denominators):
            if year in df.index:
                if complication in df.loc[year].index:
                    rates = (df.loc[year, complication].mean() / denominator) * rate
                    lq = (df.loc[year, complication].quantile(0.025) / denominator) * rate
                    uq = (df.loc[year, complication].quantile(0.925) / denominator) * rate
                    rates_per_year.append(rates)
                    lq_per_year.append(lq)
                    uq_per_year.append(uq)

                else:
                    rates_per_year.append(0)
                    lq_per_year.append(0)
                    uq_per_year.append(0)
            else:
                rates_per_year.append(0)
                lq_per_year.append(0)
                uq_per_year.append(0)

        return [rates_per_year, lq_per_year, uq_per_year]

    if len(dataframes) == 2:
        df_1_data = get_list_of_rates_and_quants(dataframes[0])
        df_2_data = get_list_of_rates_and_quants(dataframes[1])

        total_rates = [x + y for x, y in zip(df_1_data[0], df_2_data[0])]
        total_lq = [x + y for x, y in zip(df_1_data[1], df_2_data[1])]
        total_uq = [x + y for x, y in zip(df_1_data[2], df_2_data[2])]

    else:
        df_1_data = get_list_of_rates_and_quants(dataframes[0])
        df_2_data = get_list_of_rates_and_quants(dataframes[1])
        df_3_data = get_list_of_rates_and_quants(dataframes[2])

        total_rates = [x + y + z for x, y, z in zip(df_1_data[0], df_2_data[0], df_3_data[0])]
        total_lq = [x + y + z for x, y, z in zip(df_1_data[1], df_2_data[1], df_3_data[1])]
        total_uq = [x + y + z for x, y, z in zip(df_1_data[2], df_2_data[2], df_3_data[2])]

    return [total_rates, total_lq, total_uq]


# =========================================== FUNCTIONS TO PRODUCE PLOTS  ============================================
def basic_comparison_graph(intervention_years, bdata, idata, y_label, title, graph_location, save_name):
    fig, ax = plt.subplots()
    ax.plot(intervention_years, bdata[0], label="Baseline (mean)", color='deepskyblue')
    ax.fill_between(intervention_years, bdata[1], bdata[2], color='b', alpha=.1, label="UI (2.5-92.5)")
    ax.plot(intervention_years, idata[0], label="Intervention (mean)", color='olivedrab')
    ax.fill_between(intervention_years, idata[1], idata[2], color='g', alpha=.1, label="UI (2.5-92.5)")
    plt.ylabel(y_label)
    plt.xlabel('Year')
    plt.title(title)
    plt.gca().set_ylim(bottom=0)
    plt.legend()
    plt.savefig(f'./{graph_location}/{save_name}.png')
    plt.show()


def simple_line_chart(sim_years, model_rate, y_title, title, file_name, graph_location):
    plt.plot(sim_years, model_rate, 'o-g', label="Model", color='deepskyblue')
    plt.xlabel('Year')
    plt.ylabel(y_title)
    plt.title(title)
    plt.legend()
    plt.savefig(f'./outputs/sejjj49@ucl.ac.uk/{graph_location}/{file_name}.png')
    plt.show()


def simple_line_chart_with_ci(sim_years, data, y_label, title, file_name, graph_location):
    fig, ax = plt.subplots()
    ax.plot(sim_years, data[0], label="Model (mean)", color='deepskyblue')
    ax.fill_between(sim_years, data[1], data[2], color='b', alpha=.1, label="UI (2.5-92.5)")
    plt.ylabel(y_label)
    plt.xlabel('Year')
    plt.title(title)
    plt.legend()
    plt.savefig(f'{graph_location}/{file_name}.png')
    plt.show()


def simple_bar_chart(model_rates, x_title, y_title, title, file_name, sim_years, graph_location):
    bars = sim_years
    x_pos = np.arange(len(bars))
    plt.bar(x_pos, model_rates, label="Model", color='thistle')
    plt.xticks(x_pos, bars, rotation=90)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.title(title)
    plt.legend()
    plt.savefig(f'{graph_location}/{file_name}.png')
    plt.show()


def return_squeeze_plots_for_hsi(folder, hsi_string, sim_years, graph_location):

    hsi = extract_results(
        folder,
        module="tlo.methods.healthsystem",
        key="HSI_Event",
        custom_generate_series=(
            lambda df: df.loc[df['TREATMENT_ID'].str.contains(hsi_string) & df['did_run']].assign(
                year=df['date'].dt.year).groupby(['year'])['Squeeze_Factor'].mean()))

    mean_squeeze_per_year = [hsi.loc[year].to_numpy().mean() for year in sim_years]
    lq_squeeze_per_year = [np.percentile(hsi.loc[year].to_numpy(), 2.5) for year in sim_years]
    uq_squeeze_per_year = [np.percentile(hsi.loc[year].to_numpy(), 92.5) for year in sim_years]
    mean_data = [mean_squeeze_per_year, lq_squeeze_per_year, uq_squeeze_per_year]

    hsi_med = extract_results(
        folder,
        module="tlo.methods.healthsystem",
        key="HSI_Event",
        custom_generate_series=(
            lambda df: df.loc[df['TREATMENT_ID'].str.contains(hsi_string) & df['did_run']].assign(
                year=df['date'].dt.year).groupby(['year'])['Squeeze_Factor'].median()))

    median = [hsi_med.loc[year].median() for year in sim_years]

    hsi_count = extract_results(
        folder,
        module="tlo.methods.healthsystem",
        key="HSI_Event",
        custom_generate_series=(
            lambda df: df.loc[df['TREATMENT_ID'].str.contains(hsi_string) & df['did_run']].assign(
                year=df['date'].dt.year).groupby(['year'])['year'].count()))

    hsi_squeeze = extract_results(
        folder,
        module="tlo.methods.healthsystem",
        key="HSI_Event",
        custom_generate_series=(
            lambda df:
            df.loc[(df['TREATMENT_ID'].str.contains(hsi_string)) & df['did_run'] & (df['Squeeze_Factor'] > 0)
                   ].assign(year=df['date'].dt.year).groupby(['year'])['year'].count()))

    prop_squeeze_year = [(hsi_squeeze.loc[year].to_numpy().mean() / hsi_count.loc[year].to_numpy().mean()) * 100
                         for year in sim_years]
    prop_squeeze_lq = [
        (np.percentile(hsi_squeeze.loc[year].to_numpy(), 2.5) /
         np.percentile(hsi_count.loc[year].to_numpy(), 2.5)) * 100 for year in sim_years]

    prop_squeeze_uq = [
        (np.percentile(hsi_squeeze.loc[year].to_numpy(), 92.5) /
         np.percentile(hsi_count.loc[year].to_numpy(), 92.5)) * 100 for year in sim_years]

    prop_data = [prop_squeeze_year, prop_squeeze_lq, prop_squeeze_uq]

    simple_line_chart_with_ci(sim_years, mean_data, 'Mean Squeeze Factor', f'Mean Yearly Squeeze for HSI {hsi_string}',
                              f'mean_sf_{hsi_string}', graph_location)
    simple_line_chart(sim_years, median, 'Median Squeeze Factor', f'Median Yearly Squeeze for HSI {hsi_string}',
                      f'med_sf_{hsi_string}', graph_location)
    simple_line_chart_with_ci(sim_years, prop_data, '% HSIs', f'Proportion of HSI {hsi_string} where squeeze > 0',
                              f'prop_sf_{hsi_string}', graph_location)


def comparison_graph_multiple_scenarios(intervention_years, data_dict, y_label, title, graph_location, save_name):
    fig, ax = plt.subplots()

    for k, colour in zip(data_dict, ['deepskyblue', 'olivedrab', 'darksalmon']):
        ax.plot(intervention_years, data_dict[k][0], label=k, color=colour)
        ax.fill_between(intervention_years, data_dict[k][1], data_dict[k][2], color=colour, alpha=.1)

    plt.ylabel(y_label)
    plt.xlabel('Year')
    plt.title(title)
    plt.gca().set_ylim(bottom=0)
    #plt.style.use('seaborn-darkgrid')
    plt.legend()
    plt.savefig(f'./{graph_location}/{save_name}.png')
    plt.show()


def comparison_graph_multiple_scenarios_multi_level_dict(
    # todo combine with above
    intervention_years, data_dict, key, y_label, title, graph_location, save_name):
    fig, ax = plt.subplots()

    for k, colour in zip(data_dict, ['deepskyblue', 'olivedrab', 'darksalmon', 'darkviolet']):
        ax.plot(intervention_years, data_dict[k][key][0], label=k, color=colour)
        ax.fill_between(intervention_years, data_dict[k][key][1], data_dict[k][key][2], color=colour, alpha=.1)

    plt.ylabel(y_label)
    plt.xlabel('Year')
    plt.title(title)
    plt.gca().set_ylim(bottom=0)
    #plt.style.use('seaborn-darkgrid')
    plt.legend()
    plt.savefig(f'./{graph_location}/{save_name}.png')
    plt.show()


def comparison_bar_chart_multiple_bars(data, dict_name, intervention_years, y_title, title,
                                       plot_destination_folder, save_name):

    N = len(intervention_years)  # todo: will crash if change the baseline name
    ind = np.arange(N)
    width = 0.2

    for k, position, colour in zip(data, [ind - width, ind, ind + width, ind + width * 2],
                                   ['bisque', 'powderblue', 'mistyrose', 'thistle']):
        ci = [(x - y) / 2 for x, y in zip(data[k][dict_name][2], data[k][dict_name][1])]
        plt.bar(position, data[k][dict_name][0], width, label=k, yerr=ci, color=colour)

    plt.ylabel(y_title)
    plt.xlabel('Years')
    plt.title(title)
    plt.legend(loc='best')
    plt.xticks([0., 1., 2., 3., 4., 5., 6., 7., 8.], labels=intervention_years)  # todo: has the be editied with number of years
    plt.savefig(f'{plot_destination_folder}/{save_name}.png')
    plt.show()


# =========================== FUNCTIONS RETURNING DATA FROM MULTIPLE SCENARIOS =======================================
def return_birth_data_from_multiple_scenarios(results_folders, intervention_years):
        """
        Extract mean, lower and upper quantile births per year for a given scenario
        :param folder: results folder for scenario
        :return: list of total births per year of pre defined intervention period (i.e. 2020-2030)
        """

        def extract_births(folder):
            births_results = extract_results(
                folder,
                module="tlo.methods.demography",
                key="on_birth",
                custom_generate_series=(
                    lambda df: df.assign(year=df['date'].dt.year).groupby(['year'])['year'].count()),
                do_scaling=True
            )
            total_births_per_year = get_mean_and_quants(births_results, intervention_years)
            return total_births_per_year

        return {k: extract_births(results_folders[k]) for k in results_folders}


def return_pregnancy_data_from_multiple_scenarios(results_folders, intervention_years):
    """
    """

    def extract_pregnancies(folder):
        preg_results = extract_results(
            folder,
            module="tlo.methods.contraception",
            key="pregnancy",
            custom_generate_series=(
                lambda df: df.assign(year=df['date'].dt.year).groupby(['year'])['year'].count()),
            do_scaling=True
        )
        total_pregnancies_per_year = get_mean_and_quants(preg_results, intervention_years)
        return total_pregnancies_per_year

    return {k: extract_pregnancies(results_folders[k]) for k in results_folders}


def return_death_data_from_multiple_scenarios(results_folders, births_dict, intervention_years, detailed_log):
    """
    Extract mean, lower and upper quantile maternal mortality ratio, neonatal mortality ratio, crude maternal
    deaths and crude neonatal deaths per year for a given scenario
    :param folder: results folder for scenario
    :param births: list. mean number of births per year for a scenario (used as a denominator)
    :return: dict containing mean, LQ, UQ for MMR, NMR, maternal deaths and neonatal deaths
    """

    def extract_deaths(folder, births):
        death_results_labels = extract_results(
            folder,
            module="tlo.methods.demography",
            key="death",
            custom_generate_series=(
                lambda df: df.assign(year=df['date'].dt.year).groupby(['year', 'label'])['year'].count()),
            do_scaling=True)

        # TODO: if not using detailed logging we are only capturing indirect deaths during pregnancy and not postnatally
        other_preg_deaths = extract_results(
            folder,
            module="tlo.methods.demography",
            key="death",
            custom_generate_series=(
                lambda df: df.assign(year=df['date'].dt.year).groupby(['year', 'label', 'pregnancy'])['year'].count()),
            do_scaling=True
        )

        # Extract maternal mortality ratio from direct maternal causes
        mmr = get_comp_mean_and_rate('Maternal Disorders', births, death_results_labels, 100000, intervention_years)

        # Extract crude deaths due to direct maternal disorders
        crude_m_deaths = get_mean_and_quants_from_str_df(death_results_labels, 'Maternal Disorders', intervention_years)

        if detailed_log:
            indirect_deaths = extract_results(
                folder,
                module="tlo.methods.demography.detail",
                key="properties_of_deceased_persons",
                custom_generate_series=(
                    lambda df: df.loc[(df['is_pregnant'] | df['la_is_postpartum']) &
                                      df['cause_of_death'].str.contains(
                                          'AIDS|Malaria|TB|Suicide|ever_stroke|diabetes|chronic_ischemic_hd|'
                                          'ever_heart_attack|ever_stroke|chronic_kidney_disease')].assign(
                        year=df['date'].dt.year).groupby(['year'])['year'].count()),
                do_scaling=True
            )

            indirect_deaths = get_mean_and_quants(indirect_deaths, intervention_years)

        else:
            # Extract crude deaths due to indirect causes in pregnant women
            indirect_causes = ['AIDS', 'Malaria', 'TB', 'Suicide', 'Stroke', 'Depression / Self-harm', 'Heart Disease',
                               'Kidney Disease']

            indirect_deaths = list()
            id_lq = list()
            id_uq = list()

            for year in intervention_years:
                id_deaths_per_year = 0
                id_lq_py = 0
                id_uq_pu = 0
                for cause in indirect_causes:
                    if cause in other_preg_deaths.loc[year, :, True].index:
                        id_deaths_per_year += other_preg_deaths.loc[year, cause, True].mean()
                        id_lq_py += other_preg_deaths.loc[year, cause, True].quantile(0.025)
                        id_uq_pu += other_preg_deaths.loc[year, cause, True].quantile(0.925)

                indirect_deaths.append(id_deaths_per_year)
                id_lq.append(id_lq_py)
                id_uq.append(id_uq_pu)

            indirect_deaths = [indirect_deaths, id_lq, id_uq]

        # Calculate total MMR (direct + indirect deaths)
        total_mmr = [[((x + y) / z) * 100000 for x, y, z in zip(indirect_deaths[0], crude_m_deaths[0], births)],
                     [((x + y) / z) * 100000 for x, y, z in zip(indirect_deaths[1], crude_m_deaths[1], births)],
                     [((x + y) / z) * 100000 for x, y, z in zip(indirect_deaths[2], crude_m_deaths[2], births)]
                     ]

        # Extract NMR
        nmr = get_comp_mean_and_rate('Neonatal Disorders', births, death_results_labels, 1000, intervention_years)

        # And crude neonatal deaths
        crude_n_deaths = get_mean_and_quants_from_str_df(death_results_labels, 'Neonatal Disorders', intervention_years)

        return {'direct_mmr': mmr,
                'total_mmr': total_mmr,
                'nmr': nmr,
                'crude_m_deaths': crude_m_deaths, # TODO: THIS EXLUDES INDIRECT CRUDE DEATHS....
                'crude_n_deaths': crude_n_deaths}

    # Extract data from scenarios
    return {k: extract_deaths(results_folders[k], births_dict[k]) for k in results_folders}


def return_stillbirth_data_from_multiple_scenarios(results_folders, births_dict, intervention_years):
    """
    Extract antenatal and intrapartum stillbirths from a scenario and return crude numbers and stillbirth rate per
    year
    :param folder: results folder for scenario
    :param births: list. mean number of births per year for a scenario (used as a denominator)
    """

    # TODO: should we report total SBR even though only ISBR will really have been effected?

    def extract_stillbirths(folder, births):
        an_stillbirth_results = extract_results(
            folder,
            module="tlo.methods.pregnancy_supervisor",
            key="antenatal_stillbirth",
            custom_generate_series=(
                lambda df: df.assign(year=df['date'].dt.year).groupby(['year'])['year'].count()),
            do_scaling=True
        )
        ip_stillbirth_results = extract_results(
            folder,
            module="tlo.methods.labour",
            key="intrapartum_stillbirth",
            custom_generate_series=(
                lambda df: df.assign(year=df['date'].dt.year).groupby(['year'])['year'].count()),
            do_scaling=True
        )

        # Get stillbirths
        an_still_birth_data = get_mean_and_quants(an_stillbirth_results, intervention_years)
        ip_still_birth_data = get_mean_and_quants(ip_stillbirth_results, intervention_years)

        # Store mean number of stillbirths, LQ, UQ
        crude_sb = [[x + y for x, y in zip(an_still_birth_data[0], ip_still_birth_data[0])],
                    [x + y for x, y in zip(an_still_birth_data[1], ip_still_birth_data[1])],
                    [x + y for x, y in zip(an_still_birth_data[2], ip_still_birth_data[2])]]

        # Then generate SBR
        an_sbr = [[(x / y) * 1000 for x, y in zip(an_still_birth_data[0], births)],
                  [(x / y) * 1000 for x, y in zip(an_still_birth_data[1], births)],
                  [(x / y) * 1000 for x, y in zip(an_still_birth_data[2], births)]]

        ip_sbr = [[(x / y) * 1000 for x, y in zip(ip_still_birth_data[0], births)],
                  [(x / y) * 1000 for x, y in zip(ip_still_birth_data[1], births)],
                  [(x / y) * 1000 for x, y in zip(ip_still_birth_data[2], births)]]

        total_sbr = [[((x + y) / z) * 1000 for x, y, z in zip(an_still_birth_data[0], ip_still_birth_data[0], births)],
                     [((x + y) / z) * 1000 for x, y, z in zip(an_still_birth_data[1], ip_still_birth_data[1], births)],
                     [((x + y) / z) * 1000 for x, y, z in zip(an_still_birth_data[2], ip_still_birth_data[2], births)]]

        # Return as dict for graphs
        return {'an_sbr': an_sbr,
                'ip_sbr': ip_sbr,
                'sbr': total_sbr,
                'crude_sb': crude_sb}

    return {k: extract_stillbirths(results_folders[k], births_dict[k]) for k in results_folders}


def return_dalys_from_multiple_scenarios(results_folders, intervention_years):

    def get_dalys_from_scenario(results_folder, intervention_years):
        """
        Extracted stacked DALYs from logger for maternal and neonatal disorders
        :param results_folder: results folder for scenario
        :return: Maternal and neonatal dalys [Mean, LQ, UQ]
        """

        # Get DALY df
        dalys_stacked = extract_results(
            results_folder,
            module="tlo.methods.healthburden",
            key="dalys_stacked",
            custom_generate_series=(
                lambda df: df.drop(
                    columns='date').groupby(['year']).sum().stack()),
            do_scaling=True)

        # todo - should this just be at risk or total (gbd suggests total, which this calibrates well with)
        person_years_total = extract_results(
            results_folder,
            module="tlo.methods.demography",
            key="person_years",
            custom_generate_series=(
                lambda df: df.assign(total=(df['M'].apply(lambda x: sum(x.values()))) +
                                            df['F'].apply(lambda x: sum(x.values()))).assign(
                    year=df['date'].dt.year).groupby(['year'])['total'].sum()),
            do_scaling=True)

        denom = get_mean_and_quants(person_years_total, intervention_years)

        dalys_mat = get_comp_mean_and_rate('Maternal Disorders', denom[0], dalys_stacked, 100000, intervention_years)
        dalys_neo = get_comp_mean_and_rate('Neonatal Disorders', denom[0], dalys_stacked, 100000, intervention_years)

        mat_causes_death = ['ectopic_pregnancy', 'spontaneous_abortion', 'induced_abortion',
                            'severe_gestational_hypertension', 'severe_pre_eclampsia', 'eclampsia', 'antenatal_sepsis',
                            'uterine_rupture', 'intrapartum_sepsis', 'postpartum_sepsis', 'postpartum_haemorrhage',
                            'secondary_postpartum_haemorrhage', 'antepartum_haemorrhage']

        mat_causes_disab = ['maternal']

        neo_causes_death = ['early_onset_sepsis', 'late_onset_sepsis', 'encephalopathy', 'preterm_other',
                            'respiratory_distress_syndrome', 'neonatal_respiratory_depression']

        neo_causes_disab = ['Retinopathy of Prematurity', 'Neonatal Encephalopathy',
                            'Neonatal Sepsis Long term Disability', 'Preterm Birth Disability']

        yll_stacked = extract_results(
            results_folder,
            module="tlo.methods.healthburden",
            key="yll_by_causes_of_death_stacked",
            custom_generate_series=(
                lambda df: df.drop(
                    columns='date').groupby(['year']).sum().stack()),
            do_scaling=True)

        yld = extract_results(
            results_folder,
            module="tlo.methods.healthburden",
            key="yld_by_causes_of_disability",
            custom_generate_series=(
                lambda df: df.drop(
                    columns='date').groupby(['year']).sum().stack()),
            do_scaling=True)

        def get_output(causes, df):
            mean = list()
            lq = list()
            uq = list()

            for year in intervention_years:
                per_year = 0
                per_year_lq = 0
                per_year_uq = 0
                for cause in causes:
                    if cause in df.loc[year].index:
                        per_year += df.loc[year, cause].mean()
                        per_year_lq += df.loc[year, cause].quantile(0.025)
                        per_year_uq += df.loc[year, cause].quantile(0.925)

                mean.append(per_year)
                lq.append(per_year_lq)
                uq.append(per_year_uq)

            return [mean, lq, uq]

        def get_as_rate(values):
            mean = [(x / y) * 100000 for x, y in zip(values[0], denom[0])]
            lq = [(x / y) * 100000 for x, y in zip(values[1], denom[1])]
            uq = [(x / y) * 100000 for x, y in zip(values[2], denom[2])]

            return [mean, lq, uq]

        mat_yll = get_output(mat_causes_death, yll_stacked)
        mat_yll_rate = get_as_rate(mat_yll)
        mat_yld = get_output(mat_causes_disab, yld)
        mat_yld_rate = get_as_rate(mat_yld)
        neo_yll = get_output(neo_causes_death, yll_stacked)
        neo_yll_rate = get_as_rate(neo_yll)
        neo_yld = get_output(neo_causes_disab, yld)
        neo_yld_rate = get_as_rate(neo_yld)

        def extract_dalys_tlo_model(group):
            """Extract mean, LQ, UQ DALYs for maternal or neonatal disorders"""

            stacked_dalys = [dalys_stacked.loc[year, f'{group} Disorders'].mean() for year in
                             intervention_years if year in intervention_years]

            stacked_dalys_lq = [dalys_stacked.loc[year, f'{group} Disorders'].quantile(0.025) for year in
                                intervention_years if year in intervention_years]

            stacked_dalys_uq = [dalys_stacked.loc[year, f'{group} Disorders'].quantile(0.925) for year in
                                intervention_years if year in intervention_years]

            return [stacked_dalys, stacked_dalys_lq, stacked_dalys_uq]

        return {'maternal_dalys_crude': extract_dalys_tlo_model('Maternal'),
                'maternal_dalys_rate': dalys_mat,
                'maternal_yll_crude': mat_yll,
                'maternal_yll_rate': mat_yll_rate,
                'maternal_yld_crude': mat_yld,
                'maternal_yld_rate': mat_yld_rate,
                'neonatal_dalys_crude': extract_dalys_tlo_model('Neonatal'),
                'neonatal_dalys_rate': dalys_neo,
                'neonatal_yll_crude': neo_yll,
                'neonatal_yll_rate': neo_yll_rate,
                'neonatal_yld_crude': neo_yld,
                'neonatal_yld_rate': neo_yld_rate}

    # Store DALYs data for baseline and intervention
    return {k: get_dalys_from_scenario(results_folders[k], intervention_years) for k in results_folders}

