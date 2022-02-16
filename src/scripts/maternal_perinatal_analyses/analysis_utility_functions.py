from matplotlib import pyplot as plt

# ==================================================== UTILITY CODE ===================================================
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


def basic_comparison_graph(intervention_years, bdata, idata, y_label, title, graph_location, save_name):
    fig, ax = plt.subplots()
    ax.plot(intervention_years, bdata[0], label="Baseline (mean)", color='deepskyblue')
    ax.fill_between(intervention_years, bdata[1], bdata[2], color='b', alpha=.1)
    ax.plot(intervention_years, idata[0], label="Intervention (mean)", color='olivedrab')
    ax.fill_between(intervention_years, idata[1], idata[2], color='g', alpha=.1)
    plt.ylabel(y_label)
    plt.xlabel('Year')
    plt.title(title)
    plt.legend()
    plt.savefig(f'./outputs/sejjj49@ucl.ac.uk/{graph_location}/{save_name}.png')
    plt.show()

