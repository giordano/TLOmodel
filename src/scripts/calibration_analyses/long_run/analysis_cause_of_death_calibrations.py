"""
Produce comparisons between model and GBD of deaths by cause in a particular period.

This uses the results of the Scenario defined in: src/scripts/long_run/long_run.py
"""

# todo - do all the same for DALYS
# todo - change the demography analysis to avoid the issue of summarizeing and then doing groupby and leading to lower --> 0's
# todo - use GBD all ages numbers for some outputs (correct uncertainity bounds)

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from tlo.analysis.utils import (
    make_age_grp_types,
    make_age_grp_lookup,
    make_calendar_period_lookup,
    make_calendar_period_type,
    extract_params,
    extract_results,
    get_scenario_info,
    get_scenario_outputs,
    load_pickled_dataframes,
    summarize,
    format_gbd,
    create_pickles_locally
)

# %% Declare usual paths:
outputspath = Path('./outputs/tbh03@ic.ac.uk')
rfp = Path('./resources')

# ** Declare the results folder ***
results_folder = get_scenario_outputs('long_run.py', outputspath)[-1]

# If needed -- in the case that pickles were not created locally
# create_pickles_locally(results_folder)

# Declare path for output graphs from this script
make_graph_file_name = lambda stub: results_folder / f"{stub}.png"

# Define colo(u)rs to use:
colors = {
    'Model': 'royalblue',
    'Census': 'darkred',
    'WPP': 'forestgreen',
    'GBD': 'plum'
}

# %% Set the period for the analysis (comparison is made of the average annual number of deaths in this period)
period = '2010-2014'

# %% Load and process the GBD data
gbd = format_gbd(pd.read_csv(rfp / 'gbd' / 'ResourceFile_Deaths_And_DALYS_GBD2019.csv'))

# update columns name
gbd = gbd.rename(columns={
    'Sex': 'sex',
    'Age_Grp': 'age_grp',
    'Period': 'period',
    'GBD_Est': 'mean',
    'GBD_Lower': 'lower',
    'GBD_Upper': 'upper'})

# limit to deaths:
gbd = gbd.loc[gbd['measure_name'] == 'Deaths']

# %% Load modelling results:

# Extract results, summing by sex, year, age & label
results = extract_results(
    results_folder,
    module="tlo.methods.demography",
    key="death",
    custom_generate_series="assign(year = lambda x: x['date'].dt.year)"
                           ".groupby(['sex', 'year', 'age', 'label'])['person_id'].count()",
    do_scaling=True
)

# Update index to give results by five-year age-group and five-year calendar period
agegrps, agegrplookup = make_age_grp_lookup()
calperiods, calperiodlookup = make_calendar_period_lookup()
results = results.reset_index()
results['age_grp'] = results['age'].map(agegrplookup).astype(make_age_grp_types())
results['period'] = results['year'].map(calperiodlookup).astype(make_calendar_period_type())
results = results.drop(columns=['age', 'year'])

# groupby, sum and divide by five to give the average number of deaths per year within the five year period:
results = results.groupby(['period', 'sex', 'age_grp', 'label']).sum().div(5.0)


# %% Load the cause-of-deaths mappers and use them to populate the 'label' for gbd outputs
demoglog = load_pickled_dataframes(results_folder)['tlo.methods.demography']
mapper_from_gbd_causes = pd.Series(demoglog['mapper_from_gbd_cause_to_common_label'].drop(columns={'date'}).loc[0]
                                   ).to_dict()
gbd['label'] = gbd['cause_name'].map(mapper_from_gbd_causes)
assert not gbd['label'].isna().any()

# %% Make comparable pivot-tables of the GBD and Model Outputs:
# Summarize results for average number of deaths (per unified cause) per year within five-year periods and five-year
# age-groups. (index=sex/age, columns=unified_cause). (Fr the particular period specified.)

deaths_by_age_pt = dict()

# - GBD:
deaths_by_age_pt['GBD'] = gbd.loc[gbd.period == period].groupby(
    ['sex', 'age_grp', 'label'])[['mean', 'lower', 'upper']].sum().unstack().div(5.0)
# NB. division by 5.0 to make it the average number of death per year within the five-year period.

# - TLO Model:
deaths_by_age_pt['Model'] = summarize(results, collapse_columns=True).reset_index().loc[lambda x: (x.period == period)].groupby(
    by=['sex', 'age_grp', 'label']
)[['mean', 'lower', 'upper']].sum().unstack(fill_value=0.0)


# %% Make figures of overall summaries of deaths by cause
# todo - improve formatting of this one

dats = ['GBD', 'Model']
sexes = ['F', 'M']
sexname = lambda x: 'Females' if x=='F' else 'Males'

fig, axes = plt.subplots(ncols=2, nrows=2, sharey=True, sharex=True, figsize=(40, 40))

for col, sex in enumerate(sexes):
    for row, dat in enumerate(dats):

        ax = axes[row][col]
        df = deaths_by_age_pt[dat].loc[sex].loc[:, pd.IndexSlice['mean']] / 1e3

        xs = np.arange(len(df.index))
        df.plot.bar(stacked=True, ax=ax, fontsize=30)
        ax.set_xlabel('Age Group', fontsize=40)
        ax.set_title(f"{sexname(sex)}: {dat}", fontsize=60)
        ax.get_legend().remove()

# add a big axis, hide frame
bigax = fig.add_subplot(111, frameon=False)

# hide tick and tick label of the "big axis"
bigax.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
bigax.set_ylabel("Deaths per year (thousands)", fontsize=40)

fig.legend(loc="center right", fontsize=15)
fig.tight_layout()
plt.savefig(make_graph_file_name(f"Deaths_StackedBars_ModelvsGBD_{period}"))
plt.show()


# %% Plots of age-breakdown deaths patten for each cause:

sexes = ['F', 'M']
dats = ['GBD', 'Model']

all_causes = list(results.index.levels[3])
reformat_cause = lambda x: x.replace(' / ', '_')

for cause in all_causes:
    try:
        deaths_this_cause = pd.concat(
            {dat: deaths_by_age_pt[dat].loc[:, (slice(None), cause)] for dat in deaths_by_age_pt.keys()}, axis=1
        ).fillna(0.0) / 1e3

        x = list(deaths_this_cause.index.levels[1])
        xs = np.arange(len(x))

        fig, ax = plt.subplots(ncols=1, nrows=2, sharey=True, sharex=True)
        for row, sex in enumerate(sexes):
            for dat in dats:
                ax[row].plot(
                    xs,
                    deaths_this_cause.loc[(sex,),(dat, 'mean', cause)].values,
                    label=dat,
                    color=colors[dat]
                )
                ax[row].fill_between(
                    xs,
                    deaths_this_cause.loc[(sex,),(dat, 'upper', cause)].values,
                    deaths_this_cause.loc[(sex,),(dat, 'lower', cause)].values,
                    facecolor=colors[dat], alpha=0.2
                )
            ax[row].legend()
            ax[row].set_xticks(xs)
            ax[row].set_xticklabels(x, rotation=90)
            ax[row].set_xlabel('Age Group')
            ax[row].set_ylabel('Deaths per year (thousands)')
            ax[row].set_title(f"{cause}: {sexname(sex)}, {period}")
            ax[row].legend()

        fig.tight_layout()
        plt.savefig(make_graph_file_name(f"Deaths_Scatter_Plot_{period}_{reformat_cause(cause)}"))
        plt.show()

    except KeyError:
        print(f"Could not produce plot for {reformat_cause(cause)}")


# %% Plots comparing between model and actual across all ages and sex:

# - TLO Model:
tot_deaths_by_cause = pd.concat({
    'Model': summarize(results.groupby(by=['label']).sum(), collapse_columns=True).unstack(),
    'GBD': gbd.loc[gbd.Period == period].groupby(['label']).sum()[['mean', 'lower', 'upper']].unstack()
}, axis=1)
#todo - for GBD, instead use all ages and all sex numbers to get correct uncertainity bounds

select_labels = ['AIDS', 'Childhood Diarrhoea', 'Other']

fig, ax = plt.subplots()
xylim = 450
for cause in tot_deaths_by_cause.index.levels[1]:

    vals = tot_deaths_by_cause.loc[(slice(None), cause),] / 1e3

    x = vals.at[('mean', cause), 'GBD']
    xerr = np.array([
               x - vals.at[('lower', cause), 'GBD'],
               vals.at[('upper', cause), 'GBD'] - x
    ]).reshape(2, 1)
    y = vals.at[('mean', cause), 'Model']
    yerr = np.array([
        y - vals.at[('lower', cause), 'Model'],
        vals.at[('upper', cause), 'Model'] - y
    ]).reshape(2,1)

    ax.errorbar(x=x, y=y, xerr=xerr, yerr=yerr, label=cause)

    # add labels to selected points
    if cause in select_labels:
        ax.annotate(cause,
                (x,y),
                textcoords="offset points",
                xytext=(0, 10),
                ha='center'
                )

line_x = np.linspace(0, xylim)
ax.plot(line_x, line_x, 'r')
ax.set(xlim=(0, xylim), ylim=(0, xylim))
ax.set_xlabel('GBD (thousands)')
ax.set_ylabel('Model (thousands)')
ax.set_title(f'Deaths per year by Cause {period}')
plt.savefig(make_graph_file_name(f"Deaths_Scatter_Plot_{period}"))
plt.show()
