# todo: run the model and make plots of prevalence and incidence of each conditons wrt age/sex
#  make helpfer functions to make the plots for each condition and then loop through them all

import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file
from tlo.methods import (
    contraception,
    demography,
    dx_algorithm_child,
    enhanced_lifestyle,
    healthburden,
    healthseekingbehaviour,
    healthsystem,
    labour,
    ncds,
    pregnancy_supervisor,
    symptommanager,
)

# %%
resourcefilepath = Path("./resources")
outputpath = Path("./outputs")  # folder for convenience of storing outputs
datestamp = datetime.date.today().strftime("__%Y_%m_%d")

scenarios = dict()
scenarios['No_Treatment'] = []

# Create dict to capture the outputs
output_files = dict()

# %% Run the Simulation

start_date = Date(2010, 1, 1)
end_date = Date(2015, 1, 2)
popsize = 2000

for label, service_avail in scenarios.items():
    log_config = {'filename': 'LogFile'}
    # add file handler for the purpose of logging
    sim = Simulation(start_date=start_date, seed=0, log_config=log_config)

    # run the simulation
    sim.register(demography.Demography(resourcefilepath=resourcefilepath),
                 contraception.Contraception(resourcefilepath=resourcefilepath),
                 enhanced_lifestyle.Lifestyle(resourcefilepath=resourcefilepath),
                 healthsystem.HealthSystem(resourcefilepath=resourcefilepath, disable=True),
                 symptommanager.SymptomManager(resourcefilepath=resourcefilepath),
                 healthseekingbehaviour.HealthSeekingBehaviour(resourcefilepath=resourcefilepath),
                 healthburden.HealthBurden(resourcefilepath=resourcefilepath),
                 labour.Labour(resourcefilepath=resourcefilepath),
                 pregnancy_supervisor.PregnancySupervisor(resourcefilepath=resourcefilepath),
                 ncds.Ncds(resourcefilepath=resourcefilepath),
                 dx_algorithm_child.DxAlgorithmChild(resourcefilepath=resourcefilepath)
                 )

    sim.make_initial_population(n=popsize)
    sim.simulate(end_date=end_date)

    # Save the full set of results:
    output_files[label] = sim.log_filepath

# --------------------------------PLOTTING-------------------------------------------

# list all of the conditions in the module to loop through
conditions = ['nc_diabetes', 'nc_hypertension', 'nc_depression',
              'nc_chronic_lower_back_pain', 'nc_chronic_kidney_disease', 'nc_chronic_ischemic_hd', 'nc_cancers']

transform_output = lambda x: pd.concat([
    pd.Series(name=x.iloc[i]['date'], data=x.iloc[i]['data']) for i in range(len(x))
], axis=1, sort=False).transpose()

def restore_multi_index(dat):
    """restore the multi-index that had to be flattened to pass through the logger"""
    cols = dat.columns
    index_value_list = list()
    for col in cols.str.split('__'):
        index_value_list.append(tuple(component.split('=')[1] for component in col))
    index_name_list = tuple(component.split('=')[0] for component in cols[0].split('__'))
    dat.columns = pd.MultiIndex.from_tuples(index_value_list, names=index_name_list)
    return dat


# Plot prevalence by age and sex
def get_prevalence_from_logfile(logfile):
    output = parse_log_file(logfile)

    for condition in conditions:

        # give conditions readable names for plotting
        condition_name = condition.replace('nc_', '')
        condition_name = condition_name.replace('_', ' ')
        condition_name = condition_name.title()

        prevalence_by_age_and_sex = restore_multi_index(
            transform_output(
                output['tlo.methods.ncds'][f'{condition}_prevalence_by_age_and_sex']
            )
        )

        prevalence_by_age_and_sex.iloc[-1].transpose().plot.bar()
        plt.ylabel(f'Proportion With {condition_name}')
        plt.title(f'Prevalence of {condition_name} by Age')
        plt.savefig(outputpath / f'prevalence_{condition_name}_by_age.pdf')
        plt.show()


# %% Extract the relevant outputs and make a graph:
def get_incidence_rate_and_death_numbers_from_logfile(logfile):
    output = parse_log_file(logfile)

    # Calculate the "incidence rate" from the output counts of incidence
    counts = output['tlo.methods.ncds']['incidence_count_by_condition']
    counts['year'] = pd.to_datetime(counts['date']).dt.year
    counts.drop(columns='date', inplace=True)
    counts.set_index(
        'year',
        drop=True,
        inplace=True
    )

    # get person-years of time lived without condition
    # conditions hard-coded in for now
    conditions = ['nc_diabetes', 'nc_hypertension', 'nc_depression',
                  'nc_chronic_lower_back_pain', 'nc_chronic_kidney_disease', 'nc_chronic_ischemic_hd', 'nc_cancers']

    py_ = output['tlo.methods.demography']['person_years']

    # condition = 'nc_ldl_hdl'
    # py_ = output['tlo.methods.ncds'][f'person_years_{condition}']
    years = pd.to_datetime(py_['date']).dt.year
    py = pd.DataFrame(index=years, columns=['0-4', '5-9', '10-14', '15-19', '20-24', '25-29',
                                            '30-34', '35-39', '40-44', '45-49', '50-54',
                                            '55-59', '60-64', '65-69', '70-74', '75-79',
                                            '80-84', '85-89', '90-94', '95-99', '100+'])
    py_adults = pd.DataFrame(index=years, columns=['15+'])
    for year in years:
        tot_py = (
            (py_.loc[pd.to_datetime(py_['date']).dt.year == year]['M']).apply(pd.Series) +
            (py_.loc[pd.to_datetime(py_['date']).dt.year == year]['F']).apply(pd.Series)
        ).transpose()
        tot_py.index = tot_py.index.astype(int)
        py.loc[year, '0-4'] = tot_py.loc[0:4].sum().values[0]
        py.loc[year, '5-9'] = tot_py.loc[5:9].sum().values[0]
        py.loc[year, '10-14'] = tot_py.loc[10:14].sum().values[0]
        py.loc[year, '15-19'] = tot_py.loc[15:19].sum().values[0]
        py.loc[year, '20-24'] = tot_py.loc[20:24].sum().values[0]
        py.loc[year, '25-29'] = tot_py.loc[25:29].sum().values[0]
        py.loc[year, '30-34'] = tot_py.loc[30:34].sum().values[0]
        py.loc[year, '35-39'] = tot_py.loc[35:39].sum().values[0]
        py.loc[year, '40-44'] = tot_py.loc[40:44].sum().values[0]
        py.loc[year, '45-49'] = tot_py.loc[45:49].sum().values[0]
        py.loc[year, '50-54'] = tot_py.loc[50:54].sum().values[0]
        py.loc[year, '55-59'] = tot_py.loc[55:59].sum().values[0]
        py.loc[year, '60-64'] = tot_py.loc[60:64].sum().values[0]
        py.loc[year, '65-69'] = tot_py.loc[65:69].sum().values[0]
        py.loc[year, '70-74'] = tot_py.loc[70:74].sum().values[0]
        py.loc[year, '75-79'] = tot_py.loc[75:79].sum().values[0]
        py.loc[year, '80-84'] = tot_py.loc[80:84].sum().values[0]
        py.loc[year, '85-89'] = tot_py.loc[85:89].sum().values[0]
        py.loc[year, '90-94'] = tot_py.loc[90:94].sum().values[0]
        py.loc[year, '95-99'] = tot_py.loc[95:99].sum().values[0]
        py.loc[year, '100+'] = tot_py.loc[100:120].sum().values[0]

        py_adults.loc[year, '15+'] = tot_py.loc[15:120].sum().values[0]

    # Incidence rate among 0, 1, 2-4 year-olds
    inc_rate = dict()
    for age_grp in ['0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49',
                    '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80-84', '85-89', '90-94',
                    '95-99', '100+']:
        # inc_rate[age_grp][condition] = counts[age_grp].apply(pd.Series).div(f'py_{condition}'[age_grp], axis=0).dropna()
        inc_rate[age_grp] = counts[age_grp].apply(pd.Series).div(py[age_grp],
                                                                 axis=0).dropna()

    # Produce mean incidence rates of incidence rate during the simulation:
    inc_mean = pd.DataFrame()
    inc_mean['0-4y_model_output'] = inc_rate['0-4'].mean()
    inc_mean['5-9y_model_output'] = inc_rate['5-9'].mean()
    inc_mean['10-14y_model_output'] = inc_rate['10-14'].mean()
    inc_mean['15-19y_model_output'] = inc_rate['15-19'].mean()
    inc_mean['20-24y_model_output'] = inc_rate['20-24'].mean()
    inc_mean['25-29y_model_output'] = inc_rate['25-29'].mean()
    inc_mean['30-34y_model_output'] = inc_rate['30-34'].mean()
    inc_mean['35-39y_model_output'] = inc_rate['35-39'].mean()
    inc_mean['40-44y_model_output'] = inc_rate['40-44'].mean()
    inc_mean['45-49y_model_output'] = inc_rate['45-49'].mean()
    inc_mean['50-54y_model_output'] = inc_rate['50-54'].mean()
    inc_mean['55-59y_model_output'] = inc_rate['55-59'].mean()
    inc_mean['60-64y_model_output'] = inc_rate['60-64'].mean()
    inc_mean['65-69y_model_output'] = inc_rate['65-69'].mean()
    inc_mean['70-74y_model_output'] = inc_rate['70-74'].mean()
    inc_mean['75-79y_model_output'] = inc_rate['75-79'].mean()
    inc_mean['80-84y_model_output'] = inc_rate['80-84'].mean()
    inc_mean['85-89y_model_output'] = inc_rate['85-89'].mean()
    inc_mean['90-94y_model_output'] = inc_rate['90-94'].mean()
    inc_mean['95-99y_model_output'] = inc_rate['95-99'].mean()
    inc_mean['100+y_model_output'] = inc_rate['100+'].mean()

    return inc_mean


inc_by_condition = dict()
for label, file in output_files.items():
    inc_by_condition[label] = get_incidence_rate_and_death_numbers_from_logfile(file)


def plot_for_column_of_interest(results, column_of_interest):
    summary_table = dict()
    for label in results.keys():
        summary_table.update({label: results[label][column_of_interest]})
    data = 100 * pd.concat(summary_table, axis=1)
    data.plot.bar()
    plt.title(f'Incidence rate (/100 py): {column_of_interest}')
    plt.savefig(outputpath / ("NCDs_inc_rate_by_scenario" + datestamp + ".pdf"), format='pdf')
    plt.show()


# Plot incidence by condition

for column_of_interest in inc_by_condition[list(inc_by_condition.keys())[0]].columns:
    plot_for_column_of_interest(inc_by_condition, column_of_interest)
