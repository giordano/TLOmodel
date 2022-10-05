# %% Import Statements
import datetime
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import ticker

from tlo import Date, Simulation, logging
from tlo.analysis.utils import parse_log_file, unflatten_flattened_multi_index_in_logging
from tlo.methods import demography, enhanced_lifestyle, simplified_births


class LifeStylePlots:
    """ a class for for plotting lifestyle properties by both gender and age groups """

    def __init__(self, logs=None, path: str = None):

        # create a dictionary for lifestyle property description to be used as plot descriptors. Here we are
        # excluding two properties circumcision and sex workers as these are logged differently
        self.en_props = {'li_urban': 'currently urban', 'li_wealth': 'wealth level',
                         'li_low_ex': 'currently low exercise', 'li_tob': 'current using tobacco',
                         'li_ex_alc': 'current excess alcohol', 'li_mar_stat': 'marital status',
                         'li_in_ed': 'currently in education', 'li_ed_lev': 'education level',
                         'li_unimproved_sanitation': 'uninproved sanitation',
                         'li_no_clean_drinking_water': 'no clean drinking water',
                         'li_wood_burn_stove': 'wood burn stove', 'li_no_access_handwashing': ' no access hand washing',
                         'li_high_salt': 'high salt', 'li_high_sugar': 'high sugar', 'li_bmi': 'bmi',
                         'li_is_circ': 'Male circumcision', 'li_is_sexworker': 'sex workers'
                         }

        self.categories_desc: dict = {
            'li_bmi': ["bmi category 1", "bmi category 2", "bmi category 3", "bmi category 4", "bmi category 5"],
            'li_wealth': ['wealth level 1', 'wealth level 2', 'wealth level 3', 'wealth level 4', 'wealth level 5'],
            'li_mar_stat': ['Never Married', 'Married', 'Divorced'],
            'li_ed_lev': ['Not in education', 'Primary edu', 'secondary education']
        }

        # date-stamp to label log files and any other outputs
        self.datestamp: str = datetime.date.today().strftime("__%Y_%m_%d")

        # a dictionary for gender descriptions. to be used when plotting by gender
        self.gender_des: Dict[str, str] = {'M': 'Males', 'F': 'Females'}

        # get all logs
        self.all_logs = logs

        # store un flattened logs
        self.dfs = self.construct_dfs(self.all_logs)

        self.outputpath = Path(path)  # folder for convenience of storing outputs

    def construct_dfs(self, lifestyle_log) -> dict:
        """ Create dict of pd.DataFrames containing counts of different lifestyle properties by date, sex and
        age-group """
        return {
            k: unflatten_flattened_multi_index_in_logging(v.set_index('date'))
            for k, v in lifestyle_log.items() if k in self.en_props.keys()
        }

    def custom_axis_formatter(self, df: pd.DataFrame, ax, li_property: str):
        """
        create a custom date formatter since the default pandas date formatter works well with line graphs. see an
        accepted solution to issue https://stackoverflow.com/questions/30133280/pandas-bar-plot-changes-date-format

        :param df: pandas dataframe or series
        :param ax: matplotlib AxesSubplot object
        :param li_property: one of the lifestyle properties
        """
        # make the tick labels empty so the labels don't get too crowded
        tick_labels = [''] * len(df.index)
        # Every 12th tick label includes the year
        tick_labels[::12] = [item.strftime('%Y') for item in df.index[::12]]
        ax.xaxis.set_major_formatter(ticker.FixedFormatter(tick_labels))
        ax.legend(self.categories_desc[li_property] if li_property in self.categories_desc.keys()
                  else [self.en_props[li_property]], bbox_to_anchor=(0.5, -0.27), loc='lower center')
        plt.gcf().autofmt_xdate()

    # 1. GENDER PLOTS
    # --------------------------------------------------------------------------------------------------------
    def plot_categorical_properties_by_gender(self, li_property: str, categories: list):
        """ a function to plot all categorical properties of lifestyle module grouped by gender. Available
        categories per property include;

        1. bmi
            bmi is categorised as follows
                  category 1: <18
                  category 2: 18-24.9
                  category 3: 25-29.9
                  category 4: 30-34.9
                  category 5: 35+
            bmi is 0 until age 15

        2. wealth level
            wealth level is categorised as follows as follows;

                    Urban                               |         Rural
                    ------------------------------------|----------------------------------------------
                    level 1 = 75% wealth level           |  level 1 = 11% wealth level
                    level 2 = 16% wealth level          |  level 2 = 21% wealth level
                    level 3 = 5% wealth level           |  level 3 = 23% wealth level
                    level 4 = 2% wealth level           |  level 4 = 23% wealth level
                    level 5 = 2% wealth level           |  level 5 = 23% wealth level

        3. education level
             education level is categorised as follows
                    level 1: not in education
                    level 2: primary education
                    level 3 : secondary+ education )

        4. marital status
            marital status is categorised as follows
                    category 1: never married
                    category 2: married
                    category 3: widowed or divorced

        :param li_property: any other categorical property defined in lifestyle module
        :param categories: a list of categories """

        if li_property == 'li_wealth':
            self.display_wealth_level_plots_by_gender(li_property, categories)
        else:
            # a new dataframe to contain data of property categories grouped by gender
            gc_df = pd.DataFrame()

            counter: int = 0  # counter for indexing purposes
            # create subplot figure having two side by side plots
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
            for gender, desc in self.gender_des.items():
                for cat in categories:
                    gc_df[f'cat_{cat}'] = self.dfs[li_property][gender][cat].sum(axis=1)

                # normalise the probabilities
                gc_df = gc_df.apply(lambda row: row / row.sum(), axis=1)

                ax = gc_df.plot(kind='bar', stacked=True, ax=axes[counter],
                                title=f"{desc} {self.en_props[li_property]}  categories",
                                ylabel=f"{self.en_props[li_property]} proportions", xlabel="Year"
                                )
                self.custom_axis_formatter(gc_df, ax, li_property)
                # increase counter
                counter += 1
            # save and display plots for property categories by gender
            plt.savefig(self.outputpath / (li_property + self.datestamp + '.png'), format='png')
            plt.show()

    def plot_non_categorical_properties_by_gender(self, _property):
        """ a function to plot non categorical properties of lifestyle module grouped by gender

         :param _property: any other non categorical property defined in lifestyle module """
        # create a dataframe that will hold male female proportions per each lifestyle property
        totals_df = pd.DataFrame()

        # plot for male circumcision and female sex workers
        if _property in ['li_is_circ', 'li_is_sexworker']:
            # plot male circumcision
            g_plots.male_circumcision_and_sex_workers_plot(_property)
        else:
            for gender, desc in self.gender_des.items():
                totals_df[gender] = self.dfs[_property][gender]["True"].sum(axis=1)

            # normalise the probabilities
            totals_df = totals_df.apply(lambda row: row / row.sum(), axis=1)

            # plot figure
            _counter: int = 0  # a counter for indexing purposes
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
            for gender, desc in self.gender_des.items():
                ax = totals_df.iloc[:, _counter].plot(kind='bar', ax=axes[_counter], ylim=(0, 1.0),
                                                      ylabel=f'{self.en_props[_property]} proportions', xlabel="Year",
                                                      color='darkturquoise', title=f"{desc} {self.en_props[_property]}")
                # format x-axis
                self.custom_axis_formatter(totals_df, ax, _property)
                # increase counter
                _counter += 1
            # save and display plots for property categories by gender
            plt.savefig(self.outputpath / (_property + self.datestamp + '.png'), format='png')
            plt.show()

    def display_all_categorical_and_non_categorical_plots_by_gender(self):
        """ a function to display plots for both categorical and non categorical properties grouped by gender """
        for _property in self.en_props.keys():
            if _property in ['li_bmi', 'li_wealth']:
                self.plot_categorical_properties_by_gender(_property, ['1', '2', '3', '4', '5'])
            elif _property in ['li_mar_stat', 'li_ed_lev']:
                self.plot_categorical_properties_by_gender(_property, ['1', '2', '3'])
            else:
                # pass
                self.plot_non_categorical_properties_by_gender(_property)

    # 2. AGE GROUP PLOTS
    # -------------------------------------------------------------------------------------------------------------
    def plot_categorical_properties_by_age_group(self, _property: str, categories: list):
        """ a function to plot all categorical properties of lifestyle module grouped by age group. Available
        categories per property include;

        1. bmi
            bmi is categorised as follows
                  category 1: <18
                  category 2: 18-24.9
                  category 3: 25-29.9
                  category 4: 30-34.9
                  category 5: 35+
            bmi is 0 until age 15

        2. wealth level
            wealth level is categorised as follows as follows;

                    Urban                               |         Rural
                    ------------------------------------|----------------------------------------------
                    level 1 = 75% wealth level           |  level 1 = 11% wealth level
                    level 2 = 16% wealth level          |  level 2 = 21% wealth level
                    level 3 = 5% wealth level           |  level 3 = 23% wealth level
                    level 4 = 2% wealth level           |  level 4 = 23% wealth level
                    level 5 = 2% wealth level           |  level 5 = 23% wealth level

        3. education level
             education level is categorised as follows
                    level 1: not in education
                    level 2: primary education
                    level 3 : secondary+ education )

        4. marital status
            marital status is categorised as follows
                    category 1: never married
                    category 2: married
                    category 3: widowed or divorced

        :param _property: any other categorical property defined in lifestyle module
        :param categories: a list of categories """

        if _property == "li_wealth":
            self.display_wealth_level_plots_by_age_group(_property, categories)
        else:
            # create a new dataframe to contain data of age groups against categories
            new_df = pd.DataFrame()
            # loop through categories and get data into age groups categories dataframe
            for cat in categories:
                new_df[f'cat_{cat}'] = self.dfs[_property]['M'][cat].sum(axis=0) + self.dfs[_property]['F'][cat].sum(
                    axis=0)

            # convert values to proportions
            new_df = new_df.apply(lambda row: row / row.sum(), axis=1)

            ax = new_df.plot(kind='bar', stacked=True, title=f"{self.en_props[_property]}  categories",
                             ylabel=f"{self.en_props[_property]} proportions", xlabel="Age Range"
                             )
            ax.legend(self.categories_desc[_property], loc='upper right')
            # save and display plots for property categories by gender
            plt.savefig(self.outputpath / (_property + self.datestamp + '.png'), format='png')
            plt.show()

    def plot_non_categorical_properties_by_age_group(self, _property: str):
        """ a function to plot non categorical properties of lifestyle module grouped by age group

         :param _property: any other non categorical property defined in lifestyle module """
        # sum whole dataframe per each property
        get_totals = (self.dfs[_property]["M"]["True"].sum(axis=1) + self.dfs[_property]["F"]["True"].sum(
            axis=1)).sum()
        # age and store age groups from dataframe. age groups are the same for both males and females so choosing
        # either doesn't matter
        get_age_group = self.dfs[_property]["M"]["True"].columns
        # loop through age groups and get plot data
        for age_group in get_age_group:
            get_age_group_props = (self.dfs[_property]['M']['True'][age_group] +
                                   self.dfs[_property]['F']['True'][age_group]).sum() / get_totals
            plt.bar(age_group, get_age_group_props, color='darkturquoise')

        plt.title(f"{self.en_props[_property]} by age groups")
        plt.xlabel("age groups")
        plt.ylabel("proportions")
        plt.ylim(0, )
        plt.legend([self.en_props[_property]], loc='upper right')
        plt.savefig(self.outputpath / (_property + 'by_age_group' + self.datestamp + '.png'), format='png')
        plt.show()

    def display_all_categorical_and_non_categorical_plots_by_age_group(self):
        """ a function that will display plots of all enhanced lifestyle properties grouped by age group """
        for _property in self.en_props.keys():
            if _property == 'li_bmi':
                self.plot_categorical_properties_by_age_group(_property, ['1', '2', '3', '4', '5'])
            elif _property == 'li_wealth':
                self.plot_categorical_properties_by_age_group(_property, ['1', '2', '3', '4', '5'])
            elif _property in ['li_mar_stat', 'li_ed_lev']:
                self.plot_categorical_properties_by_age_group(_property, ['1', '2', '3'])
            else:
                self.plot_non_categorical_properties_by_age_group(_property)

    def male_circumcision_and_sex_workers_plot(self, _property: str = None):
        """ a function to plot for men circumcised and female sex workers

        :param _property: circumcision or female sex worker property defined in enhanced lifestyle module """

        # create a dataframe that will hold proportions per each lifestyle property
        totals_df = pd.DataFrame()
        gender: str = 'M'  # key in the logs file for men circumcised
        max_ylim = 0.30  # define y limit in plot

        # check property if it is not circumcision( if not circumcision then its female sex workers therefore we have
        # to update gender and y limit values)
        if not _property == 'li_is_circ':
            gender = 'F'
            max_ylim = 0.01
        # get proportions per property
        totals_df[_property] = self.dfs[_property][gender]["True"].sum(axis=1) / self.dfs[_property][gender].sum(axis=1)

        ax = totals_df.plot(kind='bar', ylim=(0, max_ylim), ylabel=f'{self.en_props[_property]} proportions',
                            xlabel="Year",
                            color='darkturquoise', title=f"{self.en_props[_property]}")
        # format x-axis
        self.custom_axis_formatter(totals_df, ax, _property)

        # save and display plots
        plt.savefig(self.outputpath / (_property + self.datestamp + '.png'), format='png')
        plt.show()

    def display_wealth_level_plots_by_gender(self, li_property: str, categories: list):
        """ A function to display urban and rural wealth level plots by gender """
        # a dict to hold individual's urban rural status
        _rural_urban_state = {
            'True': 'Urban',
            'False': 'Rural'
        }
        for urban_rural in ['True', 'False']:
            # a new dataframe to contain data of property categories grouped by gender
            wealth_df = pd.DataFrame()
            # create subplot figure having two side by side plots
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
            counter: int = 0  # counter for indexing purposes
            for gender, desc in self.gender_des.items():
                for cat in categories:
                    wealth_df[f'cat_{cat}'] = self.dfs[li_property][urban_rural][gender][cat].sum(axis=1)

                # normalise the probabilities
                wealth_df = wealth_df.apply(lambda row: row / row.sum(), axis=1)

                ax = wealth_df.plot(kind='bar', stacked=True, ax=axes[counter],
                                    title=f"{_rural_urban_state[urban_rural]}"
                                          f" {self.en_props[li_property]} categories in {desc}",
                                    ylabel=f"{self.en_props[li_property]} proportions", xlabel="Year"
                                    )
                self.custom_axis_formatter(wealth_df, ax, li_property)
                # increase counter
                counter += 1
            # save and display plots for property categories by gender
            plt.savefig(self.outputpath / (li_property + self.datestamp + '.png'), format='png')
            plt.show()

    def display_wealth_level_plots_by_age_group(self, li_property: str, categories: list):
        """ A function to display urban and rural wealth level plots by age groups"""
        # a dict to hold individual's urban rural status
        _rural_urban_state = {
            'True': 'Urban',
            'False': 'Rural'
        }
        # create a new dataframe to contain data of age groups against categories
        new_df = pd.DataFrame()
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
        counter: int = 0  # counter for indexing purposes
        for urban_rural in ['True', 'False']:
            # loop through categories and get data into age groups categories dataframe
            for cat in categories:
                new_df[f'cat_{cat}'] = self.dfs[li_property][urban_rural]['M'][cat].sum(axis=0) + \
                                       self.dfs[li_property][urban_rural]['F'][cat].sum(axis=0)

            # convert values to proportions
            new_df = new_df.apply(lambda row: row / row.sum(), axis=1)

            ax = new_df.plot(kind='bar', stacked=True, ax=axes[counter],
                             title=f"{_rural_urban_state[urban_rural]} "
                                   f"{self.en_props[li_property]} categories",
                             ylabel=f"{self.en_props[li_property]} proportions", xlabel="Age Range"
                             )
            ax.legend(self.categories_desc[li_property], loc='upper right')
            # increase counter
            counter += 1
        # save and display plots for property categories by gender
        plt.savefig(self.outputpath / (li_property + self.datestamp + '.png'), format='png')
        plt.show()


def extract_formatted_series(df):
    return pd.Series(index=pd.to_datetime(df['date']), data=df.iloc[:, 1].values)


def run():
    # To reproduce the results, you need to set the seed for the Simulation instance. The Simulation
    # will seed the random number generators for each module when they are registered.
    # If a seed argument is not given, one is generated. It is output in the log and can be
    # used to reproduce results of a run
    seed = 1

    # By default, all output is recorded at the "INFO" level (and up) to standard out. You can
    # configure the behaviour by passing options to the `log_config` argument of
    # Simulation.
    log_config = {
        "filename": "enhanced_lifestyle",  # The prefix for the output file. A timestamp will be added to this.
        "custom_levels": {  # Customise the output of specific loggers. They are applied in order:
            "tlo.methods.demography": logging.WARNING,
            "tlo.methods.enhanced_lifestyle": logging.INFO
        }
    }
    # For default configuration, uncomment the next line
    # log_config = dict()

    # Basic arguments required for the simulation
    start_date = Date(2010, 1, 1)
    end_date = Date(2050, 1, 1)
    pop_size = 20000

    # This creates the Simulation instance for this run. Because we"ve passed the `seed` and
    # `log_config` arguments, these will override the default behaviour.
    sim = Simulation(start_date=start_date, seed=seed, log_config=log_config)

    # Path to the resource files used by the disease and intervention methods
    resources = "./resources"

    # We register all modules in a single call to the register method, calling once with multiple
    # objects. This is preferred to registering each module in multiple calls because we will be
    # able to handle dependencies if modules are registered together
    sim.register(
        demography.Demography(resourcefilepath=resources),
        enhanced_lifestyle.Lifestyle(resourcefilepath=resources),
        simplified_births.SimplifiedBirths(resourcefilepath=resources)

    )

    sim.make_initial_population(n=pop_size)
    sim.simulate(end_date=end_date)
    return sim


# catch warnings
# pd.set_option('mode.chained_assignment', 'raise')
# %% Run the Simulation
sim = run()

# %% read the results
output = parse_log_file(sim.log_filepath)
# output = parse_log_file(Path("./outputs/enhanced_lifestyle__2022-10-05T123515.log"))

# construct a dict of dataframes using lifestyle logs
logs_df = output['tlo.methods.enhanced_lifestyle']

# # initialise LifestylePlots class
g_plots = LifeStylePlots(logs=logs_df, path="./outputs")

# plot by gender
g_plots.display_all_categorical_and_non_categorical_plots_by_gender()
#
# plot by age groups
g_plots.display_all_categorical_and_non_categorical_plots_by_age_group()
