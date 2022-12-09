"""
This is Lifestyle analyses file.  It seeks to show trends in different lifestyle properties. It plots properties by
gender and age groups
"""
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


def add_footnote(footnote: str):
    """ A function that adds a footnote below each plot. Here we are explaining what a denominator for every
    graph is """
    plt.figtext(0.5, 0.01, footnote, ha="center", fontsize=10,
                bbox={"facecolor": "gray", "alpha": 0.3, "pad": 5})


class LifeStylePlots:
    """ a class for for plotting lifestyle properties by both gender and age groups """

    def __init__(self, logs=None, path: str = None):

        # create a dictionary for lifestyle properties and their descriptions, to be used as plot descriptors.
        self.en_props = {'li_urban': ['currently urban', 'Denominator: Sum of all individuals per gender',
                                      'Denominator: Sum of all individuals in each age-group'],

                         'li_wealth': ['wealth level', 'Denominator: Sum of individuals in all wealth levels per urban '
                                                       'or rural per gender',
                                       'Denominator: Sum of all individuals in all wealth levels per urban or rural '
                                       'per each age-group'],

                         'li_low_ex': ['currently low exercise', 'Denominator: Sum of all individuals aged 15+ per '
                                                                 'gender', 'Denominator: Sum of all individuals aged '
                                                                           '15+ in each age-group'],

                         'li_tob': ['current using tobacco', 'Denominator: Sum of all individuals aged 15+ per gender',
                                    'Denominator: Sum of all individuals aged 15+ in each age-group'],

                         'li_ex_alc': ['current excess alcohol', 'Denominator: Sum of all individuals aged 15+ per '
                                                                 'urban or rural per gender',
                                       'Denominator: Sum of all individuals aged 15+ per urban or rural in each '
                                       'age-group'],

                         'li_mar_stat': ['marital status', 'Denominator: Sum of individuals in all marital status '
                                                           'categories per gender in 15+ individuals',
                                         'Denominator: Sum  of 15+ individuals in all marital status categories per '
                                         'each age-group'],

                         'li_in_ed': ['currently in education', 'Denominator: Sum of all individuals aged between '
                                                                '5-19 per gender',
                                      'Denominator: Sum of all individuals aged between 5-19 in all age-group'
                                      ],

                         'li_ed_lev': ['education level', 'Denominator: Sum of individuals from age 0 in all '
                                                          'education levels per gender',
                                       'Denominator: Sum of individuals from age 0 in all education levels in each '
                                       'age-group'],

                         'li_unimproved_sanitation': ['unimproved sanitation', 'Denominator: Sum of all individuals '
                                                                               'per urban or rural per gender',
                                                      'Denominator: Sum of individuals per rural or urban in each '
                                                      'age-group '
                                                      ],

                         'li_no_clean_drinking_water': ['no clean drinking water', 'Denominator: Sum of all '
                                                                                   'individuals per '
                                                                                   'urban or rural per gender',
                                                        'Denominator: Sum of all individuals per urban or rural in '
                                                        'each age-group '
                                                        ],
                         'li_wood_burn_stove': ['wood burn stove', 'Denominator: Sum of all individuals per urban or '
                                                                   'rural per gender',
                                                'Denominator: Sum of all individuals per urban or rural in each '
                                                'age-group'],

                         'li_no_access_handwashing': [' no access hand washing', 'Denominator: Sum of all individuals '
                                                                                 'per gender',
                                                      'Denominator: Sum of all individuals in each age-group',
                                                      ],

                         'li_high_salt': ['high salt', 'Denominator: Sum of all individuals per gender',
                                          'Denominator: Sum of individuals in each age-group'],

                         'li_high_sugar': ['high sugar', 'Denominator: Sum of all individuals per gender',
                                           'Denominator: Sum of individuals in each age-group'],

                         'li_bmi': ['bmi', 'Denominator: Sum of individuals in all bmi categories in 15+ individuals '
                                           'per urban or rural per gender',
                                    'Denominator: Sum of individuals in all bmi categories in 15+ individuals per '
                                    'rural or urban in age-group'
                                    ],

                         'li_is_circ': ['Male circumcision', 'Denominator: Sum of all males',
                                        'Denominator: Sum of males in each age-group'],

                         'li_is_sexworker': ['sex workers', 'Denominator: Sum of all females aged between 15-49',
                                             'Denominator: Sum of females in each age-group ranging from 15-49'],
                         }

        # A dictionary to map properties and their description. Useful when setting plot legend
        self.categories_desc: dict = {
            'li_bmi': ["bmi category 1", "bmi category 2", "bmi category 3", "bmi category 4", "bmi category 5"],
            'li_wealth': ['wealth level 1', 'wealth level 2', 'wealth level 3', 'wealth level 4', 'wealth level 5'],
            'li_mar_stat': ['Never Married', 'Married', 'Divorced or Widowed'],
            'li_ed_lev': ['No education', 'Primary edu', 'secondary education']
        }

        # create a dictionary to defining individual's rural or urban state
        self._rural_urban_state = {
            'True': 'Urban',
            'False': 'Rural'
        }

        # define all properties that are categorised by rural or urban in addition to age and sex
        self.cat_by_rural_urban_props = ['li_wealth', 'li_bmi', 'li_low_ex', 'li_ex_alc', 'li_wood_burn_stove',
                                         'li_unimproved_sanitation',
                                         'li_no_clean_drinking_water']

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
                  else [self.en_props[li_property][0]], loc='upper center')
        # else [self.en_props[li_property]], bbox_to_anchor=(0.5, -0.27), loc='upper center')
        plt.gcf().autofmt_xdate()

    # 1. PLOT BY GENDER
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
            bmi is np.nan until age 15

        2. wealth level
            wealth level is categorised as follows;

                    Urban                               |         Rural
                    ------------------------------------|----------------------------------------------
                    level 1 = 75% wealth level          |  level 1 = 11% wealth level
                    level 2 = 16% wealth level          |  level 2 = 21% wealth level
                    level 3 = 5% wealth level           |  level 3 = 23% wealth level
                    level 4 = 2% wealth level           |  level 4 = 23% wealth level
                    level 5 = 2% wealth level           |  level 5 = 23% wealth level

        3. education level
             education level is categorised as follows
                    level 1: no education
                    level 2: primary education
                    level 3 : secondary+ education

        4. marital status
            marital status is categorised as follows
                    category 1: never married
                    category 2: married
                    category 3: widowed or divorced

        :param li_property: any other categorical property defined in lifestyle module
        :param categories: a list of categories """

        # a new dataframe to contain data of property categories grouped by gender
        gc_df = pd.DataFrame()

        # 1. check if property is in a group of those that need plotting by urban and rural
        if li_property in self.cat_by_rural_urban_props:
            _cols_counter: int = 0  # a counter for plotting. setting rows
            fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 5))  # plot setup
            for urban_rural in self._rural_urban_state.keys():
                _rows_counter: int = 0  # a counter for plotting. setting columns
                for gender, desc in self.gender_des.items():
                    for cat in categories:
                        gc_df[f'cat_{cat}'] = self.dfs[li_property][urban_rural][gender][cat].sum(axis=1)

                    # normalise the probabilities
                    gc_df = gc_df.apply(lambda row: row / row.sum(), axis=1)
                    # do plotting
                    ax = gc_df.plot(kind='bar', stacked=True, ax=axes[_cols_counter, _rows_counter],
                                    title=f"{self._rural_urban_state[urban_rural]}"
                                          f" {self.en_props[li_property][0]} categories in {desc}",
                                    ylabel=f"{self.en_props[li_property][0]} proportions", xlabel="Year"
                                    )
                    self.custom_axis_formatter(gc_df, ax, li_property)
                    # increase counter
                    _rows_counter += 1

                _cols_counter += 1
            # save and display plots
            add_footnote(f'{self.en_props[li_property][1]}')
            plt.tight_layout()
            plt.savefig(self.outputpath / (
                li_property + '_' + self.datestamp + '.png'),
                        format='png')
            plt.show()

        # 2. if property is not in a group of those that need plotting by urban and rural plot them by age category only
        else:
            counter: int = 0  # counter for indexing purposes
            # create subplots
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
            for gender, desc in self.gender_des.items():
                for cat in categories:
                    gc_df[f'cat_{cat}'] = self.dfs[li_property][gender][cat].sum(axis=1)

                # normalise the probabilities
                gc_df = gc_df.apply(lambda row: row / row.sum(), axis=1)

                # do plotting
                ax = gc_df.plot(kind='bar', stacked=True, ax=axes[counter],
                                title=f"{desc} {self.en_props[li_property][0]}  categories",
                                ylabel=f"{self.en_props[li_property][0]} proportions", xlabel="Year"
                                )
                self.custom_axis_formatter(gc_df, ax, li_property)
                # increase counter
                counter += 1

            # save and display plots for property categories by gender
            add_footnote(f'{self.en_props[li_property][1]}')
            plt.tight_layout()
            plt.savefig(self.outputpath / (li_property + self.datestamp + '.png'), format='png')
            plt.show()

    def plot_non_categorical_properties_by_gender(self, _property):
        """ a function to plot non categorical properties of lifestyle module grouped by gender

         :param _property: any other non categorical property defined in lifestyle module """
        # set y-axis limit.
        y_lim: float = 0.8
        if _property in ['li_no_access_handwashing', 'li_high_salt', 'li_wood_burn_stove', 'li_in_ed']:
            y_lim = 1.0

        if _property in ['li_tob', 'li_ex_alc']:
            y_lim = 0.3
        # create a dataframe that will hold male female proportions per each lifestyle property
        totals_df = pd.DataFrame()

        # plot for male circumcision and female sex workers
        if _property in ['li_is_circ', 'li_is_sexworker']:
            self.male_circumcision_and_sex_workers_plot(_property)

        # check if property is in a group of those that need plotting by urban and rural
        elif _property in self.cat_by_rural_urban_props:
            _cols_counter: int = 0  # a counter for plotting. setting rows
            fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 5))  # plot setup
            for urban_rural in self._rural_urban_state.keys():
                _rows_counter: int = 0  # a counter for plotting. setting columns
                for gender, desc in self.gender_des.items():
                    # compute proportions on each property per gender
                    totals_df[gender] = self.dfs[_property][urban_rural][gender]["True"].sum(axis=1) / \
                                        self.dfs[_property][urban_rural][gender].sum(axis=1)

                    # do plotting
                    ax = totals_df.iloc[:, _rows_counter].plot(kind='bar', ax=axes[_cols_counter, _rows_counter],
                                                               ylim=(0, y_lim),
                                                               ylabel=f'{self.en_props[_property][0]} proportions',
                                                               xlabel="Year",
                                                               color='darkturquoise',
                                                               title=f"{self._rural_urban_state[urban_rural]}"
                                                                     f" {desc} {self.en_props[_property][0]}")
                    # format x-axis
                    self.custom_axis_formatter(totals_df, ax, _property)
                    # increase counter
                    _rows_counter += 1

                _cols_counter += 1
            # save and display plots for property categories by gender
            add_footnote(f'{self.en_props[_property][1]}')
            plt.tight_layout()
            plt.savefig(self.outputpath / (_property + self.datestamp + '.png'), format='png')
            plt.show()

        # if property is not in a group of those that need plotting by urban and rural plot them by age category only
        else:
            for gender, desc in self.gender_des.items():
                # compute proportions on each property per gender
                totals_df[gender] = self.dfs[_property][gender]["True"].sum(axis=1) / \
                                    self.dfs[_property][gender].sum(axis=1)

            # plot figure
            _rows_counter: int = 0  # a counter for plotting. setting rows
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
            for gender, desc in self.gender_des.items():
                # do plotting
                ax = totals_df.iloc[:, _rows_counter].plot(kind='bar', ax=axes[_rows_counter],
                                                           ylim=(0, y_lim),
                                                           ylabel=f'{self.en_props[_property][0]} proportions',
                                                           xlabel="Year",
                                                           color='darkturquoise',
                                                           title=f"{desc} {self.en_props[_property][0]}")
                # format x-axis
                self.custom_axis_formatter(totals_df, ax, _property)
                # increase counter
                _rows_counter += 1
            # save and display plots for property categories by gender
            add_footnote(f'{self.en_props[_property][1]}')
            plt.tight_layout()
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
            bmi is np.nan until age 15

        2. wealth level
            wealth level is categorised as follows as follows;

                    Urban                               |         Rural
                    ------------------------------------|----------------------------------------------
                    level 1 = 75% wealth level          |  level 1 = 11% wealth level
                    level 2 = 16% wealth level          |  level 2 = 21% wealth level
                    level 3 = 5% wealth level           |  level 3 = 23% wealth level
                    level 4 = 2% wealth level           |  level 4 = 23% wealth level
                    level 5 = 2% wealth level           |  level 5 = 23% wealth level

        3. education level
             education level is categorised as follows
                    level 1: not in education
                    level 2: primary education
                    level 3 : secondary+ education

        4. marital status
            marital status is categorised as follows
                    category 1: never married
                    category 2: married
                    category 3: widowed or divorced

        :param _property: any other categorical property defined in lifestyle module
        :param categories: a list of categories """

        # select logs from the latest year. In this case we are selecting year 2021
        all_logs_df = self.dfs[_property]
        mask = (all_logs_df.index > pd.to_datetime('2021-01-01')) & (all_logs_df.index <= pd.to_datetime('2022-01-01'))
        self.dfs[_property] = self.dfs[_property].loc[mask]

        # 1. check if property is in a group of those that need plotting by urban and rural
        if _property in self.cat_by_rural_urban_props:
            # create a new dataframe to contain data of age groups against categories
            new_df = pd.DataFrame()

            _rows_counter: int = 0  # a counter to set the number of rows when plotting
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
            for urban_rural in self._rural_urban_state.keys():
                # loop through categories and get age groups data per each category
                for cat in categories:
                    new_df[f'cat_{cat}'] = self.dfs[_property][urban_rural]['M'][cat].sum(axis=0) + \
                                           self.dfs[_property][urban_rural]['F'][cat].sum(axis=0)

                # convert values to proportions
                # new_df = new_df.apply(lambda row: row / row.sum(), axis=0)
                new_df = new_df.apply(lambda row: row / row.sum(), axis=1)
                # do plotting
                ax = new_df.plot(kind='bar', stacked=True, ax=axes[_rows_counter],
                                 title=f"{self._rural_urban_state[urban_rural]} {self.en_props[_property][0]} "
                                       f" categories (Year 2021)",
                                 ylabel=f"{self.en_props[_property][0]} proportions", xlabel="Age Range",
                                 ylim=(0, 1))
                ax.legend(self.categories_desc[_property], loc='upper right')
                _rows_counter += 1
            # save and display plots
            add_footnote(f'{self.en_props[_property][2]}')
            plt.tight_layout()
            plt.savefig(self.outputpath / (_property + self.datestamp + '.png'), format='png')
            plt.show()

        # 2. if property is not in a group of those that need plotting by urban and rural plot them by age category only
        else:
            # create a new dataframe to contain data of age groups against categories
            new_df = pd.DataFrame()
            # loop through categories and get age groups data per each category
            for cat in categories:
                new_df[f'cat_{cat}'] = self.dfs[_property]['M'][cat].sum(axis=0) + self.dfs[_property]['F'][cat].sum(
                    axis=0)
            # convert values to proportions
            # new_df = new_df.apply(lambda row: row / row.sum(), axis=0)
            new_df = new_df.apply(lambda row: row / row.sum(), axis=1)
            # do plotting
            ax = new_df.plot(kind='bar', stacked=True, title=f"{self.en_props[_property][0]}  categories (Year 2021)",
                             ylabel=f"{self.en_props[_property][0]} proportions", xlabel="Age Range"
                             )
            ax.legend(self.categories_desc[_property], loc='upper right')
            # save and display plots
            add_footnote(f'{self.en_props[_property][2]}')
            plt.tight_layout()
            plt.savefig(self.outputpath / (_property + self.datestamp + '.png'), format='png')
            plt.show()

    def plot_non_categorical_properties_by_age_group(self, _property: str):
        """ a function to plot non categorical properties of lifestyle module grouped by age group

         :param _property: any other non categorical property defined in lifestyle module """

        # select logs from the latest year. In this case we are selecting year 2021
        all_logs_df = self.dfs[_property]
        mask = (all_logs_df.index > pd.to_datetime('2021-01-01')) & (all_logs_df.index <= pd.to_datetime('2022-01-01'))
        self.dfs[_property] = self.dfs[_property].loc[mask]

        # a dataframe that will hold total number of individuals per each age group and who are positive or negative to
        # a particular lifestyle property
        get_age_group_totals = pd.DataFrame()

        # 1. check if property is in a group of those that need plotting by urban and rural
        if _property in self.cat_by_rural_urban_props:
            # create a list of age groups. age groups are the same for both males and females so choosing
            # either gender from the dataframe doesn't matter
            get_age_group = self.dfs[_property]["True"]["M"]["True"].columns

            _rows_counter: int = 0  # a counter for plotting. setting rows
            fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 5))
            for urban_rural in self._rural_urban_state.keys():
                # loop through age groups and get plot data
                for age_group in get_age_group:
                    for _bool_value in ['True', 'False']:
                        get_age_group_totals[_bool_value] = (
                            self.dfs[_property][urban_rural]['M'][_bool_value][age_group] +
                            self.dfs[_property][urban_rural]['F'][_bool_value][age_group])

                    get_age_group_props = get_age_group_totals['True'] / get_age_group_totals.sum(axis=1)

                    # do plotting
                    axes[_rows_counter].bar(age_group, get_age_group_props, color='darkturquoise')
                    axes[_rows_counter].set_title(
                        f"{self._rural_urban_state[urban_rural]} {self.en_props[_property][0]}"
                        f" by age groups (Year 2021)")
                    axes[_rows_counter].set_ylabel(f"{self.en_props[_property][0]} proportions")
                    axes[_rows_counter].set_ylim(0, 1.0)
                    axes[_rows_counter].legend([self.en_props[_property][0]], loc='upper right')
                _rows_counter += 1

                # to avoid overlapping of info on graph, set one x-label for both graphs
                plt.xlabel("age groups")
            # save and display graph
            add_footnote(f'{self.en_props[_property][2]}')
            plt.tight_layout()
            plt.savefig(self.outputpath / (_property + 'by_age_group' + self.datestamp + '.png'), format='png')
            plt.show()

        # 2. if property is not in a group of those that need plotting by urban and rural plot them by age category only
        else:
            # get individuals per each age group
            for _bool_value in ['True', 'False']:
                get_age_group_totals[_bool_value] = self.dfs[_property]['M'][_bool_value].sum(axis=0) + \
                                                    self.dfs[_property]['F'][_bool_value].sum(axis=0)

            get_age_group_props = get_age_group_totals['True'] / get_age_group_totals.sum(axis=1)
            # do plotting
            get_age_group_props.plot.bar(color='darkturquoise')
            # set plot title, labels, legends and axis limit
            plt.title(f"{self.en_props[_property][0]} by age groups (Year 2021)")
            plt.xlabel("age groups")
            plt.ylabel(f"{self.en_props[_property][0]} proportions")
            plt.ylim(0, 1 if not _property == 'li_is_sexworker' else 0.04)
            plt.legend([self.en_props[_property][0]], loc='upper right')
            add_footnote(f'{self.en_props[_property][2]}')

            plt.tight_layout()
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
        gender: str = 'M'  # key in the logs file for men circumcised
        max_ylim = 0.30  # define y limit in plot

        # check property if not circumcision. if true, update gender and y limit values
        if not _property == 'li_is_circ':
            gender = 'F'
            max_ylim = 0.01

        # get proportions per property
        totals_df = self.dfs[_property][gender]["True"].sum(axis=1) / self.dfs[_property][gender].sum(axis=1)

        ax = totals_df.plot(kind='bar', ylim=(0, max_ylim), ylabel=f'{self.en_props[_property][0]} proportions',
                            xlabel="Year",
                            color='darkturquoise', title=f"{self.en_props[_property][0]} Percentage")
        # format x-axis
        self.custom_axis_formatter(totals_df, ax, _property)

        # save and display plots
        add_footnote(f'{self.en_props[_property][1]}')
        plt.tight_layout()
        plt.savefig(self.outputpath / (_property + self.datestamp + '.png'), format='png')
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
    end_date = Date(2050, 2, 1)
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
# output = parse_log_file(Path("./outputs/enhanced_lifestyle__2022-12-07T112556.log"))

# construct a dict of dataframes using lifestyle logs
logs_df = output['tlo.methods.enhanced_lifestyle']

# initialise LifestylePlots class
g_plots = LifeStylePlots(logs=logs_df, path="./outputs")
# plot by gender
g_plots.display_all_categorical_and_non_categorical_plots_by_gender()

# plot by age groups
g_plots.display_all_categorical_and_non_categorical_plots_by_age_group()
