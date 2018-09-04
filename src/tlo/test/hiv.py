"""
Following the skeleton method for HIV

Q: should treatment be in a separate method?
"""

# import any methods from other modules, e.g. for parameter definitions
from typing import Any, Union

from pandas.tseries.offsets import Day

from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import PopulationScopeEventMixin, RegularEvent

# need to import ART, ART_Event, TB

import numpy as np
import pandas as pd

# NOTES: what should the functions be returning?
# previously they read in the population dataframe and then returned the modified population dataframe
# how to deal with the current_time variable needed in many functions?
# check use of self
# initialise population function was renamed as there were unresolved differences

# sim_size = int(100)
# current_time = 2018

# HELPER FUNCTION - should these go in class(HIV)?
# are they static methods?
# untreated HIV mortality rates - annual, adults


def log_scale(a0):
    age_scale = 2.55 - 0.025 * (a0 - 30)
    return age_scale


# read in data files #
# use function read.parameters in class HIV to do this?
file_path = '/Users/tamuri/Documents/2018/thanzi/test/HIV_test_run_data/Method_HIV.xlsx'
method_hiv_data = pd.read_excel(file_path, sheet_name=None, header=0)
HIV_prev, HIV_death, HIV_inc, CD4_base, time_CD4, initial_state_probs, \
age_distr = method_hiv_data['prevalence2018'], method_hiv_data['deaths2009_2021'], \
            method_hiv_data['incidence2009_2021'], \
            method_hiv_data['CD4_distribution2018'], method_hiv_data['Time_spent_by_CD4'], \
            method_hiv_data['Initial_state_probs'], method_hiv_data['age_distribution2018']


# this class contains all the methods required to set up the baseline population
class HIV(Module):
    """Models HIV incidence, treatment and AIDS-mortality.

    Methods required:
    * `read_parameters(data_folder)`
    * `initialise_population(population)`
    * `initialise_simulation(sim)`
    * `on_birth(mother, child)`
    """

    # Here we declare parameters for this module. Each parameter has a name, data type,
    # and longer description.
    PARAMETERS = {
        'prob_infant_fast_progressor':
            Parameter(Types.LIST, 'Probabilities that infants are fast or slow progressors'),
        'infant_progression_category':
            Parameter(Types.CATEGORICAL, 'Classification of infants into fast or slow progressors'),
        'exp_rate_mort_infant_fast_progressor':
            Parameter(Types.REAL, 'Exponential rate parameter for mortality in infants fast progressors'),
        'weibull_scale_mort_infant_slow_progressor':
            Parameter(Types.REAL, 'Weibull scale parameter for mortality in infants slow progressors'),
        'weibull_shape_mort_infant_slow_progressor':
            Parameter(Types.REAL, 'Weibull shape parameter for mortality in infants slow progressors'),
        'weibull_shape_mort_adult':
            Parameter(Types.REAL, 'Weibull shape parameter for mortality in adults'),
        'proportion_high_sexual_risk_male':
            Parameter(Types.REAL, 'proportion of men who have high sexual risk behaviour'),
        'proportion_high_sexual_risk_female':
            Parameter(Types.REAL, 'proportion of women who have high sexual risk behaviour'),
        'rr_HIV_high_sexual_risk':
            Parameter(Types.REAL, 'relative risk of acquiring HIV with high risk sexual behaviour'),
        'proportion_on_ART_infectious':
            Parameter(Types.REAL, 'proportion of people on ART contributing to transmission as not virally suppressed'),
    }

    # Next we declare the properties of individuals that this module provides.
    # Again each has a name, type and description. In addition, properties may be marked
    # as optional if they can be undefined for a given individual.
    PROPERTIES = {
        'has_HIV': Property(Types.BOOL, 'HIV status'),
        'date_HIV_infection': Property(Types.DATE, 'Date acquired HIV infection'),
        'date_AIDS_death': Property(Types.DATE, 'Projected time of AIDS death if untreated'),
        'on_ART': Property(Types.BOOL, 'Currently on ART'),
        'date_ART_start': Property(Types.DATE, 'Date ART started'),
        'ART_mortality': Property(Types.REAL, 'Mortality rates whilst on ART'),
        'sexual_risk_group': Property(Types.REAL, 'Relative risk of HIV based on sexual risk high/low'),
        'date_death': Property(Types.DATE, 'Date of death'),
        'CD4_state': Property(Types.INT, ''),

        # should be handled by other modules TODO: remove from here
        'sex': Property(Types.CATEGORICAL, categories=['M', 'F'], description='Male or female'),
        'date_of_birth': Property(Types.DATE, 'Date of birth'),
    }

    # inds = pd.read_csv('Q:/Thanzi la Onse/HIV/initial_pop_dataframe2018.csv')
    # p = inds.shape[0]  # number of rows in pop (# individuals)

    def read_parameters(self, data_folder):
        """Read parameter values from file, if required.
        :param data_folder: path of a folder supplied to the Simulation containing data files.
          Typically modules would read a particular file within here.
        """
        params = self.parameters
        params['prob_infant_fast_progressor'] = [0.36, 1 - 0.36]
        params['infant_progression_category'] = ['FAST', 'SLOW']
        params['exp_rate_mort_infant_fast_progressor'] = 1.08
        params['weibull_scale_mort_infant_slow_progressor'] = 16
        params['weibull_size_mort_infant_slow_progressor'] = 1
        params['weibull_shape_mort_infant_slow_progressor'] = 2.7
        params['weibull_shape_mort_adult'] = 2
        params['proportion_high_sexual_risk_male'] = 0.0913
        params['proportion_high_sexual_risk_female'] = 0.0095
        params['rr_HIV_high_sexual_risk'] = 2
        params['proportion_on_ART_infectious'] = 0.2

    def initialise_population(self, population):
        # TODO: this should be moved to core demography module
        initial_pop_data = pd.read_csv('/Users/tamuri/Documents/2018/thanzi/test/HIV_test_run_data/initial_pop_dataframe2018.csv')
        initial_sex_age: pd.DataFrame = initial_pop_data.groupby(['sex', 'age']).size().reset_index(name='counts')
        initial_sex_age['proportion'] = initial_sex_age.counts / len(initial_pop_data)

        sampled_sex_age = initial_sex_age.sample(n=len(population), weights='proportion', random_state=self.rng, replace=True)
        sampled_sex_age = sampled_sex_age.reset_index(drop=True)

        population.sex = sampled_sex_age['sex']
        population.date_of_birth = sampled_sex_age['age'].apply(lambda x: self.sim.date - DateOffset(years=x))

        population.has_HIV = False
        # population.date_HIV_infection = None
        # population.date_AIDS_death = None
        population.on_ART = False
        # population.date_ART_start = None
        population.ART_mortality = None
        population.sexual_risk_group = 1
        # population.date_death = None

        self.high_risk(population)
        self.prevalence(population)
        self.time_since_infection(population)
        self.initial_pop_deaths(population)

        x: pd.DataFrame = population.props
        x.to_csv("~/Documents/output.csv")

    def get_age(self, date_of_birth):
        return ((self.sim.date - date_of_birth).apply(lambda x: x.days)) / 365.25

    def high_risk(self, population):
        # should this be in initialise population?
        """ Stratify the adult (age >15) population in high or low sexual risk """
        age = self.get_age(population.date_of_birth)
        male_sample = population[(population.sex == 'M') & (age > 15)].sample(
            frac=self.parameters['proportion_high_sexual_risk_male']).index
        female_sample = population[(population.sex == 'F') & (age > 15)].sample(
            frac=self.parameters['proportion_high_sexual_risk_female']).index

        # these individuals have higher risk of hiv
        population[male_sample | female_sample, 'sexual_risk_group'] = self.parameters['rr_HIV_high_sexual_risk']


    # assign infected status using UNAIDS prevalence 2018 by age
    # randomly allocate time since infection according to CD4 distributions from spectrum
    # should do this separately for infants using CD4%
    # then could include the infant fast progressors
    # currently infant fast progressors will always have time to death shorter than time infected

    # HELPER FUNCTION - should these go in class(HIV)?
    def get_index(self, df, has_hiv, sex, age_low, age_high, CD4_state):

        index = df.index[
            (df.has_hiv == 1) &
            (df.sex == sex) &
            (df.age >= age_low) & (df.age < age_high) &
            (df.CD4_state == CD4_state)]

        return index

    def prevalence(self, population):
        """Set our property values for the initial population.

        This method is called by the simulation when creating the initial population, and is
        responsible for assigning initial values, for every individual, of those properties
        'owned' by this module, i.e. those declared in the PROPERTIES dictionary above.

        :param population: the population of individuals
        """

        now = self.sim.date

        # normalise the prevalence TODO: can this be in the excel file?
        prevalence = HIV_prev.loc[HIV_prev.year == now.year, ['age', 'sex', 'prevalence']]
        initial_pop_data = pd.read_csv('/Users/tamuri/Documents/2018/thanzi/test/HIV_test_run_data/initial_pop_dataframe2018.csv')
        initial_sex_age: pd.DataFrame = initial_pop_data.groupby(['sex', 'age']).size().reset_index(name='counts')
        prevalence = pd.merge(initial_sex_age, prevalence, how='inner', left_on=['sex', 'age'], right_on=['sex','age'])
        prevalence['proportion'] = prevalence.prevalence / 100 / prevalence.counts

        age = self.get_age(population.date_of_birth)

        for i in range(0, 81):
            # male
            # scale high/low-risk probabilities to sum to 1 for each sub-group
            idx = (age == i) & (population.sex == 'M')

            if idx.any():
                prob_i = population[idx, 'sexual_risk_group']

                # sample from uninfected df using prevalence from UNAIDS
                fraction_infected = prevalence.loc[(prevalence.sex == 'M') & (prevalence.age == i), 'proportion']
                male_infected = population[idx].sample(frac=fraction_infected, weights=prob_i).index
                population[male_infected, 'has_HIV'] = True

            # scale high/low-risk probabilities to sum to 1 for each sub-group
            idx = (age == i) & (population.sex == 'F')

            if idx.any():
                prob_i = population[idx, 'sexual_risk_group']

                # sample from uninfected df using prevalence from UNAIDS
                fraction_infected = prevalence.loc[(prevalence.sex == 'F') & (prevalence.age == i), 'proportion']
                female_infected = population[idx].sample(frac=fraction_infected, weights=prob_i).index
                population[female_infected, 'has_HIV'] = True


    def time_since_infection(self, population):
        now = self.sim.date

        cd4_states = [500, 350, 250, 200, 100, 50, 0]

        population.props['age'] = self.get_age(population.date_of_birth)
        population.props['age_int'] = population.props['age'].astype(int)

        for sex in ['M', 'F']:
            idx = (population.has_HIV == True) & (population.sex == sex)
            cd4_probs = CD4_base.loc[CD4_base.sex == sex, 'CD4_distribution2018']
            population.props.loc[idx, 'CD4_state'] = np.random.choice(cd4_states,
                                                                      size=idx.sum(),
                                                                      replace=True,
                                                                      p=cd4_probs.values)

        population.props['CD4_state'] = population.props['CD4_state'].astype(int)

        time_infected_lookup = dict()

        def add(row):
            time_infected_lookup[(row.CD4_state, row.sex, row.age)] = [(row.years1, row.years2),
                                                                       (row.prob1, row.prob2)]

        pd.read_csv('/Users/tamuri/Documents/2018/thanzi/test/HIV_test_run_data/cd4_unrolled.csv'
                    ).apply(lambda row: add(row), axis=1)

        def get_time_infected(row):
            lookup = time_infected_lookup[(row.CD4_state, row.sex, row.age)]
            return np.random.choice(lookup[0], p=lookup[1])

        time_infected: pd.Series = population[population.has_HIV].apply(lambda row: get_time_infected(row), axis=1)

        # convert to date in the past
        time_infected = time_infected.apply(lambda x: (now - DateOffset(days=int(x * 365.25))))
        population.props.loc[time_infected.index, 'date_HIV_infection'] = time_infected

        # remove temporary columns 'time_infected' and 'CD4_states'
        # del population.props['CD4_state']

        # check time infected is less than time alive (especially for infants)
        # tmp = population.props.index[(pd.notna(population.date_HIV_infection)) & ((current_time - population.date_HIV_infection).years > population.age)]
        early_doi = ((now - time_infected).dt.days / 365.25) > population.props.loc[time_infected.index, 'age'] # time infected earlier than time alive!

        if early_doi.any():
            tmp2 = now - DateOffset(years=population.props.loc[early_doi, 'age'])
            population.props.loc[early_doi, 'date_HIV_infection'] = tmp2  # replace with year of birth


    # this function needs the ART mortality rates from ART.py
    def initial_pop_deaths(self, df):

        current_time = self.sim.date

        params = self.parameters
        df.props['age'] = self.get_age(df.date_of_birth)

        # PAEDIATRIC time of death - untreated
        hiv_inf = df.props.index[(df.has_HIV) & (df.on_ART == 0) & (df.age < 3)]

        # need a two parameter Weibull with size parameter, multiply by scale instead
        time_of_death_slow = np.random.weibull(a=params['weibull_size_mort_infant_slow_progressor'],
                                               size=len(hiv_inf)) * params['weibull_scale_mort_infant_slow_progressor']


        # while time of death is shorter than time infected keep redrawing (only for the entries that need it)
        while np.any(
            time_of_death_slow < (current_time - df.props.loc[hiv_inf, 'date_HIV_infection'])):  # if any condition=TRUE for any rows

            redraw = np.argwhere(
                time_of_death_slow < (current_time - df.props.loc[self.hiv_inf, 'date_HIV_infection']))
            redraw2 = redraw.ravel()

            if len(redraw) == 0:
                break

            # redraw time of death
            time_of_death_slow[redraw2] = np.random.weibull(a=params['weibull_size_mort_infant_slow_progressor'],
                                                            size=len(redraw2)) * \
                params['weibull_scale_mort_infant_slow_progressor']

        # subtract time already spent
        time_of_death_slow = pd.to_timedelta(time_of_death_slow * 365.25, unit='d')
        # print(df.props.date_AIDS_death.dtype)
        # df.props.loc[hiv_inf, 'date_AIDS_death'] = (current_time + time_of_death_slow) - (current_time - df.props.loc[hiv_inf, 'date_HIV_infection'])
        # print(df.props.date_AIDS_death.dtype)

        # ADULT time of death, adults are all those aged >3 for untreated mortality rates
        hiv_ad = df.props.index[(df.has_HIV) & (df.on_ART == 0) & (df.age >= 3)]

        time_of_death = np.random.weibull(a=params['weibull_shape_mort_adult'], size=len(hiv_ad)) * \
            np.exp(log_scale(df.props.loc[hiv_ad, 'age']))

        time_of_death = pd.to_timedelta(time_of_death * 365.25, unit='d')

        # while time of death is shorter than time infected keep redrawing (only for entries that need it)
        while np.any(
            time_of_death < (current_time - df.props.loc[hiv_ad, 'date_HIV_infection'])):  # if any condition=TRUE for any rows


            redraw = np.argwhere(
                time_of_death < (current_time - df.props.loc[hiv_ad, 'date_HIV_infection']))
            redraw2 = redraw.ravel()

            if len(redraw) < 10:  # this condition needed for older people with long time since infection
                break

            age_index = hiv_ad[redraw2]

            time_of_death[redraw2] = np.random.weibull(a=params['weibull_shape_mort_adult'], size=len(redraw2)) * \
                np.exp(log_scale(df.props.loc[age_index, 'age']))

        # drop the hours
        time_of_death = pd.to_timedelta(time_of_death.dt.days, unit='d')

        # subtract time already spent
        df.props.loc[hiv_ad, 'date_AIDS_death'] = current_time + time_of_death # - \
        df.props.loc[hiv_ad, 'date_AIDS_death'] = df.props.loc[hiv_ad, 'date_HIV_infection'] + time_of_death # - \
       #§§h(current_time - df.props.loc[hiv_ad, 'date_HIV_infection'])

        # assign mortality rates on ART
        # NOTE: see ART.py
        # df['ART_mortality'] = self.ART_mortality_rates(df, self.current_time)

    def initialise_simulation(self, sim):
        """Get ready for simulation start.

        This method is called just before the main simulation loop begins, and after all
        modules have read their parameters and the initial population has been created.
        It is a good place to add initial events to the event queue.
        """
        event = WritePopEvent()
        sim.schedule_event(event, DateOffset(months=1))
        # raise NotImplementedError

    def on_birth(self, mother, child):
        """Initialise our properties for a newborn individual.

        This is called by the simulation whenever a new person is born.

        :param mother: the mother for this child
        :param child: the new child
        """
        raise NotImplementedError


class WritePopEvent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        super().__init__(module, frequency=DateOffset(months=12))

    def apply(self, population):
        p: pd.DataFrame = population.props
        s = sum(population.has_HIV)
        p.to_csv('~/Documents/population_' + self.sim.date + '.csv')




class HIV_Event(RegularEvent, PopulationScopeEventMixin):
    """HIV infection events

    Regular events automatically reschedule themselves at a fixed frequency,
    and thus implement discrete timestep type behaviour. The frequency is
    specified when calling the base class constructor in our __init__ method.
    """

    def __init__(self, module):
        """One line summary here

        We need to pass the frequency at which we want to occur to the base class
        constructor using super(). We also pass the module that created this event,
        so that random number generators can be scoped per-module.

        :param module: the module that created this event
        """
        super().__init__(module, frequency=DateOffset(months=1))

    def HIV_infection_adults(self, df, current_time, beta_ad):
        """Apply this event to the population.

        :param population: the current population
        """
        self.current_time = current_time

        params = self.module.parameters

        infected = len(
            df[(df.has_HIV == 1) & (df.on_ART == 0) & (
                df.age >= 15)])  # number infected untreated

        h_infected = params['proportion_on_ART_infectious'] * len(
            df[(df.has_HIV == 1) & (df.on_ART == 1) & (
                df.age >= 15)])  # number infected treated

        total_pop = len(df[(df.age >= 15)])  # whole df over 15 years

        foi = beta_ad * ((infected + h_infected) / total_pop)  # force of infection for adults

        # distribute FOI by age
        foi_m = foi * age_distr['age_distribution'][
            (age_distr.year == self.current_time) & (age_distr.sex == 'M')]  # age 15-80+
        foi_f = foi * age_distr['age_distribution'][(age_distr.year == self.current_time) & (age_distr.sex == 'F')]

        for i in range(66):  # ages 15-80
            age_value = i + 14  # adults only FOI

            # males
            susceptible_age = len(
                df[(df.age == age_value) & (df.sex == 'M') & (df.onART == 0)])

            # to determine number of new infections by age
            tmp1 = np.random.binomial(1, p=foi_m[i], size=susceptible_age)

            # allocate infections to people with high/low risk
            # scale high/low-risk probabilities to sum to 1 for each sub-group
            risk = df['sexual_risk_group'][
                       (df.age == age_value) & (df.sex == 'M') & (df.has_HIV == 0)] / \
                np.sum(df['sexual_risk_group'][
                              (df.age == age_value) & (df.sex == 'M') & (df.has_HIV == 0)])

            tmp2 = np.random.choice(
                df.index[(df.age == age_value) & (df.sex == 'M') & (df.has_HIV == 0)],
                size=len(tmp1), p=risk, replace=False)

            df.loc[tmp2, 'has_HIV'] = 1  # change status to infected
            df.loc[tmp2, 'date_HIV_infection'] = self.current_time

            df.loc[tmp2, 'date_AIDS_death'] = self.current_time + (
                np.random.weibull(a=params['weibull_shape_mort_adult'], size=len(tmp2)) * np.exp(
                    log_scale(df.age.iloc[tmp2])))

            # females
            susceptible_age = len(
                df[(df.age == age_value) & (df.sex == 'F') & (df.has_HIV == 0)])

            # to determine number of new infections by age
            tmp3 = np.random.binomial(1, p=foi_f[i], size=susceptible_age)

            # allocate infections to people with high/low risk
            # scale high/low-risk probabilities to sum to 1 for each sub-group
            risk = df['sexual_risk_group'][
                       (df.age == age_value) & (df.sex == 'F') & (df.has_HIV == 0)] / \
                np.sum(df['sexual_risk_group'][
                              (df.age == age_value) & (df.sex == 'F') & (df.has_HIV == 0)])

            tmp4 = np.random.choice(
                df.index[(df.age == age_value) & (df.sex == 'F') & (df.has_HIV == 0)],
                size=len(tmp3), p=risk, replace=False)

            df.loc[tmp4, 'has_HIV'] = 0  # change status to infected
            df.loc[tmp4, 'date_HIV_infection'] = self.current_time

            df.loc[tmp2, 'date_AIDS_death'] = self.current_time + (
                np.random.weibull(a=params['weibull_shape_mort_adult'], size=len(tmp4)) * np.exp(
                    log_scale(df.age.iloc[tmp4])))

        return df

    # run the death functions once a year
    def AIDS_death(self, df, current_time):
        self.current_time = current_time

        # choose which ones die at current_time
        current_time_int = int(round(self.current_time))  # round current_time to nearest year

        tmp = df.index[(round(df.date_AIDS_death) == current_time_int) & (df.on_ART == 0)]

        df.loc[tmp, 'date_death'] = self.current_time

        return df

    def AIDS_death_on_ART(self, df):
        tmp1 = np.random.uniform(low=0, high=1, size=df.shape[0])  # random number for every entry

        tmp2 = df.index[(pd.notna(df.ART_mortality)) & (tmp1 < df['mortality']) &
                        (df.has_HIV == 1) & (df.on_ART == 1)]

        df.loc[tmp2, 'date_death'] = self.current_time

        return df

# # to set up the baseline population
# inds.head(20)
# inds.describe(include='all')
# inds['status'].value_counts()
#
# inds = high_risk_inds(inds)
# inds = prevalence_inds(inds, current_time)
# inds = initART_inds(inds, current_time)
# inds = init_death_inds(inds, current_time)
#
# inds.head(20)
# inds.describe(include='all')
# inds['status'].value_counts()
# inds['treat'].value_counts()
#
# # to run projections set up these commands in a loop
# # example for 2019 projections with placeholder transmission rate beta_ad=0.12
# inds = inf_inds_ad(inds, 2019, 0.12)
# inds = ART_inds(inds, 2019)
# inds = killHIV_inds(inds, 2019)
# inds = killHIV_ART_inds(inds)
#
# inds.describe(include='all')
# inds['status'].value_counts()
# inds['treat'].value_counts()

# TODO: handle births, link child's risk of HIV to mother's HIV status
# TODO: include cotrimoxazole for children
# TODO: code FOI as separate function from infection function
# TODO: incorporate TB as a risk for HIV progression/mortality
