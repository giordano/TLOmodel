"""
A skeleton template for disease methods.
"""
import pandas as pd

from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import PopulationScopeEventMixin, RegularEvent, Event, IndividualScopeEventMixin
from tlo.population import logger


# ---------------------------------------------------------------------------------------------------------
#   MODULE DEFINITIONS
# ---------------------------------------------------------------------------------------------------------

class Skeleton(Module):
    """
    One line summary goes here...

    All disease modules need to be implemented as a class inheriting from Module.
    They need to provide several methods which will be called by the simulation
    framework:
    * `read_parameters(data_folder)`
    * `initialise_population(population)`
    * `initialise_simulation(sim)`
    * `on_birth(mother, child)`

    And, if the module represents a disease:
    * It must register itself: self.sim.modules['HealthSystem'].register_disease_module(self)
    * `query_symptoms_now(self)`
    * `report_qaly_values(self)`
    * `on_healthsystem_interaction(self, person_id, cue_type=None, disease_specific=None)`

    If this module represents a form of treatment:
    * TREATMENT_ID: must be defined
    * It must register the treatment: self.sim.modules['HealthSystem'].register_interventions(footprint_for_treatment)
    """

    # Here we declare parameters for this module. Each parameter has a name, data type,
    # and longer description.
    PARAMETERS = {
        'parameter_a': Parameter(
            Types.REAL, 'Description of parameter a'),
    }

    # Next we declare the properties of individuals that this module provides.
    # Again each has a name, type and description. In addition, properties may be marked
    # as optional if they can be undefined for a given individual.
    PROPERTIES = {
        'property_a': Property(Types.BOOL, 'Description of property a'),
    }

    def read_parameters(self, data_folder):
        """Read parameter values from file, if required.

        Here we do nothing.

        :param data_folder: path of a folder supplied to the Simulation containing data files.
          Typically modules would read a particular file within here.
        """
        pass

    def initialise_population(self, population):
        """Set our property values for the initial population.

        This method is called by the simulation when creating the initial population, and is
        responsible for assigning initial values, for every individual, of those properties
        'owned' by this module, i.e. those declared in the PROPERTIES dictionary above.

        :param population: the population of individuals
        """
        raise NotImplementedError

    def initialise_simulation(self, sim):
        """Get ready for simulation start.

        This method is called just before the main simulation loop begins, and after all
        modules have read their parameters and the initial population has been created.
        It is a good place to add initial events to the event queue.

        If this is a disease module, register this disease module with the healthsystem:
        e.g. self.sim.modules['HealthSystem'].register_disease_module(self)"

        If this is an interveton module: register the footprints with the healthsystem:
        e.g.    footprint_for_treatment = pd.DataFrame(index=np.arange(1), data={
                 'Name': self.TREATMENT_ID,
                 'Nurse_Time': 5,
                 'Doctor_Time': 10,
                 'Electricity': False,
                 'Water': False})
             self.sim.modules['HealthSystem'].register_interventions(footprint_for_treatment)

        """

        raise NotImplementedError

    def on_birth(self, mother_id, child_id):
        """Initialise our properties for a newborn individual.

        This is called by the simulation whenever a new person is born.

        :param mother_id: the mother for this child
        :param child_id: the new child
        """
        raise NotImplementedError

    def query_symptoms_now(self):
        """
        If this is a registered disease module, this is called by the HealthCareSeekingPoll in order to determine the
        healthlevel of each person. It can be called at any time and must return a Series with length equal to the
        number of persons alive and index matching sim.population.props. The entries encode the symptoms on the
        following "unified symptom scale":
        0=None; 1=Mild; 2=Moderate; 3=Severe; 4=Extreme_Emergency
        """

        raise NotImplementedError

    def report_daly_values(self):
        # This must send back a pd.Series or pd.DataFrame that reports on the average daly-weights that have been
        # experienced by persons in the previous month. Only rows for alive-persons must be returned.
        # The names of the series of columns is taken to be the label of the cause of this disability.
        # It will be recorded by the healthburden module as <ModuleName>_<Cause>.

        raise NotImplementedError


    def on_hsi_alert(self, person_id, treatment_id):
        """
        This is called whenever there is an HSI event commissioned by one of the other disease modules.
        """

        raise NotImplementedError


# ---------------------------------------------------------------------------------------------------------
#   DISEASE MODULE EVENTS
# ---------------------------------------------------------------------------------------------------------

class Skeleton_Event(RegularEvent, PopulationScopeEventMixin):
    """A skeleton class for an event

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

    def apply(self, population):
        """Apply this event to the population.

        :param population: the current population
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------------------------------------
#   LOGGING EVENTS
# ---------------------------------------------------------------------------------------------------------

class Skeleton_LoggingEvent(RegularEvent, PopulationScopeEventMixin):
    def __init__(self, module):
        """Produce a summary of the numbers of people with respect to the action of this module.
        This is a regular event that can output current states of people or cumulative events since last logging event.
        """

        # run this event every year
        self.repeat = 12
        super().__init__(module, frequency=DateOffset(months=self.repeat))

    def apply(self, population):
        # Make some summary statitics


        dict_to_output={
            'Metric_One': 1.0,
            'Metric_Two': 2.0
        }

        logger.info('%s|summary_12m|%s', self.sim.date, dict_to_output)


# ---------------------------------------------------------------------------------------------------------
#   HEALTH SYSTEM INTERACTION EVENTS
# ---------------------------------------------------------------------------------------------------------

class HSI_Skeleton_Example_Interaction(Event, IndividualScopeEventMixin):
    """This is a Health System Interaction Event. An interaction with the healthsystem are encapsulated in events
    like this.
    It must begin HSI_<Module_Name>_Description
    """

    def __init__(self, module, person_id):
        super().__init__(module, person_id=person_id)

        # Define the call on resources of this treatment event: Time of Officers (Appointments)
        #   - get an 'empty' footprint:
        the_appt_footprint = self.sim.modules['HealthSystem'].get_blank_appt_footprint()
        #   - update to reflect the appointments that are required
        the_appt_footprint['Over5OPD'] = 1  # This requires one out patient

        # Define the call on resources of this treatment event: Consumables
        #   - get a blank consumables footprint
        the_cons_footprint = self.sim.modules['HealthSystem'].get_blank_cons_footprint()
        #   - update with any consumables that are needed. Look in ResourceFile_Consumables.csv

        # Define the facilities at which this event can occur
        #   - this will find all the available facility levels
        the_accepted_facility_levels = \
            list(pd.unique(self.sim.modules['HealthSystem'].parameters['Facilities_For_Each_District'] \
                               ['Facility_Level']))

        # Define the necessary information for an HSI
        self.TREATMENT_ID = 'Skeleton_Example_Interaction'  # This must begin with the module name
        self.APPT_FOOTPRINT = the_appt_footprint
        self.CONS_FOOTPRINT = the_cons_footprint
        self.ACCEPTED_FACILITY_LEVELS = the_accepted_facility_levels
        self.ALERT_OTHER_DISEASES = []

    def apply(self, person_id):
        """ Do the action that take place in this health system interaction. """
        pass


