# """
# Remaining to do:
# - streamline input arguments
# - let the level of the appointment be in the log
# - let the logger give times of each hcw
# """
import datetime
import heapq as hp
import itertools
from collections import Counter, defaultdict
from collections.abc import Iterable
from itertools import repeat
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple, Union

import numpy as np
import pandas as pd

import tlo
from tlo import Date, DateOffset, Module, Parameter, Property, Types, logging
from tlo.analysis.utils import flatten_multi_index_series_into_dict_for_logging
from tlo.events import Event, PopulationScopeEventMixin, Priority, RegularEvent
from tlo.methods import Metadata
from tlo.methods.bed_days import BedDays
from tlo.methods.consumables import (
    Consumables,
    get_item_code_from_item_name,
    get_item_codes_from_package_name,
)
from tlo.methods.dxmanager import DxManager

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

logger_summary = logging.getLogger(f"{__name__}.summary")
logger_summary.setLevel(logging.INFO)


class FacilityInfo(NamedTuple):
    """Information about a specific health facility."""
    id: int
    name: str
    level: str
    region: str


class AppointmentSubunit(NamedTuple):
    """Component of an appointment relating to a specific officer type."""
    officer_type: str
    time_taken: float


class HSIEventDetails(NamedTuple):
    """Non-target specific details of a health system interaction event."""
    event_name: str
    module_name: str
    treatment_id: str
    facility_level: Optional[str]
    appt_footprint: Tuple[str]
    beddays_footprint: Tuple[Tuple[str, int]]


class HSIEventQueueItem(NamedTuple):
    """Properties of event added to health system queue.

    The order of the attributes in the tuple is important as the queue sorting is done
    by the order of the items in the tuple, i.e. first by `priority`, then `topen` and
    so on.
    """
    priority: int
    topen: Date
    queue_counter: int
    tclose: Date
    # Define HSI_Event type as string to avoid NameError exception as HSI_Event defined
    # later in module (see https://stackoverflow.com/a/36286947/4798943)
    hsi_event: 'HSI_Event'


class HSI_Event:
    """Base HSI event class, from which all others inherit.

    Concrete subclasses should also inherit from one of the EventMixin classes
    defined below, and implement at least an `apply` and `did_not_run` method.
    """

    def __init__(self, module, *args, **kwargs):
        """Create a new event.

        Note that just creating an event does not schedule it to happen; that
        must be done by calling Simulation.schedule_event.

        :param module: the module that created this event.
            All subclasses of Event take this as the first argument in their
            constructor, but may also take further keyword arguments.
        """
        self.module = module
        self.sim = module.sim
        self.target = None  # Overwritten by the mixin
        super().__init__(*args, **kwargs)  # Call the mixin's constructors

        # Defaults for the HSI information:
        self.TREATMENT_ID = ''
        self.EXPECTED_APPT_FOOTPRINT = self.make_appt_footprint({})
        self.ACCEPTED_FACILITY_LEVEL = None
        self.BEDDAYS_FOOTPRINT = self.make_beddays_footprint({})

        # Information received about this HSI:
        self._received_info_about_bed_days = None
        self.expected_time_requests = {}
        self.facility_info = None

    @property
    def bed_days_allocated_to_this_event(self):
        if self._received_info_about_bed_days is None:
            # default to the footprint if no information about bed-days is received
            return self.BEDDAYS_FOOTPRINT

        return self._received_info_about_bed_days

    def apply(self, squeeze_factor=0.0, *args, **kwargs):
        """Apply this event to the population.

        Must be implemented by subclasses.

        """
        raise NotImplementedError

    def did_not_run(self, *args, **kwargs):
        """Called when this event is due but it is not run. Return False to prevent the event being rescheduled, or True
        to allow the rescheduling. This is called each time that the event is tried to be run but it cannot be.
        """
        logger.debug(key="message", data=f"{self.__class__.__name__}: did not run.")
        return True

    def never_ran(self):
        """Called when this event is was entered to the HSI Event Queue, but was never run.
        """
        logger.debug(key="message", data=f"{self.__class__.__name__}: was never run.")

    def post_apply_hook(self):
        """Impose the bed-days footprint (if target of the HSI is a person_id)"""
        if isinstance(self.target, int):
            self.module.sim.modules['HealthSystem'].bed_days.impose_beddays_footprint(
                person_id=self.target,
                footprint=self.bed_days_allocated_to_this_event
            )

    def run(self, squeeze_factor):
        """Make the event happen."""
        updated_appt_footprint = self.apply(self.target, squeeze_factor)
        self.post_apply_hook()
        return updated_appt_footprint

    def get_consumables(self,
                        item_codes: Union[None, np.integer, int, list, set, dict] = None,
                        optional_item_codes: Union[None, np.integer, int, list, set, dict] = None,
                        to_log: Optional[bool] = True,
                        return_individual_results: Optional[bool] = False
                        ) -> Union[bool, dict]:
        """Function to allow for getting and checking of entire set of consumables. All requests for consumables should
        use this function.
        :param item_codes: The item code(s) (and quantities) of the consumables that are requested and which determine
        the summary result for availability/non-availability. This can be an `int` (the item_code needed [assume
        quantity=1]), a `list` or `set` (the collection  of item_codes [for each assuming quantity=1]), or a `dict`
        (with key:value pairs `<item_code>:<quantity>`).
        :param optional_item_codes: The item code(s) (and quantities) of the consumables that are requested and which do
         not determine the summary result for availability/non-availability. (Same format as `item_codes`). This is
         useful when a large set of items may be used, but the viability of a subsequent operation depends only on a
         subset.
        :param return_individual_results: If True returns a `dict` giving the availability of each item_code requested
        (otherwise gives a `bool` indicating if all the item_codes requested are available).
        :param to_log: If True, logs the request.
        :returns A `bool` indicating whether every item is available, or a `dict` indicating the availability of each
         item.
        Note that disease module can use the `get_item_codes_from_package_name` and `get_item_code_from_item_name`
         methods in the `HealthSystem` module to find item_codes.
        """

        def _return_item_codes_in_dict(item_codes: Union[None, np.integer, int, list, set, dict]) -> dict:
            """Convert an argument for 'item_codes` (provided as int, list, set or dict) into the format
            dict(<item_code>:quantity)."""

            if item_codes is None:
                return {}

            if isinstance(item_codes, (int, np.integer)):
                return {int(item_codes): 1}

            elif isinstance(item_codes, list):
                if not all([isinstance(i, (int, np.integer)) for i in item_codes]):
                    raise ValueError("item_codes must be integers")
                return {int(i): 1 for i in item_codes}

            elif isinstance(item_codes, dict):
                if not all(
                    [(isinstance(code, (int, np.integer)) and
                      isinstance(quantity, (float, np.floating, int, np.integer)))
                     for code, quantity in item_codes.items()]
                ):
                    raise ValueError("item_codes must be integers and quantities must be integers or floats.")
                return {int(i): float(q) for i, q in item_codes.items()}

            else:
                raise ValueError("The item_codes are given in an unrecognised format")

        hs_module = self.sim.modules['HealthSystem']

        _item_codes = _return_item_codes_in_dict(item_codes)
        _optional_item_codes = _return_item_codes_in_dict(optional_item_codes)

        # Determine if the request should be logged (over-ride argument provided if HealthSystem is disabled).
        _to_log = to_log if not hs_module.disable else False

        # Checking the availability and logging:
        rtn = hs_module.consumables._request_consumables(item_codes={**_item_codes, **_optional_item_codes},
                                                         to_log=_to_log,
                                                         facility_info=self.facility_info,
                                                         treatment_id=self.TREATMENT_ID)

        # Return result in expected format:
        if not return_individual_results:
            # Determine if all results for all the `item_codes` are True (discarding results from optional_item_codes).
            return all(v for k, v in rtn.items() if k in _item_codes)
        else:
            return rtn

    def make_beddays_footprint(self, dict_of_beddays):
        """Helper function to make a correctly-formed 'bed-days footprint'"""

        # get blank footprint
        footprint = self.sim.modules['HealthSystem'].bed_days.get_blank_beddays_footprint()

        # do checks on the dict_of_beddays provided.
        assert isinstance(dict_of_beddays, dict)
        assert all((k in footprint.keys()) for k in dict_of_beddays.keys())
        assert all(isinstance(v, (float, int)) for v in dict_of_beddays.values())

        # make footprint (defaulting to zero where a type of bed-days is not specified)
        for k, v in dict_of_beddays.items():
            footprint[k] = v

        return footprint

    def is_all_beddays_allocated(self):
        """Check if the entire footprint requested is allocated"""
        return all(
            self.bed_days_allocated_to_this_event[k] == self.BEDDAYS_FOOTPRINT[k] for k in self.BEDDAYS_FOOTPRINT
        )

    def make_appt_footprint(self, dict_of_appts):
        """Helper function to make appointment footprint in format expected downstream.

        Should be passed a dictionary keyed by appointment type codes with non-negative
        values.
        """
        health_system = self.sim.modules['HealthSystem']
        if health_system.appt_footprint_is_valid(dict_of_appts):
            return Counter(dict_of_appts)

        raise ValueError(
            "Argument to make_appt_footprint should be a dictionary keyed by "
            "appointment type code strings in Appt_Types_Table with non-negative "
            "values"
        )

    def initialise(self):
        """Initialise the HSI:
        * Set the facility_info
        * Compute appt-footprint time requirements
        """
        health_system = self.sim.modules['HealthSystem']

        if not isinstance(self.target, tlo.population.Population):
            self.facility_info = health_system.get_facility_info(self)

            # If there are bed-days specified, add (if needed) the in-patient admission and in-patient day Appointment
            # Types.
            # (HSI that require a bed for one or more days always need such appointments, but this may have been
            # missed in the declaration of the `EXPECTED_APPPT_FOOTPRINT` in the HSI.)
            # NB. The in-patient day Apppointment time is automatically applied on subsequent days.
            if sum(self.BEDDAYS_FOOTPRINT.values()):
                self.EXPECTED_APPT_FOOTPRINT = health_system.bed_days.add_first_day_inpatient_appts_to_footprint(
                    self.EXPECTED_APPT_FOOTPRINT)

            # Write the time requirements for staff of the appointments to the HSI:
            self.expected_time_requests = health_system.get_appt_footprint_as_time_request(
                facility_info=self.facility_info,
                appt_footprint=self.EXPECTED_APPT_FOOTPRINT,
            )

        # Do checks
        _ = self._check_if_appt_footprint_can_run()

    def _check_if_appt_footprint_can_run(self):
        """Check that event (if individual level) is able to run with this configuration of officers (i.e. check that
        this does not demand officers that are _never_ available), and issue warning if not."""
        health_system = self.sim.modules['HealthSystem']
        if not isinstance(self.target, tlo.population.Population):
            if health_system._officers_with_availability.issuperset(self.expected_time_requests.keys()):
                return True
            else:
                logger.warning(
                    key="message",
                    data=(f"The expected footprint of {self.TREATMENT_ID} is not possible with the configuration of "
                          f"officers.")
                )
                return False


class HSIEventWrapper(Event):
    """This is wrapper that contains an HSI event.

    It is used:
     1) When the healthsystem is in mode 'disabled=True' such that HSI events sent to the health system scheduler are
     passed to the main simulation scheduler for running on the date of `topen`. (Note, it is run with
     squeeze_factor=0.0.)
     2) When the healthsytsem is in mode `diable_and_reject_all=True` such that HSI are not run but the `never_ran`
     method is run on the date of `tclose`.
     3) When an HSI has been submitted to `schedule_hsi_event` but the service is not available.
    """

    def __init__(self, hsi_event, run_hsi=True, *args, **kwargs):
        super().__init__(hsi_event.module, *args, **kwargs)
        self.hsi_event = hsi_event
        self.target = hsi_event.target
        self.run_hsi = run_hsi  # True to call the HSI's `run` method; False to call the HSI's `never_ran` method

    def run(self):
        """Do the appropriate action on the HSI event"""

        # Check that the person is still alive (this check normally happens in the HealthSystemScheduler and silently
        # do not run the HSI event)

        if isinstance(self.hsi_event.target, tlo.population.Population) or (
            self.hsi_event.module.sim.population.props.at[self.hsi_event.target, 'is_alive']
        ):

            if self.run_hsi:
                # Run the event (with 0 squeeze_factor) and ignore the output
                _ = self.hsi_event.run(squeeze_factor=0.0)
            else:
                self.hsi_event.never_ran()


def _accepts_argument(function: callable, argument: str) -> bool:
    """Helper to test if callable object accepts an argument with a given name.

    Compared to using `inspect.signature` or `inspect.getfullargspec` the approach here
    has significantly less overhead (as a full `Signature` or `FullArgSpec` object
    does not need to constructed) but is also less readable hence why it has been
    wrapped as a helper function despite being only one-line to make its functionality
    more obvious.

    :param function: Callable object to check if argument is present in.
    :param argument: Name of argument to check.
    :returns: ``True`` is ``argument`` is an argument of ``function`` else ``False``.
    """
    # co_varnames include both arguments to function and any internally defined variable
    # names hence we check only in the first `co_argcount` items which correspond to
    # just the arguments
    return argument in function.__code__.co_varnames[:function.__code__.co_argcount]


class HealthSystem(Module):
    """
    This is the Health System Module.
    The execution of all health systems interactions are controlled through this module.
    """

    INIT_DEPENDENCIES = {'Demography'}

    PARAMETERS = {
        # Organization of the HealthSystem
        'Master_Facilities_List': Parameter(Types.DATA_FRAME, 'Listing of all health facilities.'),

        # Definitions of the officers and appointment types
        'Officer_Types_Table': Parameter(Types.DATA_FRAME, 'The names of the types of health workers ("officers")'),
        'Appt_Types_Table': Parameter(Types.DATA_FRAME, 'The names of the type of appointments with the health system'),
        'Appt_Offered_By_Facility_Level': Parameter(
            Types.DATA_FRAME, 'Table indicating whether or not each appointment is offered at each facility level.'),
        'Appt_Time_Table': Parameter(Types.DATA_FRAME,
                                     'The time taken for each appointment, according to officer and facility type.'),

        # Capabilities of the HealthSystem (under alternative assumptions)
        'Daily_Capabilities_actual': Parameter(
            Types.DATA_FRAME, 'The capabilities (minutes of time available of each type of officer in each facility) '
                              'based on the _estimated current_ number and distribution of staff estimated.'),
        'Daily_Capabilities_funded': Parameter(
            Types.DATA_FRAME, 'The capabilities (minutes of time available of each type of officer in each facility) '
                              'based on the _potential_ number and distribution of staff estimated (i.e. those '
                              'positions that can be funded).'),
        'Daily_Capabilities_funded_plus': Parameter(
            Types.DATA_FRAME, 'The capabilities (minutes of time available of each type of officer in each facility) '
                              'based on the _potential_ number and distribution of staff estimated, with adjustments '
                              'to permit each appointment type that should be run at facility level to do so in every '
                              'district.'),

        # Consumables
        'item_and_package_code_lookups': Parameter(
            Types.DATA_FRAME, 'Data imported from the OneHealth Tool on consumable items, packages and costs.'),
        'availability_estimates': Parameter(
            Types.DATA_FRAME, 'Estimated availability of consumables in the LMIS dataset.'),
        'cons_availability': Parameter(
            Types.STRING,
            "Availability of consumables. If 'default' then use the availability specified in the ResourceFile; if "
            "'none', then let no consumable be  ever be available; if 'all', then all consumables are always available."
            " When using 'all' or 'none', requests for consumables are not logged. NB. This parameter is over-ridden"
            "if an argument is provided to the module initialiser."),

        # Infrastructure and Equipment
        'BedCapacity': Parameter(
            Types.DATA_FRAME, "Data on the number of beds available of each type by facility_id"),
        'beds_availability': Parameter(
            Types.STRING,
            "Availability of beds. If 'default' then use the availability specified in the ResourceFile; if "
            "'none', then let no beds be  ever be available; if 'all', then all beds are always available. NB. This "
            "parameter is over-ridden if an argument is provided to the module initialiser."),

        # Service Availability
        'Service_Availability': Parameter(
            Types.LIST, 'List of services to be available. NB. This parameter is over-ridden if an argument is provided'
                        ' to the module initialiser.')
    }

    PROPERTIES = {
        'hs_is_inpatient': Property(
            Types.BOOL, 'Whether or not the person is currently an in-patient at any medical facility'
        ),
    }

    def __init__(
        self,
        name: Optional[str] = None,
        resourcefilepath: Optional[Path] = None,
        service_availability: Optional[List[str]] = None,
        mode_appt_constraints: int = 0,
        cons_availability: Optional[str] = None,
        beds_availability: Optional[str] = None,
        ignore_priority: bool = False,
        capabilities_coefficient: Optional[float] = None,
        use_funded_or_actual_staffing: Optional[str] = 'funded_plus',
        disable: bool = False,
        disable_and_reject_all: bool = False,
        store_hsi_events_that_have_run: bool = False,
        record_hsi_event_details: bool = False,
        compute_squeeze_factor_to_district_level: bool = True,
    ):
        """
        :param name: Name to use for module, defaults to module class name if ``None``.
        :param resourcefilepath: Path to directory containing resource files.
        :param service_availability: A list of treatment IDs to allow.
        :param mode_appt_constraints: Integer code in ``{0, 1, 2}`` determining mode of
            constraints with regards to officer numbers and time - 0: no constraints,
            all HSI events run with no squeeze factor, 1: elastic constraints, all HSI
            events run with squeeze factor, 2: hard constraints, only HSI events with
            no squeeze factor run.
        :param cons_availability: If 'default' then use the availability specified in the ResourceFile; if 'none', then
        let no consumable be ever be available; if 'all', then all consumables are always available. When using 'all'
        or 'none', requests for consumables are not logged.
        :param beds_availability: If 'default' then use the availability specified in the ResourceFile; if 'none', then
        let no beds be ever be available; if 'all', then all beds are always available.
        :param ignore_priority: If ``True`` do not use the priority information in HSI
            event to schedule
        :param capabilities_coefficient: Multiplier for the capabilities of health
            officers, if ``None`` set to ratio of initial population to estimated 2010
            population.
        :param use_funded_or_actual_staffing: If `actual`, then use the numbers and distribution of staff estimated to
        be available currently; If `funded`, then use the numbers and distribution of staff that are potentially
        available.
        :param disable: If ``True``, disables the health system (no constraints and no
            logging) and every HSI event runs.
        :param disable_and_reject_all: If ``True``, disable health system and no HSI
            events run
        :param store_hsi_events_that_have_run: Convenience flag for debugging.
        :param record_hsi_event_details: Whether to record details of HSI events used.
        :param compute_squeeze_factor_to_district_level: Whether to compute squeeze_factors to the district level, or
            the national level (which effectively pools the resources across all districts).
        """

        super().__init__(name)
        self.resourcefilepath = resourcefilepath

        assert isinstance(disable, bool)
        assert isinstance(disable_and_reject_all, bool)
        assert not (disable and disable_and_reject_all), (
            'Cannot have both disable and disable_and_reject_all selected'
        )
        self.disable = disable
        self.disable_and_reject_all = disable_and_reject_all

        assert mode_appt_constraints in {0, 1, 2}

        self.mode_appt_constraints = mode_appt_constraints

        self.ignore_priority = ignore_priority

        # Store the argument provided for service_availability
        self.arg_service_availabily = service_availability
        self.service_availability = ['*']  # provided so that there is a default even before simulation is run

        # Check that the capabilities coefficient is correct
        if capabilities_coefficient is not None:
            assert capabilities_coefficient >= 0
            assert isinstance(capabilities_coefficient, float)
        self.capabilities_coefficient = capabilities_coefficient

        # Find which resourcefile to use - those for the actual staff available or the funded staff available
        assert use_funded_or_actual_staffing in ['actual', 'funded', 'funded_plus']
        self.use_funded_or_actual_staffing = use_funded_or_actual_staffing

        # Define (empty) list of registered disease modules (filled in at `initialise_simulation`)
        self.recognised_modules_names = []

        # Define the container for calls for health system interaction events
        self.HSI_EVENT_QUEUE = []
        self.hsi_event_queue_counter = 0  # Counter to help with the sorting in the heapq

        # Check 'store_hsi_events_that_have_run': will store a running list of HSI events that have run
        # (for debugging)
        assert isinstance(store_hsi_events_that_have_run, bool)
        self.store_hsi_events_that_have_run = store_hsi_events_that_have_run
        if self.store_hsi_events_that_have_run:
            self.store_of_hsi_events_that_have_run = list()

        # If record_hsi_event_details == True, a set will be built during the simulation
        # containing HSIEventDetails tuples corresponding to all HSI_Event instances
        # used in the simulation
        self.record_hsi_event_details = record_hsi_event_details
        if record_hsi_event_details:
            self.hsi_event_details = set()

        # Store the argument provided for cons_availability
        assert cons_availability in (None, 'default', 'all', 'none')
        self.arg_cons_availability = cons_availability

        assert beds_availability in (None, 'default', 'all', 'none')
        self.arg_beds_availability = beds_availability

        # `compute_squeeze_factor_to_district_level` is a Boolean indicating whether the computation of squeeze_factors
        # should be specific to each district (when `True`), or if the computation of squeeze_factors should be on the
        # basis that resources from all districts can be effectively "pooled" (when `False).
        assert isinstance(compute_squeeze_factor_to_district_level, bool)
        self.compute_squeeze_factor_to_district_level = compute_squeeze_factor_to_district_level

        # Create the Diagnostic Test Manager to store and manage all Diagnostic Test
        self.dx_manager = DxManager(self)

        # Create the pointer that will be to the instance of BedDays used to track in-patient bed days
        self.bed_days = None

        # Create the pointer that will be to the instance of Consumables used to determine availability of consumables.
        self.consumables = None

        # Create pointer for the HealthSystemScheduler event
        self.healthsystemscheduler = None

        # Create pointer to the `HealthSystemSummaryCounter` helper class
        self._summary_counter = HealthSystemSummaryCounter()

        # Create counter for the running total of footprint of all the HSIs being run today
        self.running_total_footprint: Counter = Counter()

    def read_parameters(self, data_folder):

        path_to_resourcefiles_for_healthsystem = Path(self.resourcefilepath) / 'healthsystem'

        # Read parmaters for overall performance of the HealthSystem
        self.load_parameters_from_dataframe(pd.read_csv(
            path_to_resourcefiles_for_healthsystem / 'ResourceFile_HealthSystem_parameters.csv'
        ))

        # Load basic information about the organization of the HealthSystem
        self.parameters['Master_Facilities_List'] = pd.read_csv(
            path_to_resourcefiles_for_healthsystem / 'organisation' / 'ResourceFile_Master_Facilities_List.csv')

        # Load ResourceFiles that define appointment and officer types
        self.parameters['Officer_Types_Table'] = pd.read_csv(
            path_to_resourcefiles_for_healthsystem / 'human_resources' / 'definitions' /
            'ResourceFile_Officer_Types_Table.csv')
        self.parameters['Appt_Types_Table'] = pd.read_csv(
            path_to_resourcefiles_for_healthsystem / 'human_resources' / 'definitions' /
            'ResourceFile_Appt_Types_Table.csv')
        self.parameters['Appt_Offered_By_Facility_Level'] = pd.read_csv(
            path_to_resourcefiles_for_healthsystem / 'human_resources' / 'definitions' /
            'ResourceFile_ApptType_By_FacLevel.csv')
        self.parameters['Appt_Time_Table'] = pd.read_csv(
            path_to_resourcefiles_for_healthsystem / 'human_resources' / 'definitions' /
            'ResourceFile_Appt_Time_Table.csv')

        # Load 'Daily_Capabilities' (for both actual and funded)
        for _i in ['actual', 'funded', 'funded_plus']:
            self.parameters[f'Daily_Capabilities_{_i}'] = pd.read_csv(
                path_to_resourcefiles_for_healthsystem / 'human_resources' / f'{_i}' /
                'ResourceFile_Daily_Capabilities.csv')

        # Read in ResourceFile_Consumables
        self.parameters['item_and_package_code_lookups'] = pd.read_csv(
            path_to_resourcefiles_for_healthsystem / 'consumables' / 'ResourceFile_Consumables_Items_and_Packages.csv')
        self.parameters['availability_estimates'] = pd.read_csv(
            path_to_resourcefiles_for_healthsystem / 'consumables' / 'ResourceFile_Consumables_availability_small.csv')

        # Data on the number of beds available of each type by facility_id
        self.parameters['BedCapacity'] = pd.read_csv(
            path_to_resourcefiles_for_healthsystem / 'infrastructure_and_equipment' / 'ResourceFile_Bed_Capacity.csv')

    def pre_initialise_population(self):
        """Generate the accessory classes used by the HealthSystem and pass to them the data that has been read."""

        # Determine service_availability
        self.service_availability = self.get_service_availability()

        self.process_human_resources_files()

        # Initialise the BedDays class
        self.bed_days = BedDays(hs_module=self,
                                availability=self.get_beds_availability())
        self.bed_days.pre_initialise_population()

        # Initialise the Consumables class
        self.consumables = Consumables(data=self.parameters['availability_estimates'],
                                       rng=self.rng,
                                       availability=self.get_cons_availability())

    def initialise_population(self, population):
        self.bed_days.initialise_population(population.props)

    def initialise_simulation(self, sim):
        # If capabilities coefficient was not explicitly specified, use initial population scaling factor
        if self.capabilities_coefficient is None:
            self.capabilities_coefficient = self.sim.modules['Demography'].initial_model_to_data_popsize_ratio

        # Set the tracker in preparation for the simulation
        self.bed_days.initialise_beddays_tracker(
            model_to_data_popsize_ratio=self.sim.modules['Demography'].initial_model_to_data_popsize_ratio
        )

        # Set the consumables modules in preparation for the simulation
        self.consumables.on_start_of_day(sim.date)

        # Capture list of disease modules:
        self.recognised_modules_names = [
            m.name for m in self.sim.modules.values() if Metadata.USES_HEALTHSYSTEM in m.METADATA
        ]

        # Check that set of districts of residence in population are subset of districts from
        # `self._facilities_for_each_district`, which is derived from self.parameters['Master_Facilities_List']
        df = self.sim.population.props
        districts_of_residence = set(df.loc[df.is_alive, "district_of_residence"].cat.categories)
        assert all(
            districts_of_residence.issubset(per_level_facilities.keys())
            for per_level_facilities in self._facilities_for_each_district.values()
        ), (
            "At least one district_of_residence value in population not present in "
            "self._facilities_for_each_district resource file"
        )

        # Launch the healthsystem scheduler (a regular event occurring each day) [if not disabled]
        if not (self.disable or self.disable_and_reject_all):
            self.healthsystemscheduler = HealthSystemScheduler(self)
            sim.schedule_event(self.healthsystemscheduler, sim.date)

    def on_birth(self, mother_id, child_id):
        self.bed_days.on_birth(self.sim.population.props, mother_id, child_id)

    def on_simulation_end(self):
        """Put out to the log the information from the tracker of the last day of the simulation"""
        self.bed_days.on_simulation_end()
        self.consumables.on_simulation_end()

    def process_human_resources_files(self):
        """Create the data-structures needed from the information read into the parameters."""

        # * Define Facility Levels
        self._facility_levels = set(self.parameters['Master_Facilities_List']['Facility_Level']) - {'5'}
        assert self._facility_levels == {'0', '1a', '1b', '2', '3', '4'}  # todo soft code this?

        # * Define Appointment Types
        self._appointment_types = set(self.parameters['Appt_Types_Table']['Appt_Type_Code'])

        # * Define the Officers Needed For Each Appointment
        # (Store data as dict of dicts, with outer-dict indexed by string facility level and
        # inner-dict indexed by string type code with values corresponding to list of (named)
        # tuples of appointment officer type codes and time taken.)
        appt_time_data = self.parameters['Appt_Time_Table']
        appt_times_per_level_and_type = {_facility_level: defaultdict(list) for _facility_level in
                                         self._facility_levels}
        for appt_time_tuple in appt_time_data.itertuples():
            appt_times_per_level_and_type[
                appt_time_tuple.Facility_Level
            ][
                appt_time_tuple.Appt_Type_Code
            ].append(
                AppointmentSubunit(
                    officer_type=appt_time_tuple.Officer_Category,
                    time_taken=appt_time_tuple.Time_Taken_Mins
                )
            )
        assert (
            sum(
                len(appt_info_list)
                for level in self._facility_levels
                for appt_info_list in appt_times_per_level_and_type[level].values()
            ) == len(appt_time_data)
        )
        self._appt_times = appt_times_per_level_and_type

        # * Define Which Appointments Are Possible At Each Facility Level
        appt_type_per_level_data = self.parameters['Appt_Offered_By_Facility_Level']
        self._appt_type_by_facLevel = {
            _facility_level: set(
                appt_type_per_level_data['Appt_Type_Code'][
                    appt_type_per_level_data[f'Facility_Level_{_facility_level}']
                ]
            )
            for _facility_level in self._facility_levels
        }

        # Also store data as dict of dicts, with outer-dict indexed by string facility level and
        # inner-dict indexed by district name with values corresponding to (named) tuples of
        # facility ID and name
        # Get look-up of the districts (by name) in each region (by name)
        districts_in_region = self.sim.modules['Demography'].parameters['districts_in_region']
        all_districts = set(self.sim.modules['Demography'].parameters['district_num_to_district_name'].values())

        facilities_per_level_and_district = {_facility_level: {} for _facility_level in self._facility_levels}
        facilities_by_facility_id = dict()
        for facility_tuple in self.parameters['Master_Facilities_List'].itertuples():
            _facility_info = FacilityInfo(id=facility_tuple.Facility_ID,
                                          name=facility_tuple.Facility_Name,
                                          level=facility_tuple.Facility_Level,
                                          region=facility_tuple.Region
                                          )

            facilities_by_facility_id[facility_tuple.Facility_ID] = _facility_info

            if pd.notnull(facility_tuple.District):
                # A facility that is specific to a district:
                facilities_per_level_and_district[facility_tuple.Facility_Level][facility_tuple.District] = \
                    _facility_info

            elif pd.isnull(facility_tuple.District) and pd.notnull(facility_tuple.Region):
                # A facility that is specific to region (and not a district):
                for _district in districts_in_region[facility_tuple.Region]:
                    facilities_per_level_and_district[facility_tuple.Facility_Level][_district] = _facility_info

            elif (
                pd.isnull(facility_tuple.District) and
                pd.isnull(facility_tuple.Region) and
                (facility_tuple.Facility_Level != '5')
            ):
                # A facility that is National (not specific to a region or a district) (ignoring level 5 (headquarters))
                for _district in all_districts:
                    facilities_per_level_and_district[facility_tuple.Facility_Level][_district] = _facility_info

        # Check that there is facility of every level for every district:
        assert all(
            all_districts == facilities_per_level_and_district[_facility_level].keys()
            for _facility_level in self._facility_levels
        ), "There is not one of each facility type available to each district."

        self._facility_by_facility_id = facilities_by_facility_id
        self._facilities_for_each_district = facilities_per_level_and_district

        # * Store 'DailyCapabilities' in correct format and using the specified underlying assumptions
        self._daily_capabilities = self.format_daily_capabilities()

        # Also, store the set of officers with non-zero daily availability
        # (This is used for checking that scheduled HSI events do not make appointment requiring officers that are
        # never available.)
        self._officers_with_availability = set(self._daily_capabilities.index[self._daily_capabilities > 0])

    def format_daily_capabilities(self) -> pd.Series:
        """
        This will updates the dataframe for the self.parameters['Daily_Capabilities'] so as to include
        every permutation of officer_type_code and facility_id, with zeros against permutations where no capacity
        is available.

        It also give the dataframe an index that is useful for merging on (based on Facility_ID and Officer Type)

        (This is so that its easier to track where demands are being placed where there is no capacity)
        """

        # Get the capabilities data imported (according to the specified underlying assumptions).
        capabilities = self.parameters[f'Daily_Capabilities_{self.use_funded_or_actual_staffing}']
        capabilities = capabilities.rename(columns={'Officer_Category': 'Officer_Type_Code'})  # neaten

        # Create dataframe containing background information about facility and officer types
        facility_ids = self.parameters['Master_Facilities_List']['Facility_ID'].values
        officer_type_codes = set(self.parameters['Officer_Types_Table']['Officer_Category'].values)
        # todo - <-- avoid use of the file or define differently?

        # # naming to be not with _ within the name of an oficer
        facs = list()
        officers = list()
        for f in facility_ids:
            for o in officer_type_codes:
                facs.append(f)
                officers.append(o)

        capabilities_ex = pd.DataFrame(data={'Facility_ID': facs, 'Officer_Type_Code': officers})

        # Merge in information about facility from Master Facilities List
        mfl = self.parameters['Master_Facilities_List']
        capabilities_ex = capabilities_ex.merge(mfl, on='Facility_ID', how='left')

        # Merge in information about officers
        # officer_types = self.parameters['Officer_Types_Table'][['Officer_Type_Code', 'Officer_Type']]
        # capabilities_ex = capabilities_ex.merge(officer_types, on='Officer_Type_Code', how='left')

        # Merge in the capabilities (minutes available) for each officer type (inferring zero minutes where
        # there is no entry in the imported capabilities table)
        capabilities_ex = capabilities_ex.merge(
            capabilities[['Facility_ID', 'Officer_Type_Code', 'Total_Mins_Per_Day']],
            on=['Facility_ID', 'Officer_Type_Code'],
            how='left',
        )
        capabilities_ex = capabilities_ex.fillna(0)

        # Give the standard index:
        capabilities_ex = capabilities_ex.set_index(
            'FacilityID_'
            + capabilities_ex['Facility_ID'].astype(str)
            + '_Officer_'
            + capabilities_ex['Officer_Type_Code']
        )

        # Rename 'Total_Minutes_Per_Day'
        capabilities_ex = capabilities_ex.rename(columns={'Total_Mins_Per_Day': 'Total_Minutes_Per_Day'})

        # Checks
        assert abs(capabilities_ex['Total_Minutes_Per_Day'].sum() - capabilities['Total_Mins_Per_Day'].sum()) < 1e-7
        assert len(capabilities_ex) == len(facility_ids) * len(officer_type_codes)

        # return the pd.Series of `Total_Minutes_Per_Day' indexed for each type of officer at each facility
        return capabilities_ex['Total_Minutes_Per_Day']

    def get_service_availability(self) -> List[str]:
        """Returns service availability. (Should be equal to what is specified by the parameter, but overwrite with what
        was provided in argument if an argument was specified -- provided for backward compatibility/debugging.)"""

        if self.arg_service_availabily is None:
            service_availability = self.parameters['Service_Availability']
        else:
            service_availability = self.arg_service_availabily

        assert isinstance(service_availability, list)

        # Log the service_availability
        logger.info(key="message",
                    data=f"Running Health System With the Following Service Availability: "
                         f"{self.service_availability}"
                    )
        return service_availability

    def get_cons_availability(self) -> str:
        """Set consumables availability. (Should be equal to what is specified by the parameter, but overwrite with
        what was provided in argument if an argument was specified -- provided for backward compatibility/debugging.)"""

        if self.arg_cons_availability is None:
            _cons_availability = self.parameters['cons_availability']
        else:
            _cons_availability = self.arg_cons_availability

        # Log the service_availability
        logger.info(key="message",
                    data=f"Running Health System With the Following Consumables Availability: "
                         f"{_cons_availability}"
                    )

        return _cons_availability

    def get_beds_availability(self) -> str:
        """Set beds availability. (Should be equal to what is specified by the parameter, but overwrite with
        what was provided in argument if an argument was specified -- provided for backward compatibility/debugging.)"""

        if self.arg_beds_availability is None:
            _beds_availability = self.parameters['beds_availability']
        else:
            _beds_availability = self.arg_beds_availability

        # For logical consistency, when the HealthSystem is disabled, beds_availability should be 'all', irrespective of
        # what arguments/parameters are provided.
        if self.disable:
            _beds_availability = 'all'

        # Log the service_availability
        logger.info(key="message",
                    data=f"Running Health System With the Following Beds Availability: "
                         f"{_beds_availability}"
                    )

        return _beds_availability

    def schedule_hsi_event(
        self,
        hsi_event: 'HSI_Event',
        priority: int,
        topen: datetime.datetime,
        tclose: Optional[datetime.datetime] = None,
        do_hsi_event_checks: bool = True
    ):
        """
        Schedule a health system interaction (HSI) event.

        :param hsi_event: The HSI event to be scheduled.
        :param priority: The priority for the HSI event: 0 (highest), 1 or 2 (lowest)
        :param topen: The earliest date at which the HSI event should run.
        :param tclose: The latest date at which the HSI event should run. Set to one week after ``topen`` if ``None``.
        :param do_hsi_event_checks: Whether to perform sanity checks on the passed ``hsi_event`` argument to check that
         it constitutes a valid HSI event. This is intended for allowing disabling of these checks when scheduling
         multiple HSI events of the same ``HSI_Event`` subclass together, in which case typically performing these
         checks for each individual HSI event of the shared type will be redundant.
        """

        # If there is no specified tclose time then set this to a week after topen
        if tclose is None:
            tclose = topen + DateOffset(days=7)

        # Check topen is not in the past
        assert topen >= self.sim.date

        # Check that priority is in valid range
        assert priority in (0, 1, 2)

        # Check that topen is strictly before tclose
        assert topen < tclose

        # If ignoring the priority in scheduling, then over-write the provided priority information with 0.
        if self.ignore_priority:
            priority = 0

        # Check if healthsystem is disabled/disable_and_reject_all and, if so, schedule a wrapped event:
        if self.disable and (not self.disable_and_reject_all):
            # If healthsystem is disabled (meaning that HSI can still run), schedule for the `run` method on `topen`.
            self.sim.schedule_event(HSIEventWrapper(hsi_event=hsi_event, run_hsi=True), topen)
            return

        if self.disable_and_reject_all:
            # If healthsystem is disabled the HSI will never run: schedule for the `never_ran` method on `tclose`.
            self.sim.schedule_event(HSIEventWrapper(hsi_event=hsi_event, run_hsi=False), tclose)
            return

        # Check that this is a legitimate health system interaction (HSI) event.
        # These checks are only performed when the flag `do_hsi_event_checks` is set to ``True`` to allow disabling
        # when the checks are redundant for example when scheduling multiple HSI events of same `HSI_Event` subclass.
        if do_hsi_event_checks:
            self.check_hsi_event_is_valid(hsi_event)

        # Check that this request is allowable under current policy (i.e. included in service_availability).
        if not self.is_treatment_id_allowed(hsi_event.TREATMENT_ID, self.service_availability):
            # HSI is not allowable under the services_available parameter: run the HSI's 'never_ran' method on the date
            # of tclose.
            self.sim.schedule_event(HSIEventWrapper(hsi_event=hsi_event, run_hsi=False), tclose)

        else:
            # The HSI is allowed and will be added to the HSI_EVENT_QUEUE.
            # Let the HSI gather information about itself (facility_id and appt-footprint time requirements):
            hsi_event.initialise()
            # Add the event to the queue:
            self._add_hsi_event_queue_item_to_hsi_event_queue(
                priority=priority, topen=topen, tclose=tclose, hsi_event=hsi_event)

    def _add_hsi_event_queue_item_to_hsi_event_queue(self, priority, topen, tclose, hsi_event) -> None:
        """Add an event to the HSI_EVENT_QUEUE."""
        # Create HSIEventQueue Item, including a counter for the number of HSI_Events, to assist with sorting in the
        # queue (NB. the sorting is done ascending and by the order of the items in the tuple).
        self.hsi_event_queue_counter += 1
        _new_item: HSIEventQueueItem = HSIEventQueueItem(
            priority, topen, self.hsi_event_queue_counter, tclose, hsi_event)
        # Add to queue:
        hp.heappush(self.HSI_EVENT_QUEUE, _new_item)

    def check_hsi_event_is_valid(self, hsi_event):
        """Check the integrity of an HSI_Event."""
        assert isinstance(hsi_event, HSI_Event)

        # Check that non-empty treatment ID specified
        assert hsi_event.TREATMENT_ID != ''

        if not isinstance(hsi_event.target, tlo.population.Population):
            # This is an individual-scoped HSI event.
            # It must have EXPECTED_APPT_FOOTPRINT, BEDDAYS_FOOTPRINT and ACCEPTED_FACILITY_LEVELS.

            # Correct formatted EXPECTED_APPT_FOOTPRINT
            assert self.appt_footprint_is_valid(hsi_event.EXPECTED_APPT_FOOTPRINT)

            # That it has an acceptable 'ACCEPTED_FACILITY_LEVEL' attribute
            assert hsi_event.ACCEPTED_FACILITY_LEVEL in self._facility_levels, \
                f"In the HSI with TREATMENT_ID={hsi_event.TREATMENT_ID}, the ACCEPTED_FACILITY_LEVEL (=" \
                f"{hsi_event.ACCEPTED_FACILITY_LEVEL}) is not recognised."

            self.bed_days.check_beddays_footprint_format(hsi_event.BEDDAYS_FOOTPRINT)

            # Check that this can accept the squeeze argument
            assert _accepts_argument(hsi_event.run, 'squeeze_factor')

            # Check that the event does not request an appointment at a facility
            # level which is not possible
            appt_type_to_check_list = hsi_event.EXPECTED_APPT_FOOTPRINT.keys()
            facility_appt_types = self._appt_type_by_facLevel[
                hsi_event.ACCEPTED_FACILITY_LEVEL
            ]
            assert facility_appt_types.issuperset(appt_type_to_check_list), (
                f"An appointment type has been requested at a facility level for "
                f"which it is not possible: TREATMENT_ID={hsi_event.TREATMENT_ID}"
            )

    @staticmethod
    def is_treatment_id_allowed(treatment_id: str, service_availability: list) -> bool:
        """Determine if a treatment_id (specified as a string) can be run (i.e., is within the allowable set of
         treatments, given by `self.service_availability`. The rules are as follows:
          * An empty list means nothing is allowed
          * A list that contains only an asteriks ['*'] means run anything
          * If the list is not empty, then a treatment_id with a first part "FirstAttendance_" is also allowed
          * An entry in the list of the form "A_B_C" means a treatment_id that matches exactly is allowed
          * An entry in the list of the form "A_B_*" means that a treatment_id that begins "A_B_" or "A_B" is allowed
        """
        def _treatment_matches_pattern(_treatment_id, _service_availability):
            """Check if treatment_id matches any services specified with wildcard * patterns"""

            def _matches_this_pattern(_treatment_id, _s):
                """Returns True if this treatment_id is consistent with this component of service_availability"""
                if '*' in _s:
                    assert _s[-1] == '*', f"Component of service_availability has an asteriks not at the end: {_s}"
                    _s_split = _s.split('_')  # split the matching pattern at '_' knowing that the last component is '*'
                    _treatment_id_split = _treatment_id.split('_', len(_s_split) - 1)  # split treatment_id at '_' into
                    # as many component as there as non-asteriks component of _s.
                    # Check if all the components (that are not asteriks) are the same:
                    return all(
                        [(a == b) or (b == "*") for a, b in itertools.zip_longest(_treatment_id_split, _s_split)]
                    )
                else:
                    # If not "*", comparison is ordinary match between strings
                    return _treatment_id == _s

            for _s in service_availability:
                if _matches_this_pattern(_treatment_id, _s):
                    return True
            return False

        if not service_availability:
            # Empty list --> nothing is allowable
            return False

        if service_availability == ['*']:
            # Wildcard --> everything is allowed
            return True
        elif treatment_id in service_availability:
            # Explicit inclusion of this treatment_id --> allowed
            return True
        elif treatment_id.startswith('FirstAttendance_'):
            # FirstAttendance* --> allowable
            return True
        else:
            if _treatment_matches_pattern(treatment_id, service_availability):
                return True
        return False

    def schedule_batch_of_individual_hsi_events(
        self, hsi_event_class, person_ids, priority, topen, tclose=None, **event_kwargs
    ):
        """Schedule a batch of individual-scoped HSI events of the same type.

        Only performs sanity checks on the HSI event for the first scheduled event
        thus removing the overhead of multiple redundant checks.

        :param hsi_event_class: The ``HSI_Event`` subclass of the events to schedule.
        :param person_ids: A sequence of person ID index values to use as the targets
            of the HSI events being scheduled.
        :param priority: The priority for the HSI events: 0 (highest), 1 or 2 (lowest).
            Either a single value for all events or an iterable of per-target values.
        :param topen: The earliest date at which the HSI events should run. Either a
            single value for all events or an iterable of per-target values.
        :param tclose: The latest date at which the HSI events should run. Set to one
           week after ``topen`` if ``None``. Either a single value for all events or an
           iterable of per-target values.
        :param event_kwargs: Any additional keyword arguments to pass to the
            ``hsi_event_class`` initialiser in addition to ``person_id``.
        """
        # If any of {priority, topen, tclose} are iterable assume correspond to per-
        # target values for corresponding arguments of schedule_hsi_event otherwise
        # use same value for all calls
        priorities = priority if isinstance(priority, Iterable) else repeat(priority)
        topens = topen if isinstance(topen, Iterable) else repeat(topen)
        tcloses = tclose if isinstance(tclose, Iterable) else repeat(tclose)
        for i, (person_id, priority, topen, tclose) in enumerate(
            zip(person_ids, priorities, topens, tcloses)
        ):
            self.schedule_hsi_event(
                hsi_event=hsi_event_class(person_id=person_id, **event_kwargs),
                priority=priority,
                topen=topen,
                tclose=tclose,
                # Only perform checks for first event
                do_hsi_event_checks=(i == 0)
            )

    def appt_footprint_is_valid(self, appt_footprint):
        """
        Checks an appointment footprint to ensure it is in the correct format.
        :param appt_footprint: Appointment footprint to check.
        :return: True if valid and False otherwise.
        """
        # Check that all keys known appointment types and all values non-negative
        return isinstance(appt_footprint, dict) and all(
            k in self._appointment_types and v > 0
            for k, v in appt_footprint.items()
        )

    @property
    def capabilities_today(self) -> pd.Series:
        """
        Returns the capabilities of the health system today.
        returns: pd.Series giving minutes available for each officer type in each facility type

        Functions can go in here in the future that could expand the time available,
        simulating increasing efficiency (the concept of a productivity ratio raised
        by Martin Chalkley).

        For now this method only multiplies the estimated minutes available by the `capabilities_coefficient` scale
        factor.
        """
        return self._daily_capabilities * self.capabilities_coefficient

    def get_blank_appt_footprint(self):
        """
        This is a helper function so that disease modules can easily create their appt_footprints.
        It returns an empty Counter instance.

        """
        return Counter()

    def get_facility_info(self, hsi_event) -> FacilityInfo:
        """Helper function to find the facility at which an HSI event will take place based on their district of
        residence and the level of the facility of the HSI."""
        the_district = self.sim.population.props.at[hsi_event.target, 'district_of_residence']
        the_level = hsi_event.ACCEPTED_FACILITY_LEVEL
        return self._facilities_for_each_district[the_level][the_district]

    def get_appt_footprint_as_time_request(self, facility_info: FacilityInfo, appt_footprint: dict):
        """
        This will take an APPT_FOOTPRINT and return the required appointments in terms of the
        time required of each Officer Type in each Facility ID.
        The index will identify the Facility ID and the Officer Type in the same format
        as is used in Daily_Capabilities.
        :params facility_info: The FacilityInfo describing the facility at which the appointment occurs
        :param appt_footprint: The actual appt footprint (optional) if different to that in the HSI event.
        :return: A Counter that gives the times required for each officer-type in each facility_ID, where this time
         is non-zero.
        """
        # Accumulate appointment times for specified footprint using times from appointment times table.
        appt_footprint_times = Counter()
        for appt_type in appt_footprint:
            try:
                appt_info_list = self._appt_times[facility_info.level][appt_type]
            except KeyError as e:
                raise KeyError(
                    f"The time needed for an appointment is not defined for the specified facility level: "
                    f"appt_type={appt_type}, "
                    f"facility_level={facility_info.level}."
                ) from e

            for appt_info in appt_info_list:
                appt_footprint_times[
                    f"FacilityID_{facility_info.id}_Officer_{appt_info.officer_type}"
                ] += appt_info.time_taken

        return appt_footprint_times

    def get_squeeze_factors(self, footprints_per_event, total_footprint, current_capabilities,
                            compute_squeeze_factor_to_district_level: bool
                            ):
        """
        This will compute the squeeze factors for each HSI event from the list of all
        the calls on health system resources for the day.
        The squeeze factor is defined as (call/available - 1). ie. the highest
        fractional over-demand among any type of officer that is called-for in the
        appt_footprint of an HSI event.
        A value of 0.0 signifies that there is no squeezing (sufficient resources for
        the EXPECTED_APPT_FOOTPRINT).
        A value of 99.99 signifies that the call is for an officer_type in a
        health-facility that is not available.

        :param footprints_per_event: List, one entry per HSI event, containing the
            minutes required from each health officer in each health facility as a
            Counter (using the standard index)
        :param total_footprint: Counter, containing the total minutes required from
            each health officer in each health facility when non-zero, (using the
            standard index)
        :param current_capabilities: Series giving the amount of time available for
            each health officer in each health facility (using the standard index)
        :param compute_squeeze_factor_to_district_level: Boolean indicating whether
            the computation of squeeze_factors should be specific to each district
            (when `True`), or if the computation of squeeze_factors should be on
            the basis that resources from all districts can be effectively "pooled"
            (when `False).

        :return: squeeze_factors: an array of the squeeze factors for each HSI event
            (position in array matches that in the all_call_today list).
        """

        def get_total_minutes_of_this_officer_in_this_district(_officer):
            """Returns the minutes of current capabilities for the officer identified (this officer type in this
            facility_id)."""
            return current_capabilities.get(_officer)

        def get_total_minutes_of_this_officer_in_all_district(_officer):
            """Returns the minutes of current capabilities for the officer identified in all districts (this officer
            type in this all facilities of the same level in all districts)."""

            def split_officer_compound_string(cs) -> Tuple[int, str]:
                """Returns (facility_id, officer_type) for the officer identified in the string of the form:
                 'FacilityID_{facility_id}_Officer_{officer_type}'."""
                _, _facility_id, _, _officer_type = cs.split('_', 3)  # (NB. Some 'officer_type' include "_")
                return int(_facility_id), _officer_type

            def _match(_this_officer, facility_ids: List[int], officer_type: str):
                """Returns True if the officer identified is of the identified officer_type and is in one of the
                facility_ids."""
                this_facility_id, this_officer_type = split_officer_compound_string(_this_officer)
                return (this_officer_type == officer_type) and (this_facility_id in facility_ids)

            facility_id, officer_type = split_officer_compound_string(_officer)
            facility_level = self._facility_by_facility_id[int(facility_id)].level
            facilities_of_same_level_in_all_district = [
                _fac.id for _fac in self._facilities_for_each_district[facility_level].values()
            ]

            officers_in_the_same_level_in_all_districts = [
                _officer for _officer in current_capabilities.keys() if
                _match(_officer, facility_ids=facilities_of_same_level_in_all_district, officer_type=officer_type)
            ]

            return sum(current_capabilities.get(_o) for _o in officers_in_the_same_level_in_all_districts)

        # 1) Compute the load factors for each officer type at each facility that is
        # called-upon in this list of HSIs
        load_factor = {}
        for officer, call in total_footprint.items():
            if compute_squeeze_factor_to_district_level:
                availability = get_total_minutes_of_this_officer_in_this_district(officer)
            else:
                availability = get_total_minutes_of_this_officer_in_all_district(officer)

            if availability is None:  # todo - does this ever happen?
                load_factor[officer] = 99.99
            elif availability == 0:
                load_factor[officer] = float('inf')
            else:
                load_factor[officer] = max(call / availability - 1, 0)

        # 2) Convert these load-factors into an overall 'squeeze' signal for each HSI,
        # based on the highest load-factor of any officer required (or zero if event
        # has an empty footprint)
        squeeze_factor_per_hsi_event = np.array([
            max((load_factor[officer] for officer in footprint), default=0)
            for footprint in footprints_per_event
        ])

        assert (squeeze_factor_per_hsi_event >= 0).all()

        return squeeze_factor_per_hsi_event

    def record_hsi_event(self, hsi_event, actual_appt_footprint=None, squeeze_factor=None, did_run=True):
        """
        Record the processing of an HSI event.
        If this is an individual-level HSI_Event, it will also record the actual appointment footprint
        :param hsi_event: The HSI_Event (containing the initial expectations of footprints)
        :param actual_appt_footprint: The actual Appointment Footprint (if individual event)
        :param squeeze_factor: The squeeze factor (if individual event)
        """

        if isinstance(hsi_event.target, tlo.population.Population):
            # Population HSI-Event (N.B. This is not actually logged.)
            log_info = dict()
            log_info['TREATMENT_ID'] = hsi_event.TREATMENT_ID
            log_info['Number_By_Appt_Type_Code'] = 'Population'  # remove the appt-types with zeros
            log_info['Person_ID'] = -1  # Junk code
            log_info['Squeeze_Factor'] = 0
            log_info['did_run'] = did_run

        else:
            # Individual HSI-Event
            _squeeze_factor = squeeze_factor if squeeze_factor != np.inf else 100.0

            self.write_to_hsi_log(
                treatment_id=hsi_event.TREATMENT_ID,
                number_by_appt_type_code=actual_appt_footprint,
                person_id=hsi_event.target,
                squeeze_factor=_squeeze_factor,
                did_run=did_run,
                facility_level=hsi_event.ACCEPTED_FACILITY_LEVEL,
                facility_id=hsi_event.facility_info.id,
            )

            # Storage for the purpose of testing / documentation
            if self.store_hsi_events_that_have_run:
                self.store_of_hsi_events_that_have_run.append(
                    {
                        'HSI_Event': hsi_event.__class__.__name__,
                        'date': self.sim.date,
                        'TREATMENT_ID': hsi_event.TREATMENT_ID,
                        'did_run': did_run,
                        'Appt_Footprint': actual_appt_footprint,
                        'Squeeze_Factor': _squeeze_factor,
                        'Person_ID': hsi_event.target
                    }
                )

            if self.record_hsi_event_details:
                self.hsi_event_details.add(
                    HSIEventDetails(
                        event_name=type(hsi_event).__name__,
                        module_name=type(hsi_event.module).__name__,
                        treatment_id=hsi_event.TREATMENT_ID,
                        facility_level=getattr(
                            hsi_event, 'ACCEPTED_FACILITY_LEVEL', None
                        ),
                        appt_footprint=(
                            tuple(actual_appt_footprint)
                            if actual_appt_footprint is not None
                            else tuple(getattr(hsi_event, 'EXPECTED_APPT_FOOTPRINT', {}))
                        ),
                        beddays_footprint=tuple(sorted(hsi_event.BEDDAYS_FOOTPRINT.items()))
                    )
                )

    def write_to_hsi_log(self,
                         treatment_id,
                         number_by_appt_type_code,
                         person_id,
                         squeeze_factor,
                         did_run,
                         facility_level,
                         facility_id,
                         ):
        """Write the log `HSI_Event` and add to the summary counter."""

        logger.info(key="HSI_Event",
                    data={
                        'TREATMENT_ID': treatment_id,
                        'Number_By_Appt_Type_Code': number_by_appt_type_code,
                        'Person_ID': person_id,
                        'Squeeze_Factor': squeeze_factor,
                        'did_run': did_run,
                        'Facility_Level': facility_level if facility_level is not None else -99,
                        'Facility_ID': facility_id if facility_id is not None else -99,
                    },
                    description="record of each HSI event"
                    )

        if did_run:
            self._summary_counter.record_hsi_event(
                treatment_id=treatment_id,
                appt_footprint=number_by_appt_type_code,
                level=facility_level,
            )

    def log_current_capabilities_and_usage(self):
        """
        This will log the percentage of the current capabilities that is used at each Facility Type, according the
        `runnning_total_footprint`.
        """
        current_capabilities = self.capabilities_today
        total_footprint = self.running_total_footprint

        # Combine the current_capabilities and total_footprint per-officer totals
        comparison = pd.DataFrame(index=current_capabilities.index)
        comparison['Total_Minutes_Per_Day'] = current_capabilities
        comparison['Minutes_Used'] = pd.Series(total_footprint, dtype='float64')
        comparison['Minutes_Used'] = comparison['Minutes_Used'].fillna(0.0)
        assert len(comparison) == len(current_capabilities)

        # Compute Fraction of Time Used Overall
        total_available = comparison['Total_Minutes_Per_Day'].sum()
        fraction_time_used_overall = (
            comparison['Minutes_Used'].sum() / total_available if total_available > 0 else 0
        )

        # Compute Fraction of Time Used In Each Facility
        facility_id = [_f.split('_')[1] for _f in comparison.index]
        summary_by_fac_id = comparison.groupby(by=facility_id)[['Total_Minutes_Per_Day', 'Minutes_Used']].sum()
        summary_by_fac_id['Fraction_Time_Used'] = (
            summary_by_fac_id['Minutes_Used'] / summary_by_fac_id['Total_Minutes_Per_Day']
        ).replace([np.inf, -np.inf, np.nan], 0.0)

        # Compute Fraction of Time For Each Officer and level
        officer = [_f.rsplit('Officer_')[1] for _f in comparison.index]
        level = [self._facility_by_facility_id[int(_fac_id)].level for _fac_id in facility_id]
        summary_by_officer = comparison.groupby(by=[officer, level])[['Total_Minutes_Per_Day', 'Minutes_Used']].sum()
        summary_by_officer['Fraction_Time_Used'] = (
            summary_by_officer['Minutes_Used'] / summary_by_officer['Total_Minutes_Per_Day']
        ).replace([np.inf, -np.inf, np.nan], 0.0)
        summary_by_officer.index.names = ['Officer_Type', 'Facility_Level']

        logger.info(key='Capacity',
                    data={
                        'Frac_Time_Used_Overall': fraction_time_used_overall,
                        'Frac_Time_Used_By_Facility_ID': summary_by_fac_id['Fraction_Time_Used'].to_dict(),
                        'Frac_Time_Used_By_OfficerType':  flatten_multi_index_series_into_dict_for_logging(
                            summary_by_officer['Fraction_Time_Used']
                        ),
                    },
                    description='daily summary of utilisation and capacity of health system resources')

        self._summary_counter.record_hs_status(
            fraction_time_used_across_all_facilities=fraction_time_used_overall)

    def remove_beddays_footprint(self, person_id):
        # removing bed_days from a particular individual if any
        self.bed_days.remove_beddays_footprint(person_id=person_id)

    def find_events_for_person(self, person_id: int):
        """Find the events in the HSI_EVENT_QUEUE for a particular person.
        :param person_id: the person_id of interest
        :returns list of tuples (date_of_event, event) for that person_id in the HSI_EVENT_QUEUE.

        NB. This is for debugging and testing only - not for use in real simulations as it is slow
        """
        list_of_events = list()

        for ev_tuple in self.HSI_EVENT_QUEUE:
            date = ev_tuple[1]  # this is the 'topen' value
            event = ev_tuple[4]
            if isinstance(event.target, (int, np.integer)):
                if event.target == person_id:
                    list_of_events.append((date, event))

        return list_of_events

    def reset_queue(self):
        """Set the HSI event queue to be empty"""
        self.HSI_EVENT_QUEUE = []
        self.hsi_event_queue_counter = 0

    def get_item_codes_from_package_name(self, package: str) -> dict:
        """Helper function to provide the item codes and quantities in a dict of the form {<item_code>:<quantity>} for
         a given package name."""
        return get_item_codes_from_package_name(self.parameters['item_and_package_code_lookups'], package)

    def get_item_code_from_item_name(self, item: str) -> int:
        """Helper function to provide the item_code (an int) when provided with the name of the item"""
        return get_item_code_from_item_name(self.parameters['item_and_package_code_lookups'], item)

    def override_availability_of_consumables(self, item_codes) -> None:
        """Over-ride the availability (for all months and all facilities) of certain consumables item_codes.
        :param item_codes: Dictionary of the form {<item_code>: probability_that_item_is_available}
        :return: None
        """
        self.consumables.override_availability(item_codes)

    def on_end_of_day(self) -> None:
        """Do jobs to be done at the end of the day (after all HSI run)"""
        self.bed_days.on_end_of_day()

    def on_end_of_year(self) -> None:
        """Write to log the current states of the summary counters and reset them."""
        self._summary_counter.write_to_log_and_reset_counters()
        self.consumables.on_end_of_year()
        self.bed_days.on_end_of_year()

    def run_population_level_events(self, _list_of_population_hsi_event_tuples: List[HSIEventQueueItem]) -> None:
        """Run a list of population level events."""
        while len(_list_of_population_hsi_event_tuples) > 0:
            pop_level_hsi_event_tuple = _list_of_population_hsi_event_tuples.pop()
            pop_level_hsi_event = pop_level_hsi_event_tuple.hsi_event
            pop_level_hsi_event.run(squeeze_factor=0)
            self.record_hsi_event(hsi_event=pop_level_hsi_event)

    def run_individual_level_events(self, _list_of_individual_hsi_event_tuples: List[HSIEventQueueItem]) -> List:
        """Run a list of individual level events. Returns: list of events that did not run (maybe an empty a list)."""
        _to_be_held_over = list()

        if _list_of_individual_hsi_event_tuples:
            # Examine total call on health officers time from the HSI events in the list:

            # For all events in the list, expand the appt-footprint of the event to give the demands on each
            # officer-type in each facility_id.
            footprints_of_all_individual_level_hsi_event = [
                event_tuple.hsi_event.expected_time_requests
                for event_tuple in _list_of_individual_hsi_event_tuples
            ]

            # Compute total appointment footprint across all events
            for footprint in footprints_of_all_individual_level_hsi_event:
                # Counter.update method when called with dict-like argument adds counts
                # from argument to Counter object called from
                self.running_total_footprint.update(footprint)

            # Estimate Squeeze-Factors for today
            if self.mode_appt_constraints == 0:
                # For Mode 0 (no Constraints), the squeeze factors are all zero.
                squeeze_factor_per_hsi_event = np.zeros(
                    len(footprints_of_all_individual_level_hsi_event))
            else:
                # For Other Modes, the squeeze factors must be computed
                squeeze_factor_per_hsi_event = self.get_squeeze_factors(
                    footprints_per_event=footprints_of_all_individual_level_hsi_event,
                    total_footprint=self.running_total_footprint,
                    current_capabilities=self.capabilities_today,
                    compute_squeeze_factor_to_district_level=self.compute_squeeze_factor_to_district_level,
                )

            # For each event, determine to run or not, and run if so.
            for ev_num, event in enumerate(_list_of_individual_hsi_event_tuples):
                event = event.hsi_event
                squeeze_factor = squeeze_factor_per_hsi_event[ev_num]                  # todo use zip here!

                ok_to_run = (
                    (self.mode_appt_constraints == 0)
                    or (self.mode_appt_constraints == 1)
                    or ((self.mode_appt_constraints == 2) and (squeeze_factor == 0.0))
                )

                # Mode 0: All HSI Event run, with no squeeze
                # Mode 1: All Run With Squeeze
                # Mode 2: Only if squeeze <1

                if ok_to_run:
                    # Compute the bed days that are allocated to this HSI and provide this information to the HSI
                    # todo - only do this if some bed-days declared
                    event._received_info_about_bed_days = \
                        self.bed_days.issue_bed_days_according_to_availability(
                            facility_id=self.bed_days.get_facility_id_for_beds(persons_id=event.target),
                            footprint=event.BEDDAYS_FOOTPRINT
                        )

                    # Check that a facility has been assigned to this HSI
                    assert event.facility_info is not None, \
                        f"Cannot run HSI {event.TREATMENT_ID} without facility_info being defined."

                    # Run the HSI event (allowing it to return an updated appt_footprint)
                    actual_appt_footprint = event.run(squeeze_factor=squeeze_factor)

                    # Check if the HSI event returned updated appt_footprint
                    if actual_appt_footprint is not None:
                        # The returned footprint is different to the expected footprint: so must update load factors

                        # check its formatting:
                        assert self.appt_footprint_is_valid(actual_appt_footprint)

                        # Update load factors:
                        updated_call = self.get_appt_footprint_as_time_request(
                            facility_info=event.facility_info,
                            appt_footprint=actual_appt_footprint
                        )
                        original_call = footprints_of_all_individual_level_hsi_event[ev_num]
                        footprints_of_all_individual_level_hsi_event[ev_num] = updated_call
                        self.running_total_footprint -= original_call
                        self.running_total_footprint += updated_call

                        if self.mode_appt_constraints != 0:
                            # only need to recompute squeeze factors if running with constraints
                            # i.e. mode != 0
                            squeeze_factor_per_hsi_event = self.get_squeeze_factors(
                                footprints_per_event=footprints_of_all_individual_level_hsi_event,
                                total_footprint=self.running_total_footprint,
                                current_capabilities=self.capabilities_today,
                                compute_squeeze_factor_to_district_level=self.compute_squeeze_factor_to_district_level,
                            )
                    else:
                        # no actual footprint is returned so take the expected initial declaration as the actual
                        actual_appt_footprint = event.EXPECTED_APPT_FOOTPRINT

                    # Write to the log
                    self.record_hsi_event(
                        hsi_event=event,
                        actual_appt_footprint=actual_appt_footprint,
                        squeeze_factor=squeeze_factor,
                        did_run=True,
                    )

                else:
                    # Do not run,
                    # Call did_not_run for the hsi_event
                    rtn_from_did_not_run = event.did_not_run()

                    # If received no response from the call to did_not_run, or a True signal, then
                    # add to the hold-over queue.
                    # Otherwise (disease module returns "FALSE") the event is not rescheduled and will not run.

                    if not (rtn_from_did_not_run is False):
                        # reschedule event
                        hp.heappush(_to_be_held_over, _list_of_individual_hsi_event_tuples[ev_num])

                    # Log that the event did not run
                    self.record_hsi_event(
                        hsi_event=event,
                        actual_appt_footprint=event.EXPECTED_APPT_FOOTPRINT,
                        squeeze_factor=squeeze_factor,
                        did_run=False,
                    )

        return _to_be_held_over


class HealthSystemScheduler(RegularEvent, PopulationScopeEventMixin):
    """
    This is the HealthSystemScheduler. It is an event that occurs every day and must be the LAST event of the day.
    It inspects the calls on the healthsystem and commissions event to occur that are consistent with the
    healthsystem's capabilities for the following day, given assumptions about how this decision is made.

    N.B. Events scheduled for the same day will occur that day, but after those which were scheduled on an earlier date.

        The overall Prioritization algorithm is:
        * Look at events in order (the order is set by the heapq: see `schedule_hsi_event`)
        * Ignore if the current data is before topen
        * Remove and do nothing if tclose has expired
        * Run any  population-level HSI events
        * For an individual-level HSI event, check if there are sufficient health system capabilities to run the event

    If the event is to be run, then the following events occur:
        * The HSI event itself is run.
        * The occurence of the event is logged
        * The resources used are 'occupied' (if individual level HSI event)
        * Other disease modules are alerted of the occurence of the HSI event (if individual level HSI event)

    Here is where we can have multiple types of assumption regarding how these capabilities are modelled.
    """

    def __init__(self, module: HealthSystem):
        super().__init__(module, frequency=DateOffset(days=1), priority=Priority.END_OF_DAY)

    @staticmethod
    def _is_today_last_day_of_the_year(date):
        return (date.month == 12) and (date.day == 31)

    def _get_events_due_today(self,) -> Tuple[List, List]:
        """Interrogate the HSI_EVENT queue object to remove from it the events due today, and to return these in two
        lists:
         * list_of_individual_hsi_event_tuples_due_today
         * list_of_population_hsi_event_tuples_due_today
        """
        _list_of_individual_hsi_event_tuples_due_today = list()
        _list_of_population_hsi_event_tuples_due_today = list()
        _list_of_events_not_due_today = list()

        # To avoid repeated dataframe accesses in subsequent loop, assemble set of alive
        # person IDs as  one-off operation, exploiting the improved efficiency of
        # boolean-indexing of a Series compared to row-by-row access. From benchmarks
        # converting Series to list before converting to set is ~2x more performant than
        # direct conversion to set, while checking membership of set is ~10x quicker
        # than checking membership of Pandas Index object and ~25x quicker than checking
        # membership of list
        alive_persons = set(
            self.sim.population.props.index[self.sim.population.props.is_alive].to_list()
        )

        # Traverse the queue and split events into the three lists (due-individual, due-population, not_due)
        while len(self.module.HSI_EVENT_QUEUE) > 0:

            next_event_tuple = hp.heappop(self.module.HSI_EVENT_QUEUE)
            # Read the tuple and assemble into a dict 'next_event'

            event = next_event_tuple.hsi_event

            if self.sim.date > next_event_tuple.tclose:
                # The event has expired (after tclose) having never been run. Call the 'never_ran' function
                event.never_ran()

            elif not (
                isinstance(event.target, tlo.population.Population)
                or event.target in alive_persons
            ):
                # if individual level event and the person who is the target is no longer alive, do nothing more
                pass

            elif self.sim.date < next_event_tuple.topen:
                # The event is not yet due (before topen)
                hp.heappush(_list_of_events_not_due_today, next_event_tuple)

                if next_event_tuple.priority == 2:
                    # Check the priority
                    # If the next event is not due and has low priority, then stop looking through the heapq
                    # as all other events will also not be due.
                    break

            else:
                # The event is now due to run today and the person is confirmed to be still alive
                # Add it to the list of events due today (individual or population level)
                # NB. These list is ordered by priority and then due date

                is_pop_level_hsi_event = isinstance(event.target, tlo.population.Population)
                if is_pop_level_hsi_event:
                    _list_of_population_hsi_event_tuples_due_today.append(next_event_tuple)
                else:
                    _list_of_individual_hsi_event_tuples_due_today.append(next_event_tuple)

        # add events from the _list_of_events_not_due_today back into the queue
        while len(_list_of_events_not_due_today) > 0:
            hp.heappush(self.module.HSI_EVENT_QUEUE, hp.heappop(_list_of_events_not_due_today))

        return _list_of_individual_hsi_event_tuples_due_today, _list_of_population_hsi_event_tuples_due_today

    def apply(self, population):

        # Refresh information ready for new day:
        self.module.bed_days.on_start_of_day()
        self.module.consumables.on_start_of_day(self.sim.date)

        # Compute footprint that arise from in-patient bed-days
        inpatient_appts = self.module.bed_days.get_inpatient_appts()
        inpatient_footprints = Counter()
        for _fac_id, _footprint in inpatient_appts.items():
            inpatient_footprints.update(self.module.get_appt_footprint_as_time_request(
                facility_info=self.module._facility_by_facility_id[_fac_id], appt_footprint=_footprint)
            )

        # Write to the log that these in-patient appointments were needed:
        if len(inpatient_appts):
            for _fac_id, _inpatient_appts in inpatient_appts.items():
                self.module.write_to_hsi_log(
                    treatment_id='Inpatient_Care',
                    number_by_appt_type_code=dict(_inpatient_appts),
                    person_id=-1,
                    squeeze_factor=0.0,
                    did_run=True,
                    facility_level=self.module._facility_by_facility_id[_fac_id].level,
                    facility_id=_fac_id,
                )

        # Restart the total footprint of all calls today, beginning with those due to existing in-patients.
        self.module.running_total_footprint = inpatient_footprints

        # Create hold-over list. This will hold events that cannot occur today before they are added back to the queue.
        hold_over = list()

        # 3) Run all events due today, repeating the check for due events until none are due (this allows for HSI that
        # are added to the queue in the course of other HSI for this today to be run this day).
        while True:
            # Get the events that are due today:
            (
                list_of_individual_hsi_event_tuples_due_today,
                list_of_population_hsi_event_tuples_due_today
             ) = self._get_events_due_today()

            if (
                (len(list_of_individual_hsi_event_tuples_due_today) == 0)
                and (len(list_of_population_hsi_event_tuples_due_today) == 0)
            ):
                break

            # Run the list of population-level HSI events
            self.module.run_population_level_events(list_of_population_hsi_event_tuples_due_today)

            # Run the list of individual-level events
            _to_be_held_over = self.module.run_individual_level_events(
                list_of_individual_hsi_event_tuples_due_today
            )
            hold_over.extend(_to_be_held_over)

        # -- End-of-day activities --
        # Add back to the HSI_EVENT_QUEUE heapq all those events which are still eligible to run but which did not run
        while len(hold_over) > 0:
            hp.heappush(self.module.HSI_EVENT_QUEUE, hp.heappop(hold_over))

        # Log total usage of the facilities
        self.module.log_current_capabilities_and_usage()

        # Trigger jobs to be done at the end of the day (after all HSI run)
        self.module.on_end_of_day()

        # Do activities that are required at end of year (if today is the last day of the year)
        if self._is_today_last_day_of_the_year(self.sim.date):
            self.module.on_end_of_year()

# ---------------------------------------------------------------------------
#   Logging
# ---------------------------------------------------------------------------


class HealthSystemSummaryCounter:
    """Helper class to keep running counts of HSI and the state of the HealthSystem and logging summaries."""

    def __init__(self):
        self._reset_internal_stores()

    def _reset_internal_stores(self) -> None:
        """Create empty versions of the data structures used to store a running records."""

        self._treatment_ids = defaultdict(int)  # Running record of the `TREATMENT_ID`s of `HSI_Event`s
        self._appts = defaultdict(int)  # Running record of the Appointments of `HSI_Event`s that have run
        self._appts_by_level = {_level: defaultdict(int) for _level in ('0', '1a', '1b', '2', '3', '4')}
        # <--Same as `self._appts` but also split by facility_level
        self._frac_time_used_overall = []  # Running record of the usage of the healthcare system

    def record_hsi_event(self, treatment_id: str, appt_footprint: Counter, level: str) -> None:
        """Add information about an `HSI_Event` to the running summaries."""

        # Count the treatment_id:
        self._treatment_ids[treatment_id] += 1

        # Count each type of appointment:
        for _appt_type, _number in appt_footprint.items():
            self._appts[_appt_type] += _number
            self._appts_by_level[level][_appt_type] += _number

    def record_hs_status(self, fraction_time_used_across_all_facilities: float) -> None:
        """Record a current status metric of the HealthSystem."""

        # The fraction of all healthcare worker time that is used:
        self._frac_time_used_overall.append(fraction_time_used_across_all_facilities)

    def write_to_log_and_reset_counters(self):
        """Log summary statistics reset the data structures."""

        logger_summary.info(
            key="HSI_Event",
            description="Counts of the HSI_Events that have occurred in this calendar year by TREATMENT_ID, "
                        "and counts of the 'Appt_Type's that have occurred in this calendar year.",
            data={
                "TREATMENT_ID": self._treatment_ids,
                "Number_By_Appt_Type_Code": self._appts,
                "Number_By_Appt_Type_Code_And_Level": self._appts_by_level,
            },
        )

        logger_summary.info(
            key="Capacity",
            description="The fraction of all the healthcare worker time that is used each day, averaged over this "
                        "calendar year.",
            data={
                "average_Frac_Time_Used_Overall": np.mean(self._frac_time_used_overall),
                # <-- leaving space here for additional summary measures that may be needed in the future.
            },
        )

        self._reset_internal_stores()


class HealthSystemChangeParameters(Event, PopulationScopeEventMixin):
    """Event that causes certain internal parameters of the HealthSystem to be changed; specifically:
        * `mode_appt_constraints`
        * `ignore_priority`
        * `capabilities_coefficient`
        * `cons_availability`
        * `beds_availability`
    Note that no checking is done here on the suitability of values of each parameter."""

    def __init__(self, module: HealthSystem, parameters: Dict):
        super().__init__(module)
        self._parameters = parameters
        assert isinstance(module, HealthSystem)

    def apply(self, population):
        if 'mode_appt_constraints' in self._parameters:
            self.module.mode_appt_constraints = self._parameters['mode_appt_constraints']

        if 'ignore_priority' in self._parameters:
            self.module.ignore_priority = self._parameters['ignore_priority']

        if 'capabilities_coefficient' in self._parameters:
            self.module.capabilities_coefficient = self._parameters['capabilities_coefficient']

        if 'cons_availability' in self._parameters:
            self.module.consumables = Consumables(data=self.module.parameters['availability_estimates'],
                                                  rng=self.module.rng,
                                                  availability=self._parameters['cons_availability'])
            self.module.consumables.on_start_of_day(self.module.sim.date)

        if 'beds_availability' in self._parameters:
            self.module.bed_days.availability = self._parameters['beds_availability']
