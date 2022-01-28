"""
General utility functions for TLO analysis
"""
import json
import os
import pickle
import gzip
from pathlib import Path
from typing import Dict, Optional, TextIO

import numpy as np
import pandas as pd

from tlo import logging, util
from tlo.logging.reader import LogData
from tlo.util import create_age_range_lookup

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _parse_log_file_inner_loop(filepath, level: int = logging.INFO):
    """Parses the log file and returns dictionary of dataframes"""
    log_data = LogData()
    with open(filepath) as log_file:
        for line in log_file:
            # only parse json entities
            if line.startswith('{'):
                log_data.parse_log_line(line, level)
            else:
                print('FAILURE: found old-style log:')
                print(line)
                raise RuntimeError
    # convert dictionaries to dataframes
    output_logs = {**log_data.get_log_dataframes()}
    return output_logs


def parse_log_file(log_filepath, level: int = logging.INFO):
    """Parses logged output from a TLO run and returns Pandas dataframes.

    The dictionary returned has the format::

        {
            <logger 1 name>: {
                               <log key 1>: <pandas dataframe>,
                               <log key 2>: <pandas dataframe>,
                               <log key 3>: <pandas dataframe>
                             },

            <logger 2 name>: {
                               <log key 4>: <pandas dataframe>,
                               <log key 5>: <pandas dataframe>,
                               <log key 6>: <pandas dataframe>
                             },
            ...
        }

    :param log_filepath: file path to log file
    :param level: logging level to be parsed for structured logging
    :return: dictionary of parsed log data
    """
    print(f'Processing log file {log_filepath}')
    uuid_to_module_name: Dict[str, str] = dict()  # uuid to module name
    module_name_to_filehandle: Dict[str, TextIO] = dict()  # module name to file handle

    log_directory = Path(log_filepath).parent
    print(f'Writing module-specific log files to {log_directory}')

    # iterate over each line in the logfile
    with open(log_filepath) as log_file:
        for line in log_file:
            # only parse lines that are json log lines (old-style logging is not supported)
            if line.startswith('{'):
                log_data_json = json.loads(line)
                uuid = log_data_json['uuid']
                # if this is a header line (only header lines have a `type` key)
                if 'type' in log_data_json:
                    module_name = log_data_json["module"]
                    uuid_to_module_name[uuid] = module_name
                    # we only need to create the file if we don't already have one for this module
                    if module_name not in module_name_to_filehandle:
                        module_name_to_filehandle[module_name] = open(log_directory / f"{module_name}.log", mode="w")
                # copy line from log file to module-specific log file (both headers and non-header lines)
                module_name_to_filehandle[uuid_to_module_name[uuid]].write(line)

    print('Finished writing module-specific log files.')

    # close all module-specific files
    for file_handle in module_name_to_filehandle.values():
        file_handle.close()

    # parse each module-specific log file and collect the results into a single dictionary. metadata about each log
    # is returned in the same key '_metadata', so it needs to be collected separately and then merged back in.
    all_module_logs = dict()
    metadata = dict()
    for file_handle in module_name_to_filehandle.values():
        print(f'Parsing {file_handle.name}', end='', flush=True)
        module_specific_logs = _parse_log_file_inner_loop(file_handle.name, level)
        print(' - complete.')
        all_module_logs.update(module_specific_logs)
        # sometimes there is nothing to be parsed at a given level, so no metadata
        if 'metadata_' in module_specific_logs:
            metadata.update(module_specific_logs['_metadata'])

    if len(metadata) > 0:
        all_module_logs['_metadata'] = metadata

    print('Finished.')

    return all_module_logs


def write_log_to_excel(filename, log_dataframes):
    """Takes the output of parse_log_file() and creates an Excel file from dataframes"""
    sheets = list()
    sheet_count = 0
    metadata = log_dataframes['_metadata']
    for module, key_df in log_dataframes.items():
        if module != '_metadata':
            for key, df in key_df.items():
                sheet_count += 1
                sheets.append([module, key, sheet_count, metadata[module][key]['description']])

    writer = pd.ExcelWriter(filename)
    index = pd.DataFrame(data=sheets, columns=['module', 'key', 'sheet', 'description'])
    index.to_excel(writer, sheet_name='Index')

    sheet_count = 0
    for module, key_df in log_dataframes.items():
        if module != '_metadata':
            for key, df in key_df.items():
                sheet_count += 1
                df.to_excel(writer, sheet_name=f'Sheet {sheet_count}')
    writer.save()


def make_calendar_period_lookup():
    """Returns a dictionary mapping calendar year (in years) to five year period
    i.e. { 1950: '1950-1954', 1951: '1950-1954, ...}
    """

    # Recycles the code used to make age-range lookups:
    ranges, lookup = util.create_age_range_lookup(1950, 2100, 5)

    # Removes the '0-1950' category
    ranges.remove('0-1950')

    for year in range(1950):
        lookup.pop(year)

    return ranges, lookup


def make_calendar_period_type():
    """
    Make an ordered categorical type for calendar periods
    Returns CategoricalDType
    """
    keys, _ = make_calendar_period_lookup()
    return pd.CategoricalDtype(categories=keys, ordered=True)


def make_age_grp_lookup():
    """Returns a dictionary mapping age (in years) to five year period
    i.e. { 0: '0-4', 1: '0-4', ..., 119: '100+', 120: '100+' }
    """
    return create_age_range_lookup(min_age=0, max_age=100, range_size=5)


def make_age_grp_types():
    """
    Make an ordered categorical type for age-groups
    Returns CategoricalDType
    """
    keys, _ = create_age_range_lookup(min_age=0, max_age=100, range_size=5)
    return pd.CategoricalDtype(categories=keys, ordered=True)


def get_scenario_outputs(scenario_filename: str, outputs_dir: Path) -> list:
    """Returns paths of folders associated with a batch_file, in chronological order."""
    stub = scenario_filename.rstrip('.py')
    f: os.DirEntry
    folders = [Path(f.path) for f in os.scandir(outputs_dir) if f.is_dir() and f.name.startswith(stub)]
    folders.sort()
    return folders


def get_scenario_info(scenario_output_dir: Path) -> dict:
    """Utility function to get the the number draws and the number of runs in a batch set.

    TODO: read the JSON file to get further information
    """
    info = dict()
    f: os.DirEntry
    draw_folders = [f for f in os.scandir(scenario_output_dir) if f.is_dir()]

    info['number_of_draws'] = len(draw_folders)

    run_folders = [f for f in os.scandir(draw_folders[0]) if f.is_dir()]
    info['runs_per_draw'] = len(run_folders)

    return info


def load_pickled_dataframes(results_folder: Path, draw=0, run=0, name=None) -> dict:
    """Utility function to create a dict contaning all the logs from the specified run within a batch set"""
    folder = results_folder / str(draw) / str(run)
    p: os.DirEntry
    pickles = [p for p in os.scandir(folder) if p.name.endswith('.pickle')]
    if name is not None:
        pickles = [p for p in pickles if p.name in f"{name}.pickle"]

    output = dict()
    for p in pickles:
        name = os.path.splitext(p.name)[0]
        with open(p.path, "rb") as f:
            output[name] = pickle.load(f)

    return output


def extract_params_from_json(results_folder: Path, file_name: str, module_name: str, param_name: str):
    """Utility function to get overridden parameters from scenario runs from the json file

    returns the parameter values that change over the runs as a list
    """

    with open(str(results_folder) + '/' + file_name[:-3] + '_draws.json', 'r') as myfile:
        data = myfile.read()
    params_file = json.loads(data)
    params_in_draws = []
    for runs in params_file['draws']:
        params_in_draws.append(params_file['draws'][runs['draw_number']]['parameters'][module_name][param_name])
    return params_in_draws


def get_failed_batch_run_information(results_folder: Path, file_name: str, draw_number: int, run_number: int):
    """Utility function to recreate a particular draw from a batch run locally

    You give this function the results folder, file name, and particular draw number and it will return everything
    needed to recreate the model run locally. Specifically, the seed of the run, the parameters overwritten in a run,
    the population size of the run and the date the simulation started
    # todo: try and make this function a one liner:
        1) Make function call and run simulation
        2) Get a list of registered modules in simulation and order of module registration as this effects rng
        3) Get an end date for the simulation
    # todo: Find a way to make function work without unzipping the stdout.txt.gz file, I tried to do this but had a
            permission error
    """
    # get the location of the failed batch run
    file_location = str(results_folder) + '\\' + str(draw_number) + '\\' + str(run_number)
    # get the outputted log of the failed run, note that currently this file will need to be manually extracted prior to
    # this
    std_out = file_location + '\\' + 'stdout.txt'
    # create default value of seed incase actual seed not found
    seed = - 1
    # extract demographic information from simulations using extract results function
    extracted_pop_size = extract_results(results_folder,
                                         module="tlo.methods.demography",
                                         key="population",
                                         column="total",
                                         index="date")
    # get the first logged (initial) population size of the first draw, in the first run
    popsize = int(extracted_pop_size[0][0][0])
    # from these results get the start date
    start_date = extracted_pop_size.index[0]
    # read the text file and search for simulation seed value
    with open(std_out, 'r') as f:
        text = f.read()
        # search for the seed
        if 'Simulation RNG user entropy = ' in text:
            # find the index of txt prior to the start of the seed number
            seed_start_index = text.find('Simulation RNG user entropy = ') + len('Simulation RNG user entropy = ')
            # find the index of the txt file at the end of the seen number
            seed_end_index = text.find('"', seed_start_index)
            # from this get the seed number
            seed = int(text[seed_start_index: seed_end_index])
        # search for modules registered
        if 'RNG auto' in text:
            # find last registered module
            last_occurence_index = text.rfind(" RNG auto")
            # the text to search for registered modules is between the initial assignment of the seed and the last
            # registered module, filter text to search through by these bounds
            text_filtered = text[seed_end_index: last_occurence_index]
            # create a list to store the name and order of the registered modules
            module_names = []
            # create an index point to start the search at, which is updated in the while loop below
            index_searched_through = 0
            # loop over the filtered text while the index searched through is less than the length of the text
            while index_searched_through < len(text_filtered) - 1:
                # find the index of the character after the module name is registered starting from the value of the
                # last index searched from
                character_index_after_module_name = text_filtered.find(" RNG auto", index_searched_through + 1)
                # find the index of the character before the name is registered starting from the value of the
                # last index searched from
                character_index_before_module_name = text_filtered.find(': ["', index_searched_through)
                # append the module name to the list of module names
                module_names.append(text_filtered[character_index_before_module_name + len(': ["'):
                                                  character_index_after_module_name])
                # update the index searched through parameter so in the next loop, the search will begin from the
                # character after the last module
                index_searched_through = character_index_after_module_name
                # Use a property of the str.find function to break the loop (str.find returns -1 if the substring you
                # are searching for isn't found, in this context it means that all module names have been found)
                if character_index_before_module_name == -1:
                    break
        # search for the last date logged in text file
        if '"date": "' in text:
            # Find the index of the last logged "date": " in the txt file and then get the index after
            last_recorded_date_index = text.rfind('"date": "') + len('"date": "')
            last_recorded_values_index = text.rfind('", "values"')
            last_recorded_date = text[last_recorded_date_index: last_recorded_values_index]
            end_date = pd.to_datetime(last_recorded_date, format = '%Y-%m-%d %H:%M:%S', errors='coerce')



    # get the parameter value(s) used in this particular run
    # open json file as dictionary
    with open(str(results_folder) + '/' + file_name[:-3] + '_draws.json', 'r') as myfile:
        data = myfile.read()
    params_file = json.loads(data)
    # get the parameters overwritten in this simulation
    params_in_draw = params_file['draws'][draw_number]['parameters']
    return seed, params_in_draw, popsize, start_date


def extract_params(results_folder: Path) -> Optional[pd.DataFrame]:
    """Utility function to get overridden parameters from scenario runs

    Returns dateframe summarizing parameters that change across the draws. It produces a dataframe with index of draw
    and columns of each parameters that is specified to be varied in the batch. NB. This does the extraction from run 0
    in each draw, under the assumption that the over-written parameters are the same in each run.
    """

    try:
        f: os.DirEntry
        # Get the paths for the draws
        draws = [f for f in os.scandir(results_folder) if f.is_dir()]

        list_of_param_changes = list()

        for d in draws:
            p = load_pickled_dataframes(results_folder, d.name, 0, name="tlo.scenario")
            p = p["tlo.scenario"]["override_parameter"]

            p['module_param'] = p['module'] + ':' + p['name']
            p.index = [int(d.name)] * len(p.index)

            list_of_param_changes.append(p[['module_param', 'new_value']])

        params = pd.concat(list_of_param_changes)
        params.index.name = 'draw'
        params = params.rename(columns={'new_value': 'value'})
        params = params.sort_index()
        return params

    except KeyError:
        print("No parameters changed between the runs")
        return None


def extract_results(results_folder: Path,
                    module: str,
                    key: str,
                    column: str = None,
                    index: str = None,
                    custom_generate_series=None,
                    do_scaling: bool = False,
                    ) -> pd.DataFrame:
    """Utility function to unpack results

    Produces a dataframe that summaries one series from the log, with column multi-index for the draw/run. If an 'index'
    component of the log_element is provided, the dataframe uses that index (but note that this will only work if the
    index is the same in each run).
    Optionally, instead of a series that exists in the dataframe already, a function can be provided that, when applied
    to the dataframe indicated, yields a new pd.Series.
    Optionally, with `do_scaling`, each element is multiplied by the the scaling_factor recorded in the simulation
    (if available)
    """

    # get number of draws and numbers of runs
    info = get_scenario_info(results_folder)

    cols = pd.MultiIndex.from_product(
        [range(info['number_of_draws']), range(info['runs_per_draw'])],
        names=["draw", "run"]
    )

    def get_multiplier(_draw, _run):
        """Helper function to get the multiplier from the simulation, if it's specified and do_scaling=True"""
        if not do_scaling:
            return 1.0
        else:
            try:
                return load_pickled_dataframes(results_folder, _draw, _run, 'tlo.methods.demography'
                                               )['tlo.methods.demography']['scaling_factor']['scaling_factor'].values[0]
            except KeyError:
                return 1.0

    if custom_generate_series is None:

        assert column is not None, "Must specify which column to extract"

        results_index = None
        if index is not None:
            # extract the index from the first log, and use this ensure that all other are exactly the same.
            filename = f"{module}.pickle"
            df: pd.DataFrame = load_pickled_dataframes(results_folder, draw=0, run=0, name=filename)[module][key]
            results_index = df[index]

        results = pd.DataFrame(columns=cols)
        for draw in range(info['number_of_draws']):
            for run in range(info['runs_per_draw']):

                try:
                    df: pd.DataFrame = load_pickled_dataframes(results_folder, draw, run, module)[module][key]
                    results[draw, run] = df[column] * get_multiplier(draw, run)

                    if index is not None:
                        idx = df[index]
                        assert idx.equals(results_index), "Indexes are not the same between runs"

                except KeyError:
                    results[draw, run] = np.nan

        # if 'index' is provided, set this to be the index of the results
        if index is not None:
            results.index = results_index

        return results

    else:
        # A custom commaand to generate a series has been provided.
        # No other arguements should be provided.
        assert index is None, "Cannot specify an index if using custom_generate_series"
        assert column is None, "Cannot specify a column if using custom_generate_series"

        # Collect results and then use pd.concat as indicies may be different betweeen runs
        res = dict()
        for draw in range(info['number_of_draws']):
            for run in range(info['runs_per_draw']):
                df: pd.DataFrame = load_pickled_dataframes(results_folder, draw, run, module)[module][key]
                output_from_eval = custom_generate_series(df)
                assert pd.Series == type(output_from_eval), 'Custom command does not generate a pd.Series'
                res[f"{draw}_{run}"] = output_from_eval * get_multiplier(draw, run)
        results = pd.concat(res.values(), axis=1).fillna(0)
        results.columns = cols

        return results


def summarize(results: pd.DataFrame, only_mean: bool = False, collapse_columns: bool = False) -> pd.DataFrame:
    """Utility function to compute summary statistics

    Finds mean value and 95% interval across the runs for each draw.
    """

    summary = pd.DataFrame(
        columns=pd.MultiIndex.from_product(
            [
                results.columns.unique(level='draw'),
                ["mean", "lower", "upper"]
            ],
            names=['draw', 'stat']),
        index=results.index
    )

    summary.loc[:, (slice(None), "mean")] = results.groupby(axis=1, by='draw').mean().values
    summary.loc[:, (slice(None), "lower")] = results.groupby(axis=1, by='draw').quantile(0.025).values
    summary.loc[:, (slice(None), "upper")] = results.groupby(axis=1, by='draw').quantile(0.975).values

    if only_mean and (not collapse_columns):
        # Remove other metrics and simplify if 'only_mean' across runs for each draw is required:
        om: pd.DataFrame = summary.loc[:, (slice(None), "mean")]
        om.columns = [c[0] for c in om.columns.to_flat_index()]
        return om

    elif collapse_columns and (len(summary.columns.levels[0]) == 1):
        # With 'collapse_columns', if number of draws is 1, then collapse columns multi-index:
        summary_droppedlevel = summary.droplevel('draw', axis=1)
        if only_mean:
            return summary_droppedlevel['mean']
        else:
            return summary_droppedlevel

    else:
        return summary


def get_grid(params: pd.DataFrame, res: pd.Series):
    """Utility function to create the arrays needed to plot a heatmap.

    :param pd.DataFrame params: the dataframe of parameters with index=draw (made using `extract_params()`).
    :param pd.Series res: results of interest with index=draw (can be made using `extract_params()`)
    :returns: grid as dictionary
    """
    res = pd.concat([params.pivot(columns='module_param', values='value'), res], axis=1)
    piv = res.pivot_table(index=res.columns[0], columns=res.columns[1], values=res.columns[2])

    grid = dict()
    grid[res.columns[0]], grid[res.columns[1]] = np.meshgrid(piv.index, piv.columns)
    grid[res.columns[2]] = piv.values

    return grid


def format_gbd(gbd_df: pd.DataFrame):
    """Format GBD data to give standarize categories for age_group and period"""

    # Age-groups:
    gbd_df['Age_Grp'] = gbd_df['Age_Grp'].astype(make_age_grp_types())

    # label periods:
    calperiods, calperiodlookup = make_calendar_period_lookup()
    gbd_df['Period'] = gbd_df['Year'].map(calperiodlookup).astype(make_calendar_period_type())

    return gbd_df


def create_pickles_locally(scenario_output_dir):
    """For a run from the Batch system that has not resulted in the creation of the pickles, reconstruct the pickles
     locally."""

    def turn_log_into_pickles(logfile):
        print(f"Opening {logfile}")
        outputs = parse_log_file(logfile)
        for key, output in outputs.items():
            if key.startswith("tlo."):
                print(f" - Writing {key}.pickle")
                with open(logfile.parent / f"{key}.pickle", "wb") as f:
                    pickle.dump(output, f)

    f: os.DirEntry
    draw_folders = [f for f in os.scandir(scenario_output_dir) if f.is_dir()]
    for draw_folder in draw_folders:
        run_folders = [f for f in os.scandir(draw_folder) if f.is_dir()]
        for run_folder in run_folders:
            logfile = [x for x in os.listdir(run_folder) if x.endswith('.log')][0]
            turn_log_into_pickles(Path(run_folder) / logfile)


def compare_number_of_deaths(logfile: Path, resourcefilepath: Path):
    """Helper function to produce tables summarising deaths in the model run (given be a logfile) and the corresponding
    number of deaths in the GBD dataset.
    NB.
    * Requires output from the module `tlo.methods.demography`
    * Will do scaling automatically if the scaling-factor has been computed in the simulation (but not otherwise).
    """
    output = parse_log_file(logfile)

    # 1) Get model outputs:
    # - get scaling factor if it has been computed:
    if 'scaling_factor' in output['tlo.methods.demography']:
        sf = output['tlo.methods.demography']['scaling_factor']['scaling_factor'].values[0]
    else:
        sf = 1.0

    # - extract number of death by period/sex/age-group
    model = output['tlo.methods.demography']['death'].assign(
        year=lambda x: x['date'].dt.year
    ).groupby(
        ['sex', 'year', 'age', 'label']
    )['person_id'].count().mul(sf)

    # - format categories:
    agegrps, agegrplookup = make_age_grp_lookup()
    calperiods, calperiodlookup = make_calendar_period_lookup()
    model = model.reset_index()
    model['age_grp'] = model['age'].map(agegrplookup).astype(make_age_grp_types())
    model['period'] = model['year'].map(calperiodlookup).astype(make_calendar_period_type())
    model = model.drop(columns=['age', 'year'])

    # - sum over period and divide by five to give yearly averages
    model = model.groupby(['period', 'sex', 'age_grp', 'label']).sum().div(5.0).rename(
        columns={'person_id': 'model'}).replace({0: np.nan})

    # 2) Load comparator GBD datasets
    # - Load data, format and limit to deaths only:
    gbd_dat = format_gbd(pd.read_csv(resourcefilepath / 'gbd' / 'ResourceFile_Deaths_And_DALYS_GBD2019.csv'))
    gbd_dat = gbd_dat.loc[gbd_dat['measure_name'] == 'Deaths']
    gbd_dat = gbd_dat.rename(columns={
        'Sex': 'sex',
        'Age_Grp': 'age_grp',
        'Period': 'period',
        'GBD_Est': 'mean',
        'GBD_Lower': 'lower',
        'GBD_Upper': 'upper'})

    # - Label GBD causes of death by 'label' defined in the simulation
    mapper_from_gbd_causes = pd.Series(
        output['tlo.methods.demography']['mapper_from_gbd_cause_to_common_label'].drop(columns={'date'}).loc[0]
    ).to_dict()
    gbd_dat['label'] = gbd_dat['cause_name'].map(mapper_from_gbd_causes)
    assert not gbd_dat['label'].isna().any()

    # - Create comparable data structure:
    gbd = gbd_dat.groupby(['period', 'sex', 'age_grp', 'label'])[['mean', 'lower', 'upper']].sum().div(5.0)
    gbd = gbd.add_prefix('GBD_')

    # 3) Return summary
    return gbd.merge(model, on=['period', 'sex', 'age_grp', 'label'], how='left')


def flatten_multi_index_series_into_dict_for_logging(ser: pd.Series) -> dict:
    """Helper function that converts a pd.Series with multi-index into a dict format that is suitable for logging.
    It does this by converting the multi-index into keys of type `str` in a format that later be used to reconstruct
    the multi-index (using `unflatten_flattened_multi_index_in_logging`)."""

    assert not ser.index.has_duplicates, "There should not be any duplicates in the multi-index. These will be lost" \
                                         "in the conversion to a dict."

    names_of_multi_index = ser.index.names
    _df = ser.reset_index()
    flat_index = list()
    for _, row in _df.iterrows():
        flat_index.append('|'.join([f"{col}={row[col]}" for col in names_of_multi_index]))
    return dict(zip(flat_index, ser.values))


def unflatten_flattened_multi_index_in_logging(_x: pd.DataFrame) -> pd.DataFrame:
    """Helper function that recreate the multi-index of logged results from a pd.DataFrame that is generated by
    `parse_log`.
    If a pd.DataFrame created by `parse_log` is the result of repeated logging of a pd.Series with a multi-index that
    was transformed before logging using `flatten_multi_index_series_into_dict_for_logging`, then the pd.DataFrame's
    columns will be those flattened labels. This helper function recreates the original multi-index from which the
    flattened labels were created and applies it to the pd.DataFrame."""
    cols = _x.columns
    index_value_list = list()
    for col in cols.str.split('|'):
        index_value_list.append(tuple(component.split('=')[1] for component in col))
    index_name_list = tuple(component.split('=')[0] for component in cols[0].split('|'))
    _y = _x.copy()
    _y.columns = pd.MultiIndex.from_tuples(index_value_list, names=index_name_list)
    return _y
