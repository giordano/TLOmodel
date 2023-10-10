import argparse
from pathlib import Path

import pandas as pd

from tlo.analysis.utils import extract_results


# TODO: make it as a fnc usable for both levels/hsi_events
def get_annual_equipment_declarations_by_levels(results_folder: Path) -> pd.DataFrame:
    """Return pd.DataFrame gives the simulated annual equipment declaration by facility levels for each simulated
    year.
    NB. healthsystem.summary logger required to have been set at the level INFO or higher."""

    def get_equipment_declaration_by_levels(_df):
        """Get the equipment declaration by facility levels for the year."""

        def unpack_dict_in_series(_raw: pd.Series):
            # Create an empty DataFrame to store the data
            df = pd.DataFrame()

            # Iterate through the dictionary items
            for col_name, mydict in _raw.items():
                for date, inner_dict in mydict.items():
                    # Convert the inner_dict to a list of dictionaries with 'date'
                    data = [{'date': date, 'fac_level': inner_dict_key, 'value': inner_dict_set} for
                            inner_dict_key, inner_dict_set in inner_dict.items()]
                    # Create a DataFrame from the list with date & fac_level as indexes
                    temp_df = pd.DataFrame(data)
                    temp_df.set_index(['date', 'fac_level'], inplace=True)
                    temp_df.columns = [None]

                    # Concatenate the temporary DataFrame to the result DataFrame
                    df = pd.concat([df, temp_df])

            df.columns = [None]

            return df

        return _df \
            .set_index('date') \
            .pipe(unpack_dict_in_series) \
            .stack() \
            .droplevel(level=2)

    return extract_results(
        results_folder,
        module='tlo.methods.healthsystem.summary',
        key='Equipment By Facility Level',
        custom_generate_series=get_equipment_declaration_by_levels
        )


def get_annual_equipment_declarations_by_hsi_event_names(results_folder: Path) -> pd.DataFrame:
    """Return pd.DataFrame gives the simulated annual equipment declaration by hsi event names for each simulated year.
    NB. healthsystem.summary logger required to have been set at the level INFO or higher."""

    def get_equipment_declaration_by_hsi_event_names(_df):
        """Get the equipment declaration by hsi event names for the year."""

        def unpack_dict_in_series(_raw: pd.Series):
            # Create an empty DataFrame to store the data
            df = pd.DataFrame()

            # Iterate through the dictionary items
            for col_name, mydict in _raw.items():
                for date, inner_dict in mydict.items():
                    # Convert the inner_dict to a list of dictionaries with 'date'
                    data = [{'date': date, 'hsi_event_name': inner_dict_key, 'value': inner_dict_set} for
                            inner_dict_key, inner_dict_set in inner_dict.items()]
                    # Create a DataFrame from the list with date & hsi_event_name as indexes
                    temp_df = pd.DataFrame(data)
                    temp_df.set_index(['date', 'hsi_event_name'], inplace=True)
                    temp_df.columns = [None]

                    # Concatenate the temporary DataFrame to the result DataFrame
                    df = pd.concat([df, temp_df])

            df.columns = [None]

            return df

        return _df \
            .set_index('date') \
            .pipe(unpack_dict_in_series) \
            .stack() \
            .droplevel(level=2)

    return extract_results(
        results_folder,
        module='tlo.methods.healthsystem.summary',
        key='Equipment by HSI event name',
        custom_generate_series=get_equipment_declaration_by_hsi_event_names
        )


def create_equipment_catalogues(results_folder: Path, output_folder: Path):

    # %% Catalogue equipment by Facility Levels
    # Declare output file names
    output_file_name = 'equipment_catalogue_by_level.csv'
    output_detailed_file_name = 'equipment_catalogue_by_date_level_sim.csv'

    sim_equipment = get_annual_equipment_declarations_by_levels(results_folder)
    sim_equipment_df = pd.DataFrame(sim_equipment)
    sim_equipment_df.index.names = ['date', 'fac_level']

    # Save the detailed equipment catalogue by levels
    sim_equipment_df.to_csv(output_folder / output_detailed_file_name)
    print('equipment_catalogue_by_date_level_sim.csv saved.')
    # ---

    # Prepare a catalogue only by facility levels
    def combine_sets(row):
        """Combines sets in columns for each row."""
        combined_set = set()
        for col in row:
            combined_set.update(col)
        return combined_set

    # Combine sets from all draws and runs
    sim_equipment_by_level_df = sim_equipment_df.copy()
    sim_equipment_by_level_df['equipment'] = sim_equipment_by_level_df.apply(combine_sets, axis=1)
    # Group by 'fac_level' and join rows with the same 'fac_level' into one set
    sim_equipment_by_level_df.reset_index(inplace=True)
    sim_equipment_by_level_df = sim_equipment_by_level_df.groupby('fac_level')['equipment'].apply(
        lambda x: list(set.union(*x))
    ).reset_index()

    # Explode the 'equipment' column to separate rows, each row containing one item from the equipment set
    sim_equipment_by_level_df = sim_equipment_by_level_df.explode('equipment', ignore_index=True).set_index('fac_level')

    # Save the CSV equipment catalogue by level
    sim_equipment_by_level_df.to_csv(output_folder / output_file_name)
    print('equipment_catalogue_by_level.csv saved.')
    # ---

    # %% Catalogue equipment by HSI Event Names
    # Declare output file names
    output_file_name = 'equipment_catalogue_by_hsi_event_name.csv'
    output_detailed_file_name = 'equipment_catalogue_by_date_hsi_sim.csv'

    sim_equipment = get_annual_equipment_declarations_by_hsi_event_names(results_folder)
    sim_equipment_df = pd.DataFrame(sim_equipment)
    sim_equipment_df.index.names = ['date', 'hsi_event_name']

    # Save the detailed equipment catalogue by hsi events
    sim_equipment_df.to_csv(output_folder / output_detailed_file_name)
    print('equipment_catalogue_by_date_hsi_sim.csv saved.')
    # ---

    # Prepare a catalogue only by hsi events
    sim_equipment_by_hsi_df = sim_equipment_df.copy()

    # Replace missing equipment (in cases when HSI didn't run in that sim for that date) with empty list
    def replace_nan_with_empty_list(val):
        if isinstance(val, list):
            return val
        return []
    sim_equipment_by_hsi_df = sim_equipment_by_hsi_df.applymap(replace_nan_with_empty_list)

    # Combine sets from all draws and runs
    def combine_sets(row):
        """Combines sets in columns for each row."""
        combined_set = set()
        for col in row:
            combined_set.update(col)
        return combined_set
    sim_equipment_by_hsi_df['equipment'] = sim_equipment_by_hsi_df.apply(combine_sets, axis=1)

    # Group by 'hsi_event_name' and join rows with the same 'hsi_event_name' into one set
    sim_equipment_by_hsi_df.reset_index(inplace=True)
    sim_equipment_by_hsi_df = sim_equipment_by_hsi_df.groupby('hsi_event_name')['equipment'].apply(
        lambda x: list(set.union(*x))
    ).reset_index()

    # Explode the 'equipment' column to separate rows, each row containing one item from the equipment set
    sim_equipment_by_hsi_df = \
        sim_equipment_by_hsi_df.explode('equipment', ignore_index=True).set_index('hsi_event_name')

    # Save the CSV equipment catalogue by hsi_event_name
    sim_equipment_by_hsi_df.to_csv(output_folder / output_file_name)
    print('equipment_catalogue_by_hsi_event_name.csv saved.')
    # ---

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_folder", type=Path)
    args = parser.parse_args()

    create_equipment_catalogues(
        results_folder=args.results_folder,
        output_folder=args.results_folder,
    )
# NB. Edit run configuration, the Parameters: "./outputs/sejjej5@ucl.ac.uk/long_run_all_diseases-2023-09-04T233551Z"
