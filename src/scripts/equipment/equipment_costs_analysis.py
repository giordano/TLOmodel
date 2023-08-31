import argparse
import pandas as pd
from pathlib import Path
from tlo.analysis.utils import extract_results, summarize


def get_monthly_num_events(results_folder: Path) -> pd.DataFrame:
    """Return pd.DataFrame gives the (mean) simulated monthly number of hsi events."""

    def get_counts_of_events_detailed(_df):
        """Get the mean number of appointments with the same (event_name, module_name, treatment_id", facility_level,
        appt_footprint and beddays_footprint) for each month."""

        def unpack_nested_dict_in_series(_raw: pd.Series):
            # Create an empty DataFrame to store the data
            df = pd.DataFrame()

            # Iterate through the dictionary items
            for col_name, mydict in _raw.items():
                for key, inner_dict in mydict.items():
                    # print("\nkey:\n", key, "\ninner_dict:\n", inner_dict)
                    # Convert the inner dictionary to a temporary DataFrame with a single row
                    temp_df = pd.DataFrame(inner_dict, index=[key])

                    # Concatenate the temporary DataFrame to the result DataFrame
                    df = pd.concat([df, temp_df])

            return df

        return _df \
            .set_index('date') \
            .pipe(unpack_nested_dict_in_series) \
            .unstack()

    return summarize(
        extract_results(
            results_folder,
            module='tlo.methods.healthsystem.summary',
            key='hsi_event_counts',
            custom_generate_series=get_counts_of_events_detailed,
            do_scaling=True
        ),
        only_mean=True,
        collapse_columns=True,
    )


def analyse_equipment_costs(results_folder: Path, output_folder: Path, resourcefilepath: Path = None):
    # Declare path for output file from this script
    output_file_name = 'equipment_monthly_costs.csv'
    equipment = pd.read_csv(Path(resourcefilepath) / 'ResourceFile_Equipment.csv')

    sim_treatments = get_monthly_num_events(results_folder)
    sim_treatments_df = pd.DataFrame(sim_treatments)
    sim_treatments_df.index.names = ['hsi_event_log_code', 'date']
    sim_treatments_df.columns = ['mean_numb_events']

    sim_treatments_df.to_csv(output_folder / output_file_name)

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_folder", type=Path)
    args = parser.parse_args()

    analyse_equipment_costs(
        results_folder=args.results_folder,
        output_folder=args.results_folder,
        resourcefilepath=Path('./resources/healthsystem/infrastructure_and_equipment')
    )
