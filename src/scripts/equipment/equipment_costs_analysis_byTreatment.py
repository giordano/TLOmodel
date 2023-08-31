import argparse
import pandas as pd
from pathlib import Path
from tlo import Date
from tlo.analysis.utils import extract_results, summarize


def get_annual_num_treatments(results_folder: Path) -> pd.DataFrame:
    """Return pd.DataFrame gives the (mean) simulated annual number of treatments."""

    def get_counts_of_treatments(_df):
        """Get the mean number of appointments with the same treatment_ID for each year at any level."""

        def unpack_nested_dict_in_series(_raw: pd.Series):
            return pd.concat(
                {
                    idx: pd.DataFrame.from_dict(mydict, orient='index').T for idx, mydict in _raw.items()
                }
            ).unstack().fillna(0.0).astype(int)

        return _df \
            .assign(year=_df['date'].dt.year) \
            .set_index('year') \
            .loc[:, 'TREATMENT_ID'] \
            .pipe(unpack_nested_dict_in_series) \
            .unstack()

    return summarize(
        extract_results(
            results_folder,
            module='tlo.methods.healthsystem.summary',
            key='HSI_Event',
            custom_generate_series=get_counts_of_treatments,
            do_scaling=True
        ),
        only_mean=True,
        collapse_columns=True,
    ).unstack().astype(int).droplevel(1)
# TODO: Do we want this as integer, or rather as float with X decimal places?


def analyse_equipment_costs(results_folder: Path, output_folder: Path, resourcefilepath: Path = None):
    # Declare path for output file from this script
    output_file_name = 'equipment_annual_cost_byTreatment.csv'
    equipment = pd.read_csv(Path(resourcefilepath) / 'ResourceFile_Equipment.csv')

    sim_treatments = get_annual_num_treatments(results_folder)

    sim_treatments.to_csv(output_folder / output_file_name)
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
