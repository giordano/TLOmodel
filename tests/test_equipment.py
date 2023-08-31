import os
import pandas as pd
from pathlib import Path

tlomodel_filepath = Path(os.path.dirname(__file__)) / '..'
resourcefilepath = tlomodel_filepath / 'resources'

hsi_events = pd.read_csv(Path(tlomodel_filepath) / 'docs/hsi_events.csv')
equipment = pd.read_csv(
    Path(resourcefilepath) / 'healthsystem/infrastructure_and_equipment/ResourceFile_Equipment.csv')


def test_equipment_for_all_events_covered():
    """
    Checks if each unique event from hsi_events is present in RF_Equipment
    """
    return equipment['Event'].isin(hsi_events['Event'].unique())


# CHECK module, facility levels, event, event_code not empty

# CHECK events with obvious equipment as surgery, xray, ... have that equipment in RF
