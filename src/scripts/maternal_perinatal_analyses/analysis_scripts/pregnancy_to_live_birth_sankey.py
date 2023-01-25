import tlo

import pandas as pd
import numpy as np
from ipysankeywidget import SankeyWidget
from matplotlib import pyplot as plt
from floweaver import *
from pathlib import Path
from ipywidgets import HBox, VBox

# get the tlo path
tlopath = Path(tlo.__file__).parent.parent.parent

# Define the path of output Sankeys
outputpath = tlopath / Path('outputs/sejjj49@ucl.ac.uk')

