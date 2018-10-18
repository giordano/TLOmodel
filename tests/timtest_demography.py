

from tlo import Simulation, Date
from tlo.methods import demography
import matplotlib

path = '/Users/tbh03/Dropbox (SPH Imperial College)/Thanzi la Onse Theme 1 SHARE/05 - Resources/Demographic data/Demography_WorkingFile.xlsx'  # Edit this path so it points to your own copy of the Demography.xlsx file
start_date = Date(2010, 1, 1)
end_date = Date(2060, 1, 1)
popsize = 10

sim = Simulation(start_date=start_date)
core_module = demography.Demography(workbook_path=path)
sim.register(core_module)

sim.seed_rngs(0)

sim.make_initial_population(n=popsize)
sim.simulate(end_date=end_date)


sim.modules['Demography'].store['Population_Total']

# Make a nice plot;
#
# import matplotlib.pyplot as plt
# import numpy as np
#
#
# stats = simulation.modules['Demography'].store['Population_Total']
# time=simulation.modules['Demography'].store['Time']
#
# # plt.plot(np.arange(0, len(stats)), stats)
#
# plt.plot(time, stats)
#
# plt.show()
#

