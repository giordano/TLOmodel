# import pytest  # this is the library for testing
#
# from tlo import Simulation, Date
# from tlo.methods import demography
# from tlo.test import mockitis
#
# path = '/Users/Tara/Desktop/TLO/Demography.xlsx'  # Edit this path so it points to your own copy of the Demography.xlsx file
# start_date = Date(2010, 1, 1)
# end_date = Date(2060, 1, 1)
# popsize = 1000
#
#
# @pytest.fixture  # this is a pytest feature
# def simulation():
#     sim = Simulation(start_date=start_date)
#
#     # create the modules
#     core_module = demography.Demography(workbook_path=path)
#     mockitis_module = mockitis.Mockitis()
#
#     # register the modules
#     sim.register(core_module)
#     sim.register(mockitis_module)
#     return sim
#
#
# def test_mockitis_simulation(simulation):
#     simulation.make_initial_population(n=popsize)
#     simulation.simulate(end_date=end_date)
#
#
# if __name__ == '__main__':
#     simulation = simulation()
#     test_mockitis_simulation(simulation)
