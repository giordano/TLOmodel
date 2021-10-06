"""This is a skeletal version of the forthcoming `Malnutrition` module. It's provided only to give the properties
required for `Diarrhoea` and `Alri`"""


from tlo import Module, Property, Types, logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Malnutrition(Module):

    INIT_DEPENDENCIES = {'Demography'}

    PROPERTIES = {
        'un_clinical_acute_malnutrition': Property(Types.CATEGORICAL, 'temporary property',
                                                   categories=['MAM', 'SAM', 'well']),
        'un_HAZ_category': Property(Types.CATEGORICAL, 'temporary property',
                                    categories=['HAZ<-3', '-3<=HAZ<-2', 'HAZ>=-2']),
    }

    def __init__(self, name=None):
        super().__init__(name)

    def read_parameters(self, data_folder):
        pass

    def initialise_population(self, population):
        df = population.props
        df.loc[df.is_alive, 'un_clinical_acute_malnutrition'] = 'well'
        df.loc[df.is_alive, 'un_HAZ_category'] = 'HAZ>=-2'

    def initialise_simulation(self, sim):
        pass

    def on_birth(self, mother, child):
        df = self.sim.population.props
        df.at[child, 'un_clinical_acute_malnutrition'] = 'well'
        df.at[child, 'un_HAZ_category'] = 'HAZ>=-2'

