from src.inputs.model_settings import ModelSettings
from src.models.model_factory import ModelFactory

from src.algo.algo_factory import AlgoFactory
from src.utils.output_manager import OutputManager
import json

class Leaspy():
    def __init__(self, model_name):
        #TODO change type to model_name
        self.type = model_name
        self.model = ModelFactory.model(model_name)

    @classmethod
    def from_model_settings(cls, path_to_model_parameters):
        reader = ModelSettings(path_to_model_parameters)
        leaspy = cls(reader.model_type)
        leaspy.model.load_parameters(reader.parameters)
        leaspy.model.load_dimension(reader.dimension)
        leaspy.model.load_source_dimension(reader.source_dimension)
        leaspy.model.adapt_shapes()
        return leaspy

    """
    def load(self, path_to_model_parameters):
        reader = ModelParametersReader(path_to_model_parameters)
        self.model.load_parameters(reader.parameters)
        # TODO assert same dimension
        self.model.load_dimension(reader.dimension)
        self.model.initialize_random_variables()"""

    def save(self, path):
        self.model.save_parameters(path)

    def fit(self, data, algo_settings, seed=0):

        # Algo settings
        #reader = AlgoReader(path_to_algorithm_settings)
        algo = AlgoFactory.algo(algo_settings.algo_type)
        algo.load_parameters(algo_settings.parameters)

        # Output Manager
        path_output = algo_settings.get_path_output()
        output_manager = OutputManager(path_output)
        algo.set_output_manager(output_manager)

        # Initialize model
        self.model.smart_initialization(data)

        # Run algo
        algo.run(data, self.model, seed)

    def predict(self, data, path_to_prediction_settings, seed=0):
        #TODO Change, use specific algorithms
        """
        Predict individual parameters of a list of patients
        :param data:
        :param path_to_prediction_settings:
        :param path output
        :param seed
        :return:
        """

        # Instanciate optimization algorithm
        reader = AlgoReader(path_to_prediction_settings)
        algo = AlgoFactory.algo(reader.algo_type)
        algo.load_parameters(reader.parameters)

        # Run optimization
        realizations = algo.run(data, self.model, seed)
        reals_pop, reals_ind = realizations
        return reals_ind


    def simulate(self, path_to_simulation_settings, seed=0):

        with open(path_to_simulation_settings) as file:
            simulation_settings = json.load(file)

        indices = ['simulated_patient_{0}'.format(i) for i in range(simulation_settings['number_patients_to_simulate'])]
        simulated_individual_parameters = self.model.simulate_individual_parameters(indices, seed=seed)
        return simulated_individual_parameters

"""
    def fit(self, data, path_to_algorithm_settings, path_output=None, seed=0):

        # Algo settings
        reader = AlgoReader(path_to_algorithm_settings)
        algo = AlgoFactory.algo(reader.algo_type)
        algo.load_parameters(reader.parameters)

        # Output manager
        if path_output is not None:
            output_manager = OutputManager(path_output)
        else:
            output_manager = None

        # Initialize model
        self.model.smart_initialization(data)

        # Run algo
        algo.run(data, self.model, output_manager, seed)"""