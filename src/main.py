from src.inputs.model_parameters_reader import ModelParametersReader
from src.models.model_factory import ModelFactory
from src.inputs.algo_reader import AlgoReader
from src.algo.algo_factory import AlgoFactory
from src.utils.output_manager import OutputManager
import json

class Leaspy():
    def __init__(self, type):
        self.type = type
        self.model = ModelFactory.model(type)

    @classmethod
    def from_parameters(cls, path_to_model_parameters):
        reader = ModelParametersReader(path_to_model_parameters)
        leaspy = cls(reader.model_type)
        leaspy.model.load_parameters(reader.parameters)
        return leaspy

    def load(self, path_to_model_parameters):
        reader = ModelParametersReader(path_to_model_parameters)
        self.model.load_parameters(reader.parameters)

    def save(self, path):
        self.model.save_parameters(path)

    def fit(self, data, path_to_algorithm_settings, path_output, seed=0):

        # Algo settings
        reader = AlgoReader(path_to_algorithm_settings)
        algo = AlgoFactory.algo(reader.algo_type)
        algo.load_parameters(reader.parameters)
        algo.set_mode('fit')

        # Output manager
        output_manager = OutputManager(path_output)
        output_manager.initialize_model_statistics(self.model)

        # Run algo
        algo.run(data, self.model, output_manager, seed)

    def predict(self, data, path_to_prediction_settings, seed=0):
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
        algo.set_mode('predict')

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
