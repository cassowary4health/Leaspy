from src.inputs.model_parameters_reader import ModelParametersReader
from src.models.model_factory import ModelFactory
from src.inputs.algo_reader import AlgoReader
from src.algo.algo_factory import AlgoFactory


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

    def fit(self, data, path_to_algorithm_settings, seed=0):

        reader = AlgoReader(path_to_algorithm_settings)
        algo = AlgoFactory.algo(reader.algo_type)
        algo.load_parameters(reader.parameters)

        # Run algo
        algo.run(data, self.model, seed)

    def predict(self, data, path_to_prediction_settings):
        return 0

    def simulate(self, data, path_to_simulation_settings):
        return 0
