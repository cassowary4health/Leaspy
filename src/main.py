from src.inputs.data.dataset import Dataset
from src.inputs.model_settings import ModelSettings

from src.models.model_factory import ModelFactory

from src.algo.algo_factory import AlgoFactory
from src.utils.output_manager import OutputManager
import json


class Leaspy:
    def __init__(self, model_name):
        self.type = model_name
        self.model = ModelFactory.model(model_name)

    @classmethod
    def load(cls, path_to_model_settings):
        reader = ModelSettings(path_to_model_settings)
        leaspy = cls(reader.name)
        leaspy.model.load_parameters(reader.parameters)
        leaspy.model.load_hyperparameters(reader.hyperparameters)
        leaspy.model.is_initialized = True
        return leaspy

    def save(self, path):
        self.model.save(path)

    def fit(self, data, algorithm_settings):

        algorithm = AlgoFactory.algo(algorithm_settings)
        dataset = Dataset(data, algo=algorithm, model=self.model)
        if not self.model.is_initialized:
            self.model.initialize(dataset)
        algorithm.run(dataset, self.model)

    def personalize(self, individual, prediction_settings, seed=0):

        # Check that in predict algo
        if prediction_settings.algo_type not in ['mcmc_predict']:
            raise ValueError("Optimization Algorithm is not adapted for predict")

        print("Load predict algorithm")
        predict_algo = AlgoFactory.algo(prediction_settings.algo_type)
        predict_algo.load_parameters(prediction_settings.parameters)

        # Predict
        print("Launch predict algo")
        res = predict_algo.run(individual, self.model, seed)
        return res

    def simulate(self, path_to_simulation_settings, seed=0):

        with open(path_to_simulation_settings) as file:
            simulation_settings = json.load(file)

        indices = ['simulated_patient_{0}'.format(i) for i in range(simulation_settings['number_patients_to_simulate'])]
        simulated_individual_parameters = self.model.simulate_individual_parameters(indices, seed=seed)
        return simulated_individual_parameters


