from src.inputs.model_settings import ModelSettings
from src.models.model_factory import ModelFactory

from src.algo.algo_factory import AlgoFactory
from src.utils.output_manager import OutputManager
import json


class Leaspy:
    def __init__(self, model_name):
        self.type = model_name
        self.model = ModelFactory.model(model_name)

    def __str__(self):
        return self.model.__str__()

    @classmethod
    def from_model_settings(cls, path_to_model_parameters):
        reader = ModelSettings(path_to_model_parameters)
        leaspy = cls(reader.model_type)
        leaspy.model.load_parameters(reader.parameters)
        leaspy.model.load_hyperparameters(reader.hyperparameters)
        return leaspy

    def save(self, path):
        self.model.save_parameters(path)

    def fit(self, data, algorithm_settings):
        algorithm = AlgoFactory.algo(algorithm_settings)
        # 1. Get the good data format  : check if it is possible, given potential previous model parameters
        # 2. (Optional) Smart initialization

        # Initialize model
        self.model.smart_initialization(data)


        # Run algo
        algo.run(data, self.model)

    def predict(self, individual, prediction_settings, seed=0, method="map"):
        #TODO Change, use specific algorithms
        """
        Predict individual parameters of a patient, or an iterable of patients
        :param data:
        :param path_to_prediction_settings:
        :param path output
        :param seed
        :param method : map or distribution
        :return:
        """

        # Instanciate optimization algorithm for predict

        # Chekc that in predict algo
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


