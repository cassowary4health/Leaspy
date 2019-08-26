import json
import copy

from leaspy.inputs.data.dataset import Dataset
from leaspy.inputs.settings.model_settings import ModelSettings
from leaspy.models.model_factory import ModelFactory
from leaspy.algo.algo_factory import AlgoFactory
from leaspy.inputs.data.result import Result


class Leaspy:
    def __init__(self, model_name):
        self.type = model_name
        self.model = ModelFactory.model(model_name)

    @classmethod
    def load(cls, path_to_model_settings):
        reader = ModelSettings(path_to_model_settings)
        leaspy = cls(reader.name)
        leaspy.model.load_hyperparameters(reader.hyperparameters)
        leaspy.model.load_parameters(reader.parameters)
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

    def personalize(self, data, settings):

        algorithm = AlgoFactory.algo(settings)
        dataset = Dataset(data, algo=algorithm, model=self.model)
        individual_parameters = algorithm.run(self.model, dataset)
        result = Result(data, individual_parameters)

        return result

    def simulate(self, results, settings):

        algorithm = AlgoFactory.algo(settings)
        simulated_data = algorithm.run(self.model, results)
        return simulated_data

    @staticmethod
    def save_individual_parameters(path, individual_parameters):
        dump = copy.deepcopy(individual_parameters)

        for key1 in dump.keys():
            for key2 in dump[key1]:
                if type(dump[key1][key2]) not in [list]:
                    dump[key1][key2] = dump[key1][key2].tolist()

        with open(path, 'w') as fp:
            json.dump(dump, fp)

    @staticmethod
    def load_individual_parameters(path):
        with open(path, 'r') as f:
            individual_parameters = json.load(f)

        return individual_parameters
