import os

from src import default_data_dir
from src.models.abstract_model import AbstractModel
from src.inputs.model_parameters_reader import ModelParametersReader

import torch
from torch.autograd import Variable


class GaussianDistributionModel(AbstractModel):
    def __init__(self):
        data_dir = os.path.join(default_data_dir, "default_gaussian_distribution_parameters.json")
        reader = ModelParametersReader(data_dir)

        if reader.model_type != 'gaussian_distribution':
            raise ValueError("The default univariate parameters are not of gaussian_distribution type")

        self.model_parameters = reader.parameters


        # TODO to Pytorch, peut être dans le reader ????
        for key in self.model_parameters.keys():
            self.model_parameters[key] = Variable(torch.tensor(self.model_parameters[key]).float(), requires_grad=True)


    def initialize_individual_realization(self, individual):
        individual.individual_parameters = {'a' : 0}

    def compute_individual(self, indiviual):
        return self.model_parameters['mu'] + indiviual.individual_parameters['a']*torch.ones_like(indiviual.tensor_timepoints)

    def compute_attachment(self, data):
        likelihood = 0
        for individual in data:
            likelihood += self.compute_individual_attachment(individual)
        return likelihood

    def compute_individual_attachment(self, individual):
         return torch.sum((self.compute_individual(individual)-individual.tensor_observations)**2)