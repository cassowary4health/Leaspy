import os

from src import default_data_dir
from src.models.abstract_model import AbstractModel
from src.inputs.model_parameters_reader import ModelParametersReader

import torch
from torch.autograd import Variable
import numpy as np


class GaussianDistributionModel(AbstractModel):
    def __init__(self):
        data_dir = os.path.join(default_data_dir, "default_gaussian_distribution_parameters.json")
        reader = ModelParametersReader(data_dir)

        if reader.model_type != 'gaussian_distribution':
            raise ValueError("The default univariate parameters are not of gaussian_distribution type")

        self.model_parameters = reader.parameters

        self.reals_pop_name = []
        self.reals_ind_name = ['intercept']



        # TODO to Pytorch, peut être dans le reader ????
        #for key in self.model_parameters.keys():
        #    self.model_parameters[key] = Variable(torch.tensor(self.model_parameters[key]).float(), requires_grad=True)

    def initialize_realizations(self, data):
        """
        Initialize the realizations.
        All individual parameters, and population parameters that need to be considered as realizations.
        TODO : initialize settings + smart initialization
        :param data:
        :return:
        """

        reals_ind_name = self.reals_ind_name

        # Population parameters
        reals_pop = dict.fromkeys(self.reals_pop_name)
        for pop_name in self.reals_pop_name:
            reals_pop[pop_name] = self.model_parameters[pop_name]

        # Instanciate individual realizations
        reals_ind = dict.fromkeys(data.indices)

        # For all patients
        for idx in data.indices:
            # Create dictionnary of individual random variables
            reals_ind[idx] = dict.fromkeys(reals_ind_name)
            # For all invididual random variables, initialize
            for ind_name in reals_ind_name:
                reals_ind[idx][ind_name] = np.random.normal(loc=self.model_parameters['{0}_mean'.format(ind_name)],
                                                            scale=np.sqrt(self.model_parameters['{0}_var'.format(ind_name)]))

        # To Torch
        for key in reals_pop.keys():
            reals_pop[key] = Variable(torch.tensor(reals_pop[key]).float(), requires_grad=True)

        for idx in reals_ind.keys():
            for key in reals_ind[idx]:
                reals_ind[idx][key] = Variable(torch.tensor(reals_ind[idx][key]).float(), requires_grad=True)

        return reals_pop, reals_ind


    def compute_individual(self, individual, reals_pop, real_ind):
        return real_ind['intercept']*torch.ones_like(individual.tensor_timepoints)

    def compute_average(self, tensor_timepoints):
        return self.model_parameters['intercept_mean'] * torch.ones_like(tensor_timepoints)

    def compute_sumsquared(self, data, reals_pop, reals_ind):
        return np.sum([self.compute_individual_sumsquared(data[idx], reals_pop, reals_ind[idx]) for idx in data.indices])

    def compute_individual_sumsquared(self, individual, reals_pop, real_ind):
         return torch.sum((self.compute_individual(individual, reals_pop, real_ind)-individual.tensor_observations)**2)

    def compute_individual_attachment(self, individual, reals_pop, real_ind):
        return self.compute_individual_sumsquared(individual, reals_pop, real_ind) + np.log(np.sqrt(2*np.pi*self.model_parameters['noise_var']))

    def compute_individual_regularity(self, real_ind):
        intercept_regularity = (real_ind['intercept']-self.model_parameters['intercept_mean'])**2/(2*self.model_parameters['intercept_var'])+np.log(self.model_parameters['intercept_var']*np.sqrt(2*np.pi))
        return intercept_regularity

    def update_sufficient_statistics(self, data, reals_ind, reals_pop):

        m_intercept = data.n_individuals/20
        sigma2_intercept_0 = 0.1 # TODO smart initialization

        # Update Parameters

        # Population parameters as realizations
        for pop_name in reals_pop.keys():
            self.model_parameters[pop_name] = reals_pop[pop_name].detach().numpy()

        # population parameters not as realizations
        intercept_array = []
        for idx in reals_ind.keys():
            intercept_array.append(reals_ind[idx]['intercept'])
        intercept_array = torch.Tensor(intercept_array)
        empirical_intercept_var = np.sum(
            ((intercept_array - self.model_parameters['intercept_mean']) ** 2).detach().numpy()) / (
                                   data.n_individuals - 1)

        intercept_var_update = (1 / (data.n_individuals + m_intercept)) * (
                    data.n_individuals * empirical_intercept_var + m_intercept * sigma2_intercept_0)
        self.model_parameters['intercept_var'] = intercept_var_update
        self.model_parameters['intercept_mean'] = np.mean(intercept_array.detach().numpy())
        # Noise
        self.model_parameters['noise_var'] = self.compute_sumsquared(data, reals_pop, reals_ind).detach().numpy()/data.n_observations


    def simulate_individual_parameters(self, indices, seed=0):

        np.random.seed(seed)

        reals_ind = dict.fromkeys(self.reals_ind_name)

        for ind_name in self.reals_ind_name:
            reals_ind_temp = dict(zip(indices, self.model_parameters['intercept_mean']+(np.sqrt(self.model_parameters['intercept_var'])*np.random.randn(1, len(indices))).reshape(-1).tolist()))
            reals_ind[ind_name] = reals_ind_temp
        return reals_ind







