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

        self._initialize_random_variables()

        # TODO to Pytorch, peut être dans le reader ????
        #for key in self.model_parameters.keys():
        #    self.model_parameters[key] = Variable(torch.tensor(self.model_parameters[key]).float(), requires_grad=True)


    ###########################
    ## Core
    ###########################

    def compute_individual(self, individual, reals_pop, real_ind):
        return real_ind['intercept']*torch.ones_like(individual.tensor_timepoints.reshape(-1,1))

    def compute_average(self, tensor_timepoints):
        return self.model_parameters['intercept_mean'] * torch.ones_like(tensor_timepoints.reshape(-1,1))

    def compute_sufficient_statistics(self, data, reals_ind, reals_pop):
        # Intercept
        intercept_array = []
        for idx in reals_ind.keys():
            intercept_array.append(reals_ind[idx]['intercept'])
        intercept_array = torch.Tensor(intercept_array)
        intercept_mean = np.mean(intercept_array.detach().numpy()).tolist()
        intercept_var = np.var(intercept_array.detach().numpy()).tolist()

        # Sufficient statistics
        sufficient_statistics = {}
        sufficient_statistics['intercept_mean'] = intercept_mean
        sufficient_statistics['intercept_var'] = intercept_var
        sufficient_statistics['sum_squared'] = float(self.compute_sumsquared(data, reals_pop, reals_ind).detach().numpy())

        return sufficient_statistics

    def update_model(self, data, sufficient_statistics):
        #TODO parameters, automatic initialization of these parameters
        m_intercept = data.n_individuals/20
        sigma2_intercept_0 = 0.1

        # Intercept
        self.model_parameters['intercept_mean'] = sufficient_statistics['intercept_mean']
        intercept_var_update = (1 / (data.n_individuals + m_intercept)) * (
                data.n_individuals * sufficient_statistics['intercept_var'] + m_intercept * sigma2_intercept_0)
        self.model_parameters['intercept_var'] = intercept_var_update

        # Noise
        self.model_parameters['noise_var'] = sufficient_statistics['sum_squared']/data.n_observations

        # Update the Random Variables
        self._update_random_variables()

    def simulate_individual_parameters(self, indices, seed=0):
        np.random.seed(seed)
        reals_ind = dict.fromkeys(self.reals_ind_name)
        for ind_name in self.reals_ind_name:
            reals_ind_temp = dict(zip(indices, self.model_parameters['intercept_mean']+(np.sqrt(self.model_parameters['intercept_var'])*np.random.randn(1, len(indices))).reshape(-1).tolist()))
            reals_ind[ind_name] = reals_ind_temp
        return reals_ind







