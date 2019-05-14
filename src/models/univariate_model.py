import os

from src import default_data_dir
from src.models.abstract_model import AbstractModel
from src.inputs.model_parameters_reader import ModelParametersReader
import torch
from torch.autograd import Variable
import numpy as np


class UnivariateModel(AbstractModel):
    def __init__(self):
        data_dir = os.path.join(default_data_dir, "default_univariate_parameters.json")
        reader = ModelParametersReader(data_dir)
        self.model_parameters = reader.parameters

        if reader.model_type != 'univariate':
            raise ValueError("The default univariate parameters are not of univariate type")


        self.reals_pop_name = ['p0']
        self.reals_ind_name = ['xi','tau']

        self._initialize_random_variables()

    ###########################
    ## Core
    ###########################

    def compute_individual(self, individual, reals_pop, real_ind):
        p0 = reals_pop['p0']
        reparametrized_time = torch.exp(real_ind['xi'])*(individual.tensor_timepoints.reshape(-1,1)-real_ind['tau'])
        return torch.pow(1+(1/p0-1)*torch.exp(-reparametrized_time/(p0*(1-p0))), -1)

    def compute_average(self, tensor_timepoints):
        p0 = self.model_parameters['p0']
        reparametrized_time = np.exp(self.model_parameters['xi_mean'])*(tensor_timepoints.reshape(-1,1)-self.model_parameters['tau_mean'])
        return torch.pow(1 + (1 / p0 - 1) * torch.exp(-reparametrized_time / (p0 * (1 - p0))), -1)

    def update_sufficient_statistics(self, data, reals_ind, reals_pop):

        #TODO parameters, automatic initialization of these parameters
        m_xi = data.n_individuals/20
        sigma2_xi_0 = 0.05

        m_tau = data.n_individuals / 20
        sigma2_tau_0 = 10


        # Update Parameters

        # Population parameters as realizations
        for pop_name in reals_pop.keys():
            self.model_parameters[pop_name] = reals_pop[pop_name].detach().numpy()

        # population parameters not as realizations

        # Tau
        tau_array = []
        for idx in reals_ind.keys():
            tau_array.append(reals_ind[idx]['tau'])
        tau_array = torch.Tensor(tau_array)
        self.model_parameters['tau_mean'] = np.mean(tau_array.detach().numpy()).tolist()
        #empirical_tau_var = np.sum(((tau_array - self.model_parameters['tau_mean'])**2).detach().numpy())/(data.n_individuals-1)
        empirical_tau_var = torch.sum(tau_array**2)/(data.n_individuals)-self.model_parameters['tau_mean']**2
        tau_var_update = (1/(data.n_individuals+m_tau))*(data.n_individuals*empirical_tau_var+m_tau*sigma2_tau_0)
        self.model_parameters['tau_var'] = tau_var_update.detach().numpy()

        # Xi
        xi_array = []
        for idx in reals_ind.keys():
            xi_array.append(reals_ind[idx]['xi'])
        xi_array = torch.Tensor(xi_array)
        self.model_parameters['xi_mean'] = np.mean(xi_array.detach().numpy()).tolist()
        #empirical_xi_var = np.sum(((xi_array - self.model_parameters['xi_mean']) ** 2).detach().numpy()) / (data.n_individuals - 1)
        empirical_xi_var = torch.sum(xi_array**2)/(data.n_individuals)-self.model_parameters['xi_mean']**2
        xi_var_update = (1/(data.n_individuals+m_xi))*(data.n_individuals*empirical_xi_var+m_xi*sigma2_xi_0)
        self.model_parameters['xi_var'] = xi_var_update.detach().numpy()


        # P0
        self.model_parameters['p0'] = reals_pop['p0'].detach().numpy()

        # Noise
        self.model_parameters['noise_var'] = self.compute_sumsquared(data, reals_pop, reals_ind).detach().numpy()/data.n_observations


        self._update_random_variables()

