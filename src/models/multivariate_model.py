import os

from src import default_data_dir
from src.models.abstract_model import AbstractModel
from src.inputs.model_parameters_reader import ModelParametersReader
import torch
from torch.autograd import Variable
import numpy as np


class MultivariateModel(AbstractModel):
    # TODO dimension in multivariate model parameters initialization ???
    def __init__(self):
        data_dir = os.path.join(default_data_dir, "default_multivariate_parameters.json")
        reader = ModelParametersReader(data_dir)
        self.model_parameters = reader.parameters
        self.dimension = None

        if reader.model_type != 'multivariate':
            raise ValueError("The default multivariate parameters are not of multivariate type")


        self.reals_pop_name = ['p0','v0']
        self.reals_ind_name = ['xi','tau']





    ###########################
    ## Core
    ###########################

    def compute_individual(self, individual, reals_pop, real_ind):
        p0 = reals_pop['p0']
        v0 = reals_pop['v0']
        reparametrized_time = v0*torch.exp(real_ind['xi'])*(individual.tensor_timepoints-real_ind['tau'])
        return torch.pow(1+(1/p0-1)*torch.exp(-reparametrized_time/(p0*(1-p0))), -1)

    def compute_average(self, tensor_timepoints):
        p0 = torch.Tensor(self.model_parameters['p0'])
        v0 = torch.Tensor(self.model_parameters['v0'])
        reparametrized_time = v0*np.exp(self.model_parameters['xi_mean'])*(tensor_timepoints-self.model_parameters['tau_mean'])
        return torch.pow(1 + (1 / p0 - 1) * torch.exp(-reparametrized_time / (p0 * (1 - p0))), -1)

    def compute_sufficient_statistics(self, data, reals_ind, reals_pop):

        # Tau
        tau_array = []
        for idx in reals_ind.keys():
            tau_array.append(reals_ind[idx]['tau'])
        tau_array = torch.Tensor(tau_array)

        tau_mean = np.mean(tau_array.detach().numpy()).tolist()
        tau_var = np.var(tau_array.detach().numpy()).tolist()

        # Ksi
        xi_array = []
        for idx in reals_ind.keys():
            xi_array.append(reals_ind[idx]['xi'])
        xi_array = torch.Tensor(xi_array)

        xi_mean = np.mean(xi_array.detach().numpy()).tolist()
        xi_var = np.var(xi_array.detach().numpy()).tolist()

        # P0
        p0 = reals_pop['p0'].detach().numpy()

        # V0
        v0 = reals_pop['v0'].detach().numpy()

        sufficient_statistics = {}
        sufficient_statistics['p0'] = p0
        sufficient_statistics['v0'] = v0
        sufficient_statistics['tau_mean'] = tau_mean
        sufficient_statistics['tau_var'] = tau_var
        sufficient_statistics['xi_mean'] = xi_mean
        sufficient_statistics['xi_var'] = xi_var
        sufficient_statistics['sum_squared'] = float(self.compute_sumsquared(data, reals_pop, reals_ind).detach().numpy())

        return sufficient_statistics



    def update_model(self, data, sufficient_statistics):

        #TODO parameters, automatic initialization of these parameters
        m_xi = data.n_individuals/20
        sigma2_xi_0 = 0.05

        m_tau = data.n_individuals / 20
        sigma2_tau_0 = 10

        self.model_parameters['p0'] = sufficient_statistics['p0']
        self.model_parameters['v0'] = sufficient_statistics['v0']

        # Tau
        self.model_parameters['tau_mean'] = sufficient_statistics['tau_mean']
        tau_var_update = (1/(data.n_individuals+m_tau))*(data.n_individuals * sufficient_statistics['tau_var']+m_tau*sigma2_tau_0)
        self.model_parameters['tau_var'] = tau_var_update

        # Xi
        self.model_parameters['xi_mean'] = sufficient_statistics['xi_mean']
        tau_var_update = (1/(data.n_individuals+m_xi))*(data.n_individuals * sufficient_statistics['xi_var']+m_tau*sigma2_xi_0)
        self.model_parameters['xi_var'] = tau_var_update

        # Noise
        self.model_parameters['noise_var'] = sufficient_statistics['sum_squared']/data.n_observations

        # Update the Random Variables
        self._update_random_variables()