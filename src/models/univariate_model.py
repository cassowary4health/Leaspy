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

    def initialize_realizations(self, data):
        """
        Initialize the realizations.
        All individual parameters, and population parameters that need to be considered as realizations.
        TODO : initialize settings + smart initialization
        :param data:
        :return:
        """

        reals_pop_name = self.reals_pop_name
        reals_ind_name = self.reals_ind_name

        # Population parameters
        reals_pop = dict.fromkeys(reals_pop_name)
        for pop_name in reals_pop_name:
            reals_pop[pop_name] = 0.3


        # Instanciate individual realizations
        reals_ind = dict.fromkeys(data.indices)

        # For all patients
        for idx in data.indices:
            # Create dictionnary of individual random variables
            reals_ind[idx] = dict.fromkeys(reals_ind_name)
            # For all invididual random variables, initialize
            for ind_name in reals_ind_name:
                reals_ind[idx][ind_name] = np.random.normal(loc = self.model_parameters['{0}_mean'.format(ind_name)],
                                                            scale = np.sqrt(self.model_parameters['{0}_var'.format(ind_name)]))

        # To Torch
        for key in reals_pop.keys():
            reals_pop[key] = Variable(torch.tensor(reals_pop[key]).float(), requires_grad=True)

        for idx in reals_ind.keys():
            for key in reals_ind[idx]:
                reals_ind[idx][key] = Variable(torch.tensor(reals_ind[idx][key]).float(), requires_grad=True)

        return reals_pop, reals_ind



    def compute_individual(self, individual, reals_pop, real_ind):
        p0 = reals_pop['p0']
        reparametrized_time = torch.exp(real_ind['xi'])*(individual.tensor_timepoints-real_ind['tau'])
        return torch.pow(1+(1/p0-1)*torch.exp(-reparametrized_time/(p0*(1-p0))), -1)

    def compute_average(self, tensor_timepoints):
        p0 = self.model_parameters['p0']
        reparametrized_time = np.exp(self.model_parameters['xi_mean'])*(tensor_timepoints-self.model_parameters['tau_mean'])
        return torch.pow(1 + (1 / p0 - 1) * torch.exp(-reparametrized_time / (p0 * (1 - p0))), -1)



    # Likelihood

    def compute_sumsquared(self, data, reals_pop, reals_ind):
        return np.sum([self.compute_individual_sumsquared(data[idx], reals_pop, reals_ind[idx]) for idx in data.indices])
    def compute_individual_sumsquared(self, individual, reals_pop, real_ind):
         return torch.sum((self.compute_individual(individual, reals_pop, real_ind)-individual.tensor_observations)**2)


    def compute_individual_attachment(self, individual, reals_pop, real_ind):
        return self.compute_individual_sumsquared(individual, reals_pop, real_ind)*np.power(2*self.model_parameters['noise_var'], -1) + np.log(np.sqrt(2*np.pi*self.model_parameters['noise_var']))

    def compute_individual_regularity(self, real_ind):
        tau_regularity = (real_ind['tau']-self.model_parameters['tau_mean'])**2/(2*self.model_parameters['tau_var'])+np.log(self.model_parameters['tau_var']*np.sqrt(2*np.pi))
        xi_regularity = (real_ind['xi']-self.model_parameters['xi_mean'])**2/(2*self.model_parameters['xi_var'])+np.log(self.model_parameters['xi_var']*np.sqrt(2*np.pi))
        return tau_regularity+xi_regularity


    def update_sufficient_statistics(self, data, reals_ind, reals_pop):

        #TODO parameters, how to tune ????
        m_xi = data.n_individuals/20
        sigma2_xi_0 = 0.05

        m_tau = data.n_individuals/20
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
        empirical_tau_var = np.sum(((tau_array - self.model_parameters['tau_mean'])**2).detach().numpy())/(data.n_individuals-1)
        tau_var_update = (1/(data.n_individuals+m_tau))*(data.n_individuals*empirical_tau_var+m_tau*sigma2_tau_0)
        self.model_parameters['tau_var'] = tau_var_update

        # Xi
        xi_array = []
        for idx in reals_ind.keys():
            xi_array.append(reals_ind[idx]['xi'])
        xi_array = torch.Tensor(xi_array)
        empirical_xi_var = np.sum(
            ((xi_array - self.model_parameters['xi_mean']) ** 2).detach().numpy()) / (
                                        data.n_individuals - 1)
        xi_var_update = (1/(data.n_individuals+m_xi))*(data.n_individuals*empirical_xi_var+m_xi*sigma2_xi_0)
        self.model_parameters['xi_var'] = xi_var_update

        self.model_parameters['tau_mean'] = np.mean(tau_array.detach().numpy())
        self.model_parameters['xi_mean'] = np.mean(xi_array.detach().numpy())

        # P0
        self.model_parameters['p0'] = reals_pop['p0'].detach().numpy()

        # Noise
        self.model_parameters['noise_var'] = self.compute_sumsquared(data, reals_pop, reals_ind).detach().numpy()/data.n_observations

"""
        reals_pop_name = self.reals_pop_name
        reals_ind_name = self.reals_ind_name

        # Population parameters
        reals_pop = dict.fromkeys(reals_pop_name)
        for pop_name in reals_pop_name:
            reals_pop[pop_name] = 0.3

        # Individual parameters
        reals_ind = dict.fromkeys(reals_ind_name)
        for ind_name in reals_ind_name:
            reals_ind_temp = dict(zip(data.indices, (self.model_parameters['{0}_mean'.format(ind_name)]+np.sqrt(self.model_parameters['{0}_var'.format(ind_name)])*np.random.randn(1,len(data.indices))).reshape(-1).tolist()))
            reals_ind[ind_name] = reals_ind_temp

        # To Torch
        for key in reals_pop.keys():
            reals_pop[key] = Variable(torch.tensor(reals_pop[key]).float(), requires_grad=True)

        for key in reals_ind.keys():
            for idx in reals_ind[key]:
                reals_ind[key][idx] = Variable(torch.tensor(reals_ind[key][idx]).float(), requires_grad=True)

        return reals_pop, reals_ind"""
