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

        return reals_pop, reals_ind


    def compute_individual(self, individual, reals_pop, reals_ind):
        p0 = reals_pop['p0']
        reparametrized_time = torch.exp(reals_ind['xi'][individual.idx])*(individual.tensor_timepoints-reals_ind['tau'][individual.idx])
        return torch.pow(1+(1/p0-1)*torch.exp(-reparametrized_time/(p0*(1-p0))), -1)

    def compute_average(self, individual, reals_pop, tensor_timepoints):
        p0 = reals_pop['p0']
        reparametrized_time = np.exp(self.model_parameters['xi_mean'])*(tensor_timepoints-self.model_parameters['tau_mean'])
        return torch.pow(1 + (1 / p0 - 1) * torch.exp(-reparametrized_time / (p0 * (1 - p0))), -1)



    # Likelihood

    def compute_sumsquared(self, data, reals_pop, reals_ind):
        return np.sum([self.compute_individual_sumsquared(individual, reals_pop, reals_ind) for _,individual in data.individuals.items()])

    def compute_individual_sumsquared(self, individual, reals_pop, reals_ind):
         return torch.sum((self.compute_individual(individual, reals_pop, reals_ind)-individual.tensor_observations)**2)


    def compute_individual_attachment(self, individual, reals_pop, reals_ind):
        return self.compute_individual_sumsquared(individual, reals_pop, reals_ind)*np.power(2*self.model_parameters['noise_var'], -1) + np.log(np.sqrt(2*np.pi*self.model_parameters['noise_var']))

    def compute_individual_regularity(self, individual, reals_ind):
        tau_regularity = (reals_ind['tau'][individual.idx]-self.model_parameters['tau_mean'])**2/(2*self.model_parameters['tau_var'])+np.log(self.model_parameters['tau_var']*np.sqrt(2*np.pi))
        xi_regularity = (reals_ind['xi'][individual.idx]-self.model_parameters['xi_mean'])**2/(2*self.model_parameters['xi_var'])+np.log(self.model_parameters['xi_var']*np.sqrt(2*np.pi))
        return tau_regularity+xi_regularity

    def compute_attachment(self, data, reals_pop, reals_ind):
        return self.compute_sumsquared(data, reals_pop, reals_ind)*np.power(2*self.model_parameters['noise_var'], -1) + data.n_observations*np.log(np.sqrt(2*np.pi*self.model_parameters['noise_var']))

    def compute_regularity(self, data, reals_pop, reals_ind):
        tau_regularity = np.var([x.detach().numpy() for x in reals_ind['tau'].values()])*np.power(self.model_parameters['tau_var'], -1) + data.n_individuals*np.log(self.model_parameters['tau_var']*np.sqrt(2*np.pi))
        xi_regularity = np.var([x.detach().numpy() for x in reals_ind['xi'].values()])*np.power(self.model_parameters['xi_var'], -1) + data.n_individuals*np.log(self.model_parameters['xi_var']*np.sqrt(2*np.pi))
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
        empirical_tau_var = np.var([x.detach().numpy() for x in reals_ind['tau'].values()])
        tau_var_update = (1/(data.n_individuals+m_tau))*(data.n_individuals*empirical_tau_var+m_tau*sigma2_tau_0)
        self.model_parameters['tau_var'] = tau_var_update

        empirical_xi_var = np.var([x.detach().numpy() for x in reals_ind['xi'].values()])
        xi_var_update = (1/(data.n_individuals+m_xi))*(data.n_individuals*empirical_xi_var+m_xi*sigma2_xi_0)
        self.model_parameters['xi_var'] = xi_var_update

        self.model_parameters['tau_mean'] = np.mean([x.detach().numpy() for x in reals_ind['tau'].values()])
        self.model_parameters['xi_mean'] = np.mean([x.detach().numpy() for x in reals_ind['xi'].values()])

        # Noise
        self.model_parameters['noise_var'] = self.compute_sumsquared(data, reals_pop, reals_ind).detach().numpy()/data.n_observations

