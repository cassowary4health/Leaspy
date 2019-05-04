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

        # Population parameters
        reals_pop = dict.fromkeys(self.reals_pop_name)
        for pop_name in self.reals_pop_name:
            reals_pop[pop_name] = self.model_parameters[pop_name]

        # Individual parameters
        reals_ind = dict.fromkeys(self.reals_ind_name)
        for ind_name in self.reals_ind_name:
            reals_ind_temp = dict(zip(data.indices, (self.model_parameters['mu']+np.sqrt(self.model_parameters['intercept_var'])*np.random.randn(1,len(data.indices))).reshape(-1).tolist()))
            reals_ind[ind_name] = reals_ind_temp

        # To Torch
        for key in reals_pop.keys():
            reals_pop[key] = Variable(torch.tensor(reals_pop[key]).float(), requires_grad=True)

        for key in reals_ind.keys():
            for idx in reals_ind[key]:
                reals_ind[key][idx] = Variable(torch.tensor(reals_ind[key][idx]).float(), requires_grad=True)

        return reals_pop, reals_ind


    def compute_individual(self, individual, reals_pop, reals_ind):
        return reals_ind['intercept'][individual.idx]*torch.ones_like(individual.tensor_timepoints)

    def compute_sumsquared(self, data, reals_pop, reals_ind):
        return np.sum([self.compute_individual_sumsquared(individual, reals_pop, reals_ind) for key,individual in data.individuals.items()])

    def compute_individual_sumsquared(self, individual, reals_pop, reals_ind):
         return torch.sum((self.compute_individual(individual, reals_pop, reals_ind)-individual.tensor_observations)**2)

    def compute_individual_attachment(self, individual, reals_pop, reals_ind):
        return self.compute_individual_sumsquared(individual, reals_pop, reals_ind) + np.log(np.sqrt(2*np.pi*self.model_parameters['noise_var']))

    def compute_attachment(self, data, reals_pop, reals_ind):
        return self.compute_sumsquared(data, reals_pop, reals_ind)*np.power(self.model_parameters['noise_var'], -1) + data.n_observations*np.log(np.sqrt(2*np.pi*self.model_parameters['noise_var']))

    def compute_individual_regularity(self, individual, reals_ind):
        intercept_regularity = (reals_ind['intercept'][individual.idx]-self.model_parameters['mu'])**2/(2*self.model_parameters['intercept_var'])+np.log(self.model_parameters['intercept_var']*np.sqrt(2*np.pi))
        return intercept_regularity

    def compute_regularity(self, data, reals_pop, reals_ind):
        return np.var([x.detach().numpy() for x in reals_ind['intercept'].values()])*np.power(self.model_parameters['intercept_var'], -1) + data.n_individuals*np.log(np.sqrt(2*np.pi*self.model_parameters['intercept_var']))

    def update_sufficient_statistics(self, data, reals_ind, reals_pop):

        m_intercept = data.n_individuals/20
        sigma2_intercept_0 = 0.1 # TODO smart initialization

        # Update Parameters

        # Population parameters as realizations
        for pop_name in reals_pop.keys():
            self.model_parameters[pop_name] = reals_pop[pop_name].detach().numpy()

        # population parameters not as realizations
        empirical_intercept_var = np.var([x.detach().numpy() for x in reals_ind['intercept'].values()])
        intercept_var_update = (1 / (data.n_individuals + m_intercept)) * (
                    data.n_individuals * empirical_intercept_var + m_intercept * sigma2_intercept_0)
        self.model_parameters['intercept_var'] = intercept_var_update
        self.model_parameters['mu'] = np.mean([x.detach().numpy() for x in reals_ind['intercept'].values()])

        # Noise
        self.model_parameters['noise_var'] = self.compute_sumsquared(data, reals_pop, reals_ind).detach().numpy()/data.n_observations


    def simulate_individual_parameters(self, indices, seed=0):

        np.random.seed(seed)

        reals_ind = dict.fromkeys(self.reals_ind_name)

        for ind_name in self.reals_ind_name:
            reals_ind_temp = dict(zip(indices, self.model_parameters['mu']+(np.sqrt(self.model_parameters['intercept_var'])*np.random.randn(1, len(indices))).reshape(-1).tolist()))
            reals_ind[ind_name] = reals_ind_temp
        return reals_ind


    def plot(self, data, iter, realizations, path_output):

        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        colors = cm.rainbow(np.linspace(0, 1, 10))

        reals_pop, reals_ind = realizations

        fig, ax = plt.subplots(1,1)


        for i, (id, individual) in enumerate(data.individuals.items()):
            model_value = self.compute_individual(individual, reals_pop, reals_ind)
            score = individual.tensor_observations

            ax.plot(individual.tensor_timepoints.detach().numpy(), model_value.detach().numpy(), c=colors[i])
            ax.plot(individual.tensor_timepoints.detach().numpy(), score.detach().numpy(), c=colors[i], linestyle='--', marker='o')


        ax.plot([70,90],[self.model_parameters['mu'], self.model_parameters['mu']], linewidth = 5, c='black', alpha = 0.3)

        if not os.path.exists(os.path.join(path_output, 'plots/')):
            os.mkdir(os.path.join(path_output, 'plots/'))

        plt.savefig(os.path.join(path_output, 'plots','plot_patients_{0}.pdf'.format(iter)))
        plt.close()






