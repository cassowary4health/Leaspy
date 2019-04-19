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

        reals_pop_name = []
        reals_ind_name = ['intercept']

        # Population parameters
        reals_pop = dict.fromkeys(reals_pop_name)
        for pop_name in reals_pop_name:
            reals_pop[pop_name] = 0

        # Individual parameters
        reals_ind = dict.fromkeys(reals_ind_name)
        for ind_name in reals_ind_name:
            reals_ind_temp = dict(zip(data.indices, (np.sqrt(self.model_parameters['intercept_var'])*np.random.randn(1,len(data.indices))).reshape(-1).tolist()))
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
        return np.sum([self.compute_individual_sumsquared(individual, reals_pop, reals_ind) for individual in data])

    def compute_attachment(self, data, reals_pop, reals_ind):
        return self.compute_sumsquared(data, reals_pop, reals_ind)*np.power(self.model_parameters['noise_var'], -1) + data.n_observations*np.log(np.sqrt(2*np.pi*self.model_parameters['noise_var']))

    def compute_individual_sumsquared(self, individual, reals_pop, reals_ind):
         return torch.sum((self.compute_individual(individual, reals_pop, reals_ind)-individual.tensor_observations)**2)

    def compute_regularity(self, data, reals_pop, reals_ind):
        return np.var([x.detach().numpy() for x in reals_ind['intercept'].values()])*np.power(self.model_parameters['intercept_var'], -1) + data.n_individuals*np.log(np.sqrt(2*np.pi*self.model_parameters['intercept_var']))

    def update_sufficient_statistics(self, data, reals_ind, reals_pop):

        # Update Parameters

        # Population parameters as realizations
        for pop_name in reals_pop.keys():
            self.model_parameters[pop_name] = reals_pop[pop_name].detach().numpy()

        # population parameters not as realizations
        self.model_parameters['intercept_var'] = np.var([x.detach().numpy() for x in reals_ind['intercept'].values()])
        self.model_parameters['mu'] = np.mean([x.detach().numpy() for x in reals_ind['intercept'].values()])

        # Noise
        self.model_parameters['noise_var'] = self.compute_sumsquared(data, reals_pop, reals_ind).detach().numpy()


    def plot(self, data, realizations, iter):
        pass

        """
        import matplotlib.pyplot as plt

        import matplotlib.cm as cm

        colors = cm.rainbow(np.linspace(0, 1, 10))

        reals_pop, reals_ind = realizations

        fig, ax = plt.subplots(1,1)


        for i, individual in enumerate(data):
            model_value = self.compute_individual(individual, reals_pop, reals_ind)
            score = individual.tensor_observations

            ax.plot(individual.tensor_timepoints.detach().numpy(), model_value.detach().numpy(), c=colors[i])
            ax.plot(individual.tensor_timepoints.detach().numpy(), score.detach().numpy(), c=colors[i], linestyle='--', marker='o')


        ax.plot([70,90],[self.model_parameters['mu'], self.model_parameters['mu']], linewidth = 5, c='black', alpha = 0.3)

        if not os.path.exists('../../plots/gaussian_distribution/'):
            os.mkdir('../../plots/gaussian_distribution/')

        plt.savefig('../../plots/gaussian_distribution/plot_patients_{0}.pdf'.format(iter))
        """





