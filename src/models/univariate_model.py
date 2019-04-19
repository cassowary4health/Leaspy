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


    def initialize_realizations(self, data):
        """
        Initialize the realizations.
        All individual parameters, and population parameters that need to be considered as realizations.
        TODO : initialize settings + smart initialization
        :param data:
        :return:
        """

        reals_pop_name = ['p0']
        reals_ind_name = ['xi','tau']

        # Population parameters
        reals_pop = dict.fromkeys(reals_pop_name)
        for pop_name in reals_pop_name:
            reals_pop[pop_name] = 0.3

        # Individual parameters
        reals_ind = dict.fromkeys(reals_ind_name)
        for ind_name in reals_ind_name:
            reals_ind_temp = dict(zip(data.indices, (self.model_parameters['{0}_mean'.format(ind_name)]+self.model_parameters['{0}_std'.format(ind_name)]*np.random.randn(1,len(data.indices))).reshape(-1).tolist()))
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
        #return p0 + reparametrized_time

    def compute_average(self, individual, reals_pop, tensor_timepoints):
        p0 = reals_pop['p0']
        reparametrized_time = np.exp(self.model_parameters['xi_mean'])*(tensor_timepoints-self.model_parameters['tau_mean'])
        return torch.pow(1 + (1 / p0 - 1) * torch.exp(-reparametrized_time / (p0 * (1 - p0))), -1)
        #return p0 + reparametrized_time


    def compute_sumsquared(self, data, reals_pop, reals_ind):
        return np.sum([self.compute_individual_sumsquared(individual, reals_pop, reals_ind) for individual in data])

    def compute_attachment(self, data, reals_pop, reals_ind):
        return self.compute_sumsquared(data, reals_pop, reals_ind)*np.power(self.model_parameters['noise_var'], -1) + data.n_observations*np.log(np.sqrt(2*np.pi*self.model_parameters['noise_var']))

    def compute_individual_sumsquared(self, individual, reals_pop, reals_ind):
         return torch.sum((self.compute_individual(individual, reals_pop, reals_ind)-individual.tensor_observations)**2)

    def compute_regularity(self, data, reals_pop, reals_ind):
        tau_regularity = np.var([x.detach().numpy() for x in reals_ind['tau'].values()])*np.power(self.model_parameters['tau_std'], -2) + data.n_individuals*np.log(self.model_parameters['tau_std']*np.sqrt(2*np.pi))
        xi_regularity = np.var([x.detach().numpy() for x in reals_ind['xi'].values()])*np.power(self.model_parameters['xi_std'], -2) + data.n_individuals*np.log(self.model_parameters['xi_std']*np.sqrt(2*np.pi))
        return tau_regularity+xi_regularity

    def update_sufficient_statistics(self, data, reals_ind, reals_pop):

        # Update Parameters

        # Population parameters as realizations
        for pop_name in reals_pop.keys():
            self.model_parameters[pop_name] = reals_pop[pop_name].detach().numpy()

        # population parameters not as realizations
        self.model_parameters['tau_std'] = np.std([x.detach().numpy() for x in reals_ind['tau'].values()])
        self.model_parameters['xi_std'] = np.std([x.detach().numpy() for x in reals_ind['xi'].values()])

        self.model_parameters['tau_mean'] = np.mean([x.detach().numpy() for x in reals_ind['tau'].values()])
        self.model_parameters['xi_mean'] = np.mean([x.detach().numpy() for x in reals_ind['xi'].values()])

        # Noise
        self.model_parameters['noise_var'] = self.compute_sumsquared(data, reals_pop, reals_ind).detach().numpy()


    def plot(self, data, realizations, iter):

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

        # Plot average model
        tensor_timepoints = torch.Tensor(np.linspace(60,90,20).reshape(-1))
        model_average = self.compute_average(individual, reals_pop, tensor_timepoints)
        ax.plot(tensor_timepoints.detach().numpy(), model_average.detach().numpy(), c='black', linewidth = 4, alpha = 0.3)


        if not os.path.exists('../../plots/univariate/'):
            os.mkdir('../../plots/univariate/')

        plt.savefig('../../plots/univariate/plot_patients_{0}.pdf'.format(iter))

