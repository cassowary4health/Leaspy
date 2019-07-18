import torch
from src.algo.abstract_algo import AbstractAlgo
import os
from src.inputs.algorithm_settings import AlgorithmSettings
from src import data_dir
import numpy as np
from torch.autograd import Variable

class GradientDescent(AbstractAlgo):

    def __init__(self):
        data_dir = os.path.join(data_dir, 'algo', "default_gradient_descent_parameters.json")
        reader = AlgorithmSettings(data_dir)

        if reader.algo_type != 'gradient_descent':
            raise ValueError("The default gradient descent parameters are not of gradient_descent type")

        self.realizations = None
        self.task = None
        self.algo_parameters = reader.parameters
        self.current_iteration = 0
        self.path_output = 'output/'

    ###########################
    ## Initialization
    ###########################

    def _initialize_algo(self, data, model, realizations):
        self._initialize_likelihood(data, model, realizations)
        realizations = self._initialize_torchvariables(realizations)
        return realizations


    def _initialize_torchvariables(self, realizations):

        reals_pop, reals_ind = realizations

        for key in reals_pop.keys():
            reals_pop[key] = Variable(reals_pop[key].float(), requires_grad=True)

        # To Torch
        for idx in reals_ind.keys():
            for key in reals_ind[idx]:
                reals_ind[idx][key] = Variable(reals_ind[idx][key].float(), requires_grad=True)
        return realizations


    ###########################
    ## Core
    ###########################

    def iteration(self, data, model, realizations):

        reals_pop, reals_ind = realizations

        # Update the intermediary variables (Q, A if needed)
        model.update_variable_info('v0', reals_pop)

        # Compute loss
        attachment = model.compute_attachment(data, reals_pop, reals_ind)
        regularity = model.compute_regularity(data, reals_pop, reals_ind)
        loss = attachment + regularity

        # Do backward and backprop on realizations
        loss.backward()

        #if self.algo_parameters['estimate_population_parameters']:
        self._gradient_update_pop(reals_pop, lr=self.algo_parameters['learning_rate']/data.n_individuals)

        #if self.algo_parameters['estimate_individual_parameters']:
        if self.current_iteration > 100:
            self._gradient_update_ind(reals_ind, lr=self.algo_parameters['learning_rate'])

        # Update the sufficient statistics
        self._maximization_step(data, model, realizations)


    def _gradient_update_pop(self, reals_pop, lr):
        with torch.no_grad():
            for key in reals_pop.keys():
                reals_pop[key] -= lr * reals_pop[key].grad
                reals_pop[key].grad.zero_()

    def _gradient_update_ind(self, reals_ind, lr):
        with torch.no_grad():
            for key in reals_ind.keys():
                for idx in reals_ind[key].keys():
                    reals_ind[key][idx] -= lr * reals_ind[key][idx].grad
                    reals_ind[key][idx].grad.zero_()





