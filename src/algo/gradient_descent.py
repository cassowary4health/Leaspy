import torch
from src.algo.abstract_algo import AbstractAlgo
import os
from src.inputs.algo_reader import AlgoReader
from src import default_algo_dir
import numpy as np

class GradientDescent(AbstractAlgo):

    def __init__(self):
        data_dir = os.path.join(default_algo_dir, "default_gradient_descent_parameters.json")
        reader = AlgoReader(data_dir)

        if reader.algo_type != 'gradient_descent':
            raise ValueError("The default gradient descent parameters are not of gradient_descent type")

        self.realizations = None
        self.task = None
        self.algo_parameters = reader.parameters
        self.iteration = 0
        self.path_output = 'output/'

    ###########################
    ## Initialization
    ###########################

    def _initialize_algo(self, data, model, realizations):
        # TODO Initialize the learning rate ???
        pass


    ###########################
    ## Core
    ###########################

    def iter(self, data, model, realizations):

        reals_pop, reals_ind = realizations

        # Compute loss
        attachment = model.compute_attachment(data, reals_pop, reals_ind)
        regularity = model.compute_regularity(data, reals_pop, reals_ind)
        loss = attachment + regularity

        # Do backward and backprop on realizations
        loss.backward()

        #if self.algo_parameters['estimate_population_parameters']:
        self._gradient_update_pop(reals_pop)

        #if self.algo_parameters['estimate_individual_parameters']:
        self._gradient_update_ind(reals_ind)

        # Update the sufficient statistics
        if self.algo_parameters['estimate_population_parameters']:
            model.update_sufficient_statistics(data, reals_ind, reals_pop)

        self.iteration += 1


    def _gradient_update_pop(self, reals_pop):
        with torch.no_grad():
            for key in reals_pop.keys():
                reals_pop[key] -= self.algo_parameters['learning_rate'] * reals_pop[key].grad
                reals_pop[key].grad.zero_()

    def _gradient_update_ind(self, reals_ind):
        with torch.no_grad():
            for key in reals_ind.keys():
                for idx in reals_ind[key].keys():
                    reals_ind[key][idx] -= self.algo_parameters['learning_rate'] * reals_ind[key][idx].grad
                    reals_ind[key][idx].grad.zero_()





