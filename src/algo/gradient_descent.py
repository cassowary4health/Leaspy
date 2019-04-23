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
        self.model = None
        self.algo_parameters = reader.parameters

    def run(self, data, model, seed=None):

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            print(" ==> Setting seed to {0}".format(seed))

        realizations = model.initialize_realizations(data)

        for iter in range(self.algo_parameters['n_iter']):
            self.iter(data, model, realizations)

            if iter%100 == 0:
                model.plot(data, iter, realizations)

        self.realizations = realizations

    def iter(self, data, model, realizations):

        reals_pop, reals_ind = realizations

        # Compute loss
        attachment = model.compute_attachment(data, reals_pop, reals_ind)
        regularity = model.compute_regularity(data, reals_pop, reals_ind)
        loss = attachment + regularity


        #print(loss, attachment, regularity)
        #print(model.model_parameters)
        #print(reals_ind)
        #print("Sigma2 {0}".format(np.var([x.detach().numpy() for x in reals_ind['intercept'].values()])))

        # Do backward and backprop on realizations
        loss.backward()

        with torch.no_grad():

            if self.algo_parameters['estimate_population_parameters']:
                for key in reals_pop.keys():
                    reals_pop[key] -= self.algo_parameters['learning_rate'] * reals_pop[key].grad
                    reals_pop[key].grad.zero_()

            if self.algo_parameters['estimate_individual_parameters']:
                for key in reals_ind.keys():
                    for idx in reals_ind[key].keys():
                        reals_ind[key][idx] -= self.algo_parameters['learning_rate'] * reals_ind[key][idx].grad
                        reals_ind[key][idx].grad.zero_()


        # Update the sufficient statistics
        if self.algo_parameters['estimate_population_parameters']:
            model.update_sufficient_statistics(data, reals_ind, reals_pop)

        return 0


    def get_realizations(self):
        return self.realizations

    def set_mode(self, task):
        self.task = task
        if self.task == 'fit':
            self.algo_parameters['estimate_individual_parameters'] = True
            self.algo_parameters['estimate_population_parameters'] = True
        elif self.task == 'predict':
            self.algo_parameters['estimate_individual_parameters'] = True
            self.algo_parameters['estimate_population_parameters'] = False




