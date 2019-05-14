import torch
from src.algo.abstract_algo import AbstractAlgo
import os
from src.inputs.algo_reader import AlgoReader
from src import default_algo_dir
from src.utils.sampler import Sampler
import matplotlib.pyplot as plt

import numpy as np

class MCMCSAEM(AbstractAlgo):

    def __init__(self):
        data_dir = os.path.join(default_algo_dir, "default_mcmc_saem_parameters.json")
        reader = AlgoReader(data_dir)

        if reader.algo_type != 'mcmc_saem':
            raise ValueError("The default mcmc saem parameters are not of random_sampling type")


        self.realizations = None
        self.task = None
        self.algo_parameters = reader.parameters

        self.samplers_pop = None
        self.samplers_ind = None

        self.iteration = 0

        self.path_output = "output/"

    ###########################
    ## Initialization
    ###########################

    def _initialize_algo(self, model):
        self._initialize_samplers(model)

    def _initialize_samplers(self, model):

        pop_name = model.reals_pop_name
        ind_name = model.reals_ind_name

        self.samplers_pop = dict.fromkeys(pop_name)
        self.samplers_ind = dict.fromkeys(ind_name)

        # TODO Change this arbitrary parameters --> samplers parameters ???
        for key in pop_name:
            self.samplers_pop[key] = Sampler(key, 0.01, 50)

        for key in ind_name:
            self.samplers_ind[key] = Sampler(key, np.sqrt(model.model_parameters["{0}_var".format(key)])/2, 200)

    ###########################
    ## Getters / Setters
    ###########################

    def set_mode(self, task):
        self.task = task
        if self.task == 'fit':
            self.algo_parameters['estimate_individual_parameters'] = True
            self.algo_parameters['estimate_population_parameters'] = True
        elif self.task == 'predict':
            self.algo_parameters['estimate_individual_parameters'] = True
            self.algo_parameters['estimate_population_parameters'] = False


    ###########################
    ## Core
    ###########################

    def iter(self, data, model, realizations):

        reals_pop, reals_ind = realizations

        # Sample step
        self._sample_population_realizations(data, model, reals_pop, reals_ind)
        self._sample_individual_realizations(data, model, reals_pop, reals_ind)

        # Maximization step
        if self.algo_parameters['estimate_population_parameters']:
            model.update_sufficient_statistics(data, reals_ind, reals_pop)

        self.iteration += 1

    def _sample_population_realizations(self, data, model, reals_pop, reals_ind):
        for key in reals_pop.keys():

            # Old loss
            previous_reals_pop = reals_pop[key]
            previous_attachment = model.compute_attachment(data, reals_pop, reals_ind)
            previous_regularity = model.compute_regularity(data, reals_pop, reals_ind)
            previous_loss = previous_attachment + previous_regularity

            # New loss
            reals_pop[key] = reals_pop[key] + self.samplers_pop[key].sample()
            new_attachment = model.compute_attachment(data, reals_pop, reals_ind)
            new_regularity = model.compute_regularity(data, reals_pop, reals_ind)
            new_loss = new_attachment + new_regularity

            alpha = np.exp(-(new_loss-previous_loss).detach().numpy())

            # Compute acceptation
            accepted = self.samplers_pop[key].acceptation(alpha)

            # Revert if not accepted
            if not accepted:
                reals_pop[key] = previous_reals_pop

    def _sample_individual_realizations(self, data, model, reals_pop, reals_ind):
        for idx in reals_ind.keys():
            for key in reals_ind[idx].keys():

                # Save previous realization
                previous_reals_ind = reals_ind[idx][key]
                # print(previous_reals_ind)

                # Compute previous loss

                previous_individual_attachment = model.compute_individual_attachment(data[idx], reals_pop,
                                                                                     reals_ind[idx])
                previous_individual_regularity = model.compute_regularity_variable(reals_ind[idx][key], key)
                previous_individual_loss = previous_individual_attachment + previous_individual_regularity

                # Sample a new realization
                reals_ind[idx][key] = reals_ind[idx][key] + self.samplers_ind[key].sample()

                # Compute new loss
                new_individual_attachment = model.compute_individual_attachment(data[idx], reals_pop, reals_ind[idx])
                new_individual_regularity = model.compute_regularity_variable(reals_ind[idx][key], key)
                new_individual_loss = new_individual_attachment + new_individual_regularity

                alpha = np.exp(-(new_individual_loss - previous_individual_loss).detach().numpy())

                # Compute acceptation
                accepted = self.samplers_ind[key].acceptation(alpha)

                # Revert if not accepted
                if not accepted:
                    reals_ind[idx][key] = previous_reals_ind

