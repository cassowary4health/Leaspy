import torch
from src.algo.abstract_algo import AbstractAlgo
import os
from src.inputs.algo_settings import AlgoSettings
from src import default_algo_dir
from src.utils.sampler import Sampler
import matplotlib.pyplot as plt


import numpy as np

class MCMCSAEM(AbstractAlgo):

    def __init__(self):
        data_dir = os.path.join(default_algo_dir, "default_mcmc_saem_parameters.json")
        reader = AlgoSettings(data_dir)

        if reader.algo_type != 'mcmc_saem':
            raise ValueError("The default mcmc saem parameters are not of mcmc_saem type")


        self.realizations = None
        self.task = None
        self.algo_parameters = reader.parameters

        self.samplers_pop = None
        self.samplers_ind = None

        self.current_iteration = 0

        self.path_output = "output/"

        self.temperature_inv = 1
        self.temperature = 1

    ###########################
    ## Initialization
    ###########################

    def _initialize_algo(self, data, model, realizations):
        self._initialize_samplers(model, data)
        self._initialize_likelihood(data, model, realizations)
        if self.algo_parameters['annealing']['do_annealing']:
            self._initialize_annealing()

    def _initialize_annealing(self):
        if self.algo_parameters['annealing']['do_annealing']:
            if self.algo_parameters['annealing']['n_iter'] is None:
                self.algo_parameters['annealing']['n_iter'] = int(self.algo_parameters['n_iter']/2)

        self.temperature = self.algo_parameters['annealing']['initial_temperature']
        self.temperature_inv = 1/self.temperature



    def _initialize_samplers(self, model, data):
        pop_name = model.reals_pop_name
        ind_name = model.reals_ind_name

        self.samplers_pop = dict.fromkeys(pop_name)
        self.samplers_ind = dict.fromkeys(ind_name)

        # TODO Change this arbitrary parameters --> samplers parameters ???
        for key in pop_name:
            self.samplers_pop[key] = Sampler(key, 0.005, 25)

        for key in ind_name:
            #self.samplers_ind[key] = Sampler(key, np.sqrt(model.model_parameters["{0}_var".format(key)])/2, 200)
            self.samplers_ind[key] = Sampler(key, 0.1, 25*data.n_individuals)




    ###########################
    ## Getters / Setters
    ###########################

    ###########################
    ## Core
    ###########################

    def iteration(self, data, model, realizations):

        reals_pop, reals_ind = realizations

        # Sample step
        self._sample_population_realizations(data, model, reals_pop, reals_ind)
        self._sample_individual_realizations(data, model, reals_pop, reals_ind)

        # Maximization step
        self._maximization_step(data, model, realizations)

        # Annealing
        if self.algo_parameters['annealing']['do_annealing']:
            self._update_temperature()

    def _update_temperature(self):

        if self.current_iteration <= self.algo_parameters['annealing']['n_iter']:
            # If we cross a plateau step
            if self.current_iteration % int(self.algo_parameters['annealing']['n_iter']/self.algo_parameters['annealing']['n_plateau']) == 0:
                # Decrease temperature linearly
                self.temperature -= self.algo_parameters['annealing']['initial_temperature']/self.algo_parameters['annealing']['n_plateau']
                self.temperature = max(self.temperature, 1)
                self.temperature_inv = 1/self.temperature


    # TODO Factorize these functions:
    # We need:
    # -per rv type
    # -per individual/population

    def _sample_population_realizations(self, data, model, reals_pop, reals_ind):

        info_variables = model.get_info_variables(data)

        for key in reals_pop.keys():

            shape_current_variable = info_variables[key]["shape"]

            for dim_1 in range(shape_current_variable[0]):
                for dim_2 in range(shape_current_variable[1]):

                    # Compute Old loss
                    previous_reals_pop = reals_pop[key][dim_1, dim_2].clone() #TODO bof
                    previous_attachment = self.likelihood.get_current_attachment()
                    previous_regularity = model.compute_regularity_arrayvariable(previous_reals_pop, key, (dim_1, dim_2))
                    previous_loss = previous_attachment + previous_regularity

                    # New loss
                    reals_pop[key][dim_1, dim_2] = reals_pop[key][dim_1, dim_2] + self.samplers_pop[key].sample()

                    # Update intermediary model variables if necessary
                    model.update_variable_info(key, reals_pop)

                    # Compute new loss
                    new_attachment = model.compute_attachment(data, reals_pop, reals_ind)
                    new_regularity = model.compute_regularity_arrayvariable(reals_pop[key][dim_1, dim_2], key, (dim_1, dim_2))
                    new_loss = new_attachment + new_regularity

                    #alpha = np.exp(-(new_loss-previous_loss).detach().numpy()*self.temperature_inv)
                    alpha = np.exp(-((new_regularity-previous_regularity)*self.temperature_inv +
                                   (new_attachment-previous_attachment)).detach().numpy())

                    # Compute acceptation
                    accepted = self.samplers_pop[key].acceptation(alpha)


                    # Revert if not accepted
                    if not accepted:
                        reals_pop[key][dim_1, dim_2] = previous_reals_pop
                        # Update intermediary model variables if necessary
                        model.update_variable_info(key, reals_pop)

    # TODO Numba this
    def _sample_individual_realizations(self, data, model, reals_pop, reals_ind):
        for idx in reals_ind.keys():
            previous_individual_attachment = self.likelihood.individual_attachment[idx]
            for key in reals_ind[idx].keys():

                # Save previous realization
                previous_reals_ind = reals_ind[idx][key]

                # Compute previous loss
                #previous_individual_attachment = model.compute_individual_attachment(data[idx], reals_pop, reals_ind[idx])
                previous_individual_regularity = model.compute_regularity_variable(reals_ind[idx][key], key)
                previous_individual_loss = previous_individual_attachment + previous_individual_regularity

                # Sample a new realization
                reals_ind[idx][key] = reals_ind[idx][key] + self.samplers_ind[key].sample()

                # Compute new loss
                new_individual_attachment = model.compute_individual_attachment(data[idx], reals_pop, reals_ind[idx])
                new_individual_regularity = model.compute_regularity_variable(reals_ind[idx][key], key)
                new_individual_loss = new_individual_attachment + new_individual_regularity

                #alpha = np.exp(-(new_individual_loss - previous_individual_loss).detach().numpy()*self.temperature_inv)

                alpha = np.exp(-((new_individual_regularity-previous_individual_regularity)*self.temperature_inv +
                               (new_individual_attachment-previous_individual_attachment)).detach().numpy())

                # Compute acceptation
                accepted = self.samplers_ind[key].acceptation(alpha)

                #TODO Handle here if dim sources > 1

                # Revert if not accepted
                if not accepted:
                    reals_ind[idx][key] = previous_reals_ind
                # Keep new attachment if accepted
                else:
                    self.likelihood.individual_attachment[idx] = new_individual_attachment



    def __str__(self):
        out = ""
        out += "=== ALGO ===\n"
        out += "Iteration {0}\n".format(self.current_iteration)
        out += "=Samplers \n"

        for sampler_name, sampler in self.samplers_pop.items():
            acceptation_rate = np.mean(sampler.acceptation_temp)
            out += "    {0} rate : {1}%, std: {2}\n".format(sampler_name, 100*acceptation_rate,
                                                            sampler.std)

        for sampler_name, sampler in self.samplers_ind.items():
            acceptation_rate = np.mean(sampler.acceptation_temp)
            out += "    {0} rate : {1}%, std: {2}\n".format(sampler_name, 100 * acceptation_rate,
                                                            sampler.std)

        if self.algo_parameters['annealing']['do_annealing']:
            out += "Annealing \n"
            out += "Temperature : {0}".format(self.temperature)

        return out