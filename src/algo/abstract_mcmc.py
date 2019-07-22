import torch
from src.algo.abstract_algo import AbstractAlgo
import os
from src.inputs.algorithm_settings import AlgorithmSettings
from src import default_algo_dir
from src.utils.sampler import Sampler
import numpy as np




class AbstractMCMC(AbstractAlgo):

    def __init__(self, settings):
        #data_dir = os.path.join(default_algo_dir, "default_mcmc_saem_parameters.json")
        #reader = AlgorithmSettings(data_dir)

        #if reader.algo_type != 'mcmc_saem':
        #    raise ValueError("The default mcmc saem parameters are not of mcmc_saem type")


        self.realizations = None
        self.task = None
        self.algo_parameters = settings.parameters

        self.samplers = None


        self.current_iteration = 0

        self.path_output = "output/"

        self.temperature_inv = 1
        self.temperature = 1

    ###########################
    ## Initialization
    ###########################

    def _initialize_algo(self, data, model, realizations):
        model.initialize_MCMC_toolbox(data)
        self._initialize_samplers(model, data)
        self._initialize_likelihood(data, model, realizations)
        if self.algo_parameters['annealing']['do_annealing']:
            self._initialize_annealing()

        return realizations

    def _initialize_annealing(self):
        if self.algo_parameters['annealing']['do_annealing']:
            if self.algo_parameters['annealing']['n_iter'] is None:
                self.algo_parameters['annealing']['n_iter'] = int(self.algo_parameters['n_iter']/2)

        self.temperature = self.algo_parameters['annealing']['initial_temperature']
        self.temperature_inv = 1/self.temperature

    def _initialize_samplers(self, model, data):
        infos_variables = model.random_variable_informations()
        self.samplers = dict.fromkeys(infos_variables.keys())
        for variable, info in infos_variables.items():
            self.samplers[variable] = Sampler(info, data.n_individuals)

    ###########################
    ## Getters / Setters
    ###########################

    ###########################
    ## Core
    ###########################

    def iteration(self, data, model, realizations):

        # Sample step
        self._sample_population_realizations(data, model, realizations)
        self._sample_individual_realizations(data, model, realizations)

        # Maximization step
        self._maximization_step(data, model, realizations)
        model.update_MCMC_toolbox(['all'], realizations)

        # Update the likelihood with the new noise_var
        # TODO likelihood is computed 2 times, remove this one, and update it in maximization step ?
        # TODO or ar the update of all sufficient statistics ???
        #self.likelihood.update_likelihood(data, model, realizations)

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

    def _metropolisacceptation_step(self, new_regularity, previous_regularity, new_attachment, previous_attachment, key):

        # Compute energy difference
        alpha = np.exp(-((new_regularity - previous_regularity) * self.temperature_inv +
                         (new_attachment - previous_attachment)).detach().numpy())

        # Compute acceptation
        accepted = self.samplers[key].acceptation(alpha)
        return accepted


    def _sample_population_realizations(self, data, model, realizations):
        raise NotImplementedError

    def _sample_individual_realizations(self, data, model, reals_pop, reals_ind):
        raise NotImplementedError


    ###########################
    ## Output
    ###########################

    def __str__(self):
        out = ""
        out += "=== ALGO ===\n"
        out += "Iteration {0}\n".format(self.current_iteration)
        out += "=Samplers \n"
        for sampler_name, sampler in self.samplers.items():
            acceptation_rate = np.mean(sampler.acceptation_temp)
            out += "    {0} rate : {1}%, std: {2}\n".format(sampler_name, 100*acceptation_rate,
                                                            sampler.std)

        if self.algo_parameters['annealing']['do_annealing']:
            out += "Annealing \n"
            out += "Temperature : {0}".format(self.temperature)
        return out