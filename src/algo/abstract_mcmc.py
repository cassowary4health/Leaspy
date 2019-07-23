import torch
from src.algo.abstract_algo import AbstractAlgo
import os
from src.inputs.algorithm_settings import AlgorithmSettings
from src import default_algo_dir
from src.utils.sampler import Sampler
import numpy as np




class AbstractMCMC(AbstractAlgo):

    def __init__(self, settings):

        # Algorithm parameters
        self.algo_parameters = settings.parameters

        # Realizations and utils
        self.realizations = None
        self.task = None
        self.samplers = None
        self.current_iteration = 0

        # Annealing
        self.temperature_inv = 1
        self.temperature = 1

    ###########################
    ## Initialization
    ###########################

    def _initialize_algo(self, data, model, realizations):
        # MCMC toolbox (cache variables for speed-ups + tricks)
        model.initialize_MCMC_toolbox(data)
        # Samplers
        self._initialize_samplers(model, data)
        self._initialize_sufficient_statistics(data, model, realizations)
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

    def _initialize_sufficient_statistics(self, data, model, realizations):
        suff_stats = model.compute_sufficient_statistics(data, realizations)
        self.sufficient_statistics = {k: torch.zeros(v.shape) for k, v in suff_stats.items()}

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
            if self.current_iteration % int(
                    self.algo_parameters['annealing']['n_iter'] / self.algo_parameters['annealing'][
                        'n_plateau']) == 0:
                # Decrease temperature linearly
                self.temperature -= self.algo_parameters['annealing']['initial_temperature'] / \
                                    self.algo_parameters['annealing']['n_plateau']
                self.temperature = max(self.temperature, 1)
                self.temperature_inv = 1 / self.temperature

    def _metropolisacceptation_step(self, new_regularity, previous_regularity, new_attachment, previous_attachment, key):

        # Compute energy difference
        alpha = np.exp(-((new_regularity - previous_regularity) * self.temperature_inv +
                         (new_attachment - previous_attachment)).detach().numpy())

        # Compute acceptation
        accepted = self.samplers[key].acceptation(alpha)
        return accepted



    def _sample_population_realizations(self, data, model, realizations):

        for key in realizations.reals_pop_variable_names:
            shape_current_variable = realizations[key].shape

            # For all the dimensions
            for dim_1 in range(shape_current_variable[0]):
                for dim_2 in range(shape_current_variable[1]):

                    # Compute the attachment and regularity
                    previous_attachment = model.compute_individual_attachment_tensorized(data, realizations).sum()
                    previous_regularity = model.compute_regularity_variable(realizations[key])

                    # Keep previous realizations and sample new ones
                    previous_reals_pop = realizations[key].tensor_realizations.clone()
                    realizations[key].set_tensor_realizations_element(realizations[key].tensor_realizations[dim_1, dim_2] + self.samplers[key].sample(), (dim_1, dim_2))

                    # Update intermediary model variables if necessary
                    model.update_MCMC_toolbox([key], realizations)

                    # Compute the attachment and regularity
                    new_attachment = model.compute_individual_attachment_tensorized(data, realizations).sum()
                    new_regularity = model.compute_regularity_variable(realizations[key])

                    accepted = self._metropolisacceptation_step(new_regularity.sum(), previous_regularity.sum(),
                                                new_attachment, previous_attachment,
                                                key)

                    # Revert if not accepted
                    if not accepted:
                        # Revert realizations
                        realizations[key].tensor_realizations = previous_reals_pop
                        # Update intermediary model variables if necessary
                        model.update_MCMC_toolbox([key], realizations)


    def _sample_individual_realizations(self, data, model, realizations):


        for key_ind in realizations.reals_ind_variable_names:

            # Compute the attachment and regularity
            previous_individual_attachment = model.compute_individual_attachment_tensorized(data, realizations)
            previous_individual_regularity = model.compute_regularity_variable(realizations[key_ind])

            # Keep previous realizations and sample new ones
            previous_array_ind = realizations[key_ind].tensor_realizations
            realizations[key_ind].tensor_realizations = realizations[key_ind].tensor_realizations + self.samplers[key_ind].sample(
                shape=realizations[key_ind].tensor_realizations.shape)

            # Compute the attachment and regularity
            new_individual_attachment = model.compute_individual_attachment_tensorized(data, realizations)
            new_individual_regularity = model.compute_regularity_variable(realizations[key_ind])

            # Compute acceptation
            alpha = torch.exp(-((new_individual_attachment-previous_individual_attachment)+
                        self.temperature_inv*(new_individual_regularity- previous_individual_regularity).sum(dim=2).reshape(data.n_individuals)))
            #print(key_ind, 'Attachement {} vs {} Regularity'.format(torch.mean(pre)))
            for i, acceptation_patient in enumerate(alpha):
                accepted = self.samplers[key_ind].acceptation(acceptation_patient.detach().numpy())
                if not accepted:
                    # Update the realizations
                    realizations[key_ind].tensor_realizations[i] = previous_array_ind[i]

    ###########################
    ## Output
    ###########################

    def __str__(self):
        out = ""
        out += "=== ALGO ===\n"
        out += "Instance of {0} algo \n".format(self.name)
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


