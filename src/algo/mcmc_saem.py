import torch
from src.algo.abstract_algo import AbstractAlgo
import os
from src.inputs.algo_settings import AlgoSettings
from src import default_algo_dir
from src.utils.sampler import Sampler
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

        self.samplers = None


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
        # TODO change the parameters ????


        infos_variables = model.get_info_variables()

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

    def _metropolisacceptation_step(self, new_regularity, previous_regularity, new_attachment, previous_attachment, key):

        # Compute energy difference
        alpha = np.exp(-((new_regularity - previous_regularity) * self.temperature_inv +
                         (new_attachment - previous_attachment)).detach().numpy())

        # Compute acceptation
        accepted = self.samplers[key].acceptation(alpha)
        return accepted


    def _metropolissampling_step(self, model, real, key, dim_1, dim_2):
        pass


    def _sample_population_realizations(self, data, model, reals_pop, reals_ind):

        info_variables = model.get_info_variables()

        for key in reals_pop.keys():

            shape_current_variable = info_variables[key]["shape"]

            for dim_1 in range(shape_current_variable[0]):
                for dim_2 in range(shape_current_variable[1]):

                    # Compute Old loss
                    previous_reals_pop = reals_pop[key][dim_1, dim_2].clone() #TODO bof
                    previous_attachment = self.likelihood.get_current_attachment()
                    previous_regularity = model.compute_regularity_arrayvariable(previous_reals_pop, key, (dim_1, dim_2))

                    # New loss
                    reals_pop[key][dim_1, dim_2] = reals_pop[key][dim_1, dim_2] + self.samplers[key].sample()

                    # Update intermediary model variables if necessary
                    model.update_variable_info(key, reals_pop)

                    # Compute new loss
                    new_attachment = model.compute_attachment(data, reals_pop, reals_ind)
                    new_regularity = model.compute_regularity_arrayvariable(reals_pop[key][dim_1, dim_2], key, (dim_1, dim_2))

                    accepted = self._metropolisacceptation_step(new_regularity, previous_regularity,
                                                new_attachment, previous_attachment,
                                                key)


                    # Revert if not accepted
                    if not accepted:
                        reals_pop[key][dim_1, dim_2] = previous_reals_pop

                    # Update intermediary model variables if necessary
                    model.update_variable_info(key, reals_pop)



    # TODO Numba this
    def _sample_individual_realizations(self, data, model, reals_pop, reals_ind):

        infos_variables = model.get_info_variables()

        for idx in reals_ind.keys():
            previous_individual_attachment = self.likelihood.individual_attachment[idx]
            for key in reals_ind[idx].keys():

                shape_current_variable = infos_variables[key]["shape"]

                if shape_current_variable == (1,1):


                    # Save previous realization
                    previous_reals_ind = reals_ind[idx][key]

                    # Compute previous loss
                    #previous_individual_attachment = model.compute_individual_attachment(data[idx], reals_pop, reals_ind[idx])
                    previous_individual_regularity = model.compute_regularity_variable(reals_ind[idx][key], key)

                    # Sample a new realization
                    reals_ind[idx][key] = reals_ind[idx][key] + self.samplers[key].sample()

                    # Compute new loss
                    new_individual_attachment = model.compute_individual_attachment(data[idx], reals_pop, reals_ind[idx])
                    new_individual_regularity = model.compute_regularity_variable(reals_ind[idx][key], key)

                    accepted = self._metropolisacceptation_step(new_individual_regularity, previous_individual_regularity,
                                                     new_individual_attachment, previous_individual_attachment,
                                                     key)

                    #TODO Handle here if dim sources > 1

                    # Revert if not accepted
                    if not accepted:
                        reals_ind[idx][key] = previous_reals_ind
                    # Keep new attachment if accepted
                    else:
                        self.likelihood.individual_attachment[idx] = new_individual_attachment



                else:


                    for dim_1 in range(shape_current_variable[0]):
                        for dim_2 in range(shape_current_variable[1]):

                            # Compute Old loss
                            previous_reals_ind = reals_ind[idx][key][dim_1, dim_2].clone()  # TODO bof
                            previous_attachment = self.likelihood.get_current_attachment()
                            previous_regularity = model.compute_regularity_variable(previous_reals_ind, key)

                            # New loss
                            reals_ind[idx][key][dim_1, dim_2] = reals_ind[idx][key][dim_1, dim_2] + self.samplers[key].sample()

                            # Compute new loss
                            new_attachment = model.compute_attachment(data, reals_pop, reals_ind)
                            new_regularity = model.compute_regularity_variable(reals_ind[idx][key][dim_1, dim_2], key)

                            accepted = self._metropolisacceptation_step(new_regularity, previous_regularity,
                                                                        new_attachment, previous_attachment,
                                                                        key)

                            # Revert if not accepted
                            if not accepted:
                                reals_ind[idx][key][dim_1, dim_2] = previous_reals_ind





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