import torch
from src.algo.abstract_mcmc import AbstractMCMC
import os
from src.inputs.algo_settings import AlgoSettings
from src import default_algo_dir
from src.utils.sampler import Sampler
import numpy as np




class TunedMCMCSAEM(AbstractMCMC):

    def __init__(self):
        super().__init__()




    def _sample_population_realizations(self, data, model, reals_pop, reals_ind):

        indices = reals_ind.keys()
        subset_indices = np.random.choice(list(indices), int(data.n_individuals/4))

        data = {k: v for k, v in data.items() if k in subset_indices}
        reals_ind = {k: v for k, v in reals_ind.items() if k in subset_indices}


        info_variables = model.get_info_variables()

        for key in reals_pop.keys():

            shape_current_variable = info_variables[key]["shape"]

            for dim_1 in range(shape_current_variable[0]):
                for dim_2 in range(shape_current_variable[1]):

                    # Compute Old loss
                    previous_reals_pop = reals_pop[key][dim_1, dim_2].clone() #TODO bof
                    #previous_attachment = self.likelihood.get_current_attachment()
                    previous_attachment = model.compute_attachment(data, reals_pop, reals_ind)
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
                    else:
                        #TODO better, with the update variables infos I guess ???
                        for idx in data.indices:
                            new_individual_attachment = model.compute_individual_attachment(data[idx], reals_pop,
                                                                                            reals_ind[idx])
                            self.likelihood.individual_attachment[idx] = new_individual_attachment


                    # Update intermediary model variables if necessary
                    model.update_variable_info(key, reals_pop)


    # TODO Numba this
    def _sample_individual_realizations(self, data, model, reals_pop, reals_ind):

        infos_variables = model.get_info_variables()
        indices = reals_ind.keys()
        subset_indices = np.random.choice(list(indices), int(data.n_individuals/4))

        for idx in subset_indices:
            previous_individual_attachment = self.likelihood.individual_attachment[idx]
            for key in reals_ind[idx].keys():

                shape_current_variable = infos_variables[key]["shape"]

                if shape_current_variable == (1, 1):


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



