import torch
from src.algo.abstract_mcmc import AbstractMCMC
import copy


class MCMCPersonalize(AbstractMCMC):

    def __init__(self, settings):
        super().__init__(settings)
        self.name = "MCMC Personalize"

        self.personalize_output = {"distribution":{}}

        if self.algo_parameters['history_length']>self.algo_parameters['n_iter']:
            raise ValueError("History length of ind variables to save is greater than maximum number of iterations")


    def iteration(self, data, model, realizations):

        # Sample step
        self._sample_individual_realizations(data, model, realizations)

        # Annealing
        if self.algo_parameters['annealing']['do_annealing']:
            self._update_temperature()

        # Stack the realizations at the 100 last iterations
        iter_history = -self.algo_parameters['n_iter']+self.algo_parameters['history_length']+self.current_iteration
        if iter_history >= 0:
            self.personalize_output["distribution"][iter_history] = realizations.copy()

        """

        noise = (model.compute_sum_squared_tensorized(data, realizations).sum() / (
                    data.n_visits * data.dimension)).detach().numpy().tolist()
        print(noise)"""
