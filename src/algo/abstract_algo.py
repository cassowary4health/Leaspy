import torch
import numpy as np
from src.utils.likelihood import Likelihood
from src.utils.output_manager import OutputManager


class AbstractAlgo:
    ###########################
    ## Initialization
    ###########################

    @staticmethod
    def _initialize_seed(seed):
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            print(" ==> Setting seed to {0}".format(seed))

    ###########################
    ## Getters / Setters
    ###########################

    def load_parameters(self, parameters):
        for k, v in parameters.items():
            if k in self.algo_parameters.keys():
                previous_v = self.algo_parameters[k]
                print("Replacing {} parameter from value {} to value {}".format(k, previous_v, v))
            self.algo_parameters[k] = v


    def __str__(self):
        out = ""
        out += "=== ALGO ===\n"
        out += "Iteration {0}".format(self.current_iteration)
        return out

    def set_output_manager(self, output_path):
        if output_path is not None:
            self.output_manager = OutputManager(output_path)
        else:
            self.output_manager = None

    ###########################
    ## Core
    ###########################

    def run(self, data, model, seed=None):

        # Initialize Model
        self._initialize_seed(seed)
        # Initialize first the random variables
        # TODO : Check if needed - model.initialize_random_variables(data)
        # Then initialize the Realizations (from the random variables)
        realizations = model.initialize_realizations(data)

        # Initialize Algo
        self._initialize_algo(data, model, realizations)

        # Iterate
        for it in range(self.algo_parameters['n_iter']):
            self.iteration(data, model, realizations)
            self.output_manager.iteration(self, data, model, realizations)
            self.current_iteration += 1

        return realizations

    def iteration(self, data, model, realizations):
        raise NotImplementedError

    @staticmethod
    def _maximization_step(data, model, realizations):
        sufficient_statistics = model.compute_sufficient_statistics(data, realizations)
        model.update_model(data, sufficient_statistics)

    def _initialize_likelihood(self, data, model, realizations):
        return 0
        # TODO later
        #self.likelihood = Likelihood()
        #self.likelihood.initialize_likelihood(data, model, realizations)

