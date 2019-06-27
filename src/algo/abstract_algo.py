import torch
import numpy as np

class AbstractAlgo():


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

    def set_output_manager(self, output_manager):
        self.output_manager = output_manager

    ###########################
    ## Core
    ###########################

    def run(self, data, model, seed=None):

        # Initialize Model
        self._initialize_seed(seed)
        realizations = model.initialize_realizations(data)
        model.initialize_random_variables(data)

        #TODO because of dimension problems, maybe not here
        # First update of the model
        #sufficient_statistics = model.compute_sufficient_statistics(data, realizations)
        #model.update_model(data, sufficient_statistics)

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

