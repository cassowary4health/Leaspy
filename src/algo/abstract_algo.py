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

    def set_mode(self, task):
        self.task = task
        if self.task == 'fit':
            self.algo_parameters['estimate_individual_parameters'] = True
            self.algo_parameters['estimate_population_parameters'] = True
        elif self.task == 'predict':
            self.algo_parameters['estimate_individual_parameters'] = True
            self.algo_parameters['estimate_population_parameters'] = False


    def load_parameters(self, parameters):
        for k, v in parameters.items():
            if k in self.algo_parameters.keys():
                previous_v = self.algo_parameters[k]
                print("Replacing {} parameter from value {} to value {}".format(k, previous_v, v))
            self.algo_parameters[k] = v


    def __str__(self):
        out = ""
        out += "=== ALGO ===\n"
        out += "Iteration {0}".format(self.iteration)
        return out

    ###########################
    ## Core
    ###########################

    def run(self, data, model, output_manager, seed=None):

        # Initialize Algorithm
        self._initialize_seed(seed)
        realizations = model.initialize_realizations(data)
        self._initialize_algo(data, model, realizations)

        # Iterate
        for iteration in range(self.algo_parameters['n_iter']):
            output_manager.iter(self, data, model, realizations)
            self.iter(data, model, realizations)

        return realizations

    def iter(self, data, model, realizations):
        raise NotImplementedError

    @staticmethod
    def _maximization_step(data, model, reals_ind, reals_pop):
        sufficient_statistics = model.compute_sufficient_statistics(data, reals_ind, reals_pop)
        model.update_model(data, sufficient_statistics)

