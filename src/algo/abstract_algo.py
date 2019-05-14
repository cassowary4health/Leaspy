import torch
import numpy as np

class AbstractAlgo():


    def load_parameters(self, parameters):
        for k, v in parameters.items():
            if k in self.algo_parameters.keys():
                previous_v = self.algo_parameters[k]
                print("Replacing {} parameter from value {} to value {}".format(k, previous_v, v))
            self.algo_parameters[k] = v

    @staticmethod
    def _initialize_seed(seed):
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            print(" ==> Setting seed to {0}".format(seed))

    def run(self, data, model, output_manager, seed=None):

        # Initialize Algorithm
        self._initialize_seed(seed)
        self._initialize_algo(model)
        realizations = model.initialize_realizations(data)

        # Iterate
        for iteration in range(self.algo_parameters['n_iter']):
            output_manager.iter(self, data, model, realizations)
            self.iter(data, model, realizations)

        return realizations

    def iter(self, data, model, realizations):
        raise NotImplementedError

    def __str__(self):
        out = ""
        out += "=== ALGO ===\n"
        out += "Iteration {0}".format(self.iteration)
        return out