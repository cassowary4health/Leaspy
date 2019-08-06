import torch
import numpy as np
from src.utils.output.fit_output_manager import FitOutputManager

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

    def set_output_manager(self, output_path):
        if output_path is not None:
            self.output_manager = FitOutputManager(output_path)
        else:
            self.output_manager = None