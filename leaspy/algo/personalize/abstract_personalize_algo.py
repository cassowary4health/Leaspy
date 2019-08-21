from ..abstract_algo import AbstractAlgo
import numpy as np

class AbstractPersonalizeAlgo(AbstractAlgo):

    def __init__(self, settings):

        # Algorithm parameters
        self.algo_parameters = settings.parameters

    def run(self,model,data):
        individual_parameters = {}

        for idx in range(data.n_individuals):
            # print(idx)
            times = data.get_times_patient(idx)
            values = data.get_values_patient(idx)

            xi, tau, sources = self._get_individual_parameters(model,times,values)

            individual_parameters[data.indices[idx]] = {
                'xi': xi,
                'tau': tau,
                'sources': sources
            }

        return individual_parameters
"""
    def get_times_patient(self, i):
        return self.timepoints[i,:self.nb_observations_per_individuals[i]]

    def get_values_patient(self, i):
        return self.values[i,:self.nb_observations_per_individuals[i],:]
        
"""