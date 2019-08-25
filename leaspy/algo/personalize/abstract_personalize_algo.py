from ..abstract_algo import AbstractAlgo
import numpy as np

class AbstractPersonalizeAlgo(AbstractAlgo):

    def __init__(self, settings):

        # Algorithm parameters
        self.algo_parameters = settings.parameters

    def _get_individual_parameters(self, model, times, values):
        raise NotImplementedError('This algorithm does not present a personalization procedure')

    def run(self,model,data):
        individual_parameters = {}
        total_error = []

        for idx in range(data.n_individuals):
            times = data.get_times_patient(idx)
            values = data.get_values_patient(idx)

            xi, tau, sources, err = self._get_individual_parameters(model,times,values)

            individual_parameters[data.indices[idx]] = {
                'xi': xi,
                'tau': tau,
                'sources': sources
            }

            total_error.append(err.squeeze(0).detach().numpy())
            
        print("The standard deviation of the error is of {}".format(np.std(np.vstack(total_error))))

        return individual_parameters
"""
    def get_times_patient(self, i):
        return self.timepoints[i,:self.nb_observations_per_individuals[i]]

    def get_values_patient(self, i):
        return self.values[i,:self.nb_observations_per_individuals[i],:]
        
"""