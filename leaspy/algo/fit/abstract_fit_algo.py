from ...utils.output.fit_output_manager import FitOutputManager
from ..abstract_algo import AbstractAlgo

class AbstractFitAlgo(AbstractAlgo):

    ###########################
    ## Core
    ###########################

    def run(self, data, model, seed=None):

        # Initialize Model
        self._initialize_seed(seed)
        # Initialize first the random variables
        # TODO : Check if needed - model.initialize_random_variables(data)
        # Then initialize the Realizations (from the random variables)
        realizations = model.get_realization_object(data.n_individuals)

        # Initialize Algo
        self._initialize_algo(data, model, realizations)

        # Iterate
        for it in range(self.algo_parameters['n_iter']):
            self.iteration(data, model, realizations)
            if self.output_manager is not None: # TODO better this, should work with nones
                self.output_manager.iteration(self, data, model, realizations)
            self.current_iteration += 1

        return realizations

    def iteration(self, data, model, realizations):
        raise NotImplementedError

    def _maximization_step(self, data, model, realizations):
        burn_in_phase = self._is_burn_in()  # The burn_in is true when the maximization step is memoryless
        if burn_in_phase:
            model.update_model_parameters(data, realizations, burn_in_phase)
        else:
            sufficient_statistics = model.compute_sufficient_statistics(data, realizations)
            burn_in_step = 1. / (self.current_iteration - self.algo_parameters['n_burn_in_iter']+1)
            self.sufficient_statistics = {k: v + burn_in_step * (sufficient_statistics[k] - v)
                                          for k, v in self.sufficient_statistics.items()}
            model.update_model_parameters(data, self.sufficient_statistics, burn_in_phase)

    def _is_burn_in(self):
        return self.current_iteration < self.algo_parameters['n_burn_in_iter']

    ###########################
    ## Output
    ###########################

    def __str__(self):
        out = ""
        out += "=== ALGO ===\n"
        out += "Iteration {0}".format(self.current_iteration)
        return out

