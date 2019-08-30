
from leaspy.algo.abstract_mcmc import AbstractMCMC


class MCMCPredict(AbstractMCMC):

    def __init__(self):
        super().__init__()


    def iteration(self, data, model, realizations):
        reals_pop, reals_ind = realizations

        # Sample the individual realizations
        self._sample_individual_realizations(data, model, reals_pop, reals_ind)


    def run(self, data, model):

        # Initialize Model
        self._initialize_seed(model.parameters["seed"])
        realizations = model.get_realization_object(data)
        model.initialize_random_variables(data)

        # Initialize Algo
        self._initialize_algo(data, model, realizations)

        # Iterate
        for it in range(self.algo_parameters['n_iter']):
            self.iteration(data, model, realizations)
            self.current_iteration += 1

            reals_pop, real_ind = realizations
        return real_ind