from random import shuffle

import torch

from leaspy.algo.fit.abstract_fit_algo import AbstractFitAlgo
from leaspy.algo.utils.samplers import AlgoWithSamplersMixin


class AbstractFitMCMC(AlgoWithSamplersMixin, AbstractFitAlgo):
    """
    Abstract class containing common method for all `fit` algorithm classes based on `Monte-Carlo Markov Chains` (MCMC).

    Parameters
    ----------
    settings : :class:`.AlgorithmSettings`
        MCMC fit algorithm settings

    Attributes
    ----------
    samplers : dict[ str, :class:`~.algo.utils.samplers.abstract_sampler.AbstractSampler` ]
        Dictionary of samplers per each variable

    _random_sampling_order : bool (default True)
        This attribute controls whether we randomize the order of variables at each iteration.
        Article https://proceedings.neurips.cc/paper/2016/hash/e4da3b7fbbce2345d7772b0674a318d5-Abstract.html
        gives a rationale on why we should activate this flag.

    temperature : float
    temperature_inv : float
        Temperature and its inverse when using annealing

    See Also
    --------
    :mod:`leaspy.algo.utils.samplers`
    """

    def __init__(self, settings):

        super().__init__(settings)

        # Annealing
        # TODO? move all annealing related stuff in a dedicated mixin?
        self.temperature_inv = 1
        self.temperature = 1

        self._random_sampling_order = True

    ###########################
    ## Initialization
    ###########################

    @property
    def _do_annealing(self) -> bool:
        return self.algo_parameters.get('annealing', {}).get('do_annealing', False)

    def _initialize_algo(self, data, model, realizations):
        """
        Initialize the samplers, annealing, MCMC toolbox and sufficient statistics.

        Parameters
        ----------
        data : :class:`.Dataset`
        model : :class:`~.models.abstract_model.AbstractModel`
        realizations : :class:`~.io.realizations.collection_realization.CollectionRealization`
        """

        # MCMC toolbox (cache variables for speed-ups + tricks)
        # TODO? why not using just initialized `realizations` here in MCMC toolbox initialization?
        # TODO? we should NOT store the MCMC_toolbox in the model even if convenient, since actually it ONLY belongs to the algorithm!
        model.initialize_MCMC_toolbox()

        # Samplers
        self._initialize_samplers(model, data)
        self._initialize_sufficient_statistics(data, model, realizations)
        if self._do_annealing:
            self._initialize_annealing()

        return realizations

    def _initialize_annealing(self):
        """
        Initialize annealing, setting initial temperature and number of iterations.
        """
        if self._do_annealing:
            if self.algo_parameters['annealing']['n_iter'] is None:
                self.algo_parameters['annealing']['n_iter'] = int(self.algo_parameters['n_iter'] / 2)

        self.temperature = self.algo_parameters['annealing']['initial_temperature']
        self.temperature_inv = 1 / self.temperature

    def _initialize_sufficient_statistics(self, data, model, realizations):
        """
        Initialize the sufficient statistics.

        Parameters
        ----------
        data : :class:`.Dataset`
        model : :class:`~.models.abstract_model.AbstractModel`
        realizations : :class:`~.io.realizations.collection_realization.CollectionRealization`
        """
        # TODO: a great deal of computation for almost nothing (just to get name & shape of sufficient stats) -> refact?
        suff_stats = model.compute_sufficient_statistics(data, realizations)
        self.sufficient_statistics = {k: torch.zeros(v.shape, dtype=torch.float32) for k, v in suff_stats.items()}

    ###########################
    ## Getters / Setters
    ###########################

    ###########################
    ## Core
    ###########################

    def iteration(self, data, model, realizations):
        """
        MCMC-SAEM iteration.

        1. Sample : MC sample successively of the population and individual variables
        2. Maximization step : update model parameters from current population/individual variables values.

        Parameters
        ----------
        data : :class:`.Dataset`
        model : :class:`~.models.abstract_model.AbstractModel`
        realizations : :class:`~.io.realizations.collection_realization.CollectionRealization`
        """

        # Sample step (with order of population & individual variables shuffled)
        vars_order = realizations.reals_pop_variable_names + realizations.reals_ind_variable_names  # new list (no need to copy it)
        if self._random_sampling_order:
            # shuffle in-place!
            shuffle(vars_order)

        for key in vars_order:
            self.samplers[key].sample(data, model, realizations, self.temperature_inv)

        # Maximization step
        self._maximization_step(data, model, realizations)

        # We already updated MCMC toolbox for all population parameters during pop sampling.
        # So there is no need to update it as it once used to be.

        # Annealing
        if self._do_annealing:
            self._update_temperature()

    def _update_temperature(self):
        """
        Update the temperature according to a plateau annealing scheme.
        """
        if self.current_iteration <= self.algo_parameters['annealing']['n_iter']:
            # If we cross a plateau step
            if self.current_iteration % int(
                    self.algo_parameters['annealing']['n_iter'] / self.algo_parameters['annealing'][
                        'n_plateau']) == 0:
                # Decrease temperature linearly
                self.temperature -= self.algo_parameters['annealing']['initial_temperature'] / \
                                    self.algo_parameters['annealing']['n_plateau']
                self.temperature = max(self.temperature, 1)
                self.temperature_inv = 1 / self.temperature

    ###########################
    ## Output
    ###########################

    def __str__(self):
        out = "=== ALGO ===\n"
        out += f"Instance of {self.name} algo\n"
        out += f"Iteration {self.current_iteration}\n"

        out += "=Samplers\n"
        for sampler_name, sampler in self.samplers.items():
            acceptation_rate = torch.mean(sampler.acceptation_temp.detach()).item()
            out += f"    {sampler_name} rate : {acceptation_rate:.2%}, std: {sampler.std.mean():.5f}\n"

        if self._do_annealing:
            out += "Annealing\n"
            out += f"Temperature : {self.temperature}"

        return out
