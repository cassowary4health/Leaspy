from random import shuffle

from leaspy.algo.fit.abstract_fit_algo import AbstractFitAlgo
from leaspy.algo.utils.samplers import AlgoWithSamplersMixin
from leaspy.algo.utils.algo_with_annealing import AlgoWithAnnealingMixin
from leaspy.io.data.dataset import Dataset
from leaspy.models.abstract_model import AbstractModel
from leaspy.io.realizations.collection_realization import CollectionRealization


class AbstractFitMCMC(AlgoWithAnnealingMixin, AlgoWithSamplersMixin, AbstractFitAlgo):
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

    random_order_variables : bool (default True)
        This attribute controls whether we randomize the order of variables at each iteration.
        Article https://proceedings.neurips.cc/paper/2016/hash/e4da3b7fbbce2345d7772b0674a318d5-Abstract.html
        gives a rationale on why we should activate this flag.

    temperature : float
    temperature_inv : float
        Temperature and its inverse (modified during algorithm when using annealing)

    See Also
    --------
    :mod:`leaspy.algo.utils.samplers`
    """

    ###########################
    ## Initialization
    ###########################

    def _initialize_algo(self, dataset: Dataset, model: AbstractModel, realizations: CollectionRealization) -> None:
        """
        Initialize the samplers, annealing, MCMC toolbox and sufficient statistics.

        Parameters
        ----------
        dataset : :class:`.Dataset`
        model : :class:`~.models.abstract_model.AbstractModel`
        realizations : :class:`~.io.realizations.collection_realization.CollectionRealization`
        """

        # MCMC toolbox (cache variables for speed-ups + tricks)
        # TODO? why not using just initialized `realizations` here in MCMC toolbox initialization?
        # TODO? we should NOT store the MCMC_toolbox in the model even if convenient, since actually it ONLY belongs to the algorithm!
        model.initialize_MCMC_toolbox()

        # Samplers mixin
        self._initialize_samplers(model, dataset)

        # Annealing mixin
        self._initialize_annealing()


    ###########################
    ## Core
    ###########################

    def iteration(self, dataset: Dataset, model: AbstractModel, realizations: CollectionRealization):
        """
        MCMC-SAEM iteration.

        1. Sample : MC sample successively of the population and individual variables
        2. Maximization step : update model parameters from current population/individual variables values.

        Parameters
        ----------
        dataset : :class:`.Dataset`
        model : :class:`~.models.abstract_model.AbstractModel`
        realizations : :class:`~.io.realizations.collection_realization.CollectionRealization`
        """

        # Sample step (with order of population & individual variables shuffled)
        vars_order = realizations.reals_pop_variable_names + realizations.reals_ind_variable_names  # new list (no need to copy it)
        if self.random_order_variables:
            shuffle(vars_order)  # shuffle order in-place!

        for key in vars_order:
            self.samplers[key].sample(dataset, model, realizations, self.temperature_inv)

        # Maximization step
        self._maximization_step(dataset, model, realizations)

        # We already updated MCMC toolbox for all population parameters during pop sampling.
        # So there is no need to update it as it once used to be.

        # Annealing mixin
        self._update_temperature()
