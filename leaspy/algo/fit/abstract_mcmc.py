from __future__ import annotations
from typing import TYPE_CHECKING
from random import shuffle

from leaspy.algo.fit.abstract_fit_algo import AbstractFitAlgo
from leaspy.algo.utils.algo_with_samplers import AlgoWithSamplersMixin
from leaspy.algo.utils.algo_with_annealing import AlgoWithAnnealingMixin

from leaspy.variables.state import State
from leaspy.variables.specs import PopulationLatentVariable, IndividualLatentVariable

if TYPE_CHECKING:
    from leaspy.io.data.dataset import Dataset
    from leaspy.models.abstract_model import AbstractModel


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

    def _initialize_algo(
        self,
        state: State,
        dataset: Dataset,
    ) -> None:
        """
        Initialize the samplers & annealing.

        Parameters
        ----------
        state : :class:`.State`
        dataset : :class:`.Dataset`
        """

        # Samplers mixin
        self._initialize_samplers(state, dataset)

        # Annealing mixin
        self._initialize_annealing()

    ###########################
    ## Core
    ###########################

    def iteration(
        self,
        model: AbstractModel,
        state: State,
    ) -> None:
        """
        MCMC-SAEM iteration.

        1. Sample : MC sample successively of the population and individual variables
        2. Maximization step : update model parameters from current population/individual variables values.

        Parameters
        ----------
        model : :class:`~.models.abstract_model.AbstractModel`
        state : :class:`.State`
        """
        vars_order = (
            list(state.dag.sorted_variables_by_type[PopulationLatentVariable])
            + list(state.dag.sorted_variables_by_type[IndividualLatentVariable])
        )
        if self.random_order_variables:
            shuffle(vars_order)  # shuffle order in-place!

        for key in vars_order:
            self.samplers[key].sample(state, temperature_inv=self.temperature_inv)

        # Maximization step
        self._maximization_step(model, state)

        # Annealing mixin
        self._update_temperature()
