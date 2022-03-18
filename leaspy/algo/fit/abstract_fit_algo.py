from abc import abstractmethod

from leaspy.algo.abstract_algo import AbstractAlgo
from leaspy.io.data.dataset import Dataset
from leaspy.models.abstract_model import AbstractModel
from leaspy.io.realizations.collection_realization import CollectionRealization
from leaspy.algo.utils.algo_with_device import AlgoWithDeviceMixin

from leaspy.utils.typing import DictParamsTorch
from leaspy.exceptions import LeaspyAlgoInputError


class AbstractFitAlgo(AlgoWithDeviceMixin, AbstractAlgo):
    """
    Abstract class containing common method for all `fit` algorithm classes.

    Parameters
    ----------
    settings : :class:`.AlgorithmSettings`
        The specifications of the algorithm as a :class:`.AlgorithmSettings` instance.

    Attributes
    ----------
    algorithm_device : str
        Valid torch device
    current_iteration : int, default 0
        The number of the current iteration.
        The first iteration will be 1 and the last one `n_iter`.
    sufficient_statistics : dict[str, `torch.FloatTensor`] or None
        The previous step sufficient statistics.
        It is None during all the burn-in phase.
    Inherited attributes
        From :class:`.AbstractAlgo`

    See Also
    --------
    :meth:`.Leaspy.fit`
    """

    family = "fit"

    def __init__(self, settings):

        super().__init__(settings)

        # The algorithm is proven to converge if the sequence `burn_in_step` is positive, with an infinite sum \sum
        # (\sum_k \epsilon_k = + \infty) but a finite sum of the squares (\sum_k \epsilon_k^2 < \infty )
        # cf page 657 of the book that contains the paper
        # "Construction of Bayesian deformable models via a stochastic approximation algorithm: a convergence study"
        if not (0.5 < self.algo_parameters['burn_in_step_power'] <= 1):
            raise LeaspyAlgoInputError("The parameter `burn_in_step_power` should be in ]0.5, 1] in order to "
                                       "have theoretical guarantees on convergence of MCMC-SAEM algorithm.")

        self.current_iteration: int = 0

        self.sufficient_statistics: DictParamsTorch = None

    ###########################
    # Core
    ###########################

    def run_impl(self, model: AbstractModel, dataset: Dataset):
        """
        Main method, run the algorithm.

        Basically, it initializes the :class:`~.io.realizations.collection_realization.CollectionRealization` object,
        updates it using the `iteration` method then returns it.

        TODO fix proper abstract class

        Parameters
        ----------
        model : :class:`~.models.abstract_model.AbstractModel`
            The used model.
        dataset : :class:`.Dataset`
            Contains the subjects' observations in torch format to speed up computation.

        Returns
        -------
        2-tuple:
            * realizations : :class:`~.io.realizations.collection_realization.CollectionRealization`
                The optimized parameters.
            * None : placeholder for noise-std
        """

        with self._device_manager(model, dataset):
            # Initialize the `CollectionRealization` (from the random variables of the model)
            realizations = model.initialize_realizations_for_model(dataset.n_individuals)

            # Smart init the realizations
            realizations = model.smart_initialization_realizations(dataset, realizations)

            # Initialize Algo
            self._initialize_algo(dataset, model, realizations)

            if self.algo_parameters['progress_bar']:
                self._display_progress_bar(-1, self.algo_parameters['n_iter'], suffix='iterations')

            # Iterate
            for self.current_iteration in range(1, self.algo_parameters['n_iter']+1):

                self.iteration(dataset, model, realizations)

                if self.output_manager is not None:
                    # print/plot first & last iteration!
                    # <!> everything that will be printed/saved is AFTER iteration N (including temperature when annealing...)
                    self.output_manager.iteration(self, dataset, model, realizations)

                if self.algo_parameters['progress_bar']:
                    self._display_progress_bar(self.current_iteration - 1, self.algo_parameters['n_iter'], suffix='iterations')

            # Finally we compute model attributes once converged
            model.attributes.update(['all'], model.parameters)

        return realizations, model.parameters['noise_std']

    @abstractmethod
    def iteration(self, dataset: Dataset, model: AbstractModel, realizations: CollectionRealization):
        """
        Update the parameters (abstract method).

        Parameters
        ----------
        dataset : :class:`.Dataset`
            Contains the subjects' observations in torch format to speed-up computation.
        model : :class:`~.models.abstract_model.AbstractModel`
            The used model.
        realizations : :class:`~.io.realizations.collection_realization.CollectionRealization`
            The parameters.
        """

    @abstractmethod
    def _initialize_algo(self, dataset: Dataset, model: AbstractModel, realizations: CollectionRealization) -> None:
        """
        Initialize the fit algorithm (abstract method).

        Parameters
        ----------
        dataset : :class:`.Dataset`
        model : :class:`~.models.abstract_model.AbstractModel`
        realizations : :class:`~.io.realizations.collection_realization.CollectionRealization`
        """

    def _maximization_step(self, dataset: Dataset, model: AbstractModel, realizations: CollectionRealization):
        """
        Maximization step as in the EM algorithm. In practice parameters are set to current realizations (burn-in phase),
        or as a barycenter with previous realizations.

        Parameters
        ----------
        dataset : :class:`.Dataset`
        model : :class:`.AbstractModel`
        realizations : :class:`.CollectionRealization`
        """
        if self._is_burn_in():
            # the maximization step is memoryless
            model.update_model_parameters_burn_in(dataset, realizations)
        else:
            sufficient_statistics = model.compute_sufficient_statistics(dataset, realizations)

            burn_in_step = self.current_iteration - self.algo_parameters['n_burn_in_iter'] # min = 1, max = n_iter - n_burn_in_iter
            burn_in_step **= -self.algo_parameters['burn_in_step_power']

            if self.sufficient_statistics is None:
                # 1st iteration post burn-in
                self.sufficient_statistics = sufficient_statistics
            else:
                self.sufficient_statistics = {k: v + burn_in_step * (sufficient_statistics[k] - v)
                                              for k, v in self.sufficient_statistics.items()}

            model.update_model_parameters_normal(dataset, self.sufficient_statistics)

        # No need to update model attributes (derived from model parameters)
        # since all model computations are done with the MCMC toolbox during calibration
