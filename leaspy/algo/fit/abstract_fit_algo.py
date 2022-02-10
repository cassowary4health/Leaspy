import contextlib
import torch

from abc import abstractmethod

from leaspy.algo.abstract_algo import AbstractAlgo
from leaspy.io.data.dataset import Dataset
from leaspy.models.abstract_model import AbstractModel
from leaspy.io.realizations.collection_realization import CollectionRealization

from leaspy.utils.typing import DictParamsTorch


class AbstractFitAlgo(AbstractAlgo):
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
        The number of the current iteration
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

        self.algorithm_device = settings.device
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
            # Initialize first the random variables
            # TODO : Check if needed - model.initialize_random_variables(dataset)

            # Then initialize the Realizations (from the random variables)
            realizations = model.initialize_realizations_for_model(dataset.n_individuals)

            # Smart init the realizations
            realizations = model.smart_initialization_realizations(dataset, realizations)

            # Initialize Algo
            self._initialize_algo(dataset, model, realizations)

            if self.algo_parameters['progress_bar']:
                self._display_progress_bar(-1, self.algo_parameters['n_iter'], suffix='iterations')

            # Iterate
            for it in range(self.algo_parameters['n_iter']):

                self.iteration(dataset, model, realizations)
                self.current_iteration += 1

                if self.output_manager is not None:  # TODO better this, should work with nones
                    # do not print iteration 0 because of noise_std init pb
                    # but print first & last iteration!
                    self.output_manager.iteration(self, dataset, model, realizations)

                if self.algo_parameters['progress_bar']:
                    self._display_progress_bar(it, self.algo_parameters['n_iter'], suffix='iterations')

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
        pass

    @abstractmethod
    def _initialize_algo(self, dataset: Dataset, model: AbstractModel, realizations: CollectionRealization):
        """
        Initialize the fit algorithm (abstract method).

        Parameters
        ----------
        dataset : :class:`.Dataset`
        model : :class:`~.models.abstract_model.AbstractModel`
        realizations : :class:`~.io.realizations.collection_realization.CollectionRealization`
        """
        pass

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
            # The algorithm is proven to converge if the sequence `burn_in_step` is positive, with an infinite sum \sum
            # (\sum_k \epsilon_k = + \infty) but a finite sum of the squares (\sum_k \epsilon_k^2 < \infty )
            # cf page 657 of the book that contains the paper
            # "Construction of Bayesian deformable models via a stochastic approximation algorithm: a convergence study"
            burn_in_step = 1. / (self.current_iteration - self.algo_parameters['n_burn_in_iter'] + 1)**0.8  # TODO: hyperparameter here

            if self.sufficient_statistics is None:
                # 1st iteration post burn-in
                self.sufficient_statistics = sufficient_statistics
            else:
                self.sufficient_statistics = {k: v + burn_in_step * (sufficient_statistics[k] - v)
                                              for k, v in self.sufficient_statistics.items()}

            model.update_model_parameters_normal(dataset, self.sufficient_statistics)

        # No need to update model attributes (derived from model parameters)
        # since all model computations are done with the MCMC toolbox during calibration

    def _is_burn_in(self) -> bool:
        """
        Check if current iteration is in burn-in phase.

        Returns
        -------
        bool
        """
        return self.current_iteration < self.algo_parameters['n_burn_in_iter']

    @contextlib.contextmanager
    def _device_manager(self, model: AbstractModel, dataset: Dataset):
        """
        Context-manager to handle the "ambient device" (i.e. the device used
        to instantiate tensors and perform computations). The provided model
        and dataset will be moved to the device specified for the execution
        at the beginning of the algorithm and moved back to the original
        ('cpu') device at the end of the algorithm. The default tensor type
        will also be set accordingly.

        Parameters
        ----------
        model : :class:`~.models.abstract_model.AbstractModel`
            The used model.
        dataset : :class:`.Dataset`
            Contains the subjects' observations in torch format to speed up computation.
        """

        default_algorithm_tensor_type = 'torch.FloatTensor'
        default_algorithm_device = torch.device('cpu')

        algorithm_tensor_type = default_algorithm_tensor_type
        if self.algorithm_device != 'cpu':
            algorithm_device = torch.device(self.algorithm_device)

            dataset.move_to_device(algorithm_device)
            model.move_to_device(algorithm_device)

            algorithm_tensor_type = 'torch.cuda.FloatTensor'

        try:
            yield torch.set_default_tensor_type(algorithm_tensor_type)
        finally:
            if self.algorithm_device != 'cpu':
                dataset.move_to_device(default_algorithm_device)
                model.move_to_device(default_algorithm_device)

            torch.set_default_tensor_type(default_algorithm_tensor_type)

    ###########################
    # Output
    ###########################

    def __str__(self):
        out = "=== ALGO ===\n"
        out += f"Iteration {self.current_iteration}"
        return out
