import contextlib
import torch

from abc import abstractmethod

from leaspy.algo.abstract_algo import AbstractAlgo


class AbstractFitAlgo(AbstractAlgo):
    """
    Abstract class containing common method for all `fit` algorithm classes.

    Parameters
    ----------
    settings : :class:`.AlgorithmSettings`
        The specifications of the algorithm as a :class:`.AlgorithmSettings` instance.

    Attributes
    ----------
    current_iteration : int, default 0
        The number of the current iteration
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
        self.current_iteration = 0

    ###########################
    # Core
    ###########################

    def run_impl(self, model, dataset):
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

        with self._device_manager(dataset, model):
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

        return realizations, model.parameters['noise_std']

    @abstractmethod
    def iteration(self, dataset, model, realizations):
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
    def _initialize_algo(self, dataset, model, realizations):
        """
        Initialize the fit algorithm (abstract method).

        Parameters
        ----------
        dataset : :class:`.Dataset`
        model : :class:`~.models.abstract_model.AbstractModel`
        realizations : :class:`~.io.realizations.collection_realization.CollectionRealization`
        """
        pass

    def _maximization_step(self, dataset, model, realizations):
        """
        Maximization step as in the EM algorithm. In practice parameters are set to current realizations (burn-in phase),
        or as a barycenter with previous realizations.

        Parameters
        ----------
        dataset : :class:`.Dataset`
        model : :class:`.AbstractModel`
        realizations : :class:`.CollectionRealization`
        """
        burn_in_phase = self._is_burn_in()  # The burn_in is true when the maximization step is memoryless
        if burn_in_phase:
            model.update_model_parameters(dataset, realizations, burn_in_phase=burn_in_phase)
        else:
            sufficient_statistics = model.compute_sufficient_statistics(dataset, realizations)
            # The algorithm is proven to converge if the sequence `burn_in_step` is positive, with an infinite sum \sum
            # (\sum_k \epsilon_k = + \infty) but a finite sum of the squares (\sum_k \epsilon_k^2 < \infty )
            # cf page 657 of the book that contains the paper
            # "Construction of Bayesian deformable models via a stochastic approximation algorithm: a convergence study"
            burn_in_step = 1. / (self.current_iteration - self.algo_parameters['n_burn_in_iter'] + 1)**0.8  # TODO: hyperparameter here
            self.sufficient_statistics = {k: v + burn_in_step * (sufficient_statistics[k] - v)
                                          for k, v in self.sufficient_statistics.items()}
            model.update_model_parameters(dataset, self.sufficient_statistics, burn_in_phase=burn_in_phase)

    def _is_burn_in(self):
        """
        Check if current iteration is in burn-in phase.

        Returns
        -------
        bool
        """
        return self.current_iteration < self.algo_parameters['n_burn_in_iter']

    @contextlib.contextmanager
    def _device_manager(self, model, dataset):
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
