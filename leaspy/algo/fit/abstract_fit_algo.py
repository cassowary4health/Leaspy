import time
from abc import abstractmethod

from leaspy.algo.abstract_algo import AbstractAlgo
from leaspy.exceptions import LeaspyAlgoInputError


class AbstractFitAlgo(AbstractAlgo):
    """
    Abstract class containing common method for all `fit` algorithm classes.

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

    def __init__(self):

        super().__init__()
        self.current_iteration = 0  # TODO change to None ?

        # TODO? init from settings instead of doing it in subclasses like `AbstractFitMCMC`

    ###########################
    # Core
    ###########################

    def run(self, model, dataset):
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
        realizations : :class:`~.io.realizations.collection_realization.CollectionRealization`
            The optimized parameters.

        """

        # Check algo is well-defined
        if self.algo_parameters is None:
            raise LeaspyAlgoInputError('The fit algorithm was not properly created.')

        # Initialize Model
        time_beginning = time.time()
        self._initialize_seed(self.seed)

        # Move both the dataset and the model on the correct device, and keep the original one
        original_model_device = model.device
        dataset.move_to_device(self.device)
        model.move_to_device(self.device)

        # Initialize first the random variables
        # TODO : Check if needed - model.initialize_random_variables(dataset)

        # Then initialize the Realizations (from the random variables)
        realizations = model.get_realization_object(dataset.n_individuals)

        # Smart init the realizations
        realizations = model.smart_initialization_realizations(dataset, realizations)

        # Initialize Algo
        self._initialize_algo(dataset, model, realizations)

        if self.algo_parameters['progress_bar']:
            self.display_progress_bar(-1, self.algo_parameters['n_iter'], suffix='iterations')

        # Iterate
        for it in range(self.algo_parameters['n_iter']):

            self.iteration(dataset, model, realizations)
            self.current_iteration += 1

            if self.output_manager is not None:  # TODO better this, should work with nones
                # do not print iteration 0 because of noise_std init pb
                # but print first & last iteration!
                self.output_manager.iteration(self, dataset, model, realizations)

            if self.algo_parameters['progress_bar']:
                self.display_progress_bar(it, self.algo_parameters['n_iter'], suffix='iterations')

        if 'diag_noise' in model.loss:
            noise_map = {ft_name: f'{ft_noise:.4f}' for ft_name, ft_noise in zip(model.features, model.parameters['noise_std'].view(-1).tolist())}
            print_noise = repr(noise_map).replace("'", "").replace("{", "").replace("}", "")
            print_noise = '\n'.join(print_noise.split(', '))
        else:
            print_noise = f"{model.parameters['noise_std'].item():.4f}"

        time_end = time.time()
        diff_time = (time_end - time_beginning)

        print("\nThe standard deviation of the noise at the end of the calibration is:\n" + print_noise)
        print("\nCalibration took: " + self.convert_timer(diff_time))

        # move back both the dataset and the model to their original device
        dataset.move_to_device(original_model_device)
        model.move_to_device(original_model_device)

        return realizations

    @abstractmethod
    def iteration(self, dataset, model, realizations):
        """
        Update the parameters (abstract method).

        Parameters
        ----------
        dataset : :class:`.Dataset`
            Contains the subjects' obersvations in torch format to speed up computation.
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
        Maximization step as in the EM algorith. In practice parameters are set to current realizations (burn-in phase),
        or as a barycenter with previous realizations.

        Parameters
        ----------
        dataset : :class:`.Dataset`
        model : :class:`.AbstractModel`
        realizations : :class:`.CollectionRealization`
        """
        burn_in_phase = self._is_burn_in()  # The burn_in is true when the maximization step is memoryless
        if burn_in_phase:
            model.update_model_parameters(dataset, realizations, burn_in_phase)
        else:
            sufficient_statistics = model.compute_sufficient_statistics(dataset, realizations)
            # The algorithm is proven to converge if the sequence `burn_in_step` is positive, with an infinite sum \sum
            # (\sum_k \epsilon_k = + \infty) but a finite sum of the squares (\sum_k \epsilon_k^2 < \infty )
            # cf page 657 of the book that contains the paper
            # "Construction of Bayesian deformable models via a stochastic approximation algorithm: a convergence study"
            burn_in_step = 1. / (self.current_iteration - self.algo_parameters['n_burn_in_iter'] + 1)**0.8
            self.sufficient_statistics = {k: v + burn_in_step * (sufficient_statistics[k] - v)
                                          for k, v in self.sufficient_statistics.items()}
            model.update_model_parameters(dataset, self.sufficient_statistics, burn_in_phase)

    def _is_burn_in(self):
        """
        Check if current iteration is in burn-in phase.

        Returns
        -------
        bool
        """
        return self.current_iteration < self.algo_parameters['n_burn_in_iter']

    ###########################
    # Output
    ###########################

    def __str__(self):
        out = ""
        out += "=== ALGO ===\n"
        out += f"Iteration {self.current_iteration}"
        return out
