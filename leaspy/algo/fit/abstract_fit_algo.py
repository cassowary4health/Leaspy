import time
from abc import abstractmethod

# from ...utils.logs.fit_output_manager import FitOutputManager
from ..abstract_algo import AbstractAlgo


class AbstractFitAlgo(AbstractAlgo):
    """
    Abstract class containing common method for all `fit` algorithm classes.

    Attributes
    ----------
    current_iteration: int, default 0
        The number of the current iteration
    Inherited attributes
        From :class:`.AbstractAlgo`

    See also
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
        # Initialize Model
        time_beginning = time.time()
        self._initialize_seed(self.seed)

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
            if self.output_manager is not None:  # TODO better this, should work with nones
                self.output_manager.iteration(self, dataset, model, realizations)
            self.current_iteration += 1
            if self.algo_parameters['progress_bar']:
                self.display_progress_bar(it, self.algo_parameters['n_iter'], suffix='iterations')

        if 'diag_noise' in model.loss:
            noise_map = {ft_name: '{:.4f}'.format(ft_noise) for ft_name, ft_noise in zip(model.features, model.parameters['noise_std'].view(-1).tolist())}
            print_noise = repr(noise_map).replace("'", "").replace("{", "").replace("}", "")
            print_noise = '\n'.join(print_noise.split(', '))
        else:
            print_noise = '{:.4f}'.format(model.parameters['noise_std'].item())

        time_end = time.time()
        diff_time = (time_end - time_beginning)

        print("\nThe standard deviation of the noise at the end of the calibration is:\n" + print_noise)
        print("\nCalibration took: " + self.convert_timer(diff_time))

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
        raise NotImplementedError

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
        raise NotImplementedError

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
            burn_in_step = 1. / (self.current_iteration - self.algo_parameters['n_burn_in_iter'] + 1)
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
        out += "Iteration {0}".format(self.current_iteration)
        return out
