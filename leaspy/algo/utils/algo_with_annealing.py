import warnings

from leaspy.exceptions import LeaspyAlgoInputError


class AlgoWithAnnealingMixin:
    """
    Mixin to use in algorithms needing `temperature_inv`; inherit from this class first.

    Note that this mixin is to be used with a class inheriting from `AbstractAlgo`
    (and in particular that have a `algo_parameters` attribute)

    Parameters
    ----------
    settings : :class:`.AlgorithmSettings`
        The specifications of the algorithm as a :class:`.AlgorithmSettings` instance.

    Attributes
    ----------
    annealing_on : bool
        Is annealing activated or not?

    temperature : float >= 1
    temperature_inv : float in ]0, 1]
        Temperature and its inverse when using annealing
    """

    def __init__(self, settings):
        super().__init__(settings)

        self.temperature: float = 1.
        self.temperature_inv: float = 1.

        # useful property derived from algo parameters
        self.annealing_on: bool = self.algo_parameters.get('annealing', {}).get('do_annealing', False)
        self._annealing_period: int = None
        self._annealing_temperature_decrement: float = None

        # Dynamic number of iterations for annealing
        if self.annealing_on and self.algo_parameters['annealing'].get('n_iter', None) is None:
            self.algo_parameters['annealing']['n_iter'] = int(self.algo_parameters['annealing']['n_iter_frac'] * self.algo_parameters['n_iter'])

    def __str__(self):
        out = super().__str__()
        if self.annealing_on:
            out += "\n= Annealing =\n"
            out += f"    temperature : {self.temperature:.1f}"
        return out

    def _initialize_annealing(self):
        """
        Initialize annealing, setting initial temperature and number of iterations.
        """
        if not self.annealing_on:
            return

        self.temperature = self.algo_parameters['annealing']['initial_temperature']
        self.temperature_inv = 1 / self.temperature

        if not (isinstance(self.algo_parameters['annealing']['n_plateau'], int)
                and self.algo_parameters['annealing']['n_plateau'] > 0):
            raise LeaspyAlgoInputError('Your `annealing.n_plateau` should be a positive integer')

        if self.algo_parameters['annealing']['n_plateau'] == 1:
            warnings.warn('You defined `annealing.n_plateau` = 1, so you will stay at initial temperature. '
                          'Consider setting `annealing.n_plateau` >= 2 for a true annealing scheme.')
            return

        self._annealing_period = (
            self.algo_parameters['annealing']['n_iter']
            // (self.algo_parameters['annealing']['n_plateau'] - 1)
        )

        self._annealing_temperature_decrement = (
            (self.algo_parameters['annealing']['initial_temperature'] - 1.)
            / (self.algo_parameters['annealing']['n_plateau'] - 1)
        )
        if self._annealing_temperature_decrement <= 0:
            raise LeaspyAlgoInputError('Your `initial_temperature` should be > 1')

    def _update_temperature(self):
        """
        Update the temperature according to a plateau annealing scheme.
        """
        if not self.annealing_on or self._annealing_period is None:
            return

        if self.current_iteration <= self.algo_parameters['annealing']['n_iter']:
            # If we cross a plateau step
            if self.current_iteration % self._annealing_period == 0:
                # Decrease temperature linearly
                self.temperature -= self._annealing_temperature_decrement
                self.temperature = max(self.temperature, 1)
                self.temperature_inv = 1 / self.temperature
