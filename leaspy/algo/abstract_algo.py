from __future__ import annotations
from typing import TYPE_CHECKING
from copy import deepcopy

import sys
from abc import ABC, abstractmethod
import time
import random

import numpy as np
import torch

from leaspy.io.logs.fit_output_manager import FitOutputManager
from leaspy.utils.typing import Optional, Tuple, Any
from leaspy.exceptions import LeaspyModelInputError, LeaspyAlgoInputError

if TYPE_CHECKING:
    from leaspy.io.settings.algorithm_settings import AlgorithmSettings
    from leaspy.models.abstract_model import AbstractModel


class AbstractAlgo(ABC):
    """
    Abstract class containing common methods for all algorithm classes.
    These classes are child classes of `AbstractAlgo`.

    Parameters
    ----------
    settings : :class:`.AlgorithmSettings`
        The specifications of the algorithm as a :class:`.AlgorithmSettings` instance.

    Attributes
    ----------
    name : str
        Name of the algorithm.
    family : str
        Family of the algorithm. For now, valid families are:
            * ``'fit'```
            * ``'personalize'```
            * ``'simulate'``
    deterministic : bool
        True, if and only if algorithm does not involve in randomness.
        Setting a seed and such algorithms will be useless.
    algo_parameters : dict
        Contains the algorithm's parameters. Those are controlled by
        the :attr:`.AlgorithmSettings.parameters` class attribute.
    seed : int, optional
        Seed used by :mod:`numpy` and :mod:`torch`.
    output_manager : :class:`~.io.logs.fit_output_manager.FitOutputManager`
        Optional output manager of the algorithm
    """

    # Identifier of algorithm (classes variables)
    name: str = None
    family: str = None
    deterministic: bool = False

    # Format used to display noise std-dev values
    _log_noise_fmt = '.2%'

    def __init__(self, settings: AlgorithmSettings):

        if settings.name != self.name:
            raise LeaspyAlgoInputError(f'Inconsistent naming: {settings.name} != {self.name}')

        self.seed = settings.seed
        # we deepcopy the settings.parameters, because those algo_parameters may be
        # modified within algorithm (e.g. `n_burn_in_iter`) and we would not want the original
        # settings parameters to be also modified (e.g. to be able to re-use them without any trouble)
        self.algo_parameters = deepcopy(settings.parameters)

        self.output_manager: Optional[FitOutputManager] = None

    ###########################
    # Initialization
    ###########################
    @staticmethod
    def _initialize_seed(seed: Optional[int]):
        """
        Set :mod:`random`, :mod:`numpy` and :mod:`torch` seeds and display it (static method).

        Notes - numpy seed is needed for reproducibility for the simulation algorithm which use the scipy kernel
        density estimation function. Indeed, scipy use numpy random seed.

        Parameters
        ----------
        seed : int
            The wanted seed
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            # TODO: use logger instead (level=INFO)
            print(f" ==> Setting seed to {seed}")

    ###########################
    # Main method
    ###########################

    @abstractmethod
    def run_impl(self, model: AbstractModel, *args, **extra_kwargs) -> Tuple[Any, Optional[torch.FloatTensor]]:
        """
        Run the algorithm (actual implementation), to be implemented in children classes.

        TODO fix proper abstract class

        Parameters
        ----------
        model : :class:`~.models.abstract_model.AbstractModel`
            The used model.
        dataset : :class:`.Dataset`
            Contains all the subjects' observations with corresponding timepoints, in torch format to speed up computations.

        Returns
        -------
        A 2-tuple containing:
            * the result to send back to user
            * optional float tensor representing noise std-dev (to be printed)

        See Also
        --------
        :class:`.AbstractFitAlgo`
        :class:`.AbstractPersonalizeAlgo`
        :class:`.SimulationAlgorithm`
        """

    def run(self, model: AbstractModel, *args, return_noise: bool = False, **extra_kwargs) -> Any:
        """
        Main method, run the algorithm.

        TODO fix proper abstract class method: input depends on algorithm... (esp. simulate != from others...)

        Parameters
        ----------
        model : :class:`~.models.abstract_model.AbstractModel`
            The used model.
        dataset : :class:`.Dataset`
            Contains all the subjects' observations with corresponding timepoints, in torch format to speed up computations.
        return_noise : bool (default False), keyword only
            Should the algorithm return main output and optional noise output as a 2-tuple?

        Returns
        -------
        Depends on algorithm class: TODO change?

        See Also
        --------
        :class:`.AbstractFitAlgo`
        :class:`.AbstractPersonalizeAlgo`
        :class:`.SimulationAlgorithm`
        """

        # Check algo is well-defined
        if self.algo_parameters is None:
            raise LeaspyAlgoInputError(f'The `{self.name}` algorithm was not properly created.')

        # Set seed if needed
        self._initialize_seed(self.seed)

        # Init the run
        time_beginning = time.time()

        # Get the results (with noise)
        output, noise_std = self.run_impl(model, *args, **extra_kwargs)

        # Print run infos
        duration_in_seconds = time.time() - time_beginning
        if self.algo_parameters.get('progress_bar'):
            print()  # new line for clarity
        print(f"\n{self.family.title()} with `{self.name}` took: {self._duration_to_str(duration_in_seconds)}")

        noise_repr = self._noise_std_repr(noise_std, model=model)
        if noise_repr is not None:
            print(f"The standard deviation of the noise at the end of the {self.family} is: {noise_repr}")

        # Return only output part
        if return_noise:
            return output, noise_std
        else:
            return output

    ###########################
    # Getters / Setters
    ###########################

    def load_parameters(self, parameters: dict):
        """
        Update the algorithm's parameters by the ones in the given dictionary. The keys in the io which does not
        belong to the algorithm's parameters keys are ignored.

        Parameters
        ----------
        parameters : dict
            Contains the pairs (key, value) of the wanted parameters

        Examples
        --------
        >>> settings = leaspy.io.settings.algorithm_settings.AlgorithmSettings("mcmc_saem")
        >>> my_algo = leaspy.algo.fit.tensor_mcmcsaem.TensorMCMCSAEM(settings)
        >>> my_algo.algo_parameters
        {'n_iter': 10000,
         'n_burn_in_iter': 9000,
         'eps': 0.001,
         'L': 10,
         'sampler_ind': 'Gibbs',
         'sampler_pop': 'Gibbs',
         'annealing': {'do_annealing': False,
          'initial_temperature': 10,
          'n_plateau': 10,
          'n_iter': 200}}
        >>> parameters = {'n_iter': 5000, 'n_burn_in_iter': 4000}
        >>> my_algo.load_parameters(parameters)
        >>> my_algo.algo_parameters
        {'n_iter': 5000,
         'n_burn_in_iter': 4000,
         'eps': 0.001,
         'L': 10,
         'sampler_ind': 'Gibbs',
         'sampler_pop': 'Gibbs',
         'annealing': {'do_annealing': False,
          'initial_temperature': 10,
          'n_plateau': 10,
          'n_iter': 200}}
        """
        for k, v in parameters.items():
            if k in self.algo_parameters.keys():
                previous_v = self.algo_parameters[k]
                # TODO? log it instead (level=INFO or DEBUG)
                print(f"Replacing {k} parameter from value {previous_v} to value {v}")
            self.algo_parameters[k] = v

    def set_output_manager(self, output_settings):
        """
        Set a :class:`~.io.logs.fit_output_manager.FitOutputManager` object for the run of the algorithm

        Parameters
        ----------
        output_settings : :class:`~.io.settings.outputs_settings.OutputsSettings`
            Contains the logs settings for the computation run (console print periodicity, plot periodicity ...)

        Examples
        --------
        >>> from leaspy import AlgorithmSettings
        >>> from leaspy.io.settings.outputs_settings import OutputsSettings
        >>> from leaspy.algo.fit.tensor_mcmcsaem import TensorMCMCSAEM
        >>> algo_settings = AlgorithmSettings("mcmc_saem")
        >>> my_algo = TensorMCMCSAEM(algo_settings)
        >>> settings = {'path': 'brouillons',
                        'console_print_periodicity': 50,
                        'plot_periodicity': 100,
                        'save_periodicity': 50
                        }
        >>> my_algo.set_output_manager(OutputsSettings(settings))
        """
        if output_settings is not None:
            self.output_manager = FitOutputManager(output_settings)

    @staticmethod
    def _display_progress_bar(iteration, n_iter, suffix, n_step_default=50):
        """
        Display a progression bar while running algorithm, simply based on `sys.stdout`.

        TODO: use tqdm instead?

        Parameters
        ----------
        iteration : int >= 0 or -1
            Current iteration of the algorithm.
            The final iteration should be `n_iter - 1`
        n_iter : int
            Total iterations' number of the algorithm.
        suffix : str
            Used to differentiate types of algorithms:
                * for fit algorithms: ``suffix = 'iterations'``
                * for personalization algorithms: ``suffix = 'subjects'``.
        n_step_default : int, default 50
            The size of the progression bar.
        """
        n_step = min(n_step_default, n_iter)
        if iteration == -1:
            sys.stdout.write('\r')
            sys.stdout.write('|' + '-' * n_step + '|   0/%d ' % n_iter + suffix)
            sys.stdout.flush()
        else:
            print_every_iter = n_iter // n_step
            iteration_plus_1 = iteration + 1
            display = iteration_plus_1 % print_every_iter
            if display == 0:
                nbar = iteration_plus_1 // print_every_iter
                sys.stdout.write('\r')
                sys.stdout.write(f"|{'#'*nbar}{'-'*(n_step - nbar)}|   {iteration_plus_1}/{n_iter} {suffix}")
                sys.stdout.flush()

    def _noise_std_repr(self, noise_std: Optional[torch.FloatTensor], model: AbstractModel) -> Optional[str]:
        """
        Get a nice string representation of noise std-dev for a given model.

        TODO? move this code into a NoiseModel helper class

        Parameters
        ----------
        noise_std : :class:`torch.FloatTensor`
            Std-deviation of noise (tensor).
        model : :class:`.AbstractModel`
            Model that noise was estimated on.

        Returns
        -------
        str

        Raises
        ------
        :exc:`.LeaspyModelInputError`
            If model and noise are inconsistent.
        """
        if noise_std is None:
            return None

        noise_elts = np.array(noise_std).reshape(-1).tolist()  # can be torch tensor or numpy array (LME, constant model ...)
        noise_elts_nb = len(noise_elts)

        # in SimulationAlgorithm you may want to format a noise which is not compliant with the one from model so we don't rely on kwd
        #if getattr(model, 'noise_model', None) == 'gaussian_diagonal':
        if noise_elts_nb != 1:
            if noise_elts_nb != model.dimension:
                raise LeaspyModelInputError(f'Number of features ({model.dimension}) does not match with '
                                            f'number of diagonal terms of noise ({noise_elts_nb}).')

            noise_map = {ft_name: f'{ft_noise:{self._log_noise_fmt}}'
                         for ft_name, ft_noise in zip(model.features, noise_elts)}
            print_noise = repr(noise_map).replace("'", "").replace("{", "").replace("}", "")
            print_noise = '\n- ' + '\n- '.join(print_noise.split(', '))
        else:
            if hasattr(noise_std, 'item'):
                noise_std = noise_std.item()
            print_noise = f"{noise_std:{self._log_noise_fmt}}"

        return print_noise

    @staticmethod
    def _duration_to_str(seconds: float, *, seconds_fmt='.0f') -> str:
        """
        Convert a float representing computation time in seconds to a string giving time in hour, minutes and
        seconds ``%h %min %s``.

        If less than one hour, do not return hours. If less than a minute, do not return minutes.

        Parameters
        ----------
        seconds : float
            Computation time

        Returns
        -------
        str
            Time formatting in hour, minutes and seconds.
        """
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = seconds % 60  # float

        res = ''
        if m:
            if h:
                res += f'{h}h '
            res += f'{m}m '
        res += f'{s:{seconds_fmt}}s'

        return res

    def __str__(self):
        out = "=== ALGO ===\n"
        out += f"Instance of {self.name} algo"
        if hasattr(self, 'algorithm_device'):
            out += f" [{self.algorithm_device.upper()}]"
        return out
