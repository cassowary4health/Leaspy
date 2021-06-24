import sys
from abc import ABC, abstractmethod

import numpy as np
import torch

from leaspy.io.logs.fit_output_manager import FitOutputManager


class AbstractAlgo(ABC):
    """
    Abstract class containing common methods for all algorithm classes.
    These classes are child classes of `AbstractAlgo`.

    Attributes
    ----------
    algo_parameters: dict
        Contains the algorithm's parameters. These ones are set by a
        :class:`.AlgorithmSettings` class object.
    name: str
        Name of the algorithm.
    seed: int, optional
        Seed used by :mod:`numpy` and :mod:`torch`.
    output_manager : :class:`~.io.logs.fit_output_manager.FitOutputManager`
        Optional output manager of the algorithm
    """

    def __init__(self):
        self.algo_parameters = None
        self.name = None
        self.output_manager = None
        self.seed = None

    ###########################
    # Initialization
    ###########################
    @staticmethod
    def _initialize_seed(seed):
        """
        Set :mod:`numpy` and :mod:`torch` seeds and display it (static method).

        Notes - numpy seed is needed for reproducibility for the simulation algorithm which use the scipy kernel
        density estimation function. Indeed, scipy use numpy random seed.

        Parameters
        ----------
        seed: int
            The wanted seed
        """
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            print(" ==> Setting seed to {0}".format(seed))

    ###########################
    # Main method
    ###########################

    @abstractmethod
    def run(self, model, dataset):
        """
        Main method, run the algorithm.

        TODO fix proper abstract class

        Parameters
        ----------
        model : :class:`~.models.abstract_model.AbstractModel`
            The used model.
        dataset : :class:`.Dataset`
            Contains all the subjects' observations with corresponding timepoints, in torch format to speed up computations.

        Returns
        -------
        Depends on algorithm class: TODO change?

        See also
        --------
        :class:`.AbstractFitAlgo`
        :class:`.AbstractPersonalizeAlgo`
        :class:`.SimulationAlgorithm`
        """
        pass

    ###########################
    # Getters / Setters
    ###########################

    def load_parameters(self, parameters):
        """
        Update the algorithm's parameters by the ones in the given dictionary. The keys in the io which does not
        belong to the algorithm's parameters keys are ignored.

        Parameters
        ----------
        parameters: dict
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
                print("Replacing {} parameter from value {} to value {}".format(k, previous_v, v))
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
    def display_progress_bar(iteration, n_iter, suffix, n_step_default=50):
        """
        Display a progression bar while running algorithm, simply based on `sys.stdout`.

        Parameters
        ----------
        iteration: int
            Current iteration of the algorithm.
        n_iter: int
            Total iterations' number of the algorithm.
        suffix: str
            Used to differentiate types of algorithms:
                * for fit algorithms: ``suffix = 'iterations'``
                * for personalization algorithms: ``suffix = 'subjects'``.
        n_step_default: int, default 50
            The size of the progression bar.
        """
        n_step = min(n_step_default, n_iter)
        if iteration == -1:
            sys.stdout.write('\r')
            sys.stdout.write('|' + '-' * n_step + '|   0/%d ' % n_iter + suffix)
            sys.stdout.flush()
        else:
            print_every_iter = n_iter // n_step
            display = (iteration + 1) % print_every_iter
            if display == 0:
                nbar = (iteration + 1) // print_every_iter
                sys.stdout.write('\r')
                sys.stdout.write(
                    '|' + '#' * nbar + '-' * (n_step - nbar) + '|   %d/%d ' % (iteration + 1, n_iter) + suffix)
                sys.stdout.flush()

    @staticmethod
    def convert_timer(d):
        """
        Convert a float representing computation time in seconds to a string giving time in hour, minutes and
        seconds ``%h %min %s``.

        If less than one hour, do not return hours. If less than a minute, do not return minuts.

        Parameters
        ----------
        d: float
            Computation time

        Returns
        -------
        res: str
            Time formating in hour, minutes and seconds.
        """
        s = d % 60
        m = (d % 3600) // 60
        h = d // 3600

        res = '%ds' % s
        if m:
            res = '%dmin ' % m + res
        if h:
            res = '%dh ' % h + res
        return res
