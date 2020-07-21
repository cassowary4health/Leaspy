import numpy as np
import torch
from leaspy.io.logs.fit_output_manager import FitOutputManager


class AbstractAlgo:
    """
    AbstractAlgo class object
    Abstract class containing common method for all algorithm classes. These classes are child classes of AbstractAlgo.

    Attributes
    ----------
    algo_parameters: dict
        Contains the algorithm's parameters. These ones are set by a
        leaspy.intputs.settings.algorithm_settings.AlgorithmSettings class object.
    name: str
        Name of the algorithm.
    seed: int
        Seed used by torch.random.

    Methods
    -------
    load_parameters(parameters)
        Update the algorithm's parameters by the ones in the given dictionary. The keys in the io which does not
        belong to the algorithm's parameters keys are ignored.
    set_output_manager(output_settings)
        Set a FitOutputManager class object for the run of the algorithm.
    """

    def __init__(self):
        """
        Process initializer function that is called by class FitOutputManager.
        """
        self.algo_parameters = None
        self.name = None
        self.output_manager = None
        self.seed = None

    ###########################
    ## Initialization
    ###########################
    @staticmethod
    def _initialize_seed(seed):
        """
        @staticmethod
        Set numpy and torch seeds and display it.

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
    ## Getters / Setters
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
        Set a FitOutputManager class object for the run of the algorithm

        Parameters
        ----------
        output_settings: a leaspy.io.settings.outputs_settings.OutputsSettings class object
            Contains the logs settings for the computation run (console print periodicity, plot periodicity ...)

        Examples
        --------
        >>> from leaspy import AlgorithmSettings
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


    def iteration(self, data, model, realizations):
        raise NotImplementedError
