from leaspy.algo.fit.tensor_mcmcsaem import TensorMCMCSAEM
# from leaspy.algo.fit.gradient_descent import GradientDescent
# from leaspy.algo.fit.gradient_mcmcsaem import GradientMCMCSAEM
# from leaspy.algo.fit.hmc_saem import HMC_SAEM
from leaspy.algo.others.lme_fit import LMEFitAlgorithm

from leaspy.algo.personalize.scipy_minimize import ScipyMinimize
from leaspy.algo.personalize.mean_realisations import MeanReal
from leaspy.algo.personalize.mode_realisations import ModeReal
from leaspy.algo.personalize.gradient_descent_personalize import GradientDescentPersonalize

from leaspy.algo.others.constant_prediction_algo import ConstantPredictionAlgorithm
from leaspy.algo.others.lme_personalize import LMEPersonalizeAlgorithm

from leaspy.algo.simulate.simulate import SimulationAlgorithm


class AlgoFactory:
    """
    Return the wanted algorithm given its name.

    For developpers
    ---------------
    Add your new algorithm in corresponding category of `_algos` dictionary.
    """

    _algos = {
        'fit': {
            'mcmc_saem': TensorMCMCSAEM,
            #'mcmc_gradient_descent': GradientMCMCSAEM,
            #'gradient_descent': GradientDescent,
            # 'hmc_saem': HMC_SAEM,
            'lme_fit': LMEFitAlgorithm,
        },

        'personalize': {
            'scipy_minimize': ScipyMinimize,
            'mean_real': MeanReal,
            'mode_real': ModeReal,
            'gradient_descent_personalize': GradientDescentPersonalize, # deprecated?

            'constant_prediction': ConstantPredictionAlgorithm,
            'lme_personalize': LMEPersonalizeAlgorithm,
        },

        'simulate': {
            'simulation': SimulationAlgorithm
        }
    }

    @classmethod
    def algo(cls, algorithm_class, settings):
        """
        Return the wanted algorithm given its name.

        Parameters
        ----------
        algorithm_class: str
            Task name, used to check if the algorithm within the input `settings` is compatible with this task.
            Must be one of the following api's name:
                * `fit`
                * `personalize`
                * `simulate`

        settings : :class:`.AlgorithmSettings`
            The algorithm settings.

        Returns
        -------
        algorithm : child class of :class:`.AbstractAlgo`
            The wanted algorithm if it exists and is compatible with algorithm class.

        Raises
        ------
        ValueError
            * if the algorithm class is unknown
            * if the algorithm name is unknown / does not belong to the wanted algorithm class
        """
        name = settings.name

        if algorithm_class not in cls._algos:
            raise ValueError(f"Algorithm class '{algorithm_class}' is unknown: it must be in {set(cls._algos.keys())}.")

        if name not in cls._algos[algorithm_class]:
            raise ValueError(f"Algorithm '{name}' is unknown or does not belong to '{algorithm_class}' algorithms: it must be in {set(cls._algos[algorithm_class].keys())}.")

        # instantiate algorithm with settings and set output manager
        algorithm = cls._algos[algorithm_class][name](settings)
        algorithm.set_output_manager(settings.logs)

        return algorithm

