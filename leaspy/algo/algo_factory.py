from __future__ import annotations
from typing import TYPE_CHECKING

from leaspy.algo.fit.tensor_mcmcsaem import TensorMCMCSAEM
# from leaspy.algo.fit.gradient_descent import GradientDescent
# from leaspy.algo.fit.gradient_mcmcsaem import GradientMCMCSAEM
from leaspy.algo.others.lme_fit import LMEFitAlgorithm

from leaspy.algo.personalize.scipy_minimize import ScipyMinimize
from leaspy.algo.personalize.scipy_minimize_link import ScipyMinimizeLink
from leaspy.algo.personalize.mean_realisations import MeanReal
from leaspy.algo.personalize.mode_realisations import ModeReal

from leaspy.algo.others.constant_prediction_algo import ConstantPredictionAlgorithm
from leaspy.algo.others.lme_personalize import LMEPersonalizeAlgorithm

from leaspy.algo.simulate.simulate import SimulationAlgorithm

from leaspy.exceptions import LeaspyAlgoInputError

if TYPE_CHECKING:
    from leaspy.algo.abstract_algo import AbstractAlgo


class AlgoFactory:
    """
    Return the wanted algorithm given its name.

    Notes
    -----
    For developers: add your new algorithm in corresponding category of ``_algos`` dictionary.
    """

    _algos = {
        'fit': {
            'mcmc_saem': TensorMCMCSAEM,
            #'mcmc_gradient_descent': GradientMCMCSAEM,
            #'gradient_descent': GradientDescent,
            'lme_fit': LMEFitAlgorithm,
        },

        'personalize': {
            'scipy_minimize': ScipyMinimize,
            'scipy_minimize_link': ScipyMinimizeLink,
            'mean_real': MeanReal,
            'mode_real': ModeReal,
            #'gradient_descent_personalize': GradientDescentPersonalize, # deprecated!

            'constant_prediction': ConstantPredictionAlgorithm,
            'lme_personalize': LMEPersonalizeAlgorithm,
        },

        'simulate': {
            'simulation': SimulationAlgorithm
        }
    }

    @classmethod
    def algo(cls, algorithm_class: str, settings) -> AbstractAlgo:
        """
        Return the wanted algorithm given its name.

        Parameters
        ----------
        algorithm_class : str
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
        :exc:`.LeaspyAlgoInputError`
            * if the algorithm class is unknown
            * if the algorithm name is unknown / does not belong to the wanted algorithm class
        """
        name = settings.name

        if algorithm_class not in cls._algos:
            raise LeaspyAlgoInputError(f"Algorithm class '{algorithm_class}' is unknown: it must be in {set(cls._algos.keys())}.")

        if name not in cls._algos[algorithm_class]:
            raise LeaspyAlgoInputError(f"Algorithm '{name}' is unknown or does not belong to '{algorithm_class}' algorithms: it must be in {set(cls._algos[algorithm_class].keys())}.")

        # instantiate algorithm with settings and set output manager
        algorithm = cls._algos[algorithm_class][name](settings)
        algorithm.set_output_manager(settings.logs)

        return algorithm

