from leaspy.algo.fit.tensor_mcmcsaem import TensorMCMCSAEM
from leaspy.algo.personalize.gradient_descent_personalize import GradientDescentPersonalize
from leaspy.algo.personalize.mean_realisations import MeanReal
from leaspy.algo.personalize.mode_realisations import ModeReal
from leaspy.algo.personalize.scipy_minimize import ScipyMinimize
from leaspy.algo.simulate.simulate import SimulationAlgorithm


class AlgoFactory:
    """
    Return the wanted algorithm given its name.
    """

    @staticmethod
    def algo(algorithm_class, settings):
        """
        Return the wanted algorithm given its name.

        Parameters
        ----------
        algorithm_class: {'fit', 'simulate', 'personalize'}
            Task name, used to check if the algorithm within the input `settings` is compatible with this task.
        settings: leaspy.io.settings.algorithm_settings.AlgorithmSettings
            The algorithm settings.

        Returns
        -------
        algorithm: child class of AbstractAlgo
            The wanted algorithm.
        """
        name = settings.name

        AlgoFactory._check_compatibility(algorithm_class, name)

        # Fit Algorithm
        if name == 'mcmc_saem':
            algorithm = TensorMCMCSAEM(settings)
        # elif name == 'mcmc_gradient_descent':
        #    algorithm = GradientMCMCSAEM(settings)
        # elif name == 'gradient_descent':
        #    algorithm = GradientDescent(settings)

        # Personalize Algorithm
        elif name == 'gradient_descent_personalize':
            algorithm = GradientDescentPersonalize(settings)
        elif name == 'scipy_minimize':
            algorithm = ScipyMinimize(settings)
        elif name == 'mean_real':
            algorithm = MeanReal(settings)
        elif name == 'mode_real':
            algorithm = ModeReal(settings)
        # elif name == 'hmc_saem':
        #    algorithm = HMC_SAEM(settings)

        # Simulation agorithm
        elif name == 'simulation':
            algorithm = SimulationAlgorithm(settings)

        # Error
        else:
            raise ValueError("The name of your algorithm is unknown")

        algorithm.set_output_manager(settings.logs)
        return algorithm


    @staticmethod
    def _check_compatibility(algorithm_class, name):
        """
        Check compatibility of algorithms and API methods.

        Parameters
        ----------
        name: str
            Must be 'fit', 'simulate' or 'personalize'
            Name of the algorithm to run
        algorithm_class: str
            Must be one of the following api's name:
                * 'fit' - compatible with 'mcm_saem'
                * 'personalize' - compatible with "mode_real", "mean_real", "scipy_minimize", "gradient_descent_personalize"
                * 'simulate' - compatible with "simulation"

        Raises
        ------
        ValueError
            Raise an error if the settings' name does not belong to the wanted api methods & display the possible
            methods for that api.
        """

        if algorithm_class not in ['fit', 'simulate', 'personalize']:
            raise ValueError("The algorithm class you are using should be of class 'fit', 'simulate' or 'personalize'")

        compatibility_algorithms = {
            "fit": ["mcmc_saem"],
            "personalize": ["mode_real", "mean_real", "scipy_minimize", "gradient_descent_personalize"],
            "simulate": ["simulation"]
        }

        if name not in compatibility_algorithms[algorithm_class]:
            raise ValueError("Chosen algorithm is not compatible with method : {0} \n"
                             "please choose one in the following method list : {1}".format(name,
                                                                                           compatibility_algorithms[algorithm_class]))

