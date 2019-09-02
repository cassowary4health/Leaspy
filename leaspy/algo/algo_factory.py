from leaspy.algo.fit.gradient_descent import GradientDescent
from leaspy.algo.fit.gradient_mcmcsaem import GradientMCMCSAEM
from leaspy.algo.fit.tensor_mcmcsaem import TensorMCMCSAEM
from leaspy.algo.personalize.gradient_descent_personalize import GradientDescentPersonalize
from leaspy.algo.personalize.scipy_minimize import ScipyMinimize
from leaspy.algo.personalize.mean_realisations import MeanReal
from leaspy.algo.personalize.mcmc_personalize import MCMCPersonalize
from leaspy.algo.fit.hmc_saem import HMC_SAEM
from leaspy.algo.simulate.simulate import SimulationAlgorithm

class AlgoFactory:

    @staticmethod
    def algo(settings):
        name = settings.name

        # Fit Algorithm
        if name == 'gradient_descent':
            algorithm = GradientDescent(settings)
        elif name == 'tensor_mcmc_saem':
            algorithm = TensorMCMCSAEM(settings)
        elif name == 'mcmc_gradient_descent':
            algorithm = GradientMCMCSAEM(settings)

        # Personalize Algorithm
        #elif name == 'gradient_descent_personalize':
        #    algorithm = GradientDescentPersonalize(settings)
        #elif name == 'mcmc_personalize':
        #    algorithm = MCMCPersonalize(settings)
        elif name == 'scipy_minimize':
            algorithm = ScipyMinimize(settings)
        elif name == 'mean_real':
            algorithm = MeanReal(settings)
        #elif name == 'hmc_saem':
        #    algorithm = HMC_SAEM(settings)

        # Simulation agorithm
        elif name == 'simulation':
            algorithm = SimulationAlgorithm(settings)

        # Error
        else:
            raise ValueError("The name of your algorithm is unknown")

        algorithm.set_output_manager(settings.logs)
        return algorithm
