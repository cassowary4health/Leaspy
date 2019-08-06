from src.algo.fit.gradient_descent import GradientDescent
from src.algo.fit.gradient_mcmcsaem import GradientMCMCSAEM
from src.algo.fit.tensor_mcmcsaem import TensorMCMCSAEM
from src.algo.personalize.gradient_descent_personalize import GradientDescentPersonalize
from src.algo.personalize.scipy_minimize import ScipyMinimize
from src.algo.personalize.mcmc_personalize import MCMCPersonalize
from src.algo.fit.hmc_saem import HMC_SAEM

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
        elif name == 'gradient_descent_personalize':
            algorithm = GradientDescentPersonalize(settings)
        elif name == 'mcmc_personalize':
            algorithm = MCMCPersonalize(settings)
        elif name == 'scipy_minimize':
            algorithm = ScipyMinimize(settings)
        elif name == 'hmc_saem':
            algorithm = HMC_SAEM(settings)
        else:
            raise ValueError("The name of your algorithm is unknown")

        algorithm.load_parameters(settings.parameters)
        algorithm.set_output_manager(settings.outputs)
        return algorithm
