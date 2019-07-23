from src.algo.gradient_descent import GradientDescent
from src.algo.gradient_mcmcsaem import GradientMCMCSAEM
from src.algo.tensor_mcmcsaem import TensorMCMCSAEM
from src.algo.gradient_descent_personalize import GradientDescentPersonalize


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
        else:
            raise ValueError("The name of your algorithm is unknown")

        algorithm.load_parameters(settings.parameters)
        algorithm.set_output_manager(settings.outputs)
        return algorithm
