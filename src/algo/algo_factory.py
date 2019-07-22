from src.algo.gradient_descent import GradientDescent
from _legacy.src.mcmc_predict import MCMCPredict
from src.algo.tensor_mcmcsaem import TensorMCMCSAEM


class AlgoFactory:

    @staticmethod
    def algo(settings):
        name = settings.name

        if name == 'gradient_descent':
            algorithm = GradientDescent()
        elif name == 'mcmc_predict':
            algorithm = MCMCPredict()
        elif name == 'tensor_mcmc_saem':
            algorithm = TensorMCMCSAEM(settings)
        else:
            raise ValueError("The name of your algorithm is unknown")

        algorithm.load_parameters(settings.parameters)
        algorithm.set_output_manager(settings.outputs)
        return algorithm
