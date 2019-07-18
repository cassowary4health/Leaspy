from src.algo.gradient_descent import GradientDescent
from src._legacy.mcmc_saem import MCMCSAEM
from src._legacy.fast_mcmcsaem import FastMCMCSAEM
from src._legacy.mcmc_predict import MCMCPredict
from src.algo.tensor_mcmcsaem import TensorMCMCSAEM


class AlgoFactory:

    @staticmethod
    def algo(settings):
        name = settings.name

        if name == 'gradient_descent':
            algorithm = GradientDescent()
        elif name == 'mcmc_saem':
            algorithm = MCMCSAEM()
        elif name == 'fast_mcmc_saem':
            algorithm = FastMCMCSAEM()
        elif name == 'mcmc_predict':
            algorithm = MCMCPredict()
        elif name == 'tensor_mcmc_saem':
            algorithm = TensorMCMCSAEM(settings)
        else:
            raise ValueError("The name of your algorithm is unknown")

        algorithm.load_parameters(settings.parameters)
        algorithm.set_output_manager(settings.output_path)
        return algorithm
