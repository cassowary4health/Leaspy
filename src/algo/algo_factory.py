from src.algo.gradient_descent import GradientDescent
from src.algo.greedy_sampling import GreedySampling
from src.algo.mcmc_saem import MCMCSAEM

class AlgoFactory():

    @staticmethod
    def algo(type):
        if type.lower() == 'gradient_descent':
            return GradientDescent()
            return RandomSampling()
        elif type.lower() == 'mcmc_saem':
            return MCMCSAEM()


