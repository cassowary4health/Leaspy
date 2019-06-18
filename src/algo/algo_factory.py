from src.algo.gradient_descent import GradientDescent
from src.algo.mcmc_saem import MCMCSAEM

class AlgoFactory():

    #TODO change name of type

    @staticmethod
    def algo(type):
        if type.lower() == 'gradient_descent':
            return GradientDescent()
        elif type.lower() == 'mcmc_saem':
            return MCMCSAEM()


