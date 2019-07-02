from src.algo.gradient_descent import GradientDescent
from src.algo.mcmc_saem import MCMCSAEM
from src.algo.fast_mcmcsaem import FastMCMCSAEM
from src.algo.mcmc_predict import MCMCPredict

class AlgoFactory():

    #TODO change name of type

    @staticmethod
    def algo(type):
        if type.lower() == 'gradient_descent':
            return GradientDescent()
        elif type.lower() == 'mcmc_saem':
            return MCMCSAEM()
        elif type.lower() == 'fast_mcmc_saem':
            return FastMCMCSAEM()
        elif type.lower() == 'mcmc_predict':
            return MCMCPredict()


