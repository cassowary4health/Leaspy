from src.algo.gradient_descent import GradientDescent
from src.algo.greedy_sampling import GreedySampling

class AlgoFactory():

    @staticmethod
    def algo(type):
        if type.lower() == 'gradient_descent':
            return GradientDescent()
        elif type.lower() == 'greedy_sampling':
            return GreedySampling()


