from src.utils.random_variable.gaussian_random_variable import GaussianRandomVariable
from src.utils.random_variable.multi_gaussian_random_variable import MultiGaussianRandomVariable

class RandomVariableFactory():

    @staticmethod
    def random_variable(rv_infos):
        if rv_infos['rv_type'].lower() == 'gaussian':
            return GaussianRandomVariable(rv_infos)
        if  rv_infos['rv_type'].lower() == 'multigaussian':
            return MultiGaussianRandomVariable(rv_infos)