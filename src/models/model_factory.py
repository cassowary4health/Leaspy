from src.models.univariate_model import UnivariateModel
from src.models.multivariate_model import MultivariateModel
from src.models.multivariate_parallel_model import MultivariateModelParallel
from _legacy.src.gaussian_distribution_model import GaussianDistributionModel


class ModelFactory():

    @staticmethod
    def model(name):
        name = name.lower()
        if name == 'univariate':
            return UnivariateModel()
        elif name == 'multivariate':
            return MultivariateModel(name)
        elif name == 'multivariate_parallel':
            return MultivariateModelParallel(name)
        elif name == 'gaussian_distribution':
            return GaussianDistributionModel()
        else:
            raise ValueError("The name of the model you are trying to create does not exist")