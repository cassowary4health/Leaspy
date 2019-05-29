from src.models.univariate_model import UnivariateModel
from src.models.multivariate_model import MultivariateModel
from src.models.gaussian_distribution_model import GaussianDistributionModel


class ModelFactory():

    @staticmethod
    def model(type):
        if type.lower() == 'univariate':
            return UnivariateModel()
        if type.lower() == 'multivariate':
            return MultivariateModel()
        if type.lower() == 'gaussian_distribution':
            return GaussianDistributionModel()
