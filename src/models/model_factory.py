from src.models.univariate_model import UnivariateModel
from src.models.multivariate_model import MultivariateModel
from src.models.gaussian_distribution_model import GaussianDistributionModel


class ModelFactory():

    @staticmethod
    def model(model_type):
        if model_type.lower() == 'univariate':
            return UnivariateModel()
        if model_type.lower() == 'multivariate':
            return MultivariateModel()
        if model_type.lower() == 'gaussian_distribution':
            return GaussianDistributionModel()
