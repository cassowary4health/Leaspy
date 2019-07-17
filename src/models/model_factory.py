from src.models.univariate_model import UnivariateModel
from src.models.multivariate_model import MultivariateModel
from src.models.multivariate_model_parallel import MultivariateModelParallel
from src.models.gaussian_distribution_model import GaussianDistributionModel


class ModelFactory():

    @staticmethod
    def model(model_type):
        if model_type.lower() == 'univariate':
            return UnivariateModel()
        elif model_type.lower() == 'multivariate':
            return MultivariateModel()
        elif model_type.lower() == 'multivariate_parallel':
            return MultivariateModelParallel()
        elif model_type.lower() == 'gaussian_distribution':
            return GaussianDistributionModel()
        else:
            raise ValueError("The name of the model you are trying to create does not exist")