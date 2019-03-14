from src.models.univariate_model import UnivariateModel


class ModelFactory():

    @staticmethod
    def model(type):
        if type.lower() == 'univariate':
            return UnivariateModel()
