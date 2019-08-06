from . import UnivariateModel, MultivariateModel, MultivariateModelParallel


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
        else:
            raise ValueError("The name of the model you are trying to create does not exist")