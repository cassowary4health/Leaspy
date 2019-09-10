from . import UnivariateModel, MultivariateModel, MultivariateParallelModel


class ModelFactory():

    @staticmethod
    def model(name):
        name = name.lower()
        if name == 'univariate':
            return UnivariateModel(name)
        elif name == 'logistic' or name == 'linear':
            return MultivariateModel(name)
        elif name == 'logistic_parallel':
            return MultivariateParallelModel(name)
        else:
            raise ValueError("The name of the model you are trying to create does not exist"
                             "It should be \`logistic\` or \`logistic_parallel\`")