from . import UnivariateModel, LogisticModel, LogisticParallelModel


class ModelFactory():

    @staticmethod
    def model(name):
        name = name.lower()
        if name == 'univariate':
            return UnivariateModel(name)
        elif name == 'logistic':
            return LogisticModel(name)
        elif name == 'logistic_parallel':
            return LogisticParallelModel(name)
        else:
            raise ValueError("The name of the model you are trying to create does not exist"
                             "It should be \`logistic\` or \`logistic_parallel\`")