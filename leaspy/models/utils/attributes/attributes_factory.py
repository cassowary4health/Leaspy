from . import AttributesLogisticParallel, AttributesLogistic, AttributesLinear, AttributesUnivariate


class AttributesFactory:
    """
    Return an `Attributes` class object based on the given parameters.
    """

    @staticmethod
    def attributes(name, dimension, source_dimension):
        if type(name) == str:
            name = name.lower()
        else:
            raise AttributeError("The `name` argument must be a string!")

        if name == 'univariate':
            return AttributesUnivariate()
        elif name == 'logistic':
            return AttributesLogistic(dimension, source_dimension)
        elif name == 'logistic_parallel':
            return AttributesLogisticParallel(dimension, source_dimension)
        elif name == 'linear':
            return AttributesLinear(dimension, source_dimension)
        elif name == 'mixed_linear-logistic':
            return AttributesLogistic(dimension, source_dimension)  # TODO mixed check
        else:
            raise ValueError(
                "The name {} you provided for the attributes is not related to an attribute class".format(name))
