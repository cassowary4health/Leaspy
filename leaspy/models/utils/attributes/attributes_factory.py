from . import AttributesLogisticParallel, AttributesLogistic, AttributesLinear


class AttributesFactory:
    """
    Return an `Attributes` class object based on the given parameters.
    """

    _attributes = {
        'logistic': AttributesLogistic,
        'univariate_logistic': AttributesLogistic,

        'logistic_parallel': AttributesLogisticParallel,

        'linear': AttributesLinear,
        'univariate_linear': AttributesLinear,

        #'mixed_linear-logistic': AttributesLogistic # TODO mixed check
    }

    @classmethod
    def attributes(cls, name, dimension, source_dimension=None):
        """
        Class method to build correct model attributes depending on model `name`.

        Parameters
        ----------
        name: str
        dimension : int
        source_dimension : int, optional (default None)

        Returns
        -------
        :class:`.AttributesAbstract`
        """
        if type(name) == str:
            name = name.lower()
        else:
            raise AttributeError("The `name` argument must be a string!")

        if name in cls._attributes:
            if 'univariate' in name:
                assert dimension == 1
            return cls._attributes[name](name, dimension, source_dimension)
        else:
            raise ValueError(
                "The name {} you provided for the attributes is not related to an attribute class".format(name))
