from . import LogisticParallelAttributes, LogisticAttributes, LinearAttributes


class AttributesFactory:
    """
    Return an `Attributes` class object based on the given parameters.
    """

    _attributes = {
        'logistic': LogisticAttributes,
        'univariate_logistic': LogisticAttributes,

        'logistic_parallel': LogisticParallelAttributes,

        'linear': LinearAttributes,
        'univariate_linear': LinearAttributes,

        #'mixed_linear-logistic': ... # TODO
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
        :class:`.AbstractAttributes`
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
