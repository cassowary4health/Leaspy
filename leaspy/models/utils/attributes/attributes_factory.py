
from . import LogisticParallelAttributes, LogisticAttributes, LinearAttributes, LogisticAsymptotsAttributes, LogisticAsymptotsDelayAttributes, StannardAttributes



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
        'logistic_asymp': LogisticAsymptotsAttributes,
        'logistic_asymp_delay':LogisticAsymptotsDelayAttributes,
        'stannard': StannardAttributes,

        #'mixed_linear-logistic': AttributesLogistic # TODO mixed check
    }

    @classmethod
    def attributes(cls, name, dimension, source_dimension=None, max_asymp=None):
        if type(name) == str:
            name = name.lower()
        else:
            raise AttributeError("The `name` argument must be a string!")

        if name in cls._attributes:
            if 'univariate' in name:
                assert dimension == 1
                return cls._attributes[name](name, dimension, 0)
            else:
                return cls._attributes[name](name, dimension, source_dimension)
        else:
            raise ValueError(
                "The name {} you provided for the attributes is not related to an attribute class".format(name))

