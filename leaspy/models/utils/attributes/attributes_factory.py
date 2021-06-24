
from . import AttributesLogisticParallel, AttributesLogistic, AttributesLinear, AttributesLogisticAsymp, AttributesLogisticAsympDelay, AttributesStannard,AttributesLinearVari



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
        'logistic_asymp': AttributesLogisticAsymp,
        'logistic_asymp_delay':AttributesLogisticAsympDelay,
        'stannard': AttributesStannard,
        'linear_vari': AttributesLinearVari,

        #'mixed_linear-logistic': AttributesLogistic # TODO mixed check
    }

    @classmethod
    def attributes(cls, name, dimension, source_dimension=None,neg=None,max_asymp=None,source_dimension_direction=None):
        if type(name) == str:
            name = name.lower()
        else:
            raise AttributeError("The `name` argument must be a string!")

        if name in cls._attributes:
            if 'univariate' in name:
                assert dimension == 1
                return cls._attributes[name](name, dimension, 0)
                
            elif 'asymp' in name:
                return cls._attributes[name](name, dimension, source_dimension)
            elif 'vari' in name:
                return cls._attributes[name](name, dimension, source_dimension,source_dimension_direction)
            elif 'stannard' in name:
                return cls._attributes[name](name, dimension, source_dimension,neg)
            else:
                return cls._attributes[name](name, dimension, source_dimension)
        else:
            raise ValueError(
                "The name {} you provided for the attributes is not related to an attribute class".format(name))

