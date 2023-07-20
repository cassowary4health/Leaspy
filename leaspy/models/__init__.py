from .abstract_model import AbstractModel

from .univariate_model import UnivariateModel
from .multivariate_model import MultivariateModel
from .multivariate_parallel_model import MultivariateParallelModel
from .multivariate_ip_mixture_model import MultivariateIndividualParametersMixtureModel
from .constant_model import ConstantModel
from .lme_model import LMEModel

# flexible dictionary to have a simpler and more maintainable ModelFactory
ALL_MODELS = {
    # univariate Leaspy models
    'univariate_logistic': UnivariateModel,
    'univariate_linear': UnivariateModel,

    # multivariate Leaspy models
    'logistic': MultivariateModel,
    'linear': MultivariateModel,
    'mixed_linear-logistic': MultivariateModel,
    'logistic_parallel': MultivariateParallelModel,

    # mixture models
    'mixture_logistic': MultivariateIndividualParametersMixtureModel,
    'mixture_linear': MultivariateIndividualParametersMixtureModel,

    # naive models (for benchmarks)
    'lme': LMEModel,
    'constant': ConstantModel,
}
