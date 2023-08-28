from .abstract_model import AbstractModel

from .univariate_model import UnivariateModel
from .multivariate_model import MultivariateModel
from .multivariate_parallel_model import MultivariateParallelModel
from .constant_model import ConstantModel
from .lme_model import LMEModel
from .velocity_model import VelocityMultivariateModel

# flexible dictionary to have a simpler and more maintainable ModelFactory
all_models = {
    # univariate Leaspy models
    'univariate_logistic': UnivariateModel,
    'univariate_linear': UnivariateModel,

    # multivariate Leaspy models
    'logistic': MultivariateModel,
    'linear': MultivariateModel,
    'mixed_linear-logistic': MultivariateModel,
    'logistic_parallel': MultivariateParallelModel,
    'velocity_logistic': VelocityMultivariateModel,
    'velocity_linear': VelocityMultivariateModel,

    # naive models (for benchmarks)
    'lme': LMEModel,
    'constant': ConstantModel,
}
