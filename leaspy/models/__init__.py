from .abstract_model import AbstractModel

from .univariate_model import UnivariateModel
from .univariatev0_model import UnivariateV0Model
from .joint_univariate_model import JointUnivariateModel
from .survival_model import SurvivalModel
from .multivariate_model import MultivariateModel
from .multivariate_parallel_model import MultivariateParallelModel
from .constant_model import ConstantModel
from .lme_model import LMEModel

# flexible dictionary to have a simpler and more maintainable ModelFactory
all_models = {
    # univariate Leaspy models
    'univariate_logistic': UnivariateModel,
    'univariate_linear': UnivariateModel,
    'univariatev0_logistic': UnivariateV0Model,
    'univariatev0_linear': UnivariateV0Model,

    # multivariate Leaspy models
    'logistic': MultivariateModel,
    'linear': MultivariateModel,
    'mixed_linear-logistic': MultivariateModel,
    'logistic_parallel': MultivariateParallelModel,

    # naive models (for benchmarks)
    'lme': LMEModel,
    'constant': ConstantModel,

    #joint model
    'joint_univariate_logistic': JointUnivariateModel,

    # survival
    'univariate_survival_weibull': SurvivalModel

}
