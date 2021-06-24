from .univariate_model import UnivariateModel
from .multivariate_model import MultivariateModel
from .multivariate_parallel_model import MultivariateParallelModel
from .constant_model import ConstantModel
from .lme_model import LMEModel
from .model_logistic_asymp import LogisticAsymp
from .model_logistic_asymp_delay import LogisticAsympDelay
from .model_stannard import Stannard
from .model_linear_vari import LinearVari
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
    'logistic_asymp': LogisticAsymp,
    'logistic_asymp_delay':LogisticAsympDelay,
    'stannard':Stannard,
    #multivariate Leaspy+ models
    'linear_vari':LinearVari,

    # naive models (for benchmarks)
    'lme': LMEModel,
    'constant': ConstantModel,
    
}
