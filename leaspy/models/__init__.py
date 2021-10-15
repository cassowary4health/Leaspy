from .univariate_model import UnivariateModel
from .multivariate_model import MultivariateModel
from .multivariate_parallel_model import MultivariateParallelModel
from .constant_model import ConstantModel
from .lme_model import LMEModel
from .logistic_asymptots_model import LogisticAsymptotsModel
from .logistic_asymptots_delay_model import LogisticAsymptotsDelayModel
from .stannard_model import StannardModel

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
    'logistic_asymptots' : LogisticAsymptotsModel,
    'logistic_asymptots_delay' : LogisticAsymptotsDelayModel,
    'stannard' : StannardModel,

    # naive models (for benchmarks)
    'lme': LMEModel,
    'constant': ConstantModel,
    
}
