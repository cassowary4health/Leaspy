from .univariate_model import UnivariateModel
from .multivariate_model import MultivariateModel
from .multivariate_parallel_model import MultivariateParallelModel
from .constant_model import ConstantModel
from .lme_model import LMEModel
from .linear_B import LinearB

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
    'linearb': LinearB,
    # naive models (for benchmarks)
    'lme': LMEModel,
    'constant': ConstantModel,
    
}

initB = {"identity":lambda x:x,
"negidentity":lambda x:-x,
"logistic":lambda x:1./(1.+torch.exp(-x))}
