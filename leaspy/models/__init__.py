from .abstract_model import AbstractModel

from .univariate_model import UnivariateModel
from .multivariate_model import MultivariateModel
from .multivariate_parallel_model import MultivariateParallelModel
from .constant_model import ConstantModel
from .lme_model import LMEModel
from .univariate_treatment_model import UnivariateTreatmentModel
from .multivariate_treatment_model import MultivariateTreatmentModel

# flexible dictionary to have a simpler and more maintainable ModelFactory
all_models = {
    # univariate Leaspy models
    'univariate_logistic': UnivariateModel,
    'univariate_linear': UnivariateModel,

    # univariate treatment Leaspy models
    'univariate_treatment_logistic': UnivariateTreatmentModel,
    'univariate_treatment_linear': UnivariateTreatmentModel,

    # multivariate treatment Leaspy models
    'multivariate_treatment_logistic': MultivariateTreatmentModel,
    'multivariate_treatment_linear': MultivariateTreatmentModel,

    # multivariate Leaspy models
    'logistic': MultivariateModel,
    'linear': MultivariateModel,
    'mixed_linear-logistic': MultivariateModel,
    'logistic_parallel': MultivariateParallelModel,

    # naive models (for benchmarks)
    'lme': LMEModel,
    'constant': ConstantModel,
}
