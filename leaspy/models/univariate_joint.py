import torch
from lifelines import WeibullFitter

from leaspy.models.univariate import UnivariateModel
from leaspy.models.joint import JointModel
from leaspy.io.data.dataset import Dataset
from leaspy.exceptions import LeaspyModelInputError

from leaspy.utils.docs import doc_with_super  # doc_with_
# from leaspy.utils.subtypes import suffixed_method
from leaspy.exceptions import LeaspyInputError, LeaspyModelInputError
from leaspy.utils.weighted_tensor import WeightedTensor, TensorOrWeightedTensor
from leaspy.models.utils.initialization.model_initialization import (
    compute_patient_slopes_distribution,
    compute_patient_values_distribution,
    compute_patient_time_distribution,
    get_log_velocities,
    _torch_round)

from leaspy.variables.state import State
from leaspy.variables.specs import (
    DataVariable,
    NamedVariables,
    ModelParameter,
    PopulationLatentVariable,
    LinkedVariable,
    Hyperparameter,
    SuffStatsRW,
    VariablesValuesRO,
)
from leaspy.variables.distributions import Normal
from leaspy.utils.functional import Exp, Sqr, OrthoBasis, Sum
from leaspy.utils.weighted_tensor import unsqueeze_right

from leaspy.models.obs_models import observation_model_factory
from leaspy.utils.typing import (

    DictParams,
)
import pandas as pd
# TODO refact? implement a single function
# compute_individual_tensorized(..., with_jacobian: bool) -> returning either
# model values or model values + jacobians wrt individual parameters

# TODO refact? subclass or other proper code technique to extract model's concrete
#  formulation depending on if linear, logistic, mixed log-lin, ...


@doc_with_super()
class UnivariateJointModel(UnivariateModel, JointModel):
    """
    Manifold model for multiple variables of interest (logistic or linear formulation).

    Parameters
    ----------
    name : :obj:`str`
        The name of the model.
    **kwargs
        Hyperparameters of the model (including `noise_model`)

    Raises
    ------
    :exc:`.LeaspyModelInputError`
        * If `name` is not one of allowed sub-type: 'univariate_linear' or 'univariate_logistic'
        * If hyperparameters are inconsistent
    """

    SUBTYPES_SUFFIXES = {
        'univariate_joint': '_joint',
    }
    init_tolerance = 0.3
    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)
