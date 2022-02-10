from __future__ import annotations
from typing import TYPE_CHECKING
from dataclasses import dataclass, field
from functools import reduce
import copy

import torch

from leaspy.exceptions import LeaspyInputError
from leaspy.utils.typing import KwargsType, Tuple, Callable, Optional, Dict, DictParamsTorch

if TYPE_CHECKING:
    from leaspy.models.abstract_model import AbstractModel

# Type aliases
ValidationFunc = Callable[[KwargsType], KwargsType]


@dataclass(frozen=True)
class NoiseStruct:
    """
    Class storing all metadata of a noise structure (read-only).

    This class is not intended to be used directly, it serves as configuration for NoiseModel helper class.

    TODO? really have everything related to noise here, including stuff that is currently hardcoded in models
    (model log-likelihood...)?

    Parameters
    ----------
    distribution_factory : function [torch.Tensor, **kws] -> torch.distributions.Distribution (or None)
        A function taking a :class:`torch.Tensor` of values first, possible keyword arguments
        and returning a noise generator (instance of class :class:`torch.distributions.Distribution`),
        which can sample around these values with respect to noise structure.
    model_kws_to_dist_kws : dict[str, str]
        Mapping from naming of noise parameters in Leaspy model to the related torch distribution parameters.
    dist_kws_validators : tuple[ValidationFunc (kwargs -> kwargs)]
        Tuple of functions that sequentially (FIFO) check (& possibly clean) distribution parameters (input).
        It may raise (LeaspyAlgoInputError) if those are not appropriate for the noise structure.
        Those validators are the ones that we already may define without any need for a context
        (e.g. a 'gaussian_scalar' noise will need the scale to be of dimension 1, always)
    contextual_dist_kws_validators : tuple[**context -> ValidationFunc or None]
        Tuple of functions which are factory of validators functions, based on context parameters.
        Indeed, sometimes we may want to enforce some conditions, but we cannot enforce them without having extra contextual information
        (e.g. the scale of 'gaussian_diagonal' can be of any length in general, but if we already know the model dimension,
         then we want to make sure that the scale parameter will be of the same dimension)
        Note: if a given context is not sufficient to build a validator, factory should return None instead of a ValidationFunc.
        cf. :meth:`NoiseStruct.with_contextual_validators` for more details.

    Attributes
    ----------
    dist_kws_to_model_kws : dict[str, str] (read-only property)
        Mapping from torch distribution parameters to the related noise parameter naming in Leaspy model.

    All the previous parameters are also attributes (dataclass)
    """
    distribution_factory: Optional[Callable[..., torch.distributions.Distribution]] = None
    model_kws_to_dist_kws: Dict[str, str] = field(default_factory=dict)
    dist_kws_validators: Tuple[ValidationFunc, ...] = ()
    contextual_dist_kws_validators: Tuple[Callable[..., Optional[ValidationFunc]], ...] = ()

    @property
    def dist_kws_to_model_kws(self):
        """Shortcut for reciprocal mapping of `model_kws_to_dist_kws`"""
        return {v: k for k, v in self.model_kws_to_dist_kws.items()}

    def validate_dist_kws(self, dist_kws: KwargsType) -> KwargsType:
        """Sequentially compose all validators to validate input."""
        return reduce(
            lambda kws, V: V(kws),
            self.dist_kws_validators,  # sequence of validators (V)
            dist_kws  # initial keywords
        )

    def with_contextual_validators(self, **context_kws):
        """
        Clone the current noise structure but with the additional contextual `dist_kws_validators`.

        Note: the contextual validators will be appended, in FIFO order, to the already existing `dist_kws_validators`
        (so in particular they will be executed after them).

        Parameters
        ----------
        **context_kws
            Any relevant keyword argument which may help to define additional contextual `dist_kws_validators`.

        Returns
        -------
        NoiseStruct
            A cloned version of the current noise structure with relevant extra contextual validators set
            (they are now "static", i.e. regular validators)
        """
        # depending on context, determine which `contextual_dist_kws_validators` are relevant (= not None)
        # and those which are not (= None)
        possible_extra_validators = (ctxt_V(**context_kws) for ctxt_V in self.contextual_dist_kws_validators)

        relevant_extra_dist_kws_validators = tuple(V for V in possible_extra_validators if V is not None)
        # only keep contextual validators that were not relevant at this step (for chaining)
        remaining_contextual_dist_kws_validators = tuple(
            ctxt_V for ctxt_V, V in zip(self.contextual_dist_kws_validators, possible_extra_validators)
            if V is None
        )

        return self.__class__(
            distribution_factory=self.distribution_factory,
            model_kws_to_dist_kws=copy.deepcopy(self.model_kws_to_dist_kws),
            dist_kws_validators=self.dist_kws_validators + relevant_extra_dist_kws_validators,
            contextual_dist_kws_validators=remaining_contextual_dist_kws_validators
        )

# Helpers for validation
def convert_input_to_1D_float_tensors(d: KwargsType) -> DictParamsTorch:
    """Helper function to convert all input values into 1D torch float tensors."""
    return {
        k: (v if isinstance(v, torch.Tensor) else torch.tensor(v)).float().view(-1)
        for k, v in d.items()
    }

def validate_dimension_of_scale_factory(error_tpl: str, expected_dim: int, *,
                                        klass = LeaspyInputError):
    """Helper to produce a validator function that check dimension of scale among parameters."""
    def _validator(d: KwargsType):
        noise_scale = d['scale']  # precondition: is a tensor
        dim_noise_scale = noise_scale.numel()
        if dim_noise_scale != expected_dim:
            raise klass(error_tpl.format(noise_scale=noise_scale, dim_noise_scale=dim_noise_scale))
        return d
    return _validator

check_scale_is_univariate = validate_dimension_of_scale_factory(
    "You have provided a noise `scale` ({noise_scale}) of dimension {dim_noise_scale} "
    "whereas the `noise_struct` = 'gaussian_scalar' you requested requires a "
    "univariate scale (e.g. `scale = 0.1`).",
    expected_dim=1
)

def check_scale_is_compat_with_model_dimension(*, model: AbstractModel, **unused_extra_kws):
    """Check that scale parameter is compatible with model dimension."""
    return validate_dimension_of_scale_factory(
        "You requested a 'gaussian_diagonal' noise. However, the attribute `scale` you gave has "
        f"{{dim_noise_scale}} elements, which mismatches with model dimension of {model.dimension}. "
        f"Please give a list of std-dev for every features {model.features}, in order.",
        expected_dim=model.dimension
    )

def check_scale_is_positive(d: KwargsType):
    """Checks scale of noise is positive (component-wise if not scalar)."""
    noise_scale = d['scale']  # precondition: is a tensor
    if (noise_scale <= 0).any():
        raise LeaspyInputError(f"The noise `scale` parameter should be > 0, which is not the case in {noise_scale}.")
    return d

# Define default noise structures
NOISE_STRUCTS = {

    None: NoiseStruct(),

    'bernoulli': NoiseStruct(
        distribution_factory=torch.distributions.bernoulli.Bernoulli
    ),

    'gaussian_scalar': NoiseStruct(
        distribution_factory=torch.distributions.normal.Normal,
        model_kws_to_dist_kws={'noise_std': 'scale'},
        dist_kws_validators=(convert_input_to_1D_float_tensors, check_scale_is_positive, check_scale_is_univariate)
    ),

    'gaussian_diagonal': NoiseStruct(
        distribution_factory=torch.distributions.normal.Normal,
        model_kws_to_dist_kws={'noise_std': 'scale'},
        dist_kws_validators=(convert_input_to_1D_float_tensors, check_scale_is_positive),
        contextual_dist_kws_validators=(check_scale_is_compat_with_model_dimension,)
    ),
}
