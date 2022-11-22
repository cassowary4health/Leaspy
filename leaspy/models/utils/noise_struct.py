from __future__ import annotations
from typing import TYPE_CHECKING
from dataclasses import dataclass, field
from functools import reduce
import copy

import torch
from torch.distributions.constraints import unit_interval

from leaspy.exceptions import LeaspyInputError
from leaspy.utils.typing import KwargsType, Tuple, Callable, Optional, Dict, DictParamsTorch
from leaspy.models.utils.ordinal import OrdinalModelMixin

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

# Need to define our own Multinomial distribution for ordinal models...

class MultinomialDistribution(torch.distributions.Distribution):
    '''
    Class for a multinomial distribution with only sample method.

    Parameters
    ----------
    sf : torch.FloatTensor
        Values of the survival function [P(X > l) for l=0..L-1 where L is max_level] from which the distribution samples.
        Ordinal levels are assumed to be in the last dimension.
        Those values must be in [0, 1], and decreasing when ordinal level increases (not checked).

    Attributes
    ----------
    cdf : torch.FloatTensor
        The cumulative distribution function [P(X <= l) for l=0..L] from which the distribution samples.
        The shape of latest dimension is L+1 where L is max_level.
        We always have P(X <= L) = 1
    '''

    arg_constraints = {}
    validate_args = False

    def __init__(self, sf: torch.Tensor):
        super().__init__()
        assert unit_interval.check(sf).all(), "Bad probabilities in MultinomialDistribution"
        # shape of the sample (we discard the last dimension, used to store the different ordinal levels)
        self._sample_shape = sf.shape[:-1]
        # store the cumulative distribution function with trailing P(X <= L) = 1
        self.cdf = torch.cat((1. - sf, torch.ones((*self._sample_shape, 1))), dim=-1)

    @classmethod
    def from_pdf(cls, pdf: torch.Tensor):
        """Generate a new MultinomialDistribution from its probability density function instead of its survival function."""
        sf = OrdinalModelMixin.compute_ordinal_sf_from_ordinal_pdf(pdf)
        return cls(sf)

    def sample(self):
        """
        Multinomial sampling.

        Returns
        -------
        out : torch.IntTensor
            Vector of integer values corresponding to the multinomial sampling.
            Result is in [[0, L]]
        """
        # random sampling of cdf
        # we sample uniformly on [0, 1( but for the latest dimension corresponding to ordinal levels
        # this latest dimension will be broadcast when comparing with `cdf`
        r = torch.rand(self._sample_shape).unsqueeze(-1)
        out = (r < self.cdf).int().argmax(dim=-1) # works because it returns first index where we find a 1
        return out

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

    'ordinal': NoiseStruct(
        distribution_factory=MultinomialDistribution.from_pdf,
    ),
    'ordinal_ranking': NoiseStruct(
        distribution_factory=MultinomialDistribution, # from survival function directly
    )
}
