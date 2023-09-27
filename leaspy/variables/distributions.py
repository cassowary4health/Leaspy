from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Tuple, Any, ClassVar, Callable, Type

import torch
from torch.autograd import grad

from leaspy.utils.weighted_tensor import WeightedTensor, TensorOrWeightedTensor, sum_dim
from leaspy.exceptions import LeaspyInputError
from leaspy.utils.distributions import MultinomialDistribution
from leaspy.utils.functional import NamedInputFunction


class StatelessDistributionFamily(ABC):
    """
    Interface to represent stateless distribution families
    (i.e. no distribution parameters are stored in instance).

    TODO / WIP? allow WeightedTensor for parameters as well?
    (e.g. `batched_deltas = Normal(batched_deltas_mean, ...)` which should be masked at some indices)
    --> mask at latent pop. variable level (`batched_deltas`) or
        directly at model parameter level `batched_deltas_mean`?
    """

    parameters: ClassVar[Tuple[str, ...]]

    @classmethod
    @abstractmethod
    def validate_parameters(cls, *params: Any) -> Tuple[torch.Tensor, ...]:
        """
        Validate consistency of distribution parameters,
        returning them with out-of-place modifications if needed.
        """

    @classmethod
    def shape(cls, *params_shapes: Tuple[int, ...]) -> Tuple[int, ...]:
        """
        Shape of distribution samples (without any additional expansion),
        given shapes of distribution parameters.
        """
        # We provide a default implementation which should fit for most cases
        n_params = len(params_shapes)
        if n_params != len(cls.parameters):
            raise LeaspyInputError(
                f"Expecting {len(cls.parameters)} parameters but got {n_params}"
            )
        if n_params == 0:
            raise NotImplementedError(
                "No way to infer shape of samples since no parameter"
            )
        if n_params == 1:
            return params_shapes[0]
        return torch.broadcast_shapes(*params_shapes)

    @classmethod
    @abstractmethod
    def sample(
            cls,
            *params: torch.Tensor,
            sample_shape: Tuple[int, ...] = (),
    ) -> torch.Tensor:
        """
        Sample values, given distribution parameters (`sample_shape` is
        prepended to shape of distribution parameters).
        """

    @classmethod
    @abstractmethod
    def mode(cls, *params: torch.Tensor) -> torch.Tensor:
        """
        Mode of distribution (returning first value if discrete ties),
        given distribution parameters.
        """

    @classmethod
    @abstractmethod
    def mean(cls, *params: torch.Tensor) -> torch.Tensor:
        """Mean of distribution (if defined), given distribution parameters."""

    @classmethod
    @abstractmethod
    def stddev(cls, *params: torch.Tensor) -> torch.Tensor:
        """Standard-deviation of distribution (if defined), given distribution parameters."""

    @classmethod
    @abstractmethod
    def _nll(cls, x: torch.Tensor, *params: torch.Tensor) -> torch.Tensor:
        """Negative log-likelihood of value, given distribution parameters."""

    @classmethod
    @abstractmethod
    def _nll_jacobian(cls, x: torch.Tensor, *params: torch.Tensor) -> torch.Tensor:
        """Jacobian w.r.t. value of negative log-likelihood, given distribution parameters."""

    @classmethod
    def _nll_and_jacobian(
            cls,
            x: torch.Tensor,
            *params: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Negative log-likelihood of value and its jacobian w.r.t. value, given distribution parameters."""
        # not efficient implementation by default
        return cls._nll(x, *params), cls._nll_jacobian(x, *params)

    # AUTOMATIC COMPATIBILITY LAYER for value being a regular or a weighted tensor

    @staticmethod
    def _get_func_result_for_tensor_or_weighted_tensor(
            func: Callable,
            x: TensorOrWeightedTensor[float],
            *params: torch.Tensor,
    ) -> Any:
        """Automatic compatibility layer for value `x` being a regular or a weighted tensor."""
        if isinstance(x, WeightedTensor):
            r = func(x.value, *params)
            conv = x.valued
        else:
            r = func(x, *params)
            conv = WeightedTensor
        if isinstance(r, tuple):
            return tuple(map(conv, r))
        return conv(r)

    @classmethod
    def nll(
            cls,
            x: TensorOrWeightedTensor[float],
            *params: torch.Tensor,
    ) -> WeightedTensor[float]:
        """Negative log-likelihood of value, given distribution parameters."""
        return cls._nll(x, *params)

    @classmethod
    def nll_jacobian(
            cls,
            x: TensorOrWeightedTensor[float],
            *params: torch.Tensor,
    ) -> WeightedTensor[float]:
        """Jacobian w.r.t. value of negative log-likelihood, given distribution parameters."""
        return cls._nll_jacobian(x, *params)

    @classmethod
    def nll_and_jacobian(
            cls,
            x: TensorOrWeightedTensor[float],
            *params: torch.Tensor,
    ) -> Tuple[WeightedTensor[float], WeightedTensor[float]]:
        """Negative log-likelihood of value and its jacobian w.r.t. value, given distribution parameters."""
        return cls._nll_and_jacobian(x, *params)


class StatelessDistributionFamilyFromTorchDistribution(StatelessDistributionFamily):
    """Wrapper to build a `StatelessDistributionFamily` class from an existing torch distribution class."""

    dist_factory: ClassVar[Callable[..., torch.distributions.Distribution]]

    @classmethod
    def validate_parameters(cls, *params: Any) -> Tuple[torch.Tensor, ...]:
        """
        Validate consistency of distribution parameters,
        returning them with out-of-place modifications if needed.

        Parameters
        ----------
        params : Any
            The parameters to pass to the distribution factory.

        Returns
        -------
        Tuple[torch.Tensor, ...] :
            The validated parameters.
        """
        distribution = cls.dist_factory(*params, validate_args=True)
        return tuple(getattr(distribution, parameter) for parameter in cls.parameters)

    @classmethod
    def sample(
            cls,
            *params: torch.Tensor,
            sample_shape: Tuple[int, ...] = (),
    ) -> torch.Tensor:
        return cls.dist_factory(*params).sample(sample_shape)

    @classmethod
    def mode(cls, *params: torch.Tensor) -> torch.Tensor:
        """
        Mode of distribution (returning first value if discrete ties),
        given distribution parameters.

        Parameters
        ----------
        params : torch.Tensor
            The distribution parameters.

        Returns
        -------
        torch.Tensor :
            The value of the distribution's mode.
        """
        raise NotImplementedError("Not provided in torch.Distribution interface")

    @classmethod
    def mean(cls, *params: torch.Tensor) -> torch.Tensor:
        """
        Mean of distribution (if defined), given distribution parameters.

        Parameters
        ----------
        params : torch.Tensor
            The distribution parameters.

        Returns
        -------
        torch.Tensor :
            The value of the distribution's mean.
        """
        return cls.dist_factory(*params).mean

    @classmethod
    def stddev(cls, *params: torch.Tensor) -> torch.Tensor:
        """
        Return the standard-deviation of the distribution, given distribution parameters.

        Parameters
        ----------
        params : torch.Tensor
            The distribution parameters.

        Returns
        -------
        torch.Tensor :
            The value of the distribution's standard deviation.
        """
        return cls.dist_factory(*params).stddev

    @classmethod
    def _nll(cls, x: torch.Tensor, *params: torch.Tensor) -> torch.Tensor:
        return -cls.dist_factory(*params).log_prob(x)

    @classmethod
    def _nll_and_jacobian(
            cls,
            x: torch.Tensor,
            *params: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        nll = cls._nll(x, *params)
        (nll_grad_value,) = grad(nll, (x,), create_graph=x.requires_grad)
        return nll, nll_grad_value

    @classmethod
    def _nll_jacobian(cls, x: torch.Tensor, *params: torch.Tensor) -> torch.Tensor:
        return cls._nll_and_jacobian(x, *params)[1]

class BernoulliFamily(StatelessDistributionFamilyFromTorchDistribution):
    """Bernoulli family (stateless)."""
    parameters: ClassVar = ("loc",)
    dist_factory: ClassVar = torch.distributions.Bernoulli


class OrdinalFamily(StatelessDistributionFamilyFromTorchDistribution):
    """Ordinal family (stateless)."""
    parameters: ClassVar = ("pdf",)
    dist_factory: ClassVar = MultinomialDistribution.from_pdf


class NormalFamily(StatelessDistributionFamilyFromTorchDistribution):
    """Normal / Gaussian family (stateless)."""

    parameters: ClassVar = ("loc", "scale")
    dist_factory: ClassVar = torch.distributions.Normal
    nll_constant_standard: ClassVar = 0.5 * torch.log(2 * torch.tensor(math.pi))

    @classmethod
    def mode(cls, loc: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """
        Return the mode of the distribution given the distribution's loc and scale parameters.

        Parameters
        ----------
        loc : torch.Tensor
            The distribution loc.

        scale : torch.Tensor
            The distribution scale.

        Returns
        -------
        torch.Tensor :
            The value of the distribution's mode.
        """
        # `loc`, but with possible broadcasting of shape
        return torch.broadcast_tensors(loc, scale)[0]

    @classmethod
    def mean(cls, loc: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """
        Return the mean of the distribution, given the distribution loc and scale parameters.

        Parameters
        ----------
        loc : torch.Tensor
            The distribution loc parameters.
        scale : torch.Tensor
            The distribution scale parameters.

        Returns
        -------
        torch.Tensor :
            The value of the distribution's mean.
        """
        # Hardcode method for efficiency
        # `loc`, but with possible broadcasting of shape
        return torch.broadcast_tensors(loc, scale)[0]

    @classmethod
    def stddev(cls, loc: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """
        Return the standard-deviation of the distribution, given loc and scale of the distribution.

        Parameters
        ----------
        loc : torch.Tensor
            The distribution loc parameter.
        scale : torch.Tensor
            The distribution scale parameter.

        Returns
        -------
        torch.Tensor :
            The value of the distribution's standard deviation.
        """
        # Hardcode method for efficiency
        # `scale`, but with possible broadcasting of shape
        return torch.broadcast_tensors(loc, scale)[1]

    @classmethod
    def _nll(cls, x: torch.Tensor, loc: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        # Hardcode method for efficiency
        return (
                0.5 * ((x - loc) / scale) ** 2
                + torch.log(scale)
                + cls.nll_constant_standard
        )

    @classmethod
    def _nll_jacobian(cls, x: torch.Tensor, loc: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        # Hardcode method for efficiency
        return (x - loc) / scale ** 2

    @classmethod
    def _nll_and_jacobian(
            cls,
            x: torch.Tensor,
            loc: torch.Tensor,
            scale: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Hardcode method for efficiency
        z = (x - loc) / scale
        nll = 0.5 * z ** 2 + torch.log(scale) + cls.nll_constant_standard
        return nll, z / scale

    # @classmethod
    # def sample(cls, loc, scale, *, sample_shape = ()):
    #    # Hardcode method for efficiency? (<!> broadcasting)


class WeibullRightCensoredFamily(StatelessDistributionFamily):
    parameters: ClassVar = ("nu", "rho", 'xi', 'tau')
    dist_weibull: ClassVar = torch.distributions.weibull.Weibull

    @classmethod
    def validate_parameters(cls, *params: Any) -> Tuple[torch.Tensor, ...]:
        """
        Validate consistency of distribution parameters,
        returning them with out-of-place modifications if needed.

        Parameters
        ----------
        params : Any
            The parameters to pass to the distribution factory.

        Returns
        -------
        Tuple[torch.Tensor, ...] :
            The validated parameters.
        """
        raise NotImplementedError("Validate parameters not implemented")

    @classmethod
    def sample(
            cls,
            nu: torch.Tensor, rho: torch.Tensor, xi: torch.Tensor,
            tau: torch.Tensor,
            sample_shape: Tuple[int, ...] = (),
    ) -> torch.Tensor:
        return cls.dist_weibull(nu * torch.exp(-xi), rho).sample(sample_shape) + tau

    @classmethod
    def mode(cls, *params: torch.Tensor) -> torch.Tensor:
        """
        Mode of distribution (returning first value if discrete ties),
        given distribution parameters.

        Parameters
        ----------
        params : torch.Tensor
            The distribution parameters.

        Returns
        -------
        torch.Tensor :
            The value of the distribution's mode.
        """
        raise NotImplementedError("Mode not implemented")

    @classmethod
    def mean(cls, nu: torch.Tensor, rho: torch.Tensor, xi: torch.Tensor,
             tau: torch.Tensor) -> torch.Tensor:
        """
        Mean of distribution (if defined), given distribution parameters.

        Parameters
        ----------
        params : torch.Tensor
            The distribution parameters.

        Returns
        -------
        torch.Tensor :
            The value of the distribution's mean.
        """
        return cls.dist_weibull(nu * torch.exp(-xi), rho).mean + tau

    @classmethod
    def stddev(cls, nu: torch.Tensor, rho: torch.Tensor, xi: torch.Tensor,
               tau: torch.Tensor) -> torch.Tensor:
        """
        Return the standard-deviation of the distribution, given distribution parameters.

        Parameters
        ----------
        params : torch.Tensor
            The distribution parameters.

        Returns
        -------
        torch.Tensor :
            The value of the distribution's standard deviation.
        """
        return cls.dist_weibull(nu * torch.exp(-xi), rho).stddev

    @classmethod
    def _nll(cls, x: torch.Tensor, nu: torch.Tensor, rho: torch.Tensor, xi: torch.Tensor,
             tau: torch.Tensor) -> WeightedTensor[float]:
        # Get inputs
        event_time, event_bool = x

        # Construct reparametrized variables
        event_rep_time = torch.clamp(event_time - tau, min=0.)
        nu_rep = torch.exp(-xi) * nu

        # Survival neg log-likelihood
        n_log_survival = (event_rep_time / nu_rep) ** rho

        # Hazard neg log-likelihood only for patient with event not censored
        hazard = (rho / nu_rep) * ((event_rep_time * nu_rep) ** (rho - 1.))
        hazard = torch.where(event_bool == 0, torch.tensor(1., dtype=torch.double), hazard)

        attachment_events = n_log_survival - torch.log(hazard)

        return attachment_events

    @classmethod
    def _nll_and_jacobian(
            cls,
            x: torch.Tensor,
            nu: torch.Tensor,
            rho: torch.Tensor,
            xi: torch.Tensor,
            tau: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        nll = cls._nll(x, nu, rho)
        nll_grad_value = cls._nll_jacobian(x, nu, rho)

        return nll, nll_grad_value

    @classmethod
    def _nll_jacobian(cls, x: torch.Tensor, nu: torch.Tensor, rho: torch.Tensor, xi: torch.Tensor,
                      tau: torch.Tensor, *params: torch.Tensor) -> torch.Tensor:
        # Get inputs
        event_time, event_bool = x

        # Construct reparametrized variables
        event_rep_time = torch.clamp(event_time - tau, min=0.)
        nu_rep = torch.exp(-xi) * nu

        # Survival
        grad_xi = rho * (event_rep_time / nu_rep) ** rho - event_bool * rho
        grad_tau = (rho / nu_rep * torch.exp(xi)) * ((event_rep_time / nu_rep) ** (rho - 1.)) + event_bool * (
                rho - 1) / event_rep_time

        # Normalise as compute on normalised variables
        to_cat = [
            grad_xi * self.parameters['xi_std'],
            grad_tau * self.parameters['tau_std'],
        ]

        grads = torch.cat(to_cat, dim=-1).squeeze(0)

        return grads


@dataclass(frozen=True)
class SymbolicDistribution:
    """Class providing symbolic methods for distribution families."""

    parameters_names: Tuple[str, ...]
    dist_family: Type[StatelessDistributionFamily]

    # to hold automatic methods declared in `__post_init__`
    validate_parameters: Callable[..., Tuple[torch.Tensor, ...]] = field(
        init=False, repr=False, compare=False
    )
    """Function of named distribution parameters, to validate these parameters."""

    shape: Callable[..., Tuple[int, ...]] = field(init=False, repr=False, compare=False)
    """Function of named shapes of distribution parameters, to get shape of distribution samples."""

    mode: Callable[..., torch.Tensor] = field(init=False, repr=False, compare=False)
    """Function of named distribution parameters, to get mode of distribution."""

    mean: Callable[..., torch.Tensor] = field(init=False, repr=False, compare=False)
    """Function of named distribution parameters, to get mean of distribution."""

    stddev: Callable[..., torch.Tensor] = field(init=False, repr=False, compare=False)
    """Function of named distribution parameters, to get std-deviation of distribution."""

    def __post_init__(self):
        if len(self.parameters_names) != len(self.dist_family.parameters):
            raise ValueError(
                f"You provided {len(self.parameters_names)} names for {self.dist_family} parameters, "
                f"while expecting {len(self.dist_family.parameters)}: {self.dist_family.parameters}"
            )
        for bypass_method in {"validate_parameters", "shape", "mode", "mean", "stddev"}:
            object.__setattr__(self, bypass_method, self._get_func(bypass_method))

    def _get_func(self, func: str, *extra_args_names: str, **kws):
        """Get keyword-only function from the stateless distribution family."""
        return NamedInputFunction(
            getattr(self.dist_family, func),
            parameters=extra_args_names + self.parameters_names,
            kws=kws or None,
        )

    def get_func_sample(
            self, sample_shape: Tuple[int, ...] = ()
    ) -> NamedInputFunction[torch.Tensor]:
        """
        Factory of symbolic sampling function.

        Parameters
        ----------
        sample_shape : tuple of int, optional
            The shape of the sample.
            Default=().

        Returns
        -------
        NamedInputFunction :
            The sample function.
        """
        return self._get_func("sample", sample_shape=sample_shape)

    def get_func_nll(
            self, value_name: str
    ) -> NamedInputFunction[WeightedTensor[float]]:
        """
        Factory of symbolic function: state -> negative log-likelihood of value.

        Parameters
        ----------
        value_name : str

        Returns
        -------
        NamedInputFunction :
            The named input function to use to compute negative log likelihood.
        """
        return self._get_func("nll", value_name)

    def get_func_nll_jacobian(
            self, value_name: str
    ) -> NamedInputFunction[WeightedTensor[float]]:
        """
        Factory of symbolic function: state -> jacobian w.r.t. value of negative log-likelihood.

        Parameters
        ----------
        value_name : str

        Returns
        -------
        NamedInputFunction :
            The named input function to use to compute negative log likelihood jacobian.
        """
        return self._get_func("nll_jacobian", value_name)

    def get_func_nll_and_jacobian(
            self, value_name: str
    ) -> NamedInputFunction[Tuple[WeightedTensor[float], WeightedTensor[float]]]:
        """
        Factory of symbolic function: state -> (negative log-likelihood, its jacobian w.r.t. value).

        Parameters
        ----------
        value_name : str

        Returns
        -------
        Tuple[NamedInputFunction, NamedInputFunction] :
            The named input functions to use to compute negative log likelihood and its jacobian.
        """
        return self._get_func("nll_and_jacobian", value_name)

    @classmethod
    def bound_to(
            cls,
            dist_family: Type[StatelessDistributionFamily],
    ) -> Callable[..., SymbolicDistribution]:
        """
        Return a factory to create `SymbolicDistribution` bound to the provided distribution family.

        Parameters
        ----------
        dist_family : StatelessDistributionFamily
            The distribution family to use to create a SymbolicDistribution.

        Returns
        -------
        factory : Callable[..., SymbolicDistribution]
            The factory.
        """

        def factory(*parameters_names: str) -> SymbolicDistribution:
            """
            Factory of a `SymbolicDistribution`, bounded to the provided distribution family.

            Parameters
            ----------
            *parameters : str
                Names, in order, for distribution parameters.

            Returns
            -------
            SymbolicDistribution :
                The symbolic distribution resulting from the factory.
            """
            return SymbolicDistribution(parameters_names, dist_family)

        # Nicer runtime name and docstring for the generated factory function
        factory.__name__ = f"symbolic_{dist_family.__name__}_factory"
        factory.__qualname__ = ".".join(
            factory.__qualname__.split(".")[:-1] + [factory.__name__]
        )
        factory.__doc__ = factory.__doc__.replace(
            "the provided distribution family", f"`{dist_family.__name__}`"
        ).replace(
            "for distribution parameters",
            f"for distribution parameters: {dist_family.parameters}",
        )

        return factory


Normal = SymbolicDistribution.bound_to(NormalFamily)
Bernoulli = SymbolicDistribution.bound_to(BernoulliFamily)
Ordinal = SymbolicDistribution.bound_to(OrdinalFamily)
WeibullRightCensored = SymbolicDistribution.bound_to(WeibullRightCensoredFamily)

# INLINE UNIT TESTS
if __name__ == "__main__":
    print(Normal)
    print(Normal("mean", "std").validate_parameters(mean=0.0, std=1.0))

    nll = Normal("mean", "std").get_func_nll("val")

    args = dict(
        val=WeightedTensor(
            torch.tensor(
                [
                    [-1.0, 0.0, 0.0, 1.0],
                    [0.5, -0.5, -1.0, 0.0],
                ]
            ),
            weight=torch.tensor(
                [
                    [1, 0, 1, 1],
                    [1, 1, 0, 0],
                ]
            ),
        ),
        mean=torch.zeros((2, 4)),
        std=torch.ones(()),
    )

    r_nll = nll(**args)
    print("nll: ", r_nll)
    r_nll_sum_0 = nll.then(sum_dim, dim=1)(**args)
    r_nll_sum_1 = nll.then(sum_dim, dim=1)(**args)
    r_nll_sum_01 = nll.then(sum_dim, dim=(0, 1))(**args)  # MaskedTensor.wsum
    print("nll_sum_0: ", r_nll_sum_0)
    print("nll_sum_1: ", r_nll_sum_1)
    print("nll_sum_01: ", r_nll_sum_01)
    print("nll_sum_0,1: ", sum_dim(r_nll_sum_0, dim=0))
    print("nll_sum_1,0: ", sum_dim(r_nll_sum_1, dim=0))
