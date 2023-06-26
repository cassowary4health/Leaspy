from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Tuple, Any, ClassVar, Callable, Type

import torch
from torch.autograd import grad

from leaspy.utils.functional import NamedInputFunction
from leaspy.utils.weighted_tensor import WeightedTensor, TensorOrWeightedTensor
from leaspy.exceptions import LeaspyInputError


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
        cls, *params: torch.Tensor, sample_shape: Tuple[int, ...] = ()
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
        cls, x: torch.Tensor, *params: torch.Tensor
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
        return cls._get_func_result_for_tensor_or_weighted_tensor(cls._nll, x, *params)

    @classmethod
    def nll_jacobian(
        cls,
        x: TensorOrWeightedTensor[float],
        *params: torch.Tensor,
    ) -> WeightedTensor[float]:
        """Jacobian w.r.t. value of negative log-likelihood, given distribution parameters."""
        return cls._get_func_result_for_tensor_or_weighted_tensor(
            cls._nll_jacobian, x, *params
        )

    @classmethod
    def nll_and_jacobian(
        cls,
        x: TensorOrWeightedTensor[float],
        *params: torch.Tensor,
    ) -> Tuple[WeightedTensor[float], WeightedTensor[float]]:
        """Negative log-likelihood of value and its jacobian w.r.t. value, given distribution parameters."""
        return cls._get_func_result_for_tensor_or_weighted_tensor(
            cls._nll_and_jacobian, x, *params
        )


class StatelessDistributionFamilyFromTorchDistribution(StatelessDistributionFamily):
    """Wrapper to build a `StatelessDistributionFamily` class from an existing torch distribution class."""

    dist_factory: ClassVar[Callable[..., torch.distributions.Distribution]]

    @classmethod
    def validate_parameters(cls, *params: Any) -> Tuple[torch.Tensor, ...]:
        d = cls.dist_factory(*params, validate_args=True)
        return tuple(getattr(d, p) for p in cls.parameters)

    @classmethod
    def sample(
        cls, *params: torch.Tensor, sample_shape: Tuple[int, ...] = ()
    ) -> torch.Tensor:
        return cls.dist_factory(*params).sample(sample_shape)

    @classmethod
    def mode(cls, *params: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Not provided in torch.Distribution interface")

    @classmethod
    def mean(cls, *params: torch.Tensor) -> torch.Tensor:
        return cls.dist_factory(*params).mean

    @classmethod
    def stddev(cls, *params: torch.Tensor) -> torch.Tensor:
        return cls.dist_factory(*params).stddev

    @classmethod
    def _nll(cls, x: torch.Tensor, *params: torch.Tensor) -> torch.Tensor:
        return -cls.dist_factory(*params).log_prob(x)

    @classmethod
    def _nll_and_jacobian(
        cls, x: torch.Tensor, *params: torch.Tensor
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


class NormalFamily(StatelessDistributionFamilyFromTorchDistribution):
    """Normal / Gaussian family (stateless)."""

    parameters: ClassVar = ("loc", "scale")
    dist_factory: ClassVar = torch.distributions.Normal
    nll_constant_standard: ClassVar = 0.5 * torch.log(2 * torch.tensor(math.pi))

    @classmethod
    def mode(cls, loc, scale):
        # `loc`, but with possible broadcasting of shape
        return torch.broadcast_tensors(loc, scale)[0]

    @classmethod
    def mean(cls, loc, scale):
        # Hardcode method for efficiency
        # `loc`, but with possible broadcasting of shape
        return torch.broadcast_tensors(loc, scale)[0]

    @classmethod
    def stddev(cls, loc, scale):
        # Hardcode method for efficiency
        # `scale`, but with possible broadcasting of shape
        return torch.broadcast_tensors(loc, scale)[1]

    @classmethod
    def _nll(cls, x: torch.Tensor, loc, scale):
        # Hardcode method for efficiency
        return (
            0.5 * ((x - loc) / scale) ** 2
            + torch.log(scale)
            + cls.nll_constant_standard
        )

    @classmethod
    def _nll_jacobian(cls, x: torch.Tensor, loc, scale):
        # Hardcode method for efficiency
        return (x - loc) / scale**2

    @classmethod
    def _nll_and_jacobian(cls, x: torch.Tensor, loc, scale):
        # Hardcode method for efficiency
        z = (x - loc) / scale
        nll = 0.5 * z**2 + torch.log(scale) + cls.nll_constant_standard
        return nll, z / scale

    # @classmethod
    # def sample(cls, loc, scale, *, sample_shape = ()):
    #    # Hardcode method for efficiency? (<!> broadcasting)


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
        """Factory of symbolic sampling function."""
        return self._get_func("sample", sample_shape=sample_shape)

    def get_func_nll(
        self, value_name: str
    ) -> NamedInputFunction[WeightedTensor[float]]:
        """Factory of symbolic function: state -> negative log-likelihood of value."""
        return self._get_func("nll", value_name)

    def get_func_nll_jacobian(
        self, value_name: str
    ) -> NamedInputFunction[WeightedTensor[float]]:
        """Factory of symbolic function: state -> jacobian w.r.t. value of negative log-likelihood."""
        return self._get_func("nll_jacobian", value_name)

    def get_func_nll_and_jacobian(
        self, value_name: str
    ) -> NamedInputFunction[Tuple[WeightedTensor[float], WeightedTensor[float]]]:
        """Factory of symbolic function: state -> (negative log-likelihood, its jacobian w.r.t. value)."""
        return self._get_func("nll_and_jacobian", value_name)

    @classmethod
    def bound_to(cls, dist_family: Type[StatelessDistributionFamily]):
        """Return a factory to create `SymbolicDistribution` bound to the provided distribution family."""

        def factory(*parameters_names: str):
            """
            Factory of a `SymbolicDistribution`, bounded to the provided distribution family.

            Parameters
            ----------
            *parameters : str
                Names, in order, for distribution parameters.
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


# INLINE UNIT TESTS
if __name__ == "__main__":
    from leaspy.utils.functional import NamedInputFunction, sum_dim

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
