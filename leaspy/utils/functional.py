from __future__ import annotations

from dataclasses import dataclass
from inspect import signature
from typing import (
    Tuple,
    Iterable,
    Callable,
    Set,
    Optional,
    TypeVar,
    Generic,
    Union,
    Any,
    Mapping as TMapping,
)
import operator

import torch

from leaspy.utils.weighted_tensor import (
    WeightedTensor,
    TensorOrWeightedTensor,
    factory_weighted_tensor_unary_op,
)
from leaspy.utils.linalg import compute_orthonormal_basis
from leaspy.utils.typing import KwargsType

RT = TypeVar("RT")
S = TypeVar("S")

try:
    # Only introduced in Python 3.8
    from math import prod
except ImportError:
    # Shim for `prod` for Python < 3.8
    from functools import reduce

    def prod(iterable: Iterable[S], start: int = 1) -> S:
        """Product of all elements of the provided iterable, starting from `start`."""
        return reduce(operator.mul, iterable, start)


@dataclass(frozen=True)
class NamedInputFunction(Generic[RT]):
    """
    Bridge from a function with positional parameters to a function with keyword-only parameters.

    Nota: we do not implement the mapping of keyword-only to renamed keyword-only parameters for now
    (since not needed and would complexify the logic, in particular due to the existence of positional-only parameters)

    Parameters
    ----------
    f : Callable
        The original function.
        The named parameters to be sent in `f` should be: positional, positional-or-keyword, or variadic arguments.
        It can also have some keyword-only arguments but they should be fixed once for all with attribute `kws`.
    parameters : tuple[str, ...]
        Assigned names, in order, for positional parameters of `f`.
    kws : None (default) or dictionary[str, Any]
        Some optional fixed keyword parameters to pass upon function calls.
    """

    f: Callable[..., RT]
    parameters: Tuple[str, ...]
    kws: Optional[KwargsType] = None

    def call(self, named_params: TMapping[str, Any]) -> RT:
        """Call the underlying function with the correct positional arguments, retrieved by parameter names in input variables."""
        # we do not enforce consistency checks on `named_params` for optimization
        # this form is especially useful when provided mapping is "lazy" / "jit-computed" (like `State`)
        return self.f(*(named_params[p] for p in self.parameters), **(self.kws or {}))

    def __call__(self, **named_params) -> RT:
        """Same as `.call()` but with variadic input."""
        return self.call(named_params)

    def then(self, g: Callable[[RT], S], **g_kws) -> NamedInputFunction[S]:
        """Return a new NamedInputFunction applying (g o f) function."""

        def g_o_f(*f_args, **f_kws):
            return g(self.f(*f_args, **f_kws), **g_kws)

        # nicer for representation (too heavy otherwise)
        g_o_f.__name__ = f"{g.__name__}@{self.f.__name__}"
        g_o_f.__qualname__ = g_o_f.__name__

        return NamedInputFunction(
            f=g_o_f,
            parameters=self.parameters,
            kws=self.kws,
        )

    @staticmethod
    def bound_to(
        f: Callable[..., RT],
        check_arguments: Optional[Callable[[Tuple[str, ...], KwargsType], None]] = None,
    ):
        """Return a new factory to create new `NamedInputFunction` instances that are bound to the provided function."""

        def factory(*parameters: str, **kws) -> NamedInputFunction[RT]:
            """
            Factory of a `NamedInputFunction`, bounded to the provided function.

            Parameters
            ----------
            *parameters : str
                Names for positional parameters of the provided function.
            **kws
                Optional keyword-only arguments to pass to the provided function.
            """
            if check_arguments is not None:
                try:
                    check_arguments(parameters, kws)
                except Exception as e:
                    raise type(e)(f"{f.__name__}: {e}") from e
            return NamedInputFunction(f=f, parameters=parameters, kws=kws or None)

        # Nicer runtime name and docstring for the generated factory function
        factory.__name__ = f"symbolic_{f.__name__}_factory"
        factory.__qualname__ = ".".join(
            factory.__qualname__.split(".")[:-1] + [factory.__name__]
        )
        factory.__doc__ = factory.__doc__.replace(
            "the provided function", f"`{f.__name__}`"
        )

        return factory


def get_named_parameters(f: Callable) -> Tuple[str, ...]:
    """Get the names of parameters of the input function `f`, which should be a `NamedInputFunction` or a function with keyword-only parameters."""
    if isinstance(f, NamedInputFunction):
        return f.parameters
    params = signature(f).parameters
    non_kw_only_params = [
        p_name for p_name, p in params.items() if p.kind is not p.KEYWORD_ONLY
    ]
    # nota: we do not check annotation of returned type for now (to remain lighter)
    if len(non_kw_only_params):
        raise ValueError(non_kw_only_params)
    return tuple(params)


# Useful definitions


def _get_dim(
    x: TensorOrWeightedTensor,
    *,
    dim: Union[None, int, Tuple[int, ...]] = None,
    but_dim: Union[None, int, Tuple[int, ...]] = None,
) -> Union[int, Tuple[int, ...]]:
    assert (dim is None) or (
        but_dim is None
    ), "`dim` and `but_dim` should not be both defined"
    if but_dim is not None:
        ndim = x.ndim
        if isinstance(but_dim, int):
            but_dim = {but_dim}
        but_dim = {i if i >= 0 else ndim + i for i in but_dim}
        assert all(i >= 0 for i in but_dim), but_dim
        dim = tuple(i for i in range(ndim) if i not in but_dim)
    elif dim is None:
        # full summation by default
        dim = ()
    return dim


def sum_dim(
    x: TensorOrWeightedTensor,
    *,
    fill_value=0,
    dim: Union[None, int, Tuple[int, ...]] = None,
    but_dim: Union[None, int, Tuple[int, ...]] = None,
    **kws,
) -> torch.Tensor:
    """Sum dimension(s) of provided tensor (regular or weighted - filling with `fill_value` aggregates without any summed weighting if any)."""
    dim = _get_dim(x, dim=dim, but_dim=but_dim)
    if isinstance(x, WeightedTensor):
        return x.sum(fill_value=fill_value, dim=dim, **kws)
    return x.sum(dim=dim, **kws)


def wsum_dim(
    x: WeightedTensor,
    *,
    fill_value=0,
    dim: Union[None, int, Tuple[int, ...]] = None,
    but_dim: Union[None, int, Tuple[int, ...]] = None,
    **kws,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sum dimension(s) of provided weighted tensor (filling with `fill_value` aggregates without any summed weighting if any), and returns sum of weights as well."""
    dim = _get_dim(x, dim=dim, but_dim=but_dim)
    return x.wsum(fill_value=fill_value, dim=dim, **kws)


def sum_args(*args: torch.Tensor, **start_kw) -> torch.Tensor:
    """Summation of regular tensors with variadic input instead of standard iterable input."""
    return sum(args, **start_kw)


def prod_args(
    *args: TensorOrWeightedTensor[S], **start_kw
) -> TensorOrWeightedTensor[S]:
    """Product of tensors with variadic input instead of standard iterable input."""
    return prod(args, **start_kw)


def identity(x: S) -> S:
    """Unary identity function."""
    return x


def expand_left(
    t: TensorOrWeightedTensor[S], *, shape: Tuple[int, ...]
) -> TensorOrWeightedTensor[S]:
    """Expand shape of tensor at left with provided shape."""
    return t.expand(shape + t.shape)


def expand_right(
    t: TensorOrWeightedTensor[S], *, shape: Tuple[int, ...]
) -> TensorOrWeightedTensor[S]:
    """Expand shape of tensor at right with provided shape."""
    return t.expand(t.shape + shape)


def unsqueeze_right(
    t: TensorOrWeightedTensor[S], *, ndim: int
) -> TensorOrWeightedTensor[S]:
    """Adds `ndim` dimensions to tensor, from right-side, without copy (useful for right broadcasting which is non standard)."""
    # Nota: `unsqueeze_left` is useless since it is automatically done with standard broadcasting
    assert isinstance(ndim, int) and ndim >= 0, f"Can not unsqueeze {ndim} dimensions"
    if ndim == 0:
        return t
    return t.view(t.shape + (1,) * ndim)


def arguments_checker(
    *,
    n: Optional[int] = None,
    mandatory_kws: Optional[Set[str]] = None,
    possible_kws: Optional[Set[str]] = None,
):
    """
    Factory to check basic properties of parameters names and keyword-only arguments.

    Parameters
    ----------
    n : None or int
        Fixed number of positional arguments required by the function.
    mandatory_kws : None or set[str]
        Mandatory keyword-arguments for the function.
    possible_kws : None or set[str]
        Set of ALL possible keyword-arguments for the function.

    Returns
    -------
    function (args: tuple[str, ...], kws: dict[str, Any]) -> None
    """
    assert n is None or (isinstance(n, int) and n >= 0), n
    n_err_msg = None
    if n == 1:
        n_err_msg = f"Single name expected for positional parameters"
    elif n is not None:
        n_err_msg = f"{n} names expected for positional parameters"
    if mandatory_kws is not None and possible_kws is not None:
        unknown_mandatory_kws = mandatory_kws.difference(possible_kws)
        assert (
            len(unknown_mandatory_kws) == 0
        ), f"Some mandatory kws are not allowed: {unknown_mandatory_kws}"

    def check_arguments(args: tuple, kws: KwargsType) -> None:
        """Positional and keyword arguments checker."""
        if n_err_msg is not None:
            assert len(args) == n, n_err_msg
        if mandatory_kws is not None:
            missing_kws = mandatory_kws.difference(kws)
            assert (
                len(missing_kws) == 0
            ), f"Missing mandatory keyword-arguments: {missing_kws}"
        if possible_kws is not None:
            unknown_kws = set(kws).difference(possible_kws)
            assert len(unknown_kws) == 0, f"Unknown keyword-arguments: {unknown_kws}"

    return check_arguments


Identity = NamedInputFunction.bound_to(
    identity, arguments_checker(n=1, possible_kws=set())
)
MatMul = NamedInputFunction.bound_to(
    torch.matmul, arguments_checker(n=2, possible_kws=set())
)
OrthoBasis = NamedInputFunction.bound_to(
    compute_orthonormal_basis, arguments_checker(n=2, possible_kws=set("strip_col"))
)
Exp = NamedInputFunction.bound_to(
    factory_weighted_tensor_unary_op(torch.exp),
    arguments_checker(n=1, possible_kws=set()),
)
Sqr = NamedInputFunction.bound_to(
    factory_weighted_tensor_unary_op(torch.square),
    arguments_checker(n=1, possible_kws=set()),
)
Mean = NamedInputFunction.bound_to(
    # <!> never compute mean directly on WeightedTensor (use `.wsum()` instead)
    torch.mean,
    arguments_checker(n=1, possible_kws={"dim"}),
)
Std = NamedInputFunction.bound_to(
    # <!> never compute std directly on WeightedTensor
    torch.std,
    arguments_checker(n=1, possible_kws={"dim", "unbiased"}),
)
SumDim = NamedInputFunction.bound_to(
    sum_dim, arguments_checker(n=1)  # with `dim` XOR `but_dim`
)
Sum = NamedInputFunction.bound_to(sum_args, arguments_checker(possible_kws={"start"}))
Prod = NamedInputFunction.bound_to(prod_args, arguments_checker(possible_kws={"start"}))
# Filled = NamedInputFunction.bound_to(
#     WeightedTensor.filled, arguments_checker(n=1, mandatory_kws={"fill_value"})
# )
# Negate = NamedInputFunction.bound_to(
#     operator.neg, arguments_checker(n=1, possible_kws=set())
# )
# ItemGetter = NamedInputFunction.bound_to(
#     operator.itemgetter, arguments_checker(n=1, possible_kws=set())
# )
