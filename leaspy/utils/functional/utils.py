"""This module contains utility functions related to the functional module."""

import torch
import operator
from typing import Callable, Tuple, TypeVar, Union, Set, Optional, Iterable

from leaspy.utils.weighted_tensor import TensorOrWeightedTensor, WeightedTensor
from leaspy.utils.typing import KwargsType

from.named_input_function import NamedInputFunction


S = TypeVar("S")


try:
    # Only introduced in Python 3.8
    from math import prod as _prod
except ImportError:
    # Shim for `prod` for Python < 3.8
    from functools import reduce

    def _prod(iterable: Iterable[S], start: int = 1) -> S:
        """Product of all elements of the provided iterable, starting from `start`."""
        return reduce(operator.mul, iterable, start)


def _prod_args(
    *args: TensorOrWeightedTensor[S], **start_kw
) -> TensorOrWeightedTensor[S]:
    """Product of tensors with variadic input instead of standard iterable input."""
    return _prod(args, **start_kw)


def _identity(x: S) -> S:
    """Unary identity function."""
    return x


def get_named_parameters(f: Callable) -> Tuple[str, ...]:
    """
    Get the names of parameters of the input function `f`, which should be
    a `NamedInputFunction` or a function with keyword-only parameters.
    """
    from inspect import signature

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


def _arguments_checker(
    *,
    nb_arguments: Optional[int] = None,
    mandatory_kws: Optional[Set[str]] = None,
    possible_kws: Optional[Set[str]] = None,
) -> Callable:
    """
    Factory to check basic properties of parameters names and keyword-only arguments.

    Parameters
    ----------
    nb_arguments : int, optional
        Fixed number of positional arguments required by the function.
    mandatory_kws : set[str], optional
        Mandatory keyword-arguments for the function.
    possible_kws : set[str], optional
        Set of ALL possible keyword-arguments for the function.

    Returns
    -------
    function (args: tuple[str, ...], kws: dict[str, Any]) -> None
    """
    if nb_arguments is not None and (not isinstance(nb_arguments, int) or nb_arguments < 0):
        raise ValueError(
            "Number of arguments should be a positive or null integer or None. "
            f"You provided a {type(nb_arguments)}."
        )
    nb_arguments_error_msg = None
    if nb_arguments == 1:
        nb_arguments_error_msg = f"Single name expected for positional parameters"
    elif nb_arguments is not None:
        nb_arguments_error_msg = f"{nb_arguments} names expected for positional parameters"
    if mandatory_kws is not None and possible_kws is not None:
        unknown_mandatory_kws = mandatory_kws.difference(possible_kws)
        if len(unknown_mandatory_kws) != 0:
            raise ValueError(
                f"Some mandatory kws are not allowed: {sorted(list(unknown_mandatory_kws))}."
            )

    def check_arguments(args: tuple, kws: KwargsType) -> None:
        """Positional and keyword arguments checker."""
        if nb_arguments_error_msg is not None:
            if len(args) != nb_arguments:
                raise ValueError(nb_arguments_error_msg)
        if mandatory_kws is not None:
            missing_kws = mandatory_kws.difference(kws)
            if len(missing_kws) != 0:
                raise ValueError(f"Missing mandatory keyword-arguments: {sorted(list(missing_kws))}.")
        if possible_kws is not None:
            unknown_kws = set(kws).difference(possible_kws)
            if len(unknown_kws) != 0:
                raise ValueError(f"Unknown keyword-arguments: {sorted(list(unknown_kws))}.")

    return check_arguments


def _sum_args(*args: torch.Tensor, **start_kw) -> torch.Tensor:
    """Summation of regular tensors with variadic input instead of standard iterable input."""
    return sum(args, **start_kw)


"""
The following functions should be moved to the weighted_tensor module as
they don't have anything to do with Named input functions...
"""


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
    """
    Adds `ndim` dimensions to tensor, from right-side, without
    copy (useful for right broadcasting which is non-standard).
    """
    # Nota: `unsqueeze_left` is useless since it is automatically done with standard broadcasting
    assert isinstance(ndim, int) and ndim >= 0, f"Can not unsqueeze {ndim} dimensions"
    if ndim == 0:
        return t
    return t.view(t.shape + (1,) * ndim)


def sum_dim(
    x: TensorOrWeightedTensor,
    *,
    fill_value=0,
    dim: Union[None, int, Tuple[int, ...]] = None,
    but_dim: Union[None, int, Tuple[int, ...]] = None,
    **kws,
) -> torch.Tensor:
    """
    Sum dimension(s) of provided tensor (regular or weighted -
    filling with `fill_value` aggregates without any summed weighting if any).
    """
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
    """
    Sum dimension(s) of provided weighted tensor (filling with
    `fill_value` aggregates without any summed weighting if any),
    and returns sum of weights as well.
    """
    dim = _get_dim(x, dim=dim, but_dim=but_dim)
    return x.wsum(fill_value=fill_value, dim=dim, **kws)


def _get_dim(
    x: TensorOrWeightedTensor,
    *,
    dim: Union[None, int, Tuple[int, ...]] = None,
    but_dim: Union[None, int, Tuple[int, ...]] = None,
) -> Union[int, Tuple[int, ...]]:
    if (dim is not None) and (but_dim is not None):
        raise ValueError("`dim` and `but_dim` should not be both defined.")
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
