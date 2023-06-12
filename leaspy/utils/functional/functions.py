"""This module defines commonly used named input functions."""

from __future__ import annotations

import torch

from leaspy.utils.weighted_tensor import factory_weighted_tensor_unary_op
from leaspy.utils.linalg import compute_orthonormal_basis

from .named_input_function import NamedInputFunction
from .utils import sum_dim, _arguments_checker, _sum_args, _prod_args, _identity


Prod = NamedInputFunction.bound_to(
    _prod_args,
    _arguments_checker(
        possible_kws={"start"},
    )
)


Identity = NamedInputFunction.bound_to(
    _identity,
    _arguments_checker(
        nb_arguments=1,
        possible_kws=set(),
    ),
)


MatMul = NamedInputFunction.bound_to(
    torch.matmul,
    _arguments_checker(
        nb_arguments=2,
        possible_kws=set(),
    ),
)


OrthoBasis = NamedInputFunction.bound_to(
    compute_orthonormal_basis,
    _arguments_checker(
        nb_arguments=2,
        possible_kws=set("strip_col"),
    ),
)


Exp = NamedInputFunction.bound_to(
    factory_weighted_tensor_unary_op(torch.exp),
    _arguments_checker(
        nb_arguments=1,
        possible_kws=set(),
    ),
)


Sqr = NamedInputFunction.bound_to(
    factory_weighted_tensor_unary_op(torch.square),
    _arguments_checker(
        nb_arguments=1,
        possible_kws=set(),
    ),
)


Mean = NamedInputFunction.bound_to(
    # <!> never compute mean directly on WeightedTensor (use `.wsum()` instead)
    torch.mean,
    _arguments_checker(
        nb_arguments=1,
        possible_kws={"dim"},
    ),
)


Std = NamedInputFunction.bound_to(
    # <!> never compute std directly on WeightedTensor
    torch.std,
    _arguments_checker(
        nb_arguments=1,
        possible_kws={"dim", "unbiased"},
    ),
)


SumDim = NamedInputFunction.bound_to(
    sum_dim,
    _arguments_checker(
        nb_arguments=1,  # with `dim` XOR `but_dim`
    ),
)


Sum = NamedInputFunction.bound_to(
    _sum_args,
    _arguments_checker(
        possible_kws={"start"},
    ),
)


# Filled = NamedInputFunction.bound_to(
#     WeightedTensor.filled, arguments_checker(nb_arguments=1, mandatory_kws={"fill_value"})
# )
# Negate = NamedInputFunction.bound_to(
#     operator.neg, arguments_checker(nb_arguments=1, possible_kws=set())
# )
# ItemGetter = NamedInputFunction.bound_to(
#     operator.itemgetter, arguments_checker(nb_arguments=1, possible_kws=set())
# )
