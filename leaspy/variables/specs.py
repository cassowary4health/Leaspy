from __future__ import annotations

import torch
from abc import abstractmethod
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import (
    ClassVar,
    Callable,
    Dict,
    FrozenSet,
    Optional,
    Tuple,
    Mapping as TMapping,
    MutableMapping as TMutableMapping,
)
from collections import UserDict

from leaspy.utils.functional import (
    get_named_parameters,
    Identity,
    Sum,
    SumDim,
    Mean,
    Std,
    Sqr,
    NamedInputFunction,
)
from leaspy.models.utilities import compute_ind_param_std_from_suff_stats
from leaspy.utils.weighted_tensor import TensorOrWeightedTensor, WeightedTensor, sum_dim, expand_left
from leaspy.variables.distributions import SymbolicDistribution
from leaspy.utils.typing import KwargsType
from leaspy.exceptions import LeaspyModelInputError

VarName = str
VarValue = TensorOrWeightedTensor[float]
VariablesValuesRO = TMapping[VarName, VarValue]
VariablesLazyValuesRO = TMapping[VarName, Optional[VarValue]]
VariablesLazyValuesRW = TMutableMapping[VarName, Optional[VarValue]]
SuffStatsRO = TMapping[VarName, torch.Tensor]  # VarValue
SuffStatsRW = TMutableMapping[VarName, torch.Tensor]  # VarValue

LVL_IND = 0
LVL_FT = -1


class VariableInterface:
    """Interface for variable specifications."""

    # description: str
    """Description of variable, for documentation purposes."""

    is_settable: ClassVar[bool]
    """Is True if and only if state of variables is intended to be manually modified by user."""

    fixed_shape: ClassVar[bool]
    """Is True as soon as we guarantee that shape of variable is only dependent on model hyperparameters, not data."""

    @abstractmethod
    def compute(self, state: VariablesValuesRO) -> Optional[VarValue]:
        """
        Compute variable value from a `state` exposing a dict-like interface: var_name -> values.
        If not relevant for variable type return None.
        """

    @abstractmethod
    def get_ancestors_names(self) -> FrozenSet[VarName]:
        """Get the names of the variables that the current variable directly depends on."""

    # TODO? add a check or validate(value) method? (to be optionally called by State)
    # <!> should some extra context be passed to this method
    # (e.g. `n_individuals` or `n_timepoints` dimensions are not known during variable definition
    # but their consistency could/should be checked?)


class IndepVariable(VariableInterface):
    """Base class for variable that is not dependent on any other variable."""

    def get_ancestors_names(self) -> FrozenSet[VarName]:
        return frozenset()

    def compute(self, state: VariablesValuesRO) -> None:
        return None


@dataclass(frozen=True)
class Hyperparameter(IndepVariable):
    """Hyperparameters that can not be reset."""

    value: VarValue

    fixed_shape: ClassVar = True
    is_settable: ClassVar = False

    def __post_init__(self):
        if not isinstance(self.value, (torch.Tensor, WeightedTensor)):
            object.__setattr__(self, "value", torch.tensor(self.value))

    def to_device(self, device: torch.device) -> None:
        """Move the value to specified device (other variables never hold values so need for this method)."""
        return object.__setattr__(self, "value", self.value.to(device=device))

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.value.shape


@dataclass(frozen=True, init=False)
class Collect:
    """
    A convenient class to produce a function to collect sufficient stats that
    are existing or dedicated variables (to be automatically created).
    """

    existing_variables: Tuple[VarName, ...] = ()
    dedicated_variables: Optional[TMapping[VarName, LinkedVariable]] = None

    def __init__(
        self, *existing_variables: VarName, **dedicated_variables: LinkedVariable
    ):
        # custom init to allow more convenient variadic form
        object.__setattr__(self, "existing_variables", existing_variables)
        object.__setattr__(self, "dedicated_variables", dedicated_variables or None)

    @property
    def variables(self) -> Tuple[VarName, ...]:
        return self.existing_variables + tuple(self.dedicated_variables or ())

    def __call__(self, state: VariablesValuesRO) -> SuffStatsRW:
        return {k: state[k] for k in self.variables}


@dataclass(frozen=True)
class ModelParameter(IndepVariable):
    """
    Variable for model parameters, with a maximization rule (not to be sampled but
    is not data / nor hyperparameter, nor linked).
    """

    shape: Tuple[int, ...]
    suff_stats: Collect  # Callable[[VariablesValuesRO], SuffStatsRW]
    """
    The symbolic update functions will take variadic `suff_stats` values,
    in order to re-use NamedInputFunction logic: e.g. update_rule=Std('xi')

    <!> ISSUE: for `tau_std` and `xi_std` we also need `state` values in addition to
    `suff_stats` values (only after burn-in) since we can NOT use the variadic form
    readily for both `state` and `suff_stats` (names would be conflicting!), we sent
    `state` as a special kw variable (a bit lazy but valid) (and we prevent using this
    name for a variable as a safety)
    """

    update_rule: Callable[..., VarValue]
    """Update rule for normal phase, and memory-less (burn-in) phase unless `update_rule_burn_in` is not None."""

    update_rule_burn_in: Optional[Callable[..., VarValue]] = None
    """Specific rule for burn-in (currently implemented for some variables -> e.g. `xi_std`)"""

    # private attributes (computed in __post_init__)
    _update_rule_parameters: FrozenSet[VarName] = field(init=False, repr=False)
    _update_rule_burn_in_parameters: Optional[FrozenSet[VarName]] = field(
        default=None, init=False, repr=False
    )

    fixed_shape: ClassVar = True
    is_settable: ClassVar = True

    def __post_init__(self):
        self._check_and_store_update_rule_parameters("update_rule")
        self._check_and_store_update_rule_parameters("update_rule_burn_in")

    def _check_and_store_update_rule_parameters(self, update_method: str) -> None:
        method = getattr(self, update_method)
        if method is None:
            return
        allowed_kws = set(self.suff_stats.variables).union({"state"})
        err_msg = (
            f"Function provided in `ModelParameter.{update_method}` should be a function with keyword-only parameters "
            "(using names of this variable sufficient statistics, or the special 'state' keyword): not {}"
        )
        try:
            inferred_params = get_named_parameters(method)
        except ValueError as e:
            raise LeaspyModelInputError(err_msg.format(str(e))) from e
        forbidden_kws = set(inferred_params).difference(allowed_kws)
        if len(forbidden_kws):
            raise LeaspyModelInputError(err_msg.format(forbidden_kws))

        object.__setattr__(
            self, f"_{update_method}_parameters", frozenset(inferred_params)
        )

    def compute_update(
        self, *, state: VariablesValuesRO, suff_stats: SuffStatsRO, burn_in: bool
    ) -> VarValue:
        """Update rule (maximization step) for the model parameter."""
        update_rule, update_rule_params = self.update_rule, self._update_rule_parameters
        if burn_in and self.update_rule_burn_in is not None:
            update_rule, update_rule_params = (
                self.update_rule_burn_in,
                self._update_rule_burn_in_parameters,
            )
        state_kw = dict(state=state) if "state" in update_rule_params else {}
        # <!> it would not be clean to send all suff_stats (unfiltered) for standard kw-only functions...
        return update_rule(
            **state_kw,
            **{k: suff_stats[k] for k in self._update_rule_parameters if k != "state"},
        )

    # Special factories for typical cases (shortcuts)

    @classmethod
    def for_pop_mean(cls, pop_var_name: VarName, shape: Tuple[int, ...]):
        """
        Smart automatic definition of `ModelParameter` when it is the mean of Gaussian
        prior of a population latent variable.
        """
        return cls(
            shape,
            suff_stats=Collect(pop_var_name),
            update_rule=Identity(pop_var_name),
        )

    @classmethod
    def for_ind_mean(cls, ind_var_name: VarName, shape: Tuple[int, ...]):
        """
        Smart automatic definition of `ModelParameter` when it is the mean of Gaussian
        prior of an individual latent variable.
        """
        return cls(
            shape,
            suff_stats=Collect(ind_var_name),
            update_rule=Mean(ind_var_name, dim=LVL_IND),
        )

    @classmethod
    def for_ind_std(cls, ind_var_name: VarName, shape: Tuple[int, ...], **tol_kw):
        """
        Smart automatic definition of `ModelParameter` when it is the std-dev of Gaussian
        prior of an individual latent variable.
        """
        ind_var_sqr_name = f"{ind_var_name}_sqr"
        update_rule_normal = NamedInputFunction(
            compute_ind_param_std_from_suff_stats,
            parameters=("state", ind_var_name, ind_var_sqr_name),
            kws=dict(ip_name=ind_var_name, dim=LVL_IND, **tol_kw),
        )
        return cls(
            shape,
            suff_stats=Collect(
                ind_var_name, **{ind_var_sqr_name: LinkedVariable(Sqr(ind_var_name))}
            ),
            update_rule_burn_in=Std(ind_var_name, dim=LVL_IND),
            update_rule=update_rule_normal,
        )


@dataclass(frozen=True)
class DataVariable(IndepVariable):
    """Variables for input data, that may be reset."""

    fixed_shape: ClassVar = False
    is_settable: ClassVar = True


class LatentVariableInitType(Enum):
    """Type of initialization for latent variables."""

    PRIOR_MODE = auto()
    PRIOR_MEAN = auto()
    PRIOR_SAMPLES = auto()


@dataclass(frozen=True)
class LatentVariable(IndepVariable):
    """Unobserved variable that will be sampled, with symbolic prior distribution [e.g. Normal('xi_mean', 'xi_std')]."""

    # TODO/WIP? optional mask derive from optional masks of prior distribution parameters?
    # or should be fixed & explicit here?
    prior: SymbolicDistribution
    sampling_kws: Optional[KwargsType] = None

    is_settable: ClassVar = True

    def get_prior_shape(
        self, named_vars: TMapping[VarName, VariableInterface]
    ) -> Tuple[int, ...]:
        """Get shape of prior distribution (i.e. without any expansion for `IndividualLatentVariable`)."""
        bad_params = {
            n for n in self.prior.parameters_names if not named_vars[n].fixed_shape
        }
        if len(bad_params):
            raise LeaspyModelInputError(
                f"Shapes of some prior distribution parameters are not fixed: {bad_params}"
            )
        params_shapes = {n: named_vars[n].shape for n in self.prior.parameters_names}
        return self.prior.shape(**params_shapes)

    def _get_init_func_generic(
        self, method: LatentVariableInitType, *, sample_shape: Tuple[int, ...]
    ) -> NamedInputFunction[torch.Tensor]:
        """Return a `NamedInputFunction`: State -> Tensor, that may be used for initialization."""
        if method is LatentVariableInitType.PRIOR_SAMPLES:
            return self.prior.get_func_sample(sample_shape)
        if method is LatentVariableInitType.PRIOR_MODE:
            return self.prior.mode.then(expand_left, shape=sample_shape)
        if method is LatentVariableInitType.PRIOR_MEAN:
            return self.prior.mean.then(expand_left, shape=sample_shape)

    @abstractmethod
    def get_regularity_variables(
        self, value_name: VarName
    ) -> Dict[VarName, LinkedVariable]:
        """Automatically get extra linked variables to compute regularity term for this latent variable."""
        # return {
        #    # Not really useful... directly sum it to be memory efficient...
        #    f"nll_regul_{value_name}_full": LinkedVariable(
        #        self.prior.get_func_nll(value_name)
        #    ),
        #    # TODO: jacobian as well...
        # }
        pass


class PopulationLatentVariable(LatentVariable):
    """Population latent variable."""

    # not so easy to guarantee the fixed shape property in fact...
    # (it requires that parameters of prior distribution all have fixed shapes)
    fixed_shape: ClassVar = True

    def get_init_func(
        self, method: LatentVariableInitType
    ) -> NamedInputFunction[torch.Tensor]:
        """Return a `NamedInputFunction`: State -> Tensor, that may be used for initialization."""
        return self._get_init_func_generic(method=method, sample_shape=())

    def get_regularity_variables(
        self, value_name: VarName
    ) -> Dict[VarName, LinkedVariable]:
        # d = super().get_regularity_variables(value_name)
        d = {}
        d.update(
            {
                f"nll_regul_{value_name}": LinkedVariable(
                    # SumDim(f"nll_regul_{value_name}_full")
                    self.prior.get_func_nll(value_name).then(sum_dim)
                ),
                # TODO: jacobian as well...
            }
        )
        return d


class IndividualLatentVariable(LatentVariable):
    """Individual latent variable."""

    fixed_shape: ClassVar = False

    def get_init_func(
        self, method: LatentVariableInitType, *, n_individuals: int
    ) -> NamedInputFunction[torch.Tensor]:
        """Return a `NamedInputFunction`: State -> Tensor, that may be used for initialization."""
        return self._get_init_func_generic(method=method, sample_shape=(n_individuals,))

    def get_regularity_variables(
        self, value_name: VarName
    ) -> Dict[VarName, LinkedVariable]:
        # d = super().get_regularity_variables(value_name)
        d = {}
        d.update(
            {
                f"nll_regul_{value_name}_ind": LinkedVariable(
                    # SumDim(f"nll_regul_{value_name}_full", but_dim=LVL_IND)
                    self.prior.get_func_nll(value_name).then(sum_dim, but_dim=LVL_IND)
                ),
                f"nll_regul_{value_name}": LinkedVariable(
                    SumDim(f"nll_regul_{value_name}_ind")
                ),
                # TODO: jacobian as well...
            }
        )
        return d


@dataclass(frozen=True)
class LinkedVariable(VariableInterface):
    """
    Variable which is a deterministic expression of other variables (we directly
    use variables names instead of boring mappings: kws <-> vars).
    """

    f: Callable[..., VarValue]
    parameters: FrozenSet[VarName] = field(init=False)
    # expected_shape? (<!> some of the shape dimensions might not be known like `n_individuals` or `n_timepoints`...)
    # admissible_value? (<!> same issue than before, cf. remark on `IndividualLatentVariable`)

    is_settable: ClassVar = False
    # shape of linked variable may be fixed in some cases, but complex/boring/useless logic to guarantee it
    fixed_shape: ClassVar = False

    def __post_init__(self):
        try:
            inferred_params = get_named_parameters(self.f)
        except ValueError:
            raise LeaspyModelInputError(
                "Function provided in `LinkedVariable` should be a function with "
                "keyword-only parameters (using variables names)."
            )
        object.__setattr__(self, "parameters", frozenset(inferred_params))

    def get_ancestors_names(self) -> FrozenSet[VarName]:
        return self.parameters

    def compute(self, state: VariablesValuesRO) -> VarValue:
        return self.f(**{k: state[k] for k in self.parameters})


class NamedVariables(UserDict):
    """
    Convenient dictionary for named variables specifications.

    In particular, it:
    1. forbids the collisions in variable names when assigning/updating the collection
    2. forbids the usage of some reserved names like 'state' or 'suff_stats'
    3. automatically adds implicit variables when variables of certain kind are added
    (e.g. dedicated vars for sufficient stats of ModelParameter)
    4. automatically adds summary variables depending on all contained variables
    (e.g. `nll_regul_ind_sum` that depends on all individual latent variables contained)

    <!> For now, you should NOT update a `NamedVariables` with another one, only update with a regular mapping.
    """

    FORBIDDEN_NAMES: ClassVar = frozenset(
        {
            "all",
            "pop",
            "ind",
            "sum",
            "tot",
            "full",
            "nll",
            "attach",
            "regul",
            "state",
            "suff_stats",
        }
    )

    AUTOMATIC_VARS: ClassVar = (
        # TODO? jacobians as well
        "nll_regul_ind_sum_ind",
        "nll_regul_ind_sum",
        # "nll_regul_pop_sum" & "nll_regul_all_sum" are not really relevant so far
        # (because priors for our population variables are NOT true bayesian priors)
        # "nll_regul_pop_sum",
        # "nll_regul_all_sum",
    )

    def __init__(self, *args, **kws):
        self._latent_pop_vars = set()
        self._latent_ind_vars = set()
        super().__init__(*args, **kws)

    def __len__(self):
        return super().__len__() + len(self.AUTOMATIC_VARS)

    def __iter__(self):
        return iter(tuple(self.data) + self.AUTOMATIC_VARS)

    def __setitem__(self, name: VarName, var: VariableInterface) -> None:
        if name in self.FORBIDDEN_NAMES or name in self.AUTOMATIC_VARS:
            raise ValueError(f"Can not use the reserved name '{name}'")
        if name in self.data:
            raise ValueError(f"Can not reset the variable '{name}'")
        super().__setitem__(name, var)
        if isinstance(var, ModelParameter):
            self.update(var.suff_stats.dedicated_variables or {})
        if isinstance(var, LatentVariable):
            self.update(var.get_regularity_variables(name))
            if isinstance(var, PopulationLatentVariable):
                self._latent_pop_vars.add(name)
            else:
                self._latent_ind_vars.add(name)

    def __getitem__(self, name: VarName) -> VariableInterface:
        if name in self.AUTOMATIC_VARS:
            return self._auto_vars[name]
        return super().__getitem__(name)

    @property
    def _auto_vars(self) -> Dict[VarName, LinkedVariable]:
        # TODO? add jacobian as well?
        d = dict(
            # nll_regul_pop_sum=LinkedVariable(
            #     Sum(
            #         *(
            #             f"nll_regul_{pop_var_name}"
            #             for pop_var_name in self._latent_pop_vars
            #         )
            #     )
            # ),
            nll_regul_ind_sum_ind=LinkedVariable(
                Sum(
                    *(
                        f"nll_regul_{ind_var_name}_ind"
                        for ind_var_name in self._latent_ind_vars
                    )
                )
            ),
            nll_regul_ind_sum=LinkedVariable(SumDim("nll_regul_ind_sum_ind")),
            # nll_regul_all_sum=LinkedVariable(
            #     Sum("nll_regul_pop_sum", "nll_regul_ind_sum")
            # ),
        )
        assert d.keys() == set(self.AUTOMATIC_VARS)
        return d