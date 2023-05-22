from __future__ import annotations

from collections.abc import MutableMapping
from typing import Tuple, Optional
from contextlib import contextmanager
from enum import Enum, auto
import copy

import torch

from leaspy.variables.specs import (
    VarName,
    VarValue,
    VariablesLazyValuesRO,
    VariablesLazyValuesRW,
    Hyperparameter,
)
from leaspy.utils.functional import unsqueeze_right
from leaspy.variables.dag import VariablesDAG
from leaspy.exceptions import LeaspyInputError


class StateForkType(Enum):
    """
    The strategy used to cache forked values in :class:`.State`.

    REF : caching using references
    COPY : caching using deepcopy

    Notes
    -----
    If using `REF` beware that values will NOT be copied (it only keeps references of values),
    so do NOT mutate them directly or the behavior will be unexpected.
    """

    REF = auto()
    COPY = auto()

    def to_cache(self, d: VariablesLazyValuesRW) -> VariablesLazyValuesRO:
        """Get the values to cache, depending on forking type."""
        if self is self.REF:
            return d
        return {k: copy.deepcopy(v) for k, v in d.items()}


class State(MutableMapping):
    """
    Dictionary of cached values corresponding to the stateless DAG instance.

    Parameters
    ----------
    dag : VariablesDAG
        The stateless DAG which state will hold values for.
    auto_fork_type : :class:`.StateForkType` or None (default)
        Refer to :class:`.StateForkType` class and :attr:`auto_fork_type`

    Attributes
    ----------
    dag : VariablesDAG
        The stateless DAG which the state instante will hold values for.
    auto_fork_type : :class:`.StateForkType` or None
        If not `StateForkType.NONE` each dictionary assignment will lead to the partial caching
        of previous value and all its children, so they can be reverted without computation.
        The exact caching strategy depends on flag (caching by reference or by copy)
        Can be manually set or via `auto_fork` context manager.
    _values : MutableMapping[VarName, Optional[VarValue]]
        Private cache for values (computations are lazy thus some values may be None).
        All not None values are always self-consistent with respect to DAG dependencies.
    _last_fork : None or Mapping[VarName, Optional[VarValue]]
        If not None, holds the previous partial state values so they may be `.revert()`.
        Automatically populated on assignment operations as soon as `auto_fork_type` is not `NONE`.
        Example: if you set a new value for `a`, then value of `a` and of all its children just before assignment
        are held until either reversion or a new assignment.
    """

    def __init__(
        self, dag: VariablesDAG, *, auto_fork_type: Optional[StateForkType] = None
    ):
        self.dag = dag
        self.auto_fork_type = auto_fork_type
        self.clear()

    def clear(self) -> None:
        """Reset last forked state and reset all values to their canonical values."""
        self._values: VariablesLazyValuesRW = {
            n: var.value if isinstance(var, Hyperparameter) else None
            for n, var in self.dag.items()
        }
        self._last_fork: Optional[VariablesLazyValuesRO] = None

    def clone(self) -> State:
        """Clone current state (no copy of DAG)."""
        cloned = State(self.dag, auto_fork_type=self.auto_fork_type)
        cloned._values = copy.deepcopy(self._values)
        cloned._last_fork = copy.deepcopy(self._last_fork)
        return cloned

    @contextmanager
    def auto_fork(self, type: Optional[StateForkType] = StateForkType.REF):
        """Provide a context manager interface with temporary `auto_fork_type` set to `type`."""
        orig_auto_fork_type = self.auto_fork_type
        try:
            self.auto_fork_type = type
            yield
        finally:
            self.auto_fork_type = orig_auto_fork_type

    def __iter__(self):
        """Iterates on keys (.keys(), .values() and .items() methods are automatically provided by `MutableMapping`)."""
        return iter(self._values)

    def __len__(self) -> int:
        """Get number of variables."""
        return len(self._values)

    def _check_key_exists(self, k: VarName) -> None:
        if k not in self.dag:
            raise LeaspyInputError(f"'{k}' is not a valid variable")

    def _get_or_compute_and_cache(
        self,
        k: VarName,
        *,
        force_computation: bool = False,
        why: str = " to proceed",
    ) -> VarValue:
        """Retrieve cached value (unless `force_computation`) or compute it, assuming node exists and all its ancestors have cached values."""
        if not force_computation:
            val = self._values[k]
            if val is not None:
                return val

        val = self.dag[k].compute(self._values)
        if val is None:
            raise LeaspyInputError(
                f"'{k}' is an independent variable which is required{why}"
            )
        self._values[k] = val
        return val

    def __getitem__(self, k: VarName) -> VarValue:
        """Retrieve cached variable value or compute it and cache it (as well as all intermediate computations that were needed)."""
        self._check_key_exists(k)
        val = self._values[k]
        if val is not None:
            return val
        for a in self.dag.sorted_ancestors[k]:
            self._get_or_compute_and_cache(a, why=f" to get '{k}'")
        return self._get_or_compute_and_cache(k, force_computation=True)

    def __setitem__(self, k: VarName, v: Optional[VarValue]) -> None:
        """Smart and protected assignment of a variable value."""
        self._check_key_exists(k)
        if not self.dag[k].is_settable:
            raise LeaspyInputError(f"'{k}' is not intended to be set")
        sorted_children = self.dag.sorted_children[k]
        # automatically fork partial state to easily revert it
        if self.auto_fork_type is not None:
            self._last_fork = self.auto_fork_type.to_cache(
                {c: self._values[c] for c in (k,) + sorted_children}
            )
        # TODO? we do not "validate" / "check" input data for now (it could be a stateless variable method) to remain light
        self._values[k] = v
        # we reset values of all children of the node we just assigned a value to
        # (we postpone the evaluation of their new values when they will really be needed)
        for c in sorted_children:
            self._values[c] = None

    def put(
        self,
        k: VarName,
        v: torch.Tensor,
        *,
        indices: Tuple[int, ...] = (),
        accumulate: bool = False,
    ) -> None:
        """Smart and protected assignment of a variable value, but potentially on a subset of indices, adding (accumulating) values and OUT-OF-PLACE."""
        if indices == ():
            # `torch.index_put` is not working in this case.
            if not accumulate:
                self[k] = v
            else:
                self[k] = self[k] + v
            return
        # For now: no optimization for partial indices operations
        torch_indices = tuple(map(torch.tensor, indices))
        self[k] = self[k].index_put(
            indices=torch_indices, values=v, accumulate=accumulate
        )

    def __delitem__(self, k: VarName) -> None:
        raise NotImplementedError("Key removal is not allowed")

    def precompute_all(self) -> None:
        """Pre-compute all values of the graph (assuming leaves already have valid values)."""
        for n in self.dag:
            self._get_or_compute_and_cache(n)

    # def reset_to_admissible(self) -> None:
    #    """Reset all standard variables to their frozen or admissible values and pre-compute all other variables (forked state is cleared)."""
    #    # TODO: more generic?
    #    self.clear()
    #    for n, var in self.dag.items():
    #        if isinstance(var, StandardVariable):
    #            self._values[n] = var.admissible_value
    #    self.precompute_all()

    def revert(
        self, subset: Optional[torch.Tensor] = None, *, right_broadcasting: bool = True
    ) -> None:
        """
        Revert state to previous forked state, efficiently (forked state is then reset).

        Parameters
        ----------
        subset : None (default) or VarValue[bool]
            If not None, the reversion is only partial:
            * subset = True <=> revert previous state for those indices
            * subset = False <=> keep current state for those indices
            <!> User is responsible for having tensor values that are consistent with `subset` shape (i.e. valid broadcasting)
            for the forked node and all of its children.
           <!> When the current OR forked state is not set (value = None) on a particular node of forked DAG,
           then the reverted result is always None.
        right_broadcasting : bool (default True)
            If True and if `subset` is not None, then the subset of indices to revert uses right-broadcasting,
            instead of the standard left-broadcasting.
        """
        if self._last_fork is None:
            raise LeaspyInputError(
                "No forked state to revert from, "
                "please use within `.auto_fork()` context, or set `.auto_fork_type` to  `StateForkType.REF` or `StateForkType.COPY`."
            )

        if subset is None:
            self._values.update(self._last_fork)
            self._last_fork = None
            return

        to_revert = subset.to(torch.bool)
        to_keep = ~to_revert
        for k, old_v in self._last_fork.items():
            cur_v = self._values[k]
            if old_v is None or cur_v is None:
                self._values[k] = None
            else:
                assert (
                    old_v.shape == cur_v.shape
                ), f"Bad shapes for {k}: {old_v.shape} != {cur_v.shape}"
                if right_broadcasting:
                    add_ndim = max(old_v.ndim - to_revert.ndim, 0)
                    self._values[k] = (
                        old_v * unsqueeze_right(to_revert, ndim=add_ndim)
                        + cur_v * unsqueeze_right(to_keep, ndim=add_ndim)
                    )
                else:
                    self._values[k] = old_v * to_revert + cur_v * to_keep
        self._last_fork = None

    def to_device(self, device: torch.device) -> None:
        """
        Move values to the specified device (in-place).

        Parameters
        ----------
        device : torch.device
        """
        for k, v in self._values.items():
            if v is not None:
                self._values[k] = v.to(device=device)
        if self._last_fork is not None:
            for k, v in self._last_fork.items():
                if v is not None:
                    self._last_fork[k] = v.to(device=device)
