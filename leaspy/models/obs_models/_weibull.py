import torch

from leaspy.variables.distributions import WeibullRightCensored
from leaspy.variables.specs import VariableInterface
from leaspy.utils.weighted_tensor import WeightedTensor
from leaspy.utils.weighted_tensor import WeightedTensor, sum_dim
from leaspy.io.data.dataset import Dataset

from ._base import ObservationModel
from leaspy.variables.specs import (
    VarName,
    VariableInterface,
    LinkedVariable,
    DataVariable,
    ModelParameter,
    Collect,
    LVL_FT,
    LVL_IND,
)
from typing import (
    Dict,
    Callable,
    Optional,
    Any,
    Mapping as TMapping,
)
from leaspy.utils.functional import SumDim



class WeibullRightCensoredObservationModel(ObservationModel):
    string_for_json = "weibull-right-censored"

    def __init__(
            self,
            nu: VarName,
            rho: VarName,
            xi: VarName,
            tau: VarName,
            **extra_vars: VariableInterface,
    ):
        super().__init__(
            name="event",
            getter=self.getter,
            dist=WeibullRightCensored(nu, rho, xi, tau),
            extra_vars=extra_vars,
        )

    @staticmethod
    def getter(dataset: Dataset) -> WeightedTensor:
        if dataset.event_time is None or dataset.event_bool is None:
            raise ValueError(
                "Provided dataset is not valid. "
                "Both values and mask should be not None."
            )
        return dataset.event_time, dataset.event_bool

    def get_variables_specs(
        self,
        named_attach_vars: bool = True,
    ) -> Dict[VarName, VariableInterface]:
        """Automatic specifications of variables for this observation model."""
        # TODO change? a bit dirty? possibility of having aliases for variables?
        if named_attach_vars:
            nll_attach_var = f"nll_attach_{self.name}"
        else:
            nll_attach_var = f"nll_attach"
        return {
            self.name: DataVariable(),
            # Dependent vars
            **(self.extra_vars or {}),
            # Attachment variables
            # not really memory efficient nor useful...
            # f"{nll_attach_var}_full": LinkedVariable(self.dist.get_func_nll(self.name)),
            f"{nll_attach_var}_ind": LinkedVariable(
                # SumDim(f"{nll_attach_var}_full", but_dim=LVL_IND)
                self.dist.get_func_nll(self.name)#.then(sum_dim, but_dim=1)
            ),
            nll_attach_var: LinkedVariable(SumDim(f"{nll_attach_var}_ind")),
            # TODO jacobian of {nll_attach_var}_ind_jacobian_{self.name} wrt "y" as well? (for scipy minimize)
        }
