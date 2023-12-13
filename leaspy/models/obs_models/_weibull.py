import torch

from leaspy.variables.distributions import WeibullRightCensored, WeibullRightCensoredWithSources
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



class AbstractWeibullRightCensoredObservationModel(ObservationModel):

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

        specs = super().get_variables_specs(named_attach_vars)

        nll_attach_var = self.get_nll_attach_var_name(
            named_attach_vars
        )
        specs[f"{nll_attach_var}_ind"] = LinkedVariable(
            self.dist.get_func_nll(self.name)
        )
        specs[f"log_survival_{self.name}"] = LinkedVariable(
            self.dist._get_func("compute_log_survival", self.name)
        )
        specs[f"log_hazard_{self.name}"] = LinkedVariable(
            self.dist._get_func("compute_log_likelihood_hazard", self.name)
        )

        return specs


class WeibullRightCensoredObservationModel(AbstractWeibullRightCensoredObservationModel):
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

    @classmethod
    def default_init(self, **kwargs):
        return self(nu = kwargs.pop("nu", "nu"),
                    rho = kwargs.pop("rho", "rho"),
                    xi = kwargs.pop("xi", "xi"),
                    tau = kwargs.pop("tau", "tau"))

class WeibullRightCensoredWithSourcesObservationModel(AbstractWeibullRightCensoredObservationModel):
    string_for_json = ("weibull-right-censored-with-sources")

    def __init__(
            self,
            nu: VarName,
            rho: VarName,
            xi: VarName,
            tau: VarName,
            zeta: VarName,
            sources: VarName,
            **extra_vars: VariableInterface,
    ):
        super().__init__(
            name="event",
            getter=self.getter,
            dist=WeibullRightCensoredWithSources(nu, rho, xi, tau, zeta, sources),
            extra_vars=extra_vars,
        )

    @classmethod
    def default_init(self, **kwargs):
        return self(nu = kwargs.pop("nu", "nu"),
                    rho = kwargs.pop("rho", "rho"),
                    xi = kwargs.pop("xi", "xi"),
                    tau = kwargs.pop("tau", "tau"),
                    zeta=kwargs.pop("zeta", "zeta"),
                    sources = kwargs.pop("sources", "sources"),)