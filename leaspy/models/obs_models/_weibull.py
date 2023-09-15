import torch

from leaspy.variables.distributions import WeibullRightCensored
from leaspy.variables.specs import VariableInterface
from leaspy.utils.weighted_tensor import WeightedTensor
from leaspy.io.data.dataset import Dataset

from ._base import ObservationModel
from leaspy.variables.specs import (
    VarName,
    VariableInterface,
    LinkedVariable,
    ModelParameter,
    Collect,
    LVL_FT,
)



class WeibullRightCensoredObservationModel(ObservationModel):

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
        if dataset.values is None or dataset.mask is None:
            raise ValueError(
                "Provided dataset is not valid. "
                "Both values and mask should be not None."
            )
        return dataset.event_time, dataset.event_bool
