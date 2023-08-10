import torch

from leaspy.variables.distributions import WeibullRightCensored
from leaspy.variables.specs import VariableInterface
from leaspy.utils.weighted_tensor import WeightedTensor
from leaspy.io.data.dataset import Dataset

from ._base import ObservationModel


class WeibullObservationModel(ObservationModel):

    def __init__(
            self,
            nu_rep: VarName,
            rho: VarName,
            **extra_vars: VariableInterface,
    ):
        super().__init__(
            name="event_shifted",
            getter=self.y_getter,
            dist=WeibullRightCensored(nu_rep, rho),
            extra_vars=extra_vars,
        )

    @staticmethod
    def getter(dataset: Dataset) -> WeightedTensor:
        if dataset.values is None or dataset.mask is None:
            raise ValueError(
                "Provided dataset is not valid. "
                "Both values and mask should be not None."
            )
        return WeightedTensor(dataset.values, weight=dataset.mask.to(torch.bool))
