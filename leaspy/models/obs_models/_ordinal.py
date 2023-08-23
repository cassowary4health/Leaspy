import torch

from leaspy.io.data.dataset import Dataset
from leaspy.utils.weighted_tensor import WeightedTensor
from leaspy.variables.specs import VariableInterface
from leaspy.variables.distributions import Ordinal
from leaspy.variables.state import State

from ._base import ObservationModel


class OrdinalObservationModel(ObservationModel):
    ordinal_infos = {}

    def __init__(
            self,
            **extra_vars: VariableInterface,
    ):
        super().__init__(
            name="y",
            getter=self.y_getter,
            dist=Ordinal("model"),
            extra_vars=extra_vars,
        )

    def y_getter(self, state: State) -> WeightedTensor:
        # Why isn't it computed once for all ?
        # pdf = dataset.get_one_hot_encoding(sf=False, ordinal_infos=self.ordinal_infos)
        return state['y']
