import torch

from leaspy.variables.distributions import Bernoulli
from leaspy.variables.specs import VariableInterface
from leaspy.utils.weighted_tensor import WeightedTensor
from leaspy.variables.state import State

from ._base import ObservationModel


class BernoulliObservationModel(ObservationModel):

    def __init__(
        self,
        **extra_vars: VariableInterface,
    ):
        super().__init__(
            name="y",
            getter=self.y_getter,
            dist=Bernoulli("model"),
            extra_vars=extra_vars,
        )

    @staticmethod
    def y_getter(state: State) -> WeightedTensor:
        if state['y'] is None:
            raise ValueError(
                "Provided dataset is not valid. "
                "Both values and mask should be not None."
            )
        return state['y']
