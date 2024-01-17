from __future__ import annotations
import warnings
from typing import List

from leaspy.utils.typing import KwargsType, Tuple, DictParamsTorch
from leaspy.models.base import BaseModel
from leaspy.models.utilities import cast_value_to_tensor
from leaspy.variables.specs import VarName


class GenericModel(BaseModel):
    """
    Generic model (temporary until :class:`.AbstractModel` is really **abstract**).

    Parameters
    ----------
    name : :obj:`str`
        The name of the model.
    **kwargs
        Hyperparameters of the model.

    Attributes
    ----------
    name : :obj:`str`
        The name of the model.
    is_initialized : :obj:`bool`
        ``True`` if the model is initialized, ``False`` otherwise.
    features : :obj:`list` of :obj:`str`
        List of model features (None if not initialization).
    dimension : :obj:`int` (read-only)
        Number of features.
    parameters : :obj:`dict`
        Contains internal parameters of the model.
    """

    def __init__(self, name: str, **kwargs):
        self._parameters: DictParamsTorch = {}
        self._hyperparameters: DictParamsTorch = {}
        super().__init__(name, **kwargs)

    @property
    def parameters(self) -> DictParamsTorch:
        return self._parameters

    def _get_hyperparameters(self) -> DictParamsTorch:
        return {
            **super()._get_hyperparameters(),
            **self._hyperparameters,
        }

    @property
    def parameters_names(self) -> Tuple[VarName, ...]:
        return tuple(self.parameters.keys())

    def _get_hyperparameters_names(self) -> List[VarName]:
        return super()._get_hyperparameters_names() + list(self._hyperparameters.keys())

    def load_parameters(self, parameters: KwargsType) -> None:
        """
        Instantiate or update the model's parameters.

        Parameters
        ----------
        parameters : :obj:`dict`
            Contains the model's parameters.
        """
        for k, v in parameters:
            new_value = cast_value_to_tensor(v)
            if k in self._parameters:
                warnings.warn(
                    f"Parameter {k} was already set in model with value {self._parameters[k]}. "
                    f"Resetting it with new value {new_value}."
                )
            self._parameters[k] = new_value

    def __str__(self):
        lines = [f"=== MODEL {self.name} ==="]
        for hp_name, hp_val in self.hyperparameters.items():
            lines.append(f"{hp_name} : {hp_val}")
        lines.append('-'*len(lines[0]))
        for param_name, param_val in self.parameters.items():
            lines.append(f"{param_name} : {param_val}")

        return "\n".join(lines)
