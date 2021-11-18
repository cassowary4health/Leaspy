from abc import ABC, abstractmethod

import torch

from leaspy.exceptions import LeaspyModelInputError
from leaspy.utils.typing import DictParamsTorch, ParamType, Tuple


class AbstractAttributes(ABC):
    """
    Abstract base class for attributes of models.

    Contains the common attributes & methods of the different attributes classes.
    Such classes are used to update the models' attributes.

    Parameters
    ----------
    name : str
    dimension : int (default None)
    source_dimension : int (default None)

    Attributes
    ----------
    name : str
        Name of the associated leaspy model.
    dimension : int
        Number of features of the model
    source_dimension : int
        Number of sources of the model
        TODO? move to AbstractManifoldModelAttributes?
    univariate : bool
        Whether model is univariate or not (i.e. dimension == 1)
    has_sources : bool
        Whether model has sources or not (not univariate and source_dimension >= 1)
        TODO? move to AbstractManifoldModelAttributes?
    update_possibilities : tuple[str] (default empty)
        Contains the available parameters to update. Different models have different parameters.

    Raises
    ------
    :exc:`.LeaspyModelInputError`
        if any inconsistent parameter.
    """

    def __init__(self, name: str, dimension: int = None, source_dimension: int = None):

        if not isinstance(name, str):
            raise LeaspyModelInputError("In model attributes, you must provide a string for the parameter `name`.")
        self.name = name

        if not isinstance(dimension, int):
            raise LeaspyModelInputError("In model attributes, you must provide an integer for the parameter `dimension`.")
        self.dimension = dimension
        self.univariate = dimension == 1

        self.source_dimension = source_dimension
        self.has_sources = bool(source_dimension) # False iff None or == 0

        if self.univariate and self.has_sources:
            raise LeaspyModelInputError("Inconsistent attributes: presence of sources for a univariate model.")

        self.update_possibilities: Tuple[ParamType, ...] = () # empty tuple

    @abstractmethod
    def get_attributes(self) -> Tuple[torch.FloatTensor, ...]:
        """
        Returns the essential attributes of a given model.

        Returns
        -------
        Depends on the subclass, please refer to each specific class.
        """
        pass

    @abstractmethod
    def update(self, names_of_changes_values: Tuple[ParamType, ...], values: DictParamsTorch) -> None:
        """
        Update model group average parameter(s).

        Parameters
        ----------
        names_of_changed_values : list [str]
           Values to be updated
        values : dict [str, `torch.Tensor`]
           New values used to update the model's group average parameters

        Raises
        ------
        :exc:`.LeaspyModelInputError`
            If `names_of_changed_values` contains unknown values to update.
        """
        pass

    def _check_names(self, names_of_changed_values: Tuple[ParamType, ...]):
        """
        Check if the name of the parameter(s) to update are in the possibilities allowed by the model.

        Parameters
        ----------
        names_of_changed_values : list [str]

        Raises
        ------
        :exc:`.LeaspyModelInputError`
            If `names_of_changed_values` contains unknown values to update.
        """
        unknown_update_possibilities = set(names_of_changed_values).difference(self.update_possibilities)
        if len(unknown_update_possibilities) > 0:
            raise LeaspyModelInputError(f"{unknown_update_possibilities} not in the attributes that can be updated")
