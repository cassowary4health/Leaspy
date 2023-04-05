from .realization import (
    AbstractRealization,
    IndividualRealization,
    PopulationRealization,
)
from typing import Union
from enum import Enum, auto


class VariableType(Enum):
    INDIVIDUAL = auto()
    POPULATION = auto()


RealizationFactoryInput = Union[str, AbstractRealization, VariableType]


def realization_factory(variable_type: RealizationFactoryInput, **kws) -> AbstractRealization:
    """
    Factory for Realizations.

    Parameters
    ----------
    variable_type : str or AbstractRealization or VariableType
        If an instance of a subclass of AbstractRealization, returns the instance.
        If a string or a VariableType variant, then returns a new instance of the
        appropriate class (with optional parameters `kws`).

    **kws
        Optional parameters for initializing the requested Realization.

    Returns
    -------
    AbstractRealization :
        The desired realization.

    Raises
    ------
    ValueError:
        If variable_type is not supported.
    """
    if isinstance(variable_type, AbstractRealization):
        return variable_type
    if isinstance(variable_type, str):
        variable_type = _validate_string_variable_type(variable_type)
    return _realization_as_variable_type_factory(variable_type, **kws)


def _validate_string_variable_type(variable_type: str) -> VariableType:
    variable_type = variable_type.lower().strip()
    if variable_type == "population":
        return VariableType.POPULATION
    if variable_type == "individual":
        return VariableType.INDIVIDUAL
    raise ValueError(
        f"Invalid variable type {variable_type}"
        f"Possible values are {list(VariableType)}"
    )


def _realization_as_variable_type_factory(variable_type: VariableType, **kws) -> AbstractRealization:
    if variable_type == VariableType.POPULATION:
        return PopulationRealization(**kws)
    if variable_type == VariableType.INDIVIDUAL:
        return IndividualRealization(**kws)
