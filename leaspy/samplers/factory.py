from typing import Union, Type
from leaspy.exceptions import LeaspyAlgoInputError
from .base import (
    AbstractSampler,
    AbstractIndividualSampler,
    AbstractPopulationSampler,
)
from leaspy.samplers import INDIVIDUAL_SAMPLERS, POPULATION_SAMPLERS
from leaspy.io.realizations import VariableType


SamplerFactoryInput = Union[str, AbstractSampler]


def sampler_factory(sampler: SamplerFactoryInput, variable_type: VariableType, **kwargs) -> AbstractSampler:
    """
    Factory for Samplers.

    Parameters
    ----------
    sampler : :class:`.AbstractSampler` or :obj:`str`
        If an instance of a subclass of :class:`.AbstractSampler`, returns the instance (no copy).
        If a string, returns a new instance of the appropriate class (with optional parameters `kwargs`).

    variable_type : :class:`.VariableType`
        The type of random variable that the sampler is supposed to sample.

    **kwargs
        Optional parameters for initializing the requested Sampler
        (not used if input is a subclass of :class:`.AbstractSampler`).

    Returns
    -------
    :class:`.AbstractSampler` :
        The desired sampler.

    Raises
    ------
    :exc:`.LeaspyAlgoInputError`:
        If the sampler provided is not supported.
    """
    if isinstance(sampler, AbstractSampler):
        return sampler
    if isinstance(sampler, str):
        kls = _get_sampler_class(sampler, variable_type)
        return kls(**kwargs)
    raise LeaspyAlgoInputError(
        "The provided `sampler` should be a valid instance of `AbstractSampler`, or a string "
        f"among {set(INDIVIDUAL_SAMPLERS).union(POPULATION_SAMPLERS)}."
    )


def _get_sampler_class(sampler_name: str, variable_type: VariableType):
    if variable_type == VariableType.INDIVIDUAL:
        return _get_individual_sampler_class(sampler_name)
    if variable_type == VariableType.POPULATION:
        return _get_population_sampler_class(sampler_name)


def _get_individual_sampler_class(sampler_name: str) -> Type[AbstractIndividualSampler]:
    sampler_name = sampler_name.lower().replace("_", "-")
    kls = INDIVIDUAL_SAMPLERS.get(sampler_name, None)
    if kls is None:
        raise LeaspyAlgoInputError(
            f"Individual sampler '{sampler_name}' is not supported. "
            f"Supported samplers for individual variables are {set(INDIVIDUAL_SAMPLERS)}"
        )
    return kls


def _get_population_sampler_class(sampler_name: str) -> Type[AbstractPopulationSampler]:
    sampler_name = sampler_name.lower().replace("_", "-")
    kls = POPULATION_SAMPLERS.get(sampler_name, None)
    if kls is None:
        raise LeaspyAlgoInputError(
            f"Population sampler '{sampler_name}' is not supported. "
            f"Supported samplers for population variables are {set(POPULATION_SAMPLERS)}"
        )
    return kls