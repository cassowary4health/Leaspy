"""Defines the noise model factory."""

from typing import Union, Type, Dict
from enum import Enum

from leaspy.exceptions import LeaspyModelInputError

from ._base import ObservationModel
from ._gaussian import FullGaussianObservationModel
from ._bernoulli import BernoulliObservationModel
from ._ordinal import OrdinalObservationModel
from ._weibull import WeibullRightCensoredObservationModel


class ObservationModelNames(Enum):
    """Enumeration defining the possible names for observation models."""
    GAUSSIAN_DIAGONAL = "gaussian-diagonal"
    GAUSSIAN_SCALAR = "gaussian-scalar"
    BERNOULLI = "bernoulli"
    ORDINAL = "ordinal"
    WEIBULL_RIGHT_CENSORED = "weibull-right-censored"

    @classmethod
    def from_string(cls, model_name: str):
        try:
            return cls(model_name.lower().replace("_", "-"))
        except ValueError:
            raise NotImplementedError(
                f"The requested ObservationModel {model_name} is not implemented. "
                f"Valid observation model names are: {[elt.value for elt in cls]}."
            )


ObservationModelFactoryInput = Union[str, ObservationModelNames, ObservationModel]

OBSERVATION_MODELS: Dict[ObservationModelNames, Type[ObservationModel]] = {
    ObservationModelNames.GAUSSIAN_DIAGONAL: FullGaussianObservationModel,
    ObservationModelNames.GAUSSIAN_SCALAR: FullGaussianObservationModel,
    ObservationModelNames.BERNOULLI: BernoulliObservationModel,
    ObservationModelNames.ORDINAL: OrdinalObservationModel,
    ObservationModelNames.WEIBULL_RIGHT_CENSORED: WeibullRightCensoredObservationModel,
}


def observation_model_factory(model: ObservationModelFactoryInput, **kwargs) -> ObservationModel:
    """
    Factory for observation models.

    Parameters
    ----------
    model : :obj:`str` or :class:`.ObservationModel` or :obj:`dict` [ :obj:`str`, ...]
        - If an instance of a subclass of :class:`.ObservationModel`, returns the instance.
        - If a string, then returns a new instance of the appropriate class (with optional parameters `kws`).
        - If a dictionary, it must contain the 'name' key and other initialization parameters.
    **kwargs
        Optional parameters for initializing the requested observation model when a string.

    Returns
    -------
    :class:`.ObservationModel` :
        The desired observation model.

    Raises
    ------
    :exc:`.LeaspyModelInputError` :
        If `model` is not supported.
    """
    dimension = kwargs.pop("dimension", None)
    if isinstance(model, ObservationModel):
        return model
    if isinstance(model, str):
        model = ObservationModelNames.from_string(model)
    if isinstance(model, ObservationModelNames):
        if model == ObservationModelNames.GAUSSIAN_DIAGONAL:
            if dimension is None:
                raise NotImplementedError(
                    "WIP: dimension / features should be provided to "
                    f"init the obs_model = {ObservationModelNames.GAUSSIAN_DIAGONAL}."
                )
            return FullGaussianObservationModel.with_noise_std_as_model_parameter(dimension)
        if model == ObservationModelNames.GAUSSIAN_SCALAR:
            return FullGaussianObservationModel.with_noise_std_as_model_parameter(1)
        return OBSERVATION_MODELS[model](**kwargs)
    raise LeaspyModelInputError(
        "The provided `model` should be a valid instance of `ObservationModel`, a string "
        f"among {[c.value for c in ObservationModelNames]}."
    )
