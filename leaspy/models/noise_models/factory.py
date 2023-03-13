"""Defines the noise model factory."""

from typing import Union

from .base import BaseNoiseModel
from .bernoulli import BernoulliNoiseModel
from .gaussian import GaussianScalarNoiseModel, GaussianDiagonalNoiseModel
from .ordinal import OrdinalNoiseModel, OrdinalRankingNoiseModel

from leaspy.exceptions import LeaspyModelInputError

NOISE_MODELS = {
    "bernoulli": BernoulliNoiseModel,
    "gaussian-scalar": GaussianScalarNoiseModel,
    "gaussian-diagonal": GaussianDiagonalNoiseModel,
    "ordinal": OrdinalNoiseModel,
    "ordinal-ranking": OrdinalRankingNoiseModel,
}


def noise_model_factory(noise_model: Union[str, BaseNoiseModel], **kws) -> BaseNoiseModel:
    """
    Factory for noise models.

    Parameters
    ----------
    noise_model : str or BaseNoiseModel
        If an instance of a subclass of BaseNoiseModel, returns the instance.
        If a string, then returns a new instance of the appropriate class (with optional parameters `kws`).
    **kws
        Optional parameters for initializing the requested noise-model when a string.

    Returns
    -------
    BaseNoiseModel :
        The desired noise model.

    Raises
    ------
    LeaspyModelInputError:
        If noise_model is not supported.
    """
    if isinstance(noise_model, BaseNoiseModel):
        return noise_model
    if not isinstance(noise_model, str):
        raise LeaspyModelInputError(
            "The provided `noise_model` should be a valid instance of `BaseNoiseModel`, "
            f"or a string among {set(NOISE_MODELS.keys())}"
        )
    noise_model = noise_model.lower()
    noise_model = noise_model.replace("_", "-")
    if noise_model not in NOISE_MODELS:
        raise LeaspyModelInputError(
            f"Noise model {noise_model} is not supported."
            f"Supported noise models are {set(NOISE_MODELS.keys())}"
        )
    return NOISE_MODELS[noise_model](**kws)
