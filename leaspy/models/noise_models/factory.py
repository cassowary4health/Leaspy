"""Defines the noise model factory."""

from typing import Union

from .base import BaseNoiseModel
from .bernoulli import BernoulliNoiseModel
from .gaussian import GaussianScalarNoiseModel, GaussianDiagonalNoiseModel
from .ordinal import OrdinalNoiseModel, OrdinalRankingNoiseModel


NOISE_MODELS = {
    "bernoulli": BernoulliNoiseModel,
    "gaussian-scalar": GaussianScalarNoiseModel,
    "gaussian-diagonal": GaussianDiagonalNoiseModel,
    "ordinal": OrdinalNoiseModel,
    "ordinal-ranking": OrdinalRankingNoiseModel,
}


def noise_model_factory(noise_model: Union[str, BaseNoiseModel]) -> BaseNoiseModel:
    """
    Factory for noise models.

    Parameters
    ----------
    noise_model : str or BaseNoiseModel
        If an instance of a subclass of BaseNoiseModel, returns the instance.
        If a string, then return the appropriate class.

    Returns
    -------
    BaseNoiseModel :
        The desired noise model.

    Raises
    ------
    ValueError:
        If noise_model is not supported.
    """
    if isinstance(noise_model, BaseNoiseModel):
        return noise_model
    noise_model = noise_model.lower()
    noise_model = noise_model.replace("_", "-")
    try:
        return NOISE_MODELS[noise_model]
    except KeyError:
        raise ValueError(
            f"Noise model {noise_model} is not supported."
            f"Supported noise models are : {NOISE_MODELS.keys()}"
        )
