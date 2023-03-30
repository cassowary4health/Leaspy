"""Defines the noise model factory."""

from typing import Union, Tuple, Optional, Type

from leaspy.models.noise_models import NOISE_MODELS, BaseNoiseModel
from leaspy.exceptions import LeaspyModelInputError
from leaspy.utils.typing import KwargsType

NoiseModelFactoryInput = Union[str, BaseNoiseModel, KwargsType]


def _noise_model_class(name: str) -> Type[BaseNoiseModel]:
    """
    Get noise-model class from its code name.

    Parameters
    ----------
    name : str
        Code name of the desired noise model.

    Returns
    -------
    BaseNoiseModel :
        The requested noise model class.

    Raises
    ------
    LeaspyModelInputError
        If the noise model name is not supported.
    """
    name = name.lower().replace("_", "-")
    kls = NOISE_MODELS.get(name, None)
    if kls is None:
        raise LeaspyModelInputError(
            f"Noise model '{name}' is not supported. "
            f"Supported noise models are {set(NOISE_MODELS)}"
        )
    return kls


def _noise_model_name(kls: Type[BaseNoiseModel]) -> str:
    """
    Get code name of a noise-model class.

    Parameters
    ----------
    kls : BaseNoiseModel
        The noise model class from which to retrieve the code name.

    Returns
    -------
    str :
        The code name for the provided noise model class.

    Raises
    ------
    NotImplementedError
        If the name does not match any implemented noise model.
    """
    name = {v: k for k, v in NOISE_MODELS.items()}.get(kls, None)
    if name is None:
        raise NotImplementedError(
            f"Your noise model ({kls}) is not registered in `NOISE_MODELS`."
        )
    return name


def _noise_model_kwargs_to_params_hyperparams(
    kls: Type[BaseNoiseModel],
    kws: Optional[KwargsType],
) -> Tuple[Optional[KwargsType], KwargsType]:
    """
    Split the input keyword arguments as a tuple
    (parameters: dict | None, hyperparameters: dict).

    Parameters
    ----------
    kls : BaseNoiseModel
        The noise model class for which to split the input kwargs.
    kws : KwargsType, optional
        The input kwargs to be split.

    Returns
    -------
    tuple :
        The parameters and the hyperparameters as a tuple.
    """
    if kws is None:
        return None, {}
    params = {k: v for k, v in kws.items() if k in kls.free_parameters}
    hyperparams = {k: v for k, v in kws.items() if k not in kls.free_parameters}

    return params or None, hyperparams


def noise_model_export(noise_model: BaseNoiseModel) -> KwargsType:
    """
    Serialize a given BaseNoiseModel as a dictionary.

    Parameters
    ----------
    noise_model : BaseNoiseModel
        The noise model to serialize.

    Returns
    -------
    KwargsType :
        The noise model serialized as a dict.
    """
    return dict(
        name=_noise_model_name(noise_model.__class__),
        **noise_model.to_dict(),
    )


def noise_model_factory(noise_model: NoiseModelFactoryInput, **kws) -> BaseNoiseModel:
    """
    Factory for noise models.

    Parameters
    ----------
    noise_model : str or BaseNoiseModel or dict[str, ...]
        If an instance of a subclass of BaseNoiseModel, returns the instance.
        If a string, then returns a new instance of the appropriate class (with optional parameters `kws`).
        If a dictionary, it must contain the 'name' key and other initialization parameters.
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

    if isinstance(noise_model, str):
        kls = _noise_model_class(noise_model)
        kws = kws or None

    elif isinstance(noise_model, dict) and "name" in noise_model:
        kls = _noise_model_class(noise_model.pop("name"))
        # the optional keyword-arguments will overwrite the stored noise_model parameters and hyperparams.
        kws = {**noise_model, **kws} or None

    else:
        raise LeaspyModelInputError(
            "The provided `noise_model` should be a valid instance of `BaseNoiseModel`, a string "
            f"among {set(NOISE_MODELS)}, or a dictionary with 'name' being one of the previous options."
        )

    params, hyperparams = _noise_model_kwargs_to_params_hyperparams(kls, kws)
    return kls(params, **hyperparams)
