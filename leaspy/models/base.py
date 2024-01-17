from abc import ABC, abstractmethod
import warnings
from enum import Enum
import torch
import json
from inspect import signature
import pandas as pd

from leaspy.utils.typing import FeatureType, List, Optional, DictParams, Tuple, KwargsType, DictParamsTorch
from leaspy.exceptions import LeaspyModelInputError, LeaspyInputError
from leaspy.io.data.dataset import Dataset
from leaspy.variables.specs import VarName


class InitializationMethod(str, Enum):
    DEFAULT = "default"
    RANDOM = "random"


class BaseModel(ABC):
    """
    Base model class from which all ``Leaspy`` models should inherit.

    It defines the interface that a model should implement to be
    compatible with ``Leaspy``.

    Parameters
    ----------
    name : :obj:`str`
        The name of the model.

    **kwargs
        Hyperparameters of the model

    Attributes
    ----------
    name : :obj:`str`
        The name of the model.

    is_initialized : :obj:`bool`
        ``True``if the model is initialized, ``False`` otherwise.

    features : :obj:`list` of :obj:`str`
        List of model features (``None`` if not initialization).

    dimension : :obj:`int`
        Number of features.
    """

    def __init__(self, name: str, **kwargs):
        self.is_initialized: bool = False
        self.name = name
        self._features: Optional[List[FeatureType]] = None
        self._dimension: Optional[int] = None
        self._set_hyperparameters(kwargs)

    @property
    def settable_hyperparameters(self) -> List[str]:
        """Returns a list of settable hyperparameter names."""
        return self._get_settable_hyperparameters()

    def _get_settable_hyperparameters(self) -> List[str]:
        """BaseModel has only features and dimension as settable hyperparameters."""
        return ["features", "dimension"]

    @property
    def features(self) -> Optional[List[FeatureType]]:
        return self._features

    @features.setter
    def features(self, features: Optional[List[FeatureType]]):
        """
        Features setter.
        Ensure coherence between dimension and features attributes.
        """
        if features is None:
            # used to reset features
            self._features = None
            return

        if self.dimension is not None and len(features) != self.dimension:
            raise LeaspyModelInputError(
                f"Cannot set the model's features to {features}, because "
                f"the model has been configured with a dimension of {self.dimension}."
            )
        self._features = features

    @property
    def dimension(self) -> Optional[int]:
        """
        The dimension of the model.
        If the private attribute is defined, then it takes precedence over the feature length.
        The associated setters are responsible for their coherence.
        """
        if self._dimension is not None:
            return self._dimension
        if self.features is not None:
            return len(self.features)
        return None

    @dimension.setter
    def dimension(self, dimension: int):
        """
        Dimension setter.
        Ensures coherence between dimension and feature attributes.
        """
        if self.features is not None and len(self.features) != dimension:
            raise LeaspyModelInputError(
                f"Model has {len(self.features)} features. Cannot set the dimension to {dimension}."
            )
        self._dimension = dimension

    @property
    @abstractmethod
    def parameters_names(self) -> Tuple[VarName, ...]:
        raise NotImplementedError

    @property
    @abstractmethod
    def parameters(self) -> DictParamsTorch:
        raise NotImplementedError

    @abstractmethod
    def load_parameters(self, parameters: KwargsType) -> None:
        raise NotImplementedError

    @property
    def hyperparameters_names(self) -> Tuple[VarName, ...]:
        return tuple(self._get_hyperparameters_names())

    def _get_hyperparameters_names(self) -> List[VarName]:
        """
        Return the names of the model's hyperparameters.
        This method only retrieve the hyperparameters stored as model attributes
        like features, dimension, source_dimension...
        Stateful models need to overwrite this method to add the hyperparameters
        stored in the state.
        """
        return [
            hp for hp in self.settable_hyperparameters
            if hasattr(self, hp) and getattr(self, hp) is not None
        ]

    @property
    def hyperparameters(self) -> DictParamsTorch:
        return self._get_hyperparameters()

    def _get_hyperparameters(self) -> DictParamsTorch:
        """
        Returns a dictionary of the model's hyperparameters.
        """
        return {
            name: getattr(self, name)
            for name in self.settable_hyperparameters
            if hasattr(self, name) and getattr(self, name) is not None
        }

    def _set_hyperparameters(self, hyperparameters: KwargsType) -> None:
        """
        Updates all model hyperparameters from the provided hyperparameters.

        Parameters
        ----------
        hyperparameters : KwargsType
            The hyperparameters to be loaded.
        """
        for param in self.settable_hyperparameters:
            if param in hyperparameters:
                setattr(self, param, hyperparameters[param])
        self._raise_if_unknown_hyperparameters(hyperparameters)

    def _raise_if_unknown_hyperparameters(self, hyperparameters: KwargsType) -> None:
        """
        Raises a :exc:`.LeaspyModelInputError` if any unknown hyperparameter is provided to the model.
        """
        if len(unexpected_hyperparameters := set(hyperparameters.keys()).difference(set(self.settable_hyperparameters))) > 0:
            raise LeaspyModelInputError(
                f"Only {self.settable_hyperparameters} are valid hyperparameters for {self.__qualname__}. "
                f"Unknown hyperparameters provided: {unexpected_hyperparameters}."
            )

    def initialize(self, dataset: Optional[Dataset] = None, method: Optional[InitializationMethod] = None) -> None:
        """
        Initialize the model given a :class:`.Dataset` and an initialization method.

        After calling this method :attr:`is_initialized` should be ``True`` and model
        should be ready for use.

        Parameters
        ----------
        dataset : :class:`.Dataset`, optional
            The dataset we want to initialize from.
        method : InitializationMethod, optional
            A custom method to initialize the model
        """
        method = InitializationMethod(method or InitializationMethod.DEFAULT)
        if self.is_initialized and self.features is not None:
            # we also test that self.features is not None, since for `ConstantModel`:
            # `is_initialized`` is True but as a mock for being personalization-ready,
            # without really being initialized!
            warn_msg = '<!> Re-initializing an already initialized model.'
            if dataset and dataset.headers != self.features:
                warn_msg += (
                    f" Overwritting previous model features ({self.features}) "
                    f"with new ones ({dataset.headers})."
                )
                self.features = None  # wait validation of compatibility to store new features
            warnings.warn(warn_msg)
        self._validate_compatibility_of_dataset(dataset)
        self.features = dataset.headers if dataset else None
        self.is_initialized = True

    def _validate_compatibility_of_dataset(self, dataset: Optional[Dataset] = None) -> None:
        """
        Raise if the given :class:`.Dataset` is not compatible with the current model.

        Parameters
        ----------
        dataset : :class:`.Dataset`, optional
            The :class:`.Dataset` we want to model.

        Raises
        ------
        :exc:`.LeaspyModelInputError` :
            - If the :class:`.Dataset` has a number of dimensions smaller than 2.
            - If the :class:`.Dataset` does not have the same dimensionality as the model.
            - If the :class:`.Dataset`'s headers do not match the model's.
        """
        if not dataset:
            return
        if self.dimension is not None and dataset.dimension != self.dimension:
            raise LeaspyModelInputError(
                f"Unmatched dimensions: {self.dimension} (model) ≠ {dataset.dimension} (data)."
            )
        if self.features is not None and dataset.headers != self.features:
            raise LeaspyModelInputError(
                f"Unmatched features: {self.features} (model) ≠ {dataset.headers} (data)."
            )

    def _get_dataframe_from_dataset(self, dataset: Dataset) -> pd.DataFrame:
        df = dataset.to_pandas().dropna(how='all').sort_index()[dataset.headers]
        if not df.index.is_unique:
            raise LeaspyInputError("Index of DataFrame is not unique.")
        if not df.index.to_frame().notnull().all(axis=None):
            raise LeaspyInputError("Index of DataFrame contains unvalid values.")
        if self.features != df.columns.tolist():
            raise LeaspyInputError(
                f"Features mismatch between model and dataset: {self.features} != {df.columns}"
            )
        return df

    def save(self, path: str, **kwargs) -> None:
        """
        Save ``Leaspy`` object as json model parameter file.

        Parameters
        ----------
        path : :obj:`str`
            Path to store the model's parameters.
        **kwargs
            Keyword arguments for :meth:`.AbstractModel.to_dict` child method
            and ``json.dump`` function (default to indent=2).
        """
        export_kws = {k: kwargs.pop(k) for k in signature(self.to_dict).parameters if k in kwargs}
        model_settings = self.to_dict(**export_kws)
        kwargs = {"indent": 2, **kwargs}
        with open(path, 'w') as fp:
            json.dump(model_settings, fp, **kwargs)

    def to_dict(self, **kwargs) -> KwargsType:
        """
        Export model as a dictionary ready for export.

        Returns
        -------
        KwargsType :
            The model instance serialized as a dictionary.
        """
        from leaspy import __version__
        from .utilities import tensor_to_list

        hyperparameters = {name: tensor_to_list(value) for name, value in (self.hyperparameters or {}).items()}

        return {
            "leaspy_version": __version__,
            "name": self.name,
            "parameters": {
                k: tensor_to_list(v)
                for k, v in (self.parameters or {}).items()
            },
            **hyperparameters,
        }

    @abstractmethod
    def compute_individual_trajectory(
        self,
        timepoints,
        individual_parameters: DictParams,
        *,
        skip_ips_checks: bool = False,
    ) -> torch.Tensor:
        raise NotImplementedError
