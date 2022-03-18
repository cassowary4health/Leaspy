
from __future__ import annotations
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod
import itertools
import json
import warnings

import torch
import numpy as np

from leaspy import __version__
from leaspy.exceptions import LeaspyModelInputError
from leaspy.utils.typing import KwargsType, FeatureType, Optional, List, Tuple

if TYPE_CHECKING:
    from leaspy.io.data.dataset import Dataset


class GenericModel(ABC):
    """
    Generic model (temporary until :class:`.AbstractModel` is really **abstract**).

    TODO: change naming after AbstractModel was renamed?

    Parameters
    ----------
    name : str
        Name of the model
    **kwargs
        Hyperparameters of the model

    Attributes
    ----------
    name : str
        Name of the model
    is_initialized : bool
        Is the model initialized?
    features : list[str]
        List of model features (None if not initialization)
    dimension : int (read-only)
        Number of features
    parameters : dict
        Contains internal parameters of the model
    """

    # to be changed in sub-classes so to benefit from automatic methods

    # dict of {hyperparam_name: (default_value, type_hint)} instead?
    _hyperparameters: KwargsType = {}
    # top-level "hyperparameters" that are FULLY defined by others hyperparameters
    _properties: Tuple[str, ...] = ('dimension',)

    #_parameters = () # names may be dynamic depending on hyperparameters...
    #_attributes = () # TODO: really pertinent? why not a model parameter? cf. "mixing_matrix"

    def __init__(self, name: str, **kwargs):

        self.name = name
        #self.reset_hyperparameters()
        self.features: List[FeatureType] = None
        self.parameters: KwargsType = {}
        #self.dimension = None
        #self.noise_model = None

        self.is_initialized: bool = False # to be explicitly set as True by subclasses if so

        # Load hyperparameters at init (and set at default values when missing)
        self.load_hyperparameters(kwargs, with_defaults=True)

    """
    # TODO?
    def reset_hyperparameters(self) -> None:
        for hp_name, hp_type_hint in self._hyperparameters.items():
            setattr(self, hp_name, None)
            self.__annotations__[hp_name] = hp_type_hint #Optional[hp_type_hint]
    """

    def validate_compatibility_of_dataset(self, dataset: Dataset):
        """
        Raise if the given dataset is not compatible with the current model.

        Parameters
        ----------
        dataset : :class:`.Dataset`
            The dataset we want to model.

        Raises
        ------
        :exc:`~leaspy.exceptions.LeaspyDataInputError`
            If and only if data is incompatible with model.
        """
        return

    def initialize(self, dataset: Dataset, method: str = None):
        """
        Initialize the model given a dataset and an initialization method.

        After calling this method :attr:`is_initialized` should be True and model should be ready for use.

        Parameters
        ----------
        dataset : :class:`.Dataset`
            The dataset we want to initialize from.
        method : str, optional (default None)
            A custom method to initialize the model
        """
        if self.is_initialized and self.features is not None:
            # we also test that self.features is not None, since for `ConstantModel`:
            # `is_initialized`` is True but as a mock for being personalization-ready, without really being initialized!
            warn_msg = '<!> Re-initializing an already initialized model.'
            if dataset.headers != self.features:
                warn_msg += f' Overwritting previous model features ({self.features}) with new ones ({dataset.headers}).'
            warnings.warn(warn_msg)

        self.validate_compatibility_of_dataset(dataset)
        self.features = dataset.headers
        self.is_initialized = True

    def get_hyperparameters(self, *, with_features = True, with_properties = True, default = None) -> KwargsType:
        """
        Get all model hyperparameters

        Parameters
        ----------
        with_features, with_properties : bool (default True)
            Whether to include `features` and respectively all `_properties` (i.e. _dynamic_ hyperparameters)
            in the returned dictionary
        default : Any
            Default value is something is an hyperparameter is missing (should not!)

        Returns
        -------
        dict { hyperparam_name : str -> hyperparam_value : Any }
        """

        hps_names_iters = []

        # <!> Order of hyperparameters matters

        if with_features:
            hps_names_iters.append(['features'])

        hps_names_iters.append(self._hyperparameters.keys())

        if with_properties:
            hps_names_iters.append(self._properties)

        all_hp_names = itertools.chain(*hps_names_iters)

        return {
            hp_name: getattr(self, hp_name, default)
            for hp_name in all_hp_names
        }

    def hyperparameters_ok(self) -> bool:
        """
        Check all model hyperparameters are ok

        Returns
        -------
        bool
        """

        d_ok = {
            hp_name: hp_val is not None #and check hp_val compatible with hp_type_hint
            #for hp_name, hp_type_hint in self._hyperparameters.items()
            for hp_name, hp_val in self.get_hyperparameters(with_features=True, with_properties=True).items()
        }
        return all(d_ok.values())

    # 'features' (and 'dimension') are really core hyperparameters

    @property
    def dimension(self) -> Optional[int]:
        # read-only <-> number of modelled features
        if self.features is None:
            return None
        else:
            return len(self.features)

    """
    # if we want hyperparameters direct access without storing them in top-level
    def __getattr__(self, key: str):# -> Any:
        # overload so to mock hyperparameters on top-class level

    def __hasattr__(self, key: str) -> bool:
        # overload so to mock hyperparameters on top-class level

    def __setattr__(self, key: str, val) -> None:
        # overload so to mock hyperparameters on top-class level
    """

    def load_parameters(self, parameters, *, list_converter=np.array) -> None:
        """
        Instantiate or update the model's parameters.

        Parameters
        ----------
        parameters : dict
            Contains the model's parameters.
        list_converter : callable
            The function to convert list objects.
        """

        """
        self.parameters = {} # reset completely here
        # TODO: optional reset + warn if overwriting existing?
        # TODO: load model defaults at reset instead?
        for k, v in parameters.items():
            self.parameters[k] = v # unserialize here?
        """

        #<!> shallow copy only
        self.parameters = parameters.copy()

        # convert lists
        for k, v in self.parameters.items():
            if isinstance(v, list):
                self.parameters[k] = list_converter(v)

    def load_hyperparameters(self, hyperparameters: KwargsType, *, with_defaults: bool = False) -> None:
        """
        Load model hyperparameters from a dict

        Parameters
        ----------
        hyperparameters : dict[str, Any]
            Contains the model's hyperparameters
        with_defaults : bool (default False)
            If true, it also resets hyperparameters that are part of the model but not included
            in `hyperparameters` to their default value.

        Raises
        ------
        :exc:`.LeaspyModelInputError`
            if inconsistent hyperparameters
        """

        # no total reset of hyperparameters here unlike in load_parameters...

        # TODO change this behavior in ModelSettings? why not sending an empty dict instead of None??
        if hyperparameters is None:
            hyperparameters = {}

        settable_hps = {'features'}.union(self._hyperparameters.keys())

        # unknown hyper parameters
        non_settable_hps = set(hyperparameters.keys()).difference(settable_hps)
        # no Python method to get intersection and difference at once... so it is split
        dynamic_hps = non_settable_hps.intersection(self._properties)
        unknown_hps = non_settable_hps.difference(dynamic_hps)

        if len(unknown_hps) > 0:
            raise LeaspyModelInputError(f'Unknown hyperparameters for `{self.__class__.__qualname__}`: {unknown_hps}')

        # set "static" hyperparameters only
        if with_defaults:
            hyperparameters = {**self._hyperparameters, **hyperparameters}

        for hp_name, hp_val in hyperparameters.items():
            if hp_name in settable_hps:
                setattr(self, hp_name, hp_val) # top-level of object...

        # check that dynamic hyperparameters match if provided...
        # (check this after all "static" hyperparameters being set)
        dynamic_hps_given_value_expected_value = {
            d_hp_name: (hyperparameters[d_hp_name], getattr(self, d_hp_name))
            for d_hp_name in dynamic_hps
        }
        dynamic_hps_given_value_neq_expected_value = {
            d_hp_name: (given_v, expected_v)
            for d_hp_name, (given_v, expected_v) in dynamic_hps_given_value_expected_value.items()
            if given_v != expected_v
        }
        if len(dynamic_hps_given_value_neq_expected_value) != 0:
            raise LeaspyModelInputError(f"Dynamic hyperparameters provided do not correspond to the expected ones:\n"
                                        f"{dynamic_hps_given_value_neq_expected_value}")

    def save(self, path: str, **kwargs):
        """
        Save Leaspy object as json model parameter file.

        Default save method: it can be overwritten in child class but should be generic...

        Parameters
        ----------
        path : str
            Path to store the model's parameters.
        **kwargs
            Keyword arguments for json.dump method.
        """
        model_parameters_save = self.parameters.copy() # <!> shallow copy
        for param_name, param_val in model_parameters_save.items():
            if isinstance(param_val, (torch.Tensor, np.ndarray)):
                model_parameters_save[param_name] = param_val.tolist()

        model_settings = {
            'leaspy_version': __version__,
            'name': self.name,
            **self.get_hyperparameters(with_features=True, with_properties=True),
            'parameters': model_parameters_save
        }

        # Default json.dump kwargs:
        kwargs = {'indent': 2, **kwargs}

        with open(path, 'w') as fp:
            json.dump(model_settings, fp, **kwargs)

    @abstractmethod
    def compute_individual_trajectory(self, timepoints, individual_parameters: dict) -> torch.Tensor:
        """
        Compute scores values at the given time-point(s) given a subject's individual parameters.

        Parameters
        ----------
        timepoints : scalar or array_like[scalar] (list, tuple, :class:`numpy.ndarray`)
            Contains the age(s) of the subject.
        individual_parameters : dict[str, Any]
            Contains the individual parameters.
            Each individual parameter should be a scalar or array_like

        Returns
        -------
        :class:`torch.Tensor`
            Contains the subject's scores computed at the given age(s)
            Shape of tensor is (1, n_tpts, n_features)
        """

    def __str__(self):

        lines = [
            f"=== MODEL {self.name} ===" # header
        ]

        # hyperparameters
        for hp_name, hp_val in self.get_hyperparameters(with_features=True, with_properties=True).items():
            lines.append(f"{hp_name} : {hp_val}")

        # separation between hyperparams & params
        lines.append('-'*len(lines[0]))

        for param_name, param_val in self.parameters.items():
            lines.append(f"{param_name} : {param_val}")

        return "\n".join(lines)
