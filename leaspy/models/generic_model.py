
from abc import ABC, abstractmethod
from typing import Optional, List, Any, Dict
import itertools
import json

import torch
import numpy as np

KwargsType = Dict[str, Any]

class GenericModel(ABC):
    """
    Generic model (temporary until :class:`.AbstractModel` is really **abstract**).

    TODO: change naming after AbstractModel was renamed?

    Attributes
    ----------
    name: str
    features: list[str]
    dimension: int (read-only)
        Number of features
    parameters: dict
    is_initialized: bool
    """

    # to be changed in sub-classes so to benefit from automatic methods

    # dict of {hyperparam_name: type_hint} instead?
    _hyperparameters = ('features',)
    # top-level "hyperparameters" that are FULLY defined by others hyperparameters
    _properties = ('dimension',)

    #_parameters = () # names may be dynamic depending on hyperparameters...
    #_attributes = () # TODO: really pertinent? why not a model parameter? cf. "mixing_matrix"

    def __init__(self, name: str):

        self.name = name
        #self.reset_hyperparameters()
        self.features: Optional[List[str]] = None
        self.parameters: KwargsType = {}
        #self.dimension = None

        self.is_initialized: bool = False # to be explicitly set as True by subclasses if so

    """
    def reset_hyperparameters(self) -> None:
        for hp_name, hp_type_hint in self._hyperparameters.items():
            setattr(self, hp_name, None)
            self.__annotations__[hp_name] = hp_type_hint #Optional[hp_type_hint]
    """

    def get_hyperparameters(self, *, with_properties = True, default = None):
        """
        Get all model hyperparameters

        Returns
        -------
        dict
        """

        all_hp_names = self._hyperparameters
        if with_properties:
            all_hp_names = itertools.chain(all_hp_names, self._properties)

        return {
            hp_name: getattr(self, hp_name, default) for hp_name in all_hp_names
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
            for hp_name, hp_val in self.get_hyperparameters().items()
        }
        return all(d_ok.values())

    # 'features' (and 'dimension') are really core hyperparameters

    @property
    def dimension(self) -> Optional[int]:
        # read-only <-> number of modelled features
        if self.features is None:
            return None
        else: # TODO? use: if self.hyperparameters_ok()
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

    def load_parameters(self, parameters, list_converter=np.array) -> None:
        """
        Instantiate or update the model's parameters.

        Parameters
        ----------
        parameters: dict
            Contains the model's parameters
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

    def load_hyperparameters(self, hyperparameters) -> None:
        """
        Load model hyperparameters from a dict

        Parameters
        ----------
        hyperparameters: dict
            Contains the model's hyperparameters
        """

        # no total reset of hyperparameters here unlike in load_parameters...

        # TODO change this behavior in ModelSettings? why not sending an empty dict instead of None??
        if hyperparameters is None:
            return

        # unknown hyper parameters
        non_static_hps = set(hyperparameters.keys()).difference(self._hyperparameters)
        dynamic_hps = non_static_hps.intersection(self._properties)
        unknown_hps = non_static_hps.difference(dynamic_hps) # no Python method to get intersection and difference at once...

        if len(unknown_hps) > 0:
            raise ValueError(f'Unknown hyperparameters: {unknown_hps}...')

        # set "static" hyperparameters only
        for hp_name, hp_val in hyperparameters.items():
            if hp_name in self._hyperparameters:
                setattr(self, hp_name, hp_val) # top-level of object...

        # check that dynamic hyperparameters match if provided...
        # (check this after all "static" hyperparameters being set)
        dynamic_hps_expected = {
            d_hp_name: getattr(self, d_hp_name) for d_hp_name in dynamic_hps
        }
        dynamic_hps_provided = {
            d_hp_name: hyperparameters[d_hp_name] for d_hp_name in dynamic_hps
        }
        if dynamic_hps_provided != dynamic_hps_expected:
            raise ValueError(f"Dynamic hyperparameters provided do not correspond to the expected ones:\n{dynamic_hps_provided} != {dynamic_hps_expected}")

    def save(self, path, **kwargs):
        """
        Save Leaspy object as json model parameter file.

        Default save method: it can be overwritten in child class but should be generic...

        Parameters
        ----------
        path: str
            Path to store the model's parameters.
        **kwargs
            Keyword arguments for json.dump method.
        """
        model_parameters_save = self.parameters.copy() # <!> shallow copy
        for param_name, param_val in model_parameters_save.items():
            if isinstance(param_val, (torch.Tensor, np.ndarray)):
                model_parameters_save[param_name] = param_val.tolist()

        model_settings = {
            'name': self.name,
            **self.get_hyperparameters(with_properties=True),
            'parameters': model_parameters_save
        }
        with open(path, 'w') as fp:
            json.dump(model_settings, fp, **kwargs)

    @abstractmethod
    def compute_individual_trajectory(self, timepoints, individual_parameters, **kws) -> torch.Tensor:
        """
        Compute scores values at the given time-point(s) given a subject's individual parameters.

        Parameters
        ----------
        timepoints : scalar or array_like[scalar] (list, tuple, :class:`numpy.ndarray`)
            Contains the age(s) of the subject.
        individual_parameters: dict
            Contains the individual parameters.
            Each individual parameter should be a scalar or array_like
        **kws: Any
            extra model specific keyword-arguments

        Returns
        -------
        :class:`torch.Tensor`
            Contains the subject's scores computed at the given age(s)
            Shape of tensor is (1, n_tpts, n_features)
        """
        ...

    def __str__(self):

        lines = [
            f"=== MODEL {self.name} ===" # header
        ]

        # hyperparameters
        for hp_name, hp_val in self.get_hyperparameters(with_properties=True).items():
            lines.append(f"{hp_name} : {hp_val}")

        # separation between hyperparams & params
        lines.append('-'*len(lines[0]))

        for param_name, param_val in self.parameters.items():
            # if type(self.parameters[key]) == float:
            #    logs += "{} : {:.5f}\n".format(key, self.parameters[key])
            # else:
            lines.append(f"{param_name} : {param_val}")

        return "\n".join(lines)

