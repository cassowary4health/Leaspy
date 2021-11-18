import json

from leaspy import __version__
from leaspy.exceptions import LeaspyModelInputError
from leaspy.utils.typing import KwargsType, Union


class ModelSettings:
    """
    Used in :meth:`.Leaspy.load` to create a :class:`.Leaspy` class object from a `json` file.

    Parameters
    ----------
    path_to_model_settings_or_dict : dict or str
        * If a str: path to a json file containing model settings
        * If a dict: content of model settings

    Raises
    ------
    :exc:`.LeaspyModelInputError`
    """
    def __init__(self, path_to_model_settings_or_dict: Union[str, dict]):

        if isinstance(path_to_model_settings_or_dict, dict):
            settings = path_to_model_settings_or_dict
        elif isinstance(path_to_model_settings_or_dict, str):
            with open(path_to_model_settings_or_dict) as fp:
                settings = json.load(fp)
        else:
            raise LeaspyModelInputError(f"Bad type for model settings: should be a dict or a path as a string, not {type(path_to_model_settings_or_dict)}")

        ModelSettings._check_settings(settings)
        self._get_name(settings)
        self._get_parameters(settings)
        self._get_hyperparameters(settings)

    @staticmethod
    def _check_settings(settings):

        error_tpl = "The '{}' key is missing in the model parameters (JSON file) you are loading."

        for mandatory_key in ['name', 'parameters']:
            if mandatory_key not in settings.keys():
                raise LeaspyModelInputError(error_tpl.format(mandatory_key))

        # check leaspy_version attribute for compatibility purposes
        if 'leaspy_version' not in settings.keys():
            raise LeaspyModelInputError("The model you are trying to load was generated with a leaspy version < 1.1 "
                    f"and is not compatible with your current version of leaspy == {__version__} "
                    "because of a bug in the multivariate model which lead to under-optimal results.\n"
                    "Please consider re-calibrating your model with your current leaspy version.\n"
                    "If you really want to load it as is (at your own risk) please use leaspy == 1.0.*")
        else:
            # we will be able to add some checks here to check/adapt retro/future compatibility of models
            pass

    def _get_name(self, settings):
        self.name: str = settings['name'].lower()

    def _get_parameters(self, settings):
        self.parameters: KwargsType = settings['parameters']

    def _get_hyperparameters(self, settings):
        hyperparameters = {k.lower(): v for k, v in settings.items()
                           if k not in ['name', 'parameters', 'leaspy_version']}

        self.hyperparameters: KwargsType = hyperparameters
