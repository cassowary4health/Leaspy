import json

from leaspy import __version__


class ModelSettings:
    """
    Used in :meth:`.Leaspy.load` to create a :class:`.Leaspy` class object from a `json` file.
    """
    def __init__(self, path_to_model_settings):
        if type(path_to_model_settings) is dict:
            settings = path_to_model_settings
        else:
            with open(path_to_model_settings) as fp:
                settings = json.load(fp)

        ModelSettings._check_settings(settings)
        self._get_name(settings)
        self._get_parameters(settings)
        self._get_hyperparameters(settings)

    @staticmethod
    def _check_settings(settings):
        if 'name' not in settings.keys():
            raise ValueError("The 'name' key is missing in the model parameters (JSON file) you are loading")
        if 'parameters' not in settings.keys():
            raise ValueError("The 'parameters' key is missing in the model parameters (JSON file) you are loading")

        # check leaspy_version attribute for compatibility purposes
        if 'leaspy_version' not in settings.keys():
            raise ValueError("The model you are trying to load was generated with a leaspy version < 1.1"
                    f" and is not compatible with your current version of leaspy == {__version__}"
                    " because of a bug in the multivariate model which lead to under-optimal results.\n"
                    "Please consider re-calibrating your model with your current leaspy version.\n"
                    "If you really want to load it as is (at your own risk) please use leaspy == 1.0.*")
        else:
            # we will be able to add some checks here to check/adapt retro/future compatibility of models
            pass

    def _get_name(self, settings):
        self.name = settings['name'].lower()

    def _get_parameters(self, settings):
        self.parameters = settings['parameters']

    def _get_hyperparameters(self, settings):
        hyperparameters = {k.lower(): v for k, v in settings.items() if k not in ['name', 'parameters', 'leaspy_version']}
        if hyperparameters:
            self.hyperparameters = hyperparameters
        else:
            self.hyperparameters = None
