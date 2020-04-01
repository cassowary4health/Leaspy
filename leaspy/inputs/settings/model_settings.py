import json
import warnings

from torch import tensor, float32


class ModelSettings:
    def __init__(self, path_to_model_settings):
        self.name = None
        self.hyperparameters = None
        self.parameters = None
        self.posterior_distribution = {'mean': None, 'covariance': None}

        if type(path_to_model_settings) is dict:
            settings = path_to_model_settings
        else:
            with open(path_to_model_settings) as fp:
                settings = json.load(fp)

        ModelSettings._check_settings(settings)
        self._get_name(settings)
        self._get_parameters(settings)
        self._get_hyperparameters(settings)
        self._get_posterior_distribution(settings)

    @staticmethod
    def _check_settings(settings):
        if 'name' not in settings.keys():
            raise ValueError('The \'type\' key is missing in the model parameters (JSON file) you are loading')
        if 'parameters' not in settings.keys():
            raise ValueError('The \'parameters\' key is missing in the model parameters (JSON file) you are loading')

    def _get_name(self, settings):
        self.name = settings['name'].lower()

    def _get_parameters(self, settings):
        self.parameters = settings['parameters']

    def _get_hyperparameters(self, settings):
        hyperparameters = {k.lower(): v for k, v in settings.items() if k not in ['name', 'parameters']}
        if hyperparameters:
            self.hyperparameters = hyperparameters
        else:
            self.hyperparameters = None

    def _get_posterior_distribution(self, settings):
        if "posterior_distribution" in settings.keys():
            self.posterior_distribution.update({k.lower(): tensor(v, dtype=float32) for k, v in
                                                settings["posterior_distribution"].items()})
        else:
            warnings.warn('The loaded model file does not have the field "posterior_distribution". '
                          'This might rise an error in future release!', DeprecationWarning, stacklevel=4)
