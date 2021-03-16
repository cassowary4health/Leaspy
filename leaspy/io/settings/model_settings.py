import json


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
