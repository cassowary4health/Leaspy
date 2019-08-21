import json
import warnings

from leaspy.inputs.settings.outputs_settings import OutputsSettings


class AlgorithmSettings:
    """
    Read a algo_parameters json file and create the corresponding algo
    """
    def __init__(self, path_to_algorithm_settings):
        with open(path_to_algorithm_settings) as fp:
            settings = json.load(fp)

        AlgorithmSettings._check_settings(settings)
        self._get_type(settings)
        self._get_parameters(settings)
        self._get_outputs(settings)
        self._get_outputs_path(settings)
        self._get_seed(settings)

    @staticmethod
    def _check_settings(settings):
        if 'name' not in settings.keys():
            raise ValueError('The \'name\' key is missing in the algorithm settings (JSON file) you are loading')
        if 'parameters' not in settings.keys():
            raise ValueError('The \'parameters\' key is missing in the algorithm settings (JSON file) you are loading')

    def _get_type(self, settings):
        self.name = settings['name'].lower()

    def _get_parameters(self, settings):
        self.parameters = settings['parameters']

    def _get_outputs_path(self, settings):
        self.outputs_path = settings['outputs']['path']

    def _get_outputs(self, settings):
        # Check if the 'outputs' keys is in the settings ; otherwise no outputs
        if 'outputs' not in settings.keys():
            warnings.warn("You did not provide any output folder settings")
            self.outputs = None
            return

        # Check if the 'exists' keys exists, and if false, then no outputs
        if 'exists' in settings['outputs'] and settings['outputs']['exists'] is False:
            self.outputs = None
            return

        self.outputs = OutputsSettings(settings['outputs'])

    def _get_seed(self, settings):
        if 'seed' in settings.keys():
            try:
                self.seed = int(settings['seed'])
            except ValueError:
                print("The \'seed\' parameter you provided cannot be converted to int")
        else:
            self.seed = None
