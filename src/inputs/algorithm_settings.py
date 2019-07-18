import json


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
        self._get_output(settings)
        self._get_seed(settings)

    @staticmethod
    def _check_settings(settings):
        if 'name' not in settings.keys():
            raise ValueError('The \'name\' key is missing in the algorithm settings (JSON file) you are loading')
        if 'parameters' not in settings.keys():
            raise ValueError('The \'parameters\' key is missing in the algorithm settings (JSON file) you are loading')
        if 'output' not in settings.keys():
            raise ValueError('The \'output\' key is missing in the algorithm settings (JSON file) you are loading')
        if 'path' not in settings['output'].keys():
            print("Warning: The \'path\' key is missing in the output parameter")

    def _get_type(self, settings):
        self.name = settings['name'].lower()

    def _get_parameters(self, settings):
        self.parameters = settings['parameters']

    def _get_output(self, settings):
        self.output = settings['output']
        if 'path' in self.output.keys():
            self.output_path = self.output['path']
        else:
            self.output_path = None

    def _get_seed(self, settings):
        if 'seed' in settings.keys():
            try:
                self.seed = int(settings['seed'])
            except ValueError:
                print("The \'seed\' parameter you provided cannot be converted to int")
        else:
            self.seed = None
