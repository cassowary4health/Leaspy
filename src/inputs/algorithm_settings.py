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
        self._get_smart_initialization(settings)

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

    def _get_smart_initialization(self, settings):
        if 'smart_initialization' in settings.keys() and settings['smart_initialization'] is True:
            self.smart_initialization = True
        else:
            self.smart_initialization = False
    '''
        algo_type = parameters['type']

        # Output path
        if 'path' in parameters['output'].keys() and parameters['output']['path'] is not None:
            algo_output_path = parameters['output']['path']
        else:
            algo_output_path = None

        parameters = {k.lower(): v for k, v in parameters['parameters'].items()}

        return algo_type, parameters, algo_output_path

    def get_path_output(self):
        return self.output_path


    
        # Annealing
        #if 'annealing' in parameters.keys():
        #    print("Annealing options given")
        #    if "do_annealing" in parameters['annealing'].keys() and parameters['annealing']['do_annealing']:
        #        print("Doing annealing")

        # Annealing for some algo
        #annealing_parameters = parameters['annealing']
        #if annealing_parameters is not None:
    '''
