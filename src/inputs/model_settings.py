import json


class ModelSettings:
    def __init__(self, path_to_model_settings):
        with open(path_to_model_settings) as fp:
            settings = json.load(fp)

        ModelSettings._check_settings(settings)
        self._get_type(settings)
        self._get_parameters(settings)
        self._get_hyperparameters(settings)

    @staticmethod
    def _check_settings(settings):
        if 'name' not in settings.keys():
            raise ValueError('The \'type\' key is missing in the model parameters (JSON file) you are loading')
        if 'parameters' not in settings.keys():
            raise ValueError('The \'parameters\' key is missing in the model parameters (JSON file) you are loading')

    def _get_type(self, settings):
        self.name = settings['name'].lower()

    def _get_parameters(self, settings):
        self.parameters = settings['parameters']

    def _get_hyperparameters(self, settings):
        hyperparameters = {k.lower(): v for k, v in settings.items() if k not in ['name', 'parameters']}
        if hyperparameters:
            self.hyperparameters = hyperparameters
        else:
            self.hyperparameters = None

    '''
    @staticmethod
    def _recursive_to_float(x):
        if type(x) is list:
            return [ModelSettings._recursive_to_float(_) for _ in x]
        return x

    
        model_type = settings['type']
        dimension = settings['dimension']
        source_dimension = settings['source_dimension']
        parameters = {k.lower(): self.to_float(v) for k, v in settings['parameters'].items()}

        # Check that there is no list
        for key in parameters.keys():
            # Transform to numpy if list
            if type(parameters[key]) in [list]:
                parameters[key] = np.array(parameters[key])

        return model_type, dimension, source_dimension, parameters
    
    @staticmethod
    def to_float(x):
        if type(x) == int:
            return float(x)
        elif type(x) == list:
            #TODO make this recursive
            if type(x[0]) not in [list]:
                return [float(el) for el in x]
            else:
                return [[float(el) for el in els] for els in x]
        else:
            return x
    '''
