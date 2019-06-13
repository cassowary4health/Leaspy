import json
import numpy as np

class ModelSettings():
    def __init__(self, path_to_model_parameters):
        with open(path_to_model_parameters) as fp:
            parameters = json.load(fp)

        self.model_type, self.dimension, self.parameters = self.read_parameters(parameters)

    def read_parameters(self, parameters):
        if 'type' not in parameters.keys():
            raise ValueError('The \'type\' key is missing in the model parameters (JSON file) you are loading')
        if 'parameters' not in parameters.keys():
            raise ValueError('The \'parameters\' key is missing in the model parameters (JSON file) you are loading')
        if 'dimension' not in parameters.keys():
            raise ValueError('The \'dimension\' key is missing in the model parameters (JSON file) you are loading')

        model_type = parameters['type']
        dimension = parameters['dimension']
        parameters = {k.lower(): self.to_float(v) for k, v in parameters['parameters'].items()}

        # Check that there is no list
        for key in parameters.keys():
            # Transform to numpy if list
            if type(parameters[key]) in [list]:
                parameters[key] = np.array(parameters[key])

        return model_type, dimension, parameters

    @staticmethod
    def to_float(x):
        if type(x) == int:
            return float(x)
        elif type(x) == list:
            return [float(el) for el in x]
        else:
            return x


