import json


class ModelParametersReader():
    def __init__(self, path_to_model_parameters):
        with open(path_to_model_parameters) as fp:
            parameters = json.load(fp)

        self.model_type, self.parameters = self.read_parameters(parameters)

    def read_parameters(self, parameters):
        if 'type' not in parameters.keys():
            raise ValueError('The \'type\' key is missing in the model parameters (JSON file) you are loading')
        if 'parameters' not in parameters.keys():
            raise ValueError('The \'parameters\' key is missing in the model parameters (JSON file) you are loading')

        model_type = parameters['type']
        parameters = {k.lower(): v for k, v in parameters['parameters'].items()}

        return model_type, parameters

