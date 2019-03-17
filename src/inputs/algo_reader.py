import json

class AlgoReader():
    """
    Read a algo_parameters json file and create the corresponding algo
    """
    def __init__(self, path_to_algo_parameters):
        with open(path_to_algo_parameters) as fp:
            parameters = json.load(fp)

        self.algo_type, self.parameters = self.read_parameters(parameters)

    def read_parameters(self, parameters):
        if 'type' not in parameters.keys():
            raise ValueError('The \'type\' key is missing in the algo parameters (JSON file) you are loading')
        if 'parameters' not in parameters.keys():
            raise ValueError('The \'parameters\' key is missing in the algo parameters (JSON file) you are loading')

        algo_type = parameters['type']
        parameters = {k.lower(): v for k, v in parameters['parameters'].items()}

        return algo_type, parameters

