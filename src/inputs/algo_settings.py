import json

class AlgoSettings():
    """
    Read a algo_parameters json file and create the corresponding algo
    """
    def __init__(self, path_to_algo_parameters):
        with open(path_to_algo_parameters) as fp:
            parameters = json.load(fp)

        self.algo_type, self.parameters, self.output_path = self.read_parameters(parameters)

    def read_parameters(self, parameters):
        if 'type' not in parameters.keys():
            raise ValueError('The \'type\' key is missing in the algo parameters (JSON file) you are loading')
        if 'parameters' not in parameters.keys():
            raise ValueError('The \'parameters\' key is missing in the algo parameters (JSON file) you are loading')
        if 'output' not in parameters.keys():
            raise ValueError('The \'output\' key is missing in the algo parameters (JSON file) you are loading')
        if 'path' not in parameters['output'].keys():
            print("Warning: The \'path\' key is missing in the output parameter")

        algo_type = parameters['type']

        if 'path' in parameters['output'].keys() and parameters['output']['path'] is not None:
            algo_output_path = parameters['output']['path']
        else:
            algo_output_path = None

        parameters = {k.lower(): v for k, v in parameters['parameters'].items()}



        return algo_type, parameters, algo_output_path

    def get_path_output(self):
        return self.output_path
