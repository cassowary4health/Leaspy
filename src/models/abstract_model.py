import json


class AbstractModel():
    def __init__(self):
        self.model_parameters = {}

    def load_parameters(self, model_parameters):
        for k, v in model_parameters.items():
            if k in self.model_parameters.keys():
                previous_v = self.model_parameters[k]
                print("Replacing {} parameter from value {} to value {}".format(k, previous_v, v))
            self.model_parameters[k] = v

    def save_parameters(self, path):
        with open(path, 'w') as outfile:
            json.dump(self.model_parameters, outfile)

    def initialize_realizations(self):
        raise NotImplementedError

    def simulate_individual_parameters(self):
        raise NotImplementedError
