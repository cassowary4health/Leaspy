class AbstractAlgo():


    def load_parameters(self, parameters):
        for k, v in parameters.items():
            if k in self.parameters.keys():
                previous_v = self.parameters[k]
                print("Replacing {} parameter from value {} to value {}".format(k, previous_v, v))
            self.parameters[k] = v