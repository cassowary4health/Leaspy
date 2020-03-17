import torch
import pandas as pd
import numpy as np

class IndividualParameters:
    """
    IndividualParameters object class.
    The object holds a collection of individual parameters, that are outputs of the api personalization.
    There are used as inputs of the simulation algorithm, to provide an initial distribution of individual parameters.

    Attributes
    ----------


    Methods
    ----------

    """

    def __init__(self):
        self.indices = []
        self.individual_parameters = {}
        self.parameters_shape = {} # {p_name: p_shape}

    def add_individual_parameters(self, index, individual_parameters):
        self.indices.append(index)
        self.individual_parameters[index] = individual_parameters

        for k, v in individual_parameters.items():
            ### Keep track of the parameter shape
            if k not in self.parameters_shape.keys():
                self.parameters_shape[k] = 1 if np.ndim(v) == 0 else len(v)

    def to_dataframe(self):
        """

        """
        p_names = list(self.parameters_shape.keys())

        # Get the data, idx per idx
        arr = []
        for idx in self.indices:
            indiv_arr = [idx]
            indiv_p = self.individual_parameters[idx]

            for p_name in p_names:
                shape = self.parameters_shape[p_name]

                if shape == 1:
                    indiv_arr.append(indiv_p[p_name])
                else:
                    for i in range(shape):
                        indiv_arr.append(indiv_p[p_name][i])
            arr.append(indiv_arr)

        # Get the column names
        final_names = ['ID']
        for p_name in p_names:
            shape = self.parameters_shape[p_name]
            if shape == 1:
                final_names.append(p_name)
            else:
                for i in range(shape):
                    final_names.append(p_name+'_'+str(i))

        df = pd.DataFrame(arr, columns=final_names)
        return df.set_index('ID')



    def from_dataframe(self):
        """

        """

    def from_pytorch(self):
        """

        """

    def to_pytorch(self):
        """

        """
        ips_pytorch = {}
        p_names = list(self.parameters_shape)

        for p_name in p_names:

            p_val = [self.individual_parameters[idx][p_name] for idx in self.indices]
            p_val = torch.tensor(p_val, dtype=torch.float32)
            p_val = p_val.reshape(shape=(len(self.indices), self.parameters_shape[p_name]))

            ips_pytorch[p_name] = p_val

        return ips_pytorch


    def save_individual_parameters(self, path, extension='.csv'):
        if path == '.csv':
            save_csv()
        elif path == '.json':
            save_json()

    @staticmethod
    def load_individual_parameters(path, verbose=True, **args):
        if path == '.csv':
            load_csv()
        elif path == '.json':
            load_json()

