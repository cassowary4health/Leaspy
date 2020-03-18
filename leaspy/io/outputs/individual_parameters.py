import torch
import pandas as pd
import numpy as np
import warnings

class IndividualParameters:
    """
    IndividualParameters object class.
    The object holds a collection of individual parameters, that are outputs of the api personalization.
    There are used as io of the simulation algorithm, to provide an initial distribution of individual parameters.

    Attributes
    ----------


    Methods
    ----------

    """

    def __init__(self):
        self._indices = []
        self._individual_parameters = {}
        self._parameters_shape = {} # {p_name: p_shape}
        self._default_saving_type = 'csv'

    def add_individual_parameters(self, index, individual_parameters):
        if index in self._indices:
            raise ValueError(f'The index {index} has already been added before')
        self._indices.append(index)
        self._individual_parameters[index] = individual_parameters

        for k, v in individual_parameters.items():
            ### Keep track of the parameter shape
            if k not in self._parameters_shape.keys():
                self._parameters_shape[k] = 1 if np.ndim(v) == 0 else len(v)

    def to_dataframe(self):
        """

        """
        p_names = list(self._parameters_shape.keys())

        # Get the data, idx per idx
        arr = []
        for idx in self._indices:
            indiv_arr = [idx]
            indiv_p = self._individual_parameters[idx]

            for p_name in p_names:
                shape = self._parameters_shape[p_name]

                if shape == 1:
                    indiv_arr.append(indiv_p[p_name])
                else:
                    for i in range(shape):
                        indiv_arr.append(indiv_p[p_name][i])
            arr.append(indiv_arr)

        # Get the column names
        final_names = ['ID']
        for p_name in p_names:
            shape = self._parameters_shape[p_name]
            if shape == 1:
                final_names.append(p_name)
            else:
                final_names += [p_name+'_'+str(i) for i in range(shape)]

        df = pd.DataFrame(arr, columns=final_names)
        return df.set_index('ID')


    @staticmethod
    def from_dataframe(df):
        """

        """
        # Check the names to keep
        df_names = list(df.columns.values)

        final_names = {}
        for name in df_names:
            split = name.split('_')[0]
            if split not in final_names:
                final_names[split] = []
            final_names[split].append(name)

        final_names = {k: v if len(v) > 1 else v[0] for k, v in final_names.items()}

        # Create the individual parameters
        ip = IndividualParameters()

        for idx, v_flat in df.iterrows():
            i_d = {k: v_flat[v] if np.ndim(v) == 0 else v_flat[v].values.tolist() for k, v in final_names.items()}
            ip.add_individual_parameters(idx, i_d)

        return ip

    @staticmethod
    def from_pytorch(dict_pytorch):
        """

        """
        warnings.warn("The `from_pytorch` method of IndividualParameters object"
                      "is unaware of the subject indices. They are initialized with default values")
        ip = IndividualParameters()

        keys = list(dict_pytorch.keys())
        n_subjects = len(dict_pytorch[keys[0]])

        for idx in range(n_subjects):
            p = {k: dict_pytorch[k][idx].numpy().tolist() for k in keys}
            p = {k: v[0] if len(v) == 1 else v for k, v in p.items()}

            ip.add_individual_parameters(idx, p)

        return ip

    def to_pytorch(self):
        """

        """
        ips_pytorch = {}
        p_names = list(self._parameters_shape)

        for p_name in p_names:

            p_val = [self._individual_parameters[idx][p_name] for idx in self._indices]
            p_val = torch.tensor(p_val, dtype=torch.float32)
            p_val = p_val.reshape(shape=(len(self._indices), self._parameters_shape[p_name]))

            ips_pytorch[p_name] = p_val

        return ips_pytorch


    def save_individual_parameters(self, path, extension='.csv'):
        extension = IndividualParameters._check_and_get_extension(path)
        if not extension:
            warnings.warn(f'You did not provide a valid extension (csv or json) for the file. '
                          f'Default to {self._default_saving_type}')
            extension = self._default_saving_type

        if extension == '.csv':
            IndividualParameters._save_csv(path)
        elif extension == '.json':
            IndividualParameters._save_json(path)
        else:
            raise ValueError("Something bad happened")

    @staticmethod
    def load_individual_parameters(path, verbose=True, **args):
        extension = IndividualParameters._check_and_get_extension(path)
        if not extension or extension not in ['csv', 'json']:
            raise ValueError('The file you provide should have a `.csv` or `.json` name')

        if extension == 'csv':
            ip = IndividualParameters._load_csv(path)
        elif path == 'json':
            ip = IndividualParameters._load_json(path)
        else:
            raise ValueError("Something bad happened")

        return ip

    @staticmethod
    def _check_and_get_extension(path):

        if '.' in path:
            extension = path.split('.')[-1]
            return extension
        return False

    @staticmethod
    def _save_csv(path):
        return 0

    @staticmethod
    def _save_json(path):
        return 0

    @staticmethod
    def _load_csv(path):
        return 0

    @staticmethod
    def _load_json(path):
        return 0
