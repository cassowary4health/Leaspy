import torch
import json
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
        # Check indices
        if index in self._indices:
            raise ValueError(f'The index {index} has already been added before')
        self._indices.append(index)

        # Check the dictionary format
        if type(individual_parameters) != dict:
            raise ValueError('The `individual_parameters` argument should be a dictionary')

        individual_parameters = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in individual_parameters.items()} # Conversion from numpy to list
        for k, v in individual_parameters.items():
            valid_types = [list, int, float, np.float32, np.float64]
            if type(v) not in valid_types:
                raise ValueError(f'Incorrect dictionary value. Error for key: {k} -> type {type(v)}')

        self._individual_parameters[index] = individual_parameters

        for k, v in individual_parameters.items():
            # Keep track of the parameter shape
            if k not in self._parameters_shape.keys():
                self._parameters_shape[k] = 1 if np.ndim(v) == 0 else len(v)

    def __getitem__(self, item):
        return self._individual_parameters[item]


    def subset(self, list_of_idx):
        # TODO Igor
        return

    def get_mean(self, parameter):
        #TODO Igor
        return

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
            i_d = {k: v_flat[v].tolist() if np.ndim(v) == 0 else v_flat[v].values.tolist() for k, v in final_names.items()}
            ip.add_individual_parameters(idx, i_d)

        return ip

    @staticmethod
    def from_pytorch(indices, dict_pytorch):
        """

        """
        ip = IndividualParameters()

        keys = list(dict_pytorch.keys())

        for i, idx in enumerate(indices):
            p = {k: dict_pytorch[k][i].numpy().tolist() for k in keys}
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

        return self._indices, ips_pytorch

    def save(self, path):
        extension = IndividualParameters._check_and_get_extension(path)
        if not extension:
            warnings.warn(f'You did not provide a valid extension (csv or json) for the file. '
                          f'Default to {self._default_saving_type}')
            extension = self._default_saving_type
            path = path+'.'+extension

        if extension == 'csv':
            self._save_csv(path)
        elif extension == 'json':
            self._save_json(path)
        else:
            raise ValueError(f"Something bad happened: extension is {extension}")

    @staticmethod
    def load(path):
        extension = IndividualParameters._check_and_get_extension(path)
        if not extension or extension not in ['csv', 'json']:
            raise ValueError('The file you provide should have a `.csv` or `.json` name')

        if extension == 'csv':
            ip = IndividualParameters._load_csv(path)
        elif extension == 'json':
            ip = IndividualParameters._load_json(path)
        else:
            raise ValueError(f"Something bad happened: extension is {extension}")

        return ip

    @staticmethod
    def _check_and_get_extension(path):
        path = path.split('/')[-1]
        if '.' in path:
            extension = path.split('.')[-1]
            return extension
        return False

    def _save_csv(self, path):
        df = self.to_dataframe()
        df.to_csv(path)

    def _save_json(self, path):
        json_data = {
            'indices': self._indices,
            'individual_parameters': self._individual_parameters,
            'parameters_shape': self._parameters_shape
        }

        with open(path, 'w') as f:
            json.dump(json_data, f)

    @staticmethod
    def _load_csv(path):

        df = pd.read_csv(path, index_col=0)
        ip = IndividualParameters.from_dataframe(df)

        return ip

    @staticmethod
    def _load_json(path):
        with open(path, 'r') as f:
            json_data = json.load(f)

        ip = IndividualParameters()
        ip._indices = json_data['indices']
        ip._individual_parameters = json_data['individual_parameters']
        ip._parameters_shape = json_data['parameters_shape']

        return ip
