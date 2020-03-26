import torch
import json
import pandas as pd
import numpy as np
import warnings

class IndividualParameters:
    r"""
    IndividualParameters object class.
    The object holds a collection of individual parameters, that are outputs of the api personalization.
    There are used as io of the simulation algorithm, to provide an initial distribution of individual parameters.

    Attributes
    ----------
    _indices: list
        List of the patient indices
    _individual_parameters: dict
        Individual indices (key) with their corresponding individual parameters {parameter name: parameter value}
    _parameters_shape: dict
        Shape of each individual parameter
    _default_saving_type: str
        Default extension for saving when none is provided

    Methods
    ----------
    add_individual_parameters(index, individual_parameters)
        Adds the individual parameters of a new patient
    __getitem__[index]
        Returns the individual parameters of patient idx
    subset(indices)
        Returns a IndividualParameters object containing only the patients in indices
    get_mean(parameter_name)
        Returns the mean value of the parameter_name across all patients
    get_std(parameter_name)
        Return the standard deviation value of the parameter_name across all patients
    to_dataframe()
        Returns the dataframe of individual parameters
    from_dataframe(df)
        Static method that returns an IndividualParameters object from the dataframe
    to_pytorch()
        Returns the indices, pytorch_dict corresponding to the individual parameters
    from_pytorch(indices, pytorch_dict)
        Static method that returns an IndividualParameters object from the indices and pytorch dictionary
    save(path):
        Saves the individual parameters (json or csv) at the path location
    load(path):
        Static method that loads the individual parameters (json or csv) existing at the path locatio
    """

    def __init__(self):
        self._indices = []
        self._individual_parameters = {}
        self._parameters_shape = {} # {p_name: p_shape}
        self._default_saving_type = 'csv'

    def add_individual_parameters(self, index, individual_parameters):
        r"""
        Add the individual parameter of an individual to the IndividualParameters object

        Parameters
        ----------
        index: str
            Index of the individual
        individual_parameters: dict
            Individual parameters of the individual {name: value}

        Raises
        ------
        ValueError
            If the index is not a string or has already been added

        Examples
        --------
        Add two individual with tau, xi and sources parameters

        >>> ip = IndividualParameters()
        >>> ip.add_individual_parameters('index-1', {"xi": 0.1, "tau": 70, "sources": [0.1, -0.3]})
        >>> ip.add_individual_parameters('index-2', {"xi": 0.2, "tau": 73, "sources": [-0.4, -0.1]})
        """
        # Check indices
        if type(index) != str:
            raise ValueError(f'The index should be a string ({type(item)} provided instead)')

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
        if type(item) != str:
            raise ValueError(f'The index should be a string ({type(item)} provided instead)')
        return self._individual_parameters[item]

    def subset(self, indices):
        r"""
        Returns IndividualParameters object with a subset of the initial individuals

        Parameters
        ----------
        indices: list
            List of strings that corresponds to the indices of the individuals to return

        Returns
        -------
        IndividualParameters
            An instance of the IndividualParameters object with the selected list of individuals

        Raises
        ------
        ValueError
            Raise an error if one of the index is not in the IndividualParameters

        Examples
        --------

        >>> ip = IndividualParameters()
        >>> ip.add_individual_parameters('index-1', {"xi": 0.1, "tau": 70, "sources": [0.1, -0.3]})
        >>> ip.add_individual_parameters('index-2', {"xi": 0.2, "tau": 73, "sources": [-0.4, -0.1]})
        >>> ip.add_individual_parameters('index-3', {"xi": 0.3, "tau": 58, "sources": [-0.6, 0.2]})
        >>> ip_sub = ip.subset(['index-1', 'index-3'])
        """
        ip = IndividualParameters()

        for idx in indices:
            if idx not in self._indices:
                raise ValueError(f'The index {index} is not in the indices')
            p = self[idx].copy()
            ip.add_individual_parameters(idx, p)

        return ip

    def get_mean(self, parameter):
        r"""
        Returns the mean value of the parameter_name across all patients

        Parameters
        ----------
        parameter: str
            Name of the parameter

        Returns
        -------
        list or float
            Mean value of the parameter

        Raises
        ------
        ValueError
            If the parameter is not in the IndividualParameters

        Examples
        --------

        >>> ip = IndividualParameters.load("path/to/individual_parameters")
        >>> tau_mean = ip.get_mean("tau")
        """
        if parameter not in self._parameters_shape.keys():
            ValueError(f"Parameter {parameter} does not exist in the individual parameters")

        p = [v[parameter] for v in self._individual_parameters.values()]
        p_mean = np.mean(p, axis=0).tolist()

        return p_mean

    def get_std(self, parameter):
        r"""
        Returns the stardard deviation of the parameter_name across all patients

        Parameters
        ----------
        parameter: str
            Name of the parameter

        Returns
        -------
        list or float
            Standard value of the parameter

        Raises
        ------
        ValueError
            If the parameter is not in the IndividualParameters

        Examples
        --------

        >>> ip = IndividualParameters.load("path/to/individual_parameters")
        >>> tau_std = ip.get_std("tau")
        """
        if parameter not in self._parameters_shape.keys():
            ValueError(f"Parameter {parameter} does not exist in the individual parameters")

        p = [v[parameter] for v in self._individual_parameters.values()]
        p_std = np.std(p, axis=0).tolist()

        return p_std

    def to_dataframe(self):
        r"""
        Returns the dataframe of individual parameters

        Returns
        -------
        dataframe: pandas.DataFrame
            Each row corresponds to one individual. The index corresponds to the individual index. The columns are
            the names of the parameters.


        Examples
        --------
        Convert the individual parameters object into a dataframe

        >>> ip = IndividualParameters.load("path/to/individual_parameters")
        >>> ip_df = ip.to_dataframe()
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
        r"""
        Static method that returns an IndividualParameters object from the dataframe

        Parameters
        ----------
        df: pandas.DataFrame
            Dataframe of the invidual parameters. Each row must correspond to one individual. The index corresponds
            to the individual index. The columns are the names of the parameters.


        Returns
        -------
        IndividualParameters

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
        r"""
        Static method that returns an IndividualParameters object from the indices and pytorch dictionary

        Parameters
        ----------
        indices: list
            List of the patients indices
        dict_pytorch: dict
            Dictionary of the individual parameters

        Returns
        -------
        IndividualParameters


        Examples
        --------

        >>> indices = ['index-1', 'index-2', 'index-3']
        >>> ip_pytorch = {
        >>>    "xi": torch.tensor([[0.1], [0.2], [0.3]], dtype=torch.float32),
        >>>    "tau": torch.tensor([[70], [73], [58.]], dtype=torch.float32),
        >>>    "sources": torch.tensor([[0.1, -0.3], [-0.4, 0.1], [-0.6, 0.2]], dtype=torch.float32)
        >>> }
        >>> ip_pytorch = IndividualParameters.from_pytorch(indices, ip_pytorch)

        """

        len_p = {k: len(v) for k, v in dict_pytorch.items()}
        for k, v in len_p.items():
            if v != len(indices):
                raise ValueError(f'The parameter {k} should be of same length as the indices')

        ip = IndividualParameters()

        keys = list(dict_pytorch.keys())

        for i, idx in enumerate(indices):
            p = {k: dict_pytorch[k][i].numpy().tolist() for k in keys}
            p = {k: v[0] if len(v) == 1 else v for k, v in p.items()}

            ip.add_individual_parameters(idx, p)

        return ip

    def to_pytorch(self):
        r"""
        Returns the indices and pytorch dictionary of individual parameters

        Returns
        -------
        indices: list
            List of patient indices
        pytorch_dict: dict
            Dictionary of the individual parameters {parameters name: pytorch list of values across individuals}


        Examples
        --------
        Convert the individual parameters object into a dataframe

        >>> ip = IndividualParameters.load("path/to/individual_parameters")
        >>> indices, ip_pytorch = ip.to_pytorch()
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
        r"""
        Saves the individual parameters (json or csv) at the path location

        Parameters
        ----------
        path: str
            Path and file name of the individual parameters. The extension can be json or csv.
            If no extension, default extension (csv) is used

        """
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
        r"""
        Static method that loads the individual parameters (json or csv) existing at the path locatio

        Parameters
        ----------
        path: str
            Path and file name of the individual parameters.

        Returns
        -------
        IndividualParameters:
            Individual parameters object load from the file

        Raises
        ------
        ValueError:
            If the provided extension is not csv not json

        Examples
        --------

        >>> ip = IndividualParameters.load('/path/to/individual_parameters_1.json')
        >>> ip2 = IndividualParameters.load('/path/to/individual_parameters_2.csv')
        """
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
