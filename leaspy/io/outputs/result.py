import copy
import json
import os
import warnings
from collections.abc import Iterable

import pandas as pd
import torch

from leaspy.io.data.data import Data
from leaspy.io.data.dataset import Dataset


class Result:
    """
    Result object class.
    Used as logs by personalize algorithms & simulation algorithm.

    Attributes
    ----------
    data : :class:`.Data`
        Object containing the idx, time-points and observations of the patients.
    individual_parameters : dict [str, :class:`torch.Tensor`]
        Contains log-acceleration 'xi', time-shifts 'tau' & 'sources' (dictionary of `torch.Tensor`).
    ID_to_idx : dict
        The keys are the individual ID & the items are their respective ordered position in the data file given
        by the user. This order remains the same during the computation.
        Example - in Result.individual_parameters['xi'], the first element corresponds to the
        first patient in ID_to_idx.
    noise_std : float or :class:`torch.FloatTensor`
        Desired noise standard deviation level.
    """

    # TODO : Check consistency and ordering of sujbects ID between Data and individual parameters io.
    def __init__(self, data, individual_parameters, noise_std=None):
        """
        Process the initializer function - called by Leaspy.io.outputs.result.Result

        Parameters
        ----------
        data : :class:`.Data`
            Object containing the idx, time-points and observations of the patients
        individual_parameters : dict [str, :class:`torch.Tensor`]
            Contains log-acceleration 'xi', time-shifts 'tau' & 'sources'
        noise_std : float or :class:`torch.FloatTensor`, optional (default None)
            Desired noise standard deviation level
        """
        self.data = data
        self.individual_parameters = individual_parameters
        self.ID_to_idx = {key: i for i, key in enumerate(data.individuals)}
        self.noise_std = noise_std

    # TODO : this method is used only once in plotting => delete it ?
    def get_torch_individual_parameters(self, ID=None):
        """
        Getter function for the individual parameters.

        Parameters
        ----------
        ID : list, optional (default None)
            Contains the identifiers of the wanted subject.

        Returns
        -------
        dict [str, :class:`torch.Tensor`]
            Contains the individual parameters.
        """
        if ID is not None:
            if type(ID) != list:
                if isinstance(ID, str) or not isinstance(ID, Iterable):
                    # If ID is not a Iterable (case where ID is a int) => convert into list
                    # If ID is a str => convert into list
                    ID = [ID]
                else:
                    raise ValueError('Input argument "ID" must be a single identifier or a list or identifier!')

            list_idt = [self.ID_to_idx[id_patient] for id_patient in ID]
            ind_parameters = {key: value[list_idt] for key, value in self.individual_parameters.items()}
        else:
            ind_parameters = self.individual_parameters.copy()
        return ind_parameters

    # TODO: unit test & functional test
    def get_dataframe_individual_parameters(self, cofactors=None):
        """
        Return the dataframe of the individual parameters.

        Each row corresponds to a subject. The columns correspond
        (in this order) to the subjects' ID, the individual parameters (one column per individual parameter) & the
        cofactors (one column per cofactor).

        Parameters
        ----------
        cofactors: str or list, optional (default None)
            Contains the cofactor(s) to join to the logs dataframe.

        Notes
        -----
        The cofactors must be present in the leaspy data object stored into the .data attribute of the result instance.
        See the exemple.

        Returns
        -------
        :class:`pandas.DataFrame`
            Contains for each patient his ID & his individual parameters (optional and his cofactors states)

        Examples
        --------
        Load a longitudinal multivariate dataset & the subjects' cofactors. Compute the individual parameters for this
        dataset & get the corresponding dataframe with the genetic APOE cofactor

        >>> import pandas as pd
        >>> from leaspy import AlgorithmSettings, Data, Leaspy, Plotter
        >>> leaspy_logistic = Leaspy('logistic')
        >>> data = Data.from_csv_file('data/my_leaspy_data.csv')  # replace with your own path!
        >>> genes_cofactors = pd.read_csv('data/genes_cofactors.csv')  # replace with your own path!
        >>> print(genes_cofactors.head())
                   ID      APOE4
        0  sub-HS0102          1
        1  sub-HS0112          0
        2  sub-HS0113          0
        3  sub-HS0114          1
        4  sub-HS0115          0

        >>> data.load_cofactors(genes_cofactors, 'GENES')
        >>> model_settings = AlgorithmSettings('mcmc_saem', seed=0)
        >>> personalize_settings = AlgorithmSettings('mode_real', seed=0)
        >>> leaspy_logistic.fit(data, model_settings)
        >>> individual_results = leaspy_logistic.personalize(data, model_settings)
        >>> individual_results_df = individual_results.get_dataframe_individual_parameters('GENES')
        >>> print(individual_results_df.head())
                           tau        xi  sources_0  sources_1  APOE4
        ID
        sub-HS0102   70.329201  0.120465   5.969921  -0.245034      1
        sub-HS0112   95.156624 -0.692099   1.520273   3.477707      0
        sub-HS0113   74.900673 -1.769864  -1.222979   1.665889      0
        sub-HS0114   81.792763 -1.003620   1.021321   2.371716      1
        sub-HS0115   89.724648 -0.820971  -0.480975   0.741601      0
        """
        # Initialize patient dict with ID
        patient_dict = {'ID': list(self.ID_to_idx.keys())}

        # For each individual variable
        for variable_ind in list(self.individual_parameters.keys()):
            # Case tau / ksi --> unidimensional
            if self.individual_parameters[variable_ind].shape[1] == 1:
                patient_dict[variable_ind] = self.individual_parameters[variable_ind].numpy().reshape(-1)
            # Case sources --> multidimensional
            elif self.individual_parameters[variable_ind].shape[1] > 1:
                for dim in range(self.individual_parameters[variable_ind].shape[1]):
                    patient_dict[variable_ind + "_{}".format(dim)] = \
                        self.individual_parameters[variable_ind][:, dim].numpy().reshape(-1)

        df_individual_parameters = pd.DataFrame(patient_dict).set_index('ID')

        # If you want to load cofactors too
        if cofactors is not None:
            if type(cofactors) == str:
                cofactors = [cofactors]

            cofactor_dict = {'ID': list(self.data.individuals.keys())}

            for cofactor in cofactors:
                cofactor_dict[cofactor] = [self.data.individuals[idx].cofactors[cofactor] for
                                           idx in cofactor_dict['ID']]

            df_cofactors = pd.DataFrame(cofactor_dict).set_index('ID')
            df_individual_parameters = df_individual_parameters.join(df_cofactors)

        return df_individual_parameters

    def save_individual_parameters_csv(self, path, idx=None, cofactors=None, **args):
        """
        Save the individual parameters in a csv format.

        Parameters
        ----------
        path : str
            The logs's path.
        idx : list [str], optional (default None)
            Contain the IDs of the selected subjects. If ``None``, all the subjects are selected.
        cofactors : str or list [str], optional (default None)
            Contains the cofactor(s) to join to the logs dataframe.
        **args : Any
            Parameters to pass to :meth:`pandas.DataFrame.to_csv`.

        Notes
        -----
        The cofactors must be present in the leaspy data object stored into the :attr:`.data` attribute of the result instance.
        See the example.

        Examples
        --------
        Save the individual parameters of the twenty first subjects.

        >>> from leaspy import AlgorithmSettings, Data, Leaspy
        >>> leaspy_logistic = Leaspy('logistic')
        >>> data = Data.from_csv_file('data/my_leaspy_data.csv') # replace with your own path!
        >>> genes_cofactors = pd.read_csv('data/genes_cofactors.csv')  # replace with your own path!
        >>> data.load_cofactors(genes_cofactors, 'GENES')
        >>> model_settings = AlgorithmSettings('mcmc_saem', seed=0)
        >>> personalize_settings = AlgorithmSettings('mode_real', seed=0)
        >>> leaspy_logistic.fit(data, model_settings)
        >>> individual_results = leaspy_logistic.personalize(data, model_settings)
        >>> output_path = 'outputs/logistic_seed0-mode_real_seed0-individual_parameter.csv'
        >>> idx = list(individual_results.individual_parameters.keys())[:20]
        >>> individual_results.save_individual_parameters_csv(output_path, idx, cofactors='GENES')
        """
        self._check_folder_existancy(path)

        df_individual_parameters = self.get_dataframe_individual_parameters(cofactors=cofactors)
        if idx:
            if type(idx) != list:
                raise TypeError('Input "idx" must be a list, even if it contains only one element! '
                                'You gave idx={} which is of type {}.'.
                                format(idx, type(idx)))
            df_individual_parameters = df_individual_parameters.loc[idx]
        df_individual_parameters.to_csv(path, index=True, **args)

    def save_individual_parameters_json(self, path, idx=None, human_readable=None, **args):
        """
        Save the individual parameters in a json format.

        Parameters
        ----------
        path : str
            The logs's path.
        idx : list [str], optional (default None)
            Contain the IDs of the selected subjects. If ``None``, all the subjects are selected.
        human_readable : Any, optional (default None)
            .. deprecated:: 1.0
            TODO change to bool
                * If None (default): save as json file
                * If not None: call :meth:`.save_individual_parameters_torch`.
        **args : Any
            Arguments to pass to json.dump.

        Examples
        --------
        Save the individual parameters of the twenty first subjects.

        >>> from leaspy import AlgorithmSettings, Data, Leaspy
        >>> leaspy_logistic = Leaspy('logistic')
        >>> data = Data.from_csv_file('data/my_leaspy_data.csv')
        >>> model_settings = AlgorithmSettings('mcmc_saem', seed=0)
        >>> personalize_settings = AlgorithmSettings('mode_real', seed=0)
        >>> leaspy_logistic.fit(data, model_settings)
        >>> individual_results = leaspy_logistic.personalize(data, model_settings)
        >>> output_path = 'outputs/logistic_seed0-mode_real_seed0-individual_parameter.json'
        >>> idx = list(individual_results.individual_parameters.keys())[:20]
        >>> individual_results.save_individual_parameters_json(output_path, idx)
        """
        self._check_folder_existancy(path)
        dump = self._get_dump(idx)
        if human_readable is not None:
            warnings.warn("This parameter is deprecated! To save as a torch file, use the method "
                          "'save_individual_parameters_torch'.", DeprecationWarning, stacklevel=2)
            self.save_individual_parameters_torch(path, idx)
        else:
            with open(path, 'w') as fp:
                json.dump(dump, fp, **args)

    def save_individual_parameters_torch(self, path, idx=None, **args):
        """
        Save the individual parameters in a torch format.

        Parameters
        ----------
        path : str
            The logs's path.
        idx : list [str], optional (default None)
            Contain the IDs of the selected subjects. If ``None``, all the subjects are selected.
        args : Any
            Arguments to pass to torch.save.

        Examples
        --------
        Save the individual parameters of the twenty first subjects.

        >>> from leaspy import AlgorithmSettings, Data, Leaspy
        >>> leaspy_logistic = Leaspy('logistic')
        >>> data = Data.from_csv_file('data/my_leaspy_data.csv')
        >>> model_settings = AlgorithmSettings('mcmc_saem', seed=0)
        >>> personalize_settings = AlgorithmSettings('mode_real', seed=0)
        >>> leaspy_logistic.fit(data, model_settings)
        >>> individual_results = leaspy_logistic.personalize(data, model_settings)
        >>> output_path = 'outputs/logistic_seed0-mode_real_seed0-individual_parameter.pt'
        >>> idx = list(individual_results.individual_parameters.keys())[:20]
        >>> individual_results.save_individual_parameters_torch(output_path, idx)
        """
        self._check_folder_existancy(path)
        dump = self._get_dump(idx)
        torch.save(dump, path, **args)

    @staticmethod
    def _check_folder_existancy(path):
        # Test path's folder existence (if path contain a folder)
        if os.path.dirname(path) != '':
            if not os.path.isdir(os.path.dirname(path)):
                raise FileNotFoundError(
                    'Cannot save individual parameter at path %s - The folder does not exist!' % path)

    def _get_dump(self, idx=None):
        """
        Convert the individual_parameters attribute into a dictionary of list. The univariate parameters values
        like xi and tau are squeeze from shape (n_subjects, 1) to (n_subjects,).
        One can select only the wanted subject by specifying their ID with 'idx' parameter.

        Parameters
        ----------
        idx : list, optional (default None)
            Contains the ID of the selected subjects.

        Returns
        -------
        dict [str, list]
        """
        dump: dict = copy.deepcopy(self.individual_parameters)
        # Ex: individual_parameters = {'param1': torch.tensor([[1], [2], [3]]), ...}

        # Select only the wanted subjects
        if idx is not None:
            selected_id = [self.ID_to_idx[val] for val in idx]
            dump = {key: val[selected_id] for key, val in dump.items()}

        for key in dump.keys():
            if type(dump[key]) not in [list]:
                # For multivariate parameter - like sources
                # convert tensor([[1, 2], [2, 3]]) into [[1, 2], [2, 3]]
                if dump[key].shape[1] == 2:
                    dump[key] = dump[key].tolist()
                # for univariate parameters - like xi & tau
                # convert tensor([[1], [2], [3]]) into [1, 2, 3] => squeeze it
                elif dump[key].shape[1] == 1:
                    dump[key] = dump[key].squeeze().tolist()
        return dump

    @staticmethod
    def load_individual_parameters_from_csv(path, verbose=True, **args):
        """
        Load individual parameters from a csv.

        Parameters
        ----------
        path : str
            The file's path. The csv file musts contain two columns named 'tau' and 'xi'. If the individual parameters
            come from a multivariate model, it must also contain the columns 'sources_i' for i in [0, ..., n_sources].
        verbose : bool (default True)
        args : Any
            Parameters to pass to :func:`pandas.read_csv`.

        Returns
        -------
        dict [str, :class:`torch.Tensor`]
            A dictionary of torch.tensor which contains the individual parameters.

        Examples
        --------
        Load an individual parameters dictionary from a saved file.

        >>> from leaspy import Result
        >>> path = 'outputs/logistic_seed0-mode_real_seed0-individual_parameter.csv'
        >>> individual_parameters = Result.load_individual_parameters_from_csv(path)
        """
        df = pd.read_csv(path, **args)
        if verbose:
            print("Load from csv file ... conversion to torch")
        return Result.load_individual_parameters_from_dataframe(df)

    @staticmethod
    def load_individual_parameters_from_dataframe(df):
        """
        Load individual parameters from a :class:`pandas.DataFrame`.

        Parameters
        ----------
        df : :class:`pandas.DataFrame`
            Must contain two columns named 'tau' and 'xi'. If the individual parameters come from a multivariate model,
            it must also contain the columns 'sources_i' for i in [0, ..., n_sources].

        Returns
        -------
        dict [str, :class:`torch.Tensor`]
            A dictionary of torch.tensor which contains the individual parameters.
        """
        df.columns = [header.lower() for header in df.columns]
        sources_index = ["sources" in header for header in df.columns]
        ind_param = {'tau': torch.tensor(df['tau'].values, dtype=torch.float32).view(-1, 1),
                     'xi': torch.tensor(df['xi'].values, dtype=torch.float32).view(-1, 1)}
        if any(sources_index):
            ind_param['sources'] = torch.tensor(df.iloc[:, sources_index].values, dtype=torch.float32)
        return ind_param

    @staticmethod
    def load_individual_parameters_from_json(path, verbose=True, **args):
        """
        Load individual parameters from a json file.

        Deprecated : also load torch files.

        Parameters
        ----------
        path : str
            The file's path.
        verbose : bool (default True)
        args : Any
            Parameters to pass to json.load.

        Returns
        -------
        dict [str, :class:`torch.Tensor`]
            A dictionary of `torch.Tensor` which contains the individual parameters.

        Examples
        --------
        Load an individual parameters dictionary from a saved file.

        >>> from leaspy import Result
        >>> path = 'outputs/logistic_seed0-mode_real_seed0-individual_parameter.json'
        >>> individual_parameters = Result.load_individual_parameters_from_json(path)
        """
        # Test if file is a json file
        try:
            with open(path, 'r') as f:
                individual_parameters = json.load(f, **args)
                if verbose:
                    print("Load from json file ... conversion to torch")
                for key in individual_parameters.keys():
                    # Convert every list in torch.tensor
                    individual_parameters[key] = torch.tensor(individual_parameters[key], dtype=torch.float32)
                    # If tensor is 1-dimensional tensor([1, 2, 3]) => reshape it in tensor([[1], [2], [3]])
                    if individual_parameters[key].dim() == 1:
                        individual_parameters[key] = individual_parameters[key].view(-1, 1)
        # Else if it is a torch file
        except UnicodeDecodeError:
            warnings.warn('To load a torch file, use the static method result.load_individual_parameters_from_torch',
                          DeprecationWarning, stacklevel=2)
            individual_parameters = torch.load(path)  # load function from torch
            if verbose:
                print("Load from torch file")
        return individual_parameters

    @staticmethod
    def load_individual_parameters_from_torch(path, verbose=True, **args):
        """
        Load individual parameters from a torch file.

        Parameters
        ----------
        path : str
            The file's path.
        verbose : bool (default True)
        args : Any
            Parameters to pass to torch.load.

        Returns
        -------
        dict [str, :class:`torch.Tensor`]
            A dictionary of `torch.Tensor` which contains the individual parameters.

        Examples
        --------
        Load an individual parameters dictionary from a saved file.

        >>> from leaspy import Result
        >>> path = 'outputs/logistic_seed0-mode_real_seed0-individual_parameter.pt'
        >>> individual_parameters = Result.load_individual_parameters_from_torch(path)
        """
        if verbose:
            print("Load from torch file")
        individual_parameters = torch.load(path, **args)
        for key, val in individual_parameters.items():
            if type(val) != torch.Tensor:
                individual_parameters[key] = torch.tensor(val, dtype=torch.float32)
            if individual_parameters[key].ndim != 2:
                individual_parameters[key] = individual_parameters[key].unsqueeze(-1)
        return individual_parameters

    @staticmethod
    def load_individual_parameters(path_or_df, verbose=True, **args):
        """
        Load individual parameters from a :class:`pandas.DataFrame`, a csv, a json file or a torch file.

        Parameters
        ----------
        path_or_df : str or :class:`pandas.DataFrame`
            The file's path or a DataFrame containing the individual parameters.
        verbose : bool (default True)
        args : Any
            Parameters to pass to the corresponding load fonction.

        Returns
        -------
        dict [str, :class:`torch.Tensor`]
            A dictionary of torch.tensor which contains the individual parameters.
        """
        if type(path_or_df) == pd.DataFrame:
            return Result.load_individual_parameters_from_dataframe(path_or_df)
        elif type(path_or_df) == str:
            file_extension = os.path.splitext(path_or_df)[-1]
            if file_extension == '.csv':
                return Result.load_individual_parameters_from_csv(path_or_df, verbose=verbose, **args)
            elif file_extension == '.json':
                return Result.load_individual_parameters_from_json(path_or_df, verbose=verbose, **args)
            else:
                if file_extension not in ('.pt', '.p'):
                    warnings.warn('File extension not recognized (got "%s"). Try torch.load by default.',
                                  RuntimeWarning, stacklevel=2)
                return Result.load_individual_parameters_from_torch(path_or_df, verbose=verbose, **args)
        else:
            raise TypeError("The given input must be a pandas.DataFrame or a string giving the path of the file "
                            "containing the individual parameters!")

    @staticmethod
    def load_result(data, individual_parameters, cofactors=None, verbose=True, **args):
        """
        Load a `Result` class object from two file - one for the individual data & one for the individual parameters.

        Parameters
        ----------
        data : str or :class:`pandas.DataFrame` or :class:`.Data`
            The file's path or a DataFrame containing the features' scores.
        individual_parameters :  str or :class:`pandas.DataFrame`
            The file's path or a DataFrame containing the individual parameters.
        cofactors : str or :class:`pandas.DataFrame`, optional (default None)
            The file's path or a DataFrame containing the individual cofactors.
            The ID must be in index! Thus, the shape is (n_subjects, n_cofactors).
        verbose : bool (default True)
        args : Any
            Parameters to pass to result.load_individual_parameters static method.

        Returns
        -------
        `Result`
            A Result class object which contains the individual parameters and the individual data.

        Examples
        --------
        Launch an individual parameters estimation, save it and reload it.

        >>> from leaspy import AlgorithmSettings, Data, Leaspy, Result
        >>> leaspy_logistic = Leaspy('logistic')
        >>> data = Data.from_csv_file('data/my_leaspy_data.csv')
        >>> model_settings = AlgorithmSettings('mcmc_saem', seed=0)
        >>> personalize_settings = AlgorithmSettings('mode_real', seed=0)
        >>> leaspy_logistic.fit(data, model_settings)
        >>> individual_results = leaspy_logistic.personalize(data, model_settings)
        >>> path_data = 'data/my_leaspy_data.csv'
        >>> path_individual_parameters = 'outputs/logistic_seed0-mode_real_seed0-individual_parameter.json'
        >>> individual_results.data.to_dataframe().to_csv(path_data)
        >>> individual_results.save_individual_parameters_json(path_individual_parameters)
        >>> individual_parameters = Result.load_result(path_data, path_individual_parameters)
        """
        if type(data) == Data:
            pass
        elif type(data) == str:
            data = Data.from_csv_file(data)
        elif type(data) == pd.DataFrame:
            data = Data.from_dataframe(data)
        else:
            raise TypeError("The given `data` input must be a pandas.DataFrame or a string giving the path of the file "
                            "containing the features' scores! You gave an object of type %s" % str(type(data)))

        if cofactors is not None:
            if type(cofactors) == str:
                cofactors_df = pd.read_csv(cofactors, index_col=0)
            elif type(cofactors) == pd.DataFrame:
                cofactors_df = cofactors.copy()
            else:
                raise TypeError("The given `cofactors` input must be a pandas.DataFrame or a string giving the path of "
                                "the file containing the cofactors! You gave an object of type %s" %
                                str(type(cofactors)))
            data.load_cofactors(cofactors_df, cofactors_df.columns.to_list())

        individual_parameters = Result.load_individual_parameters(individual_parameters, verbose=verbose, **args)
        return Result(data, individual_parameters)

    def get_error_distribution_dataframe(self, model, cofactors=None):
        """
        Get signed residual distribution per patient, per sub-score & per visit. Each residual is equal to the
        modeled data minus the observed data.

        Parameters
        ----------
        model : :class:`~.models.abstract_model.AbstractModel`
        cofactors : str, list [str], optional (default None)
            Contains the cofactors' names to be included in the DataFrame. By default, no cofactors are returned.
            If cofactors == "all", all the available cofactors are returned.

        Returns
        -------
        residuals_dataframe : :class:`pandas.DataFrame`

        Examples
        --------
        Get mean absolute error per feature:

        >>> from leaspy import AlgorithmSettings, Data, Leaspy
        >>> data = Data.from_csv_file("/my/data/path")
        >>> leaspy_logistic = Leaspy('logistic')
        >>> settings = AlgorithmSettings("mcmc_saem")
        >>> leaspy_logistic.calibrate(data, settings)
        >>> settings = AlgorithmSettings("mode_real")
        >>> results = leaspy_logistic.personalize(data, settings)
        >>> residuals_dataframe = results.get_error_distribution_dataframe(model)
        >>> residuals_dataframe[results.data.headers].abs().mean()
        """
        residuals_dataset = Dataset(self.data)
        residuals_dataset.values = model.compute_individual_tensorized(residuals_dataset.timepoints,
                                                                       self.individual_parameters) \
                                   - residuals_dataset.values
        residuals_dataframe = residuals_dataset.to_pandas().set_index('ID')

        if cofactors is not None:
            if type(cofactors) == str:
                if cofactors == "all":
                    cofactors_list = self.data.cofactors
                else:
                    cofactors_list = [cofactors]
            elif type(cofactors) == list:
                cofactors_list = cofactors
            else:
                raise TypeError("The given `cofactors` input must be a string or a list of strings! "
                                "You gave an object of type %s" % str(type(cofactors)))
            cofactors_df = self.data.to_dataframe(cofactors=cofactors).groupby('ID').first()[cofactors_list]
            residuals_dataframe = residuals_dataframe.join(cofactors_df)
        return residuals_dataframe

    ###############################################################
    # DEPRECATION WARNINGS
    # These following methods will be removed in a future release
    ###############################################################

    @staticmethod
    def get_cofactor_states(cofactors):
        """
        .. deprecated:: 1.0
        Given a list of string return the list of unique elements.

        Parameters
        ----------
        cofactors : list[str]
            Distribution list of the cofactors.

        Returns
        -------
        list
            Uniques occurrence of the input vector.
        """
        warnings.warn("This method will soon be removed!", DeprecationWarning)

        result = []
        for state in cofactors:
            if state not in result:
                result.append(state)
        result.sort()
        return result

    def get_parameter_distribution(self, parameter, cofactor=None):
        """
        .. deprecated:: 1.0
        Return the wanted parameter distribution (one distribution per covariate state).

        Parameters
        ----------
        parameter : str
            The wanted parameter's name (ex: 'xi', 'tau' ...).
        cofactor : str, optional (default None)
            The wanted cofactor's name.

        Returns
        -------
        list[float] or  dict[str, *]

        Notes
        -----
        If ``cofactor is None``:
            * If the parameter is univariate => return a list the parameter's distribution:
                list[float]
            * If the parameter is multivariate => return a dictionary:
                {'parameter1': distribution of parameter variable 1, 'parameter2': ...}

        If ``cofactor is not None``:
            * If the parameter is univariate => return a dictionary:
                {'cofactor1': parameter distribution such that patient.covariate = covariate1, 'cofactor2': ...}
            * If the parameter is multivariate => return a dictionary:
                {'cofactor1': {'parameter1': ..., 'parameter2': ...}, 'cofactor2': { ...}, ...}
        """
        warnings.warn("This method will soon be removed!", DeprecationWarning)

        parameter_distribution = self.individual_parameters[parameter]  # torch.tensor class object
        # parameter_distribution is of size (N_subjects, N_dimension_of_parameter)

        # Check the tensor's dimension is <= 2
        if parameter_distribution.ndimension() > 2:
            raise ValueError('The chosen parameter %s is a tensor of dimension %d - it must be 1 or 2 dimensional!' %
                             (parameter, parameter_distribution.ndimension()))
        ##############################################
        # If there is no cofactor to take into account
        ##############################################
        if cofactor is None:
            # If parameter is 1-dimensional
            if parameter_distribution.shape[1] == 1:
                # return a list of length = N_subjects
                parameter_distribution = parameter_distribution.view(-1).tolist()
            # Else transpose it and split it in a dictionary
            else:
                # return {'parameter1': distribution of parameter variable 1, 'parameter2': ... }
                parameter_distribution = {parameter + str(i): val for i, val in
                                          enumerate(parameter_distribution.transpose(0, 1).tolist())}
            return parameter_distribution

        ############################################################
        # If the distribution as asked for different cofactor values
        ############################################################
        # Check if the cofactor exist
        if cofactor not in self.data[0].cofactors.keys():
            raise ValueError("The cofactor '%s' do not exist. Here are the available cofactors: %s" %
                             (cofactor, list(self.data[0].cofactors.keys())))
        # Get possible covariate stats
        # cofactors = [_.cofactors[cofactor] for _ in self.data if _.cofactors[cofactor] is not None]
        cofactors = self.get_cofactor_distribution(cofactor)
        cofactor_states = self.get_cofactor_states(cofactors)

        # Initialize the result
        distributions = {}

        # If parameter 1-dimensional
        if parameter_distribution.shape[1] == 1:
            parameter_distribution = parameter_distribution.view(-1).tolist()  # ex: [1, 2, 3]
            # Create one entry per cofactor state
            for p in cofactor_states:
                if p not in distributions.keys():
                    distributions[p] = []
                # For each covariate state, get parameter distribution
                for i, v in enumerate(parameter_distribution):
                    if self.data[i].cofactors[cofactor] == p:
                        distributions[p].append(v)
                        # return {'cofactor1': ..., 'cofactor2': ...}
        else:
            # Create one dictionary per cofactor state
            for p in cofactor_states:
                if p not in distributions.keys():
                    # Create one dictionary per parameter dimension
                    distributions[p] = {parameter + str(i): [] for i in range(parameter_distribution.shape[1])}
                # Fill these entries by the corresponding values of the corresponding subject
                for i, v in enumerate(parameter_distribution.tolist()):
                    if self.data[i].cofactors[cofactor] == p:
                        for j, key in enumerate(distributions[p].keys()):
                            distributions[p][key].append(v[j])
                            # return {'cofactor1': {'parameter1': .., 'parameter2': ..}, 'cofactor2': { .. }, .. }
        return distributions

    def get_cofactor_distribution(self, cofactor):
        """
        .. deprecated:: 1.0
        Get the list of the cofactor's distribution.

        Parameters
        ----------
        cofactor : str
            Cofactor's name

        Returns
        -------
        list
            Cofactor's distribution.
        """
        warnings.warn("This method will soon be removed!", DeprecationWarning)

        return [d.cofactors[cofactor] for d in self.data]

    def get_patient_individual_parameters(self, idx):
        """
        .. deprecated:: 1.0
        Get the dictionary of the wanted patient's individual parameters

        Parameters
        ----------
        idx : str
            ID of the wanted patient

        Returns
        -------
        dict[param_name:str, `torch.Tensor`]
            Patient's individual parameters
        """
        warnings.warn("This method will soon be removed!", DeprecationWarning)

        # indices = list(self.data.individuals.keys())
        # idx_number = int(
        #     np.where(np.array(indices) == idx)[0])
        idx_number = [idx_nbr for idx_nbr, idxx in self.data.iter_to_idx.items() if idxx == idx][0]

        patient_dict = dict.fromkeys(self.individual_parameters.keys())

        for variable_ind in list(self.individual_parameters.keys()):
            patient_dict[variable_ind] = self.individual_parameters[variable_ind][idx_number]

        return patient_dict
