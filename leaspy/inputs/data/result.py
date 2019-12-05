import torch
import pandas as pd


class Result:
    """
    Result object class.
    Used as output by personalize algorithm & simulation algorithm.

    Attributes
    ----------
    data: leaspy.inputs.data.data.Data class object
        Object containing the idx, time-points and observations of the patients.
    individual_parameters: dict
        Contains log-acceleration 'xi', time-shifts 'tau' & 'sources' (dictionary of torch.tensor).
    ID_to_idx: dict
        The keys are the individual ID & the items are their respective ordered position in the data file given
        by the user. This order remains the same during the computation.
        Example - in Result.individual_parameters['xi'], the first element corresponds to the first patient in ID_to_idx.
    noise_std: float
        Desired noise standard deviation level.

    Methods
    -------
    get_dataframe_individual_parameters(cofactors)
        Return the dataframe of the individual parameters.
    get_torch_individual_parameters()
        Getter function for the individual parameters.

    Depreciated in a futur release:
        get_cofactor_distribution(cofactor)
            Get the list of the cofactor's distribution.
        get_cofactor_states(cofactors)
            Given a list of string return the list of unique elements.
        get_parameter_distribution(parameter, cofactor=None)
            Return the wanted parameter distribution (one distribution per covariate state).
        get_patient_individual_parameters(idx)
            Get the dictionary of the wanted patient's individual parameters.
        get_error_distribution(model, cofactor=None, aggregate_subscores=False, aggregate_visits=False)
            Get error distribution per patient. By default, return one error value per patient & per subscore & per visit.
    """

    def __init__(self, data, individual_parameters, noise_std=None):
        """
        Process the initializer function - called by Leaspy.inputs.data.result.Result

        Parameters
        ----------
        data: leaspy.inputs.data.data class object
            Object containing the idx, time-points and observations of the patients
        individual_parameters: dictionary of torch.tensor
            Contains log-acceleration 'xi', time-shifts 'tau' & 'sources'
        noise_std: float
            Desired noise standard deviation level
        """

        self.data = data
        self.individual_parameters = individual_parameters
        self.ID_to_idx = {key: i for i, key in enumerate(data.individuals)}
        self.noise_std = noise_std

    # def load_covariables(self, covariables, csv):
    #    self.covariables = covariables

    def get_torch_individual_parameters(self):
        """
        Getter function for the individual parameters.

        Returns
        -------
        torch.Tensor
            Contains the individual parameters
        """

        # if ID is not None:
        #     liste_idt = [self.ID_to_idx[id_patient] for id_patient in ID]
        #     ind_parameters = {key : value[liste_idt] for key, value in self.individual_parameters.items()}
        # else:
        #     ind_parameters = self.individual_parameters.copy()
        return self.individual_parameters.copy()

    def get_dataframe_individual_parameters(self, cofactors=None):
        """
        Return the dataframe of the individual parameters. Each row corresponds to a subject. The columns correspond
        (in this order) to the subjects' ID, the individual parameters (one column per individual parameter) & the
        cofactors (one column per cofactor).

        Parameters
        ----------
        cofactors: str or list
            Contains the cofactor(s) to join to the output dataframe.

        Notes
        -----
        The cofactors must be present in the leaspy data object stored into the .data attribute of the result instance.
        See the exemple.

        Returns
        -------
        pandas.DataFrame
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

    @staticmethod
    def get_cofactor_states(cofactors):
        """
        Given a list of string return the list of unique elements

        Parameters
        ----------
        cofactors: list
            Distribution list of the cofactors

        Returns
        -------
        list of strings
            Uniques occurrence of the input vector
        """
        result = []
        for state in cofactors:
            if state not in result:
                result.append(state)
        result.sort()
        return result

    def get_parameter_distribution(self, parameter, cofactor=None):
        """
        Return the wanted parameter distribution (one distribution per covariate state)

        Parameters
        ----------
        parameter: string
            The wanted parameter's name (ex: 'xi', 'tau' ...)
        cofactor: string
            The wanted cofactor's name

        Returns
        -------
        list of floats, dict of list of float
            If no cofactor is given & the parameter is univariate => return a list the parameter's distribution
            If no cofactor is given & the parameter is multivariate => return a dictionary =
                {'parameter1': distribution of parameter variable 1, 'parameter2': ...}
            If a cofactor is given & the parameter is univariate => return a dictionary =
                {'cofactor1': parameter distribution such that patient.covariate = covariate1, 'cofactor2': ...}
            If a cofactor is given & the parameter is multivariate => return a dictionary =
                {'cofactor1': {'parameter1': ..., 'parameter2': ...}, 'cofactor2': { ...}, ...}
        """
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
        Get the list of the cofactor's distribution

        Parameters
        ----------
        cofactor: string
            Cofactor's name

        Returns
        -------
        list of float
            Cofactor's distribution
        """
        return [d.cofactors[cofactor] for d in self.data]

    def get_patient_individual_parameters(self, idx):
        """
        Get the dictionary of the wanted patient's individual parameters

        Parameters
        ----------
        idx: string
            ID of the wanted patient

        Returns
        -------
        dict of torch.tensor
            Patient's individual parameters
        """
        # indices = list(self.data.individuals.keys())
        # idx_number = int(
        #     np.where(np.array(indices) == idx)[0])
        idx_number = [idx_nbr for idx_nbr, idxx in self.data.iter_to_idx.items() if idxx == idx][0]

        patient_dict = dict.fromkeys(self.individual_parameters.keys())

        for variable_ind in list(self.individual_parameters.keys()):
            patient_dict[variable_ind] = self.individual_parameters[variable_ind][idx_number]

        return patient_dict

    def get_error_distribution(self, model, cofactor=None, aggregate_subscores=False, aggregate_visits=False):
        """
        Get error distribution per patient. By default, return one error value per patient & per subscore & per visit.
        Use 'aggregate_subscores' to average error values among subscores.
        Use 'aggregate_visits' to average error values among visits.
        Use both to have one error value per patient.
        Use `cofactor' to cluster the patients by their corresponding cofactor's state.

        Parameters
        ----------
        model: leaspy model class object
        cofactor: string
        aggregate_subscores: boolean (default = False)
            Use 'aggregate_subscores' to average error values among subscores.
        aggregate_visits: boolean (default = False)
            Use 'aggregate_visits' to average error values among visits.

        Returns
        -------
        dict
            If cofactor is None => return a dictionary of torch tensor {'patient1': error1, ...}
            If cofactor is not None => return a dictionary dictionary of torch tensor
            {'cofactor1': {'patient1': error1, ...}, ...}
        """
        error_distribution = {}
        get_sources = (model.name != "univariate")
        for i, (key, patient) in enumerate(self.data.individuals.items()):
            param_ind = {'tau': self.individual_parameters['tau'][i],
                         'xi': self.individual_parameters['xi'][i]}
            if get_sources:
                param_ind['sources'] = self.individual_parameters['sources'][i]

            computed_minus_observations = model.compute_individual_tensorized(
                            torch.tensor(patient.timepoints, dtype=torch.float32).unsqueeze(0), param_ind).squeeze(0)
            computed_minus_observations -= torch.tensor(patient.observations, dtype=torch.float32)

            if aggregate_subscores:
                if aggregate_visits:
                    # One value per patient
                    error_distribution[key] = torch.mean(computed_minus_observations).tolist()
                else:
                    # One value per patient & per subscore
                    error_distribution[key] = torch.mean(computed_minus_observations, 1).tolist()
            elif aggregate_visits:
                # One value per patient & per visit
                error_distribution[key] = torch.mean(computed_minus_observations, 0).tolist()
            else:
                # One value per patient & per subscore & per visit
                error_distribution[key] = computed_minus_observations.tolist()

        if cofactor:
            cofactors = self.get_cofactor_distribution(cofactor)
            result = {state: {} for state in self.get_cofactor_states(cofactors)}
            for key in result.keys():
                result[key] = {patient: error_distribution[patient] for i, patient in
                               enumerate(error_distribution.keys()) if cofactors[i] == key}
            return result  # return {'cofactor1': {'patient1': error1, ...}, ...}
        else:
            return error_distribution  # return {'patient1': error1, ...}
