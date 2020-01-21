import numpy as np
import torch
from scipy import stats
from sklearn.preprocessing import StandardScaler

from leaspy.algo.abstract_algo import AbstractAlgo
from leaspy.inputs.data.data import Data
from leaspy.inputs.data.result import Result


class SimulationAlgorithm(AbstractAlgo):
    """
    SimulationAlgorithm object class.
    This algorithm simulate new data given existing one by learning the individual parameters joined distribution.

    You can choose to only learn the distribution of a group of patient. To do so, choose the cofactor and the cofactor
    state of the wanted patient in the settings. Exemple - For Alzheimer patient, you can load a genetic cofactor
    informative of the APOE4 carriers. Choose cofactor 'genetic' and cofactor_state 'APOE4' to simulate only
    APOE4 carriers.

    Attributes
    ----------
    algo_parameters : `dict`
        Contains the algorithm's parameters.
    bandwidth_method : `float`, `str`, `callable`, optional
        Bandwidth argument used in scipy.stats.gaussian_kde in order to learn the patients' distribution.
    cofactor : `str` (default = None)
        The cofactor used to select the wanted group of patients (ex - 'genes'). It must correspond to an existing
        cofactor in the attribute `Data` of the input `result` of the ``run`` method. See ``SimulationAlgorithm.run``
        documentation.
    cofactor_state : `str` (default None) TODO: check that the loaded cofactors are converted into strings!
        The cofactor state used to select the  wanted group of patients (ex - 'APOE4'). It must correspond to an
        existing cofactor state in the attribute `Data` of the input `result` of the ``run`` method. See
        ``SimulationAlgorithm.run`` documentation.
    features_bounds : `bool`, `dict` : [`str`, `float`, `float`] (default False)
        Specify if the scores of the generated subjects must be bounded. This parameter can express in two way:
        - `bool` : the bounds are the maximum and minimum scores observed in the ``results.data`` object.
        - `dict` : the user has to set the min and max bounds for every features. For example:
                    ``{'feature1': (score_min, score_max), 'feature2': (score_min, score_max), ...}``
    mean_number_of_visits : `int` (default 6)
        Average number of visits of the simulated patients.
        Examples - choose 5 => in average, a simulated patient will have 5 visits.
    name : "simulation"
        Algorithm's name.
    noise : `float` or `bool` (default True)
        Wanted level of noise in the generated scores - noise of zero will lead to patients having "perfect progression"
        of their scores, i.e. following exactly a logistic curve. If set to ``True``, this value is set as the
        model's `noise_std` parameter.
    number_of_subjects: `int`
        Number of subject to simulate.
    seed: `int`
        Used by numpy.random & torch.random for reproducibility.
    sources_method : `str` {'full_kde', 'normal_sources'}
        - ``"full_kde"`` : the sources are also learned with the gaussian kernel density estimation.
        - ``"normal_sources"`` : the sources are generated as multivariate normal distribution linked with the other
        individual parameters.
    std_number_of_visits : `int`
        Standard deviation used into the generation of the number of visits per simulated patient.

    Methods
    -------
    run(model, results)
        Run the simulation of new patients for some given leaspy object result & model.
    """

    def __init__(self, settings):
        """
        Process initializer function that is called by Leaspy().simulate.

        Parameters
        ----------
        settings : `leaspy.inputs.algorithm_settings.AlgorithmSettings` class object
            Set the class attributes.

        Raises
        ------
        ValueError
            If ``settings.parameters['sources_method']`` is not one of the two option allowed -
            "full_kde" or "normal_sources".
        TypeError
            If the type of ``settings.parameters['features_bounds']`` is not `bool` or `dict`.
        """
        super().__init__()

        # TODO: put it in abstract_algo + add settings=None in AbstractAlgo __init__ method
        self.algo_parameters = settings.parameters
        self.name = settings.name
        self.seed = settings.seed

        self._initialize_seed(self.seed)

        self.bandwidth_method = settings.parameters['bandwidth_method']
        self.cofactor = settings.parameters['cofactor']
        self.cofactor_state = settings.parameters['cofactor_state']
        self.features_bounds = settings.parameters['features_bounds']
        self.mean_number_of_visits = settings.parameters['mean_number_of_visits']
        self.noise = settings.parameters['noise']
        self.number_of_subjects = settings.parameters['number_of_subjects']
        self.sources_method = settings.parameters['sources_method']
        self.std_number_of_visits = settings.parameters['std_number_of_visits']

        if self.sources_method not in ("full_kde", "normal_sources"):
            raise ValueError('The "sources_method" parameter must be "full_kde" or "normal_sources"!')

        if type(self.features_bounds) not in [bool, dict]:
            raise TypeError('The type of the "features_bounds" parameter must be %s or %s, not %s!'
                            % (str(bool), str(dict), str(type(self.features_bounds))))

    def _check_cofactors(self, data):
        """
        Check the value.

        Parameters
        ----------
        data : `leaspy.inputs.data.Data` class object
            Contains the cofactors and cofactors' states.

        Raises
        ------
        ValueError
            Raised if the parameters "cofactor" and "cofactor_state" do not receive a valid value.
        """
        def reformat_str(string, replace=True):
            result = string.replace('[', "").replace(']', "")
            if replace:
                result = result.replace(',', " or")
            return result

        cofactors = {}
        for ind in data.individuals.values():
            if bool(ind.cofactors):
                for key, val in ind.cofactors.items():
                    if key in cofactors.keys():
                        cofactors[key] += [val]
                    else:
                        cofactors[key] = [val]
        for key, val in cofactors.items():
            cofactors[key] = np.unique(val)

        if self.cofactor not in cofactors.keys():
            raise ValueError('The input "cofactor" parameter %s does not correspond to any cofactor in your data! '
                             'The available cofactor(s) are %s.'
                             % (self.cofactor, reformat_str(str(list(cofactors.keys())))))
        if self.cofactor_state not in cofactors[self.cofactor]:
            raise ValueError('The input "cofactor_state" parameter "%s" does not correspond to any cofactor state'
                             ' in your data! The available cofactor states for "%s" are %s.'
                             % (self.cofactor_state, self.cofactor, reformat_str(str(cofactors[self.cofactor]))))

    @staticmethod
    def _get_mean_and_covariance_matrix(m):
        """
        Compute the empirical mean and covariance matrix of the input. Twice faster than `numpy.cov`.

        Parameters
        ----------
        m : `torch.Tensor`
            Input matrix - one row per individual parameter distribution (xi, tau etc).

        Returns
        -------
        mean : `torch.Tensor`
            Mean by variable.
        covariance :  `torch.Tensor`
            Covariance matrix.
        """
        m_exp = torch.mean(m, dim=0)
        x = m - m_exp[None, :]
        cov = 1 / (x.size(0) - 1) * x.t() @ x
        return m_exp, cov

    @staticmethod
    def _sample_sources(bl, tau, xi, source_dimension, df_mean, df_cov):
        """
        Simulate individual sources given baseline age bl, time-shift tau, log-acceleration xi & sources dimension.

        Parameters
        ----------
        bl : `float`
            Baseline age of the simulated patient.
        tau : `float`
            Time-shift of the simulated patient.
        xi : `float`
            Log-acceleration of the simulated patient.
        source_dimension : `int`
            Sources' dimension of the simulated patient.
        df_mean : `torch.Tensor`
            Mean values per individual parameter type (bl_mean, tau_mean, xi_mean & sources_means) (1-dimensional).
        df_cov : `torch.Tensor`
            Empirical covariance matrix of the individual parameters (2-dimensional).

        Returns
        -------
        `torch.Tensor`
            Sources of the simulated patient (1-dimensional).
        """
        x_1 = torch.tensor([bl, tau, xi], dtype=torch.float32)

        mu_1 = df_mean[:3].clone()
        mu_2 = df_mean[3:].clone()

        sigma_11 = df_cov.narrow(0, 0, 3).narrow(1, 0, 3).clone()
        sigma_22 = df_cov.narrow(0, 3, source_dimension).narrow(1, 3, source_dimension).clone()
        sigma_12 = df_cov.narrow(0, 3, source_dimension).narrow(1, 0, 3).clone()

        mean_cond = mu_2 + sigma_12 @ sigma_11.inverse() @ (x_1 - mu_1)
        cov_cond = sigma_22 - sigma_12 @ sigma_11.inverse() @ sigma_12.transpose(0, -1)

        return torch.distributions.multivariate_normal.MultivariateNormal(mean_cond, cov_cond).sample()

    def _get_number_of_visits(self):
        """
        Simulate number of visits for a new simulated patient based of attributes `mean_number_of_visits' &
        'std_number_of_visits'.

        Returns
        -------
        number_of_visits : `int`
            Number of visits.
        """
        # Generate a number of visit around the mean_number_of_visits
        number_of_visits = int(self.mean_number_of_visits)
        if self.mean_number_of_visits != 0:
            number_of_visits += int(torch.normal(torch.tensor(0., dtype=torch.float32),
                                                 torch.tensor(self.std_number_of_visits, dtype=torch.float32)).item())
        return number_of_visits

    def _get_features_bounds(self, results_object):
        """
        Get the bound of the baseline scores of the generated patients. Each generated patient whose baseline is outside
        these bounds are discarded.

        Parameters
        ----------
        results_object : a `leaspy.inputs.data.result.Result` class object

        Returns
        -------
        features_min : `np.ndarray`
            Lowest score allowed per feature - sorted accordingly to the features in ``result.data.headers``.
        features_max : `np.ndarray`
            Highest score allowed per feature - sorted accordingly to the features in ``result.data.headers``.
        """
        features_min = np.zeros(len(results_object.data.headers))
        features_max = np.ones(len(results_object.data.headers))
        if type(self.features_bounds) is dict:
            assert results_object.data.headers == list(self.features_bounds.keys()), \
                'The keys of your input "features_bounds" do not match the headers of your data!' \
                + '\nThe data headers - %s' % str(results_object.data.headers) \
                + '\nYour "features_bounds" input - %s' % str(list(self.features_bounds.keys()))
            for i, key in enumerate(results_object.data.headers):
                features_min[i] = self.features_bounds[key][0]
                features_max[i] = self.features_bounds[key][1]
            return features_min, features_max
        else:
            df_scores = results_object.data.to_dataframe().groupby('ID').first()
            return df_scores.iloc[:, 1:].min().values, df_scores.iloc[:, 1:].max().values

    def _get_timepoints(self, bl):
        """
        Generate the time points of a subject given his baseline age.

        Parameters
        ----------
        bl : float
            The subject's baseline age.

        Returns
        -------
        ages : `list` [`float`]
            Contains the subject's time points.
        """
        number_of_visits = self._get_number_of_visits()
        if number_of_visits == 1:
            ages = [bl]
        elif number_of_visits == 2:
            ages = [bl, bl + 0.5]
        else:
            ages = [bl, bl + 0.5] + [bl + j for j in range(1, number_of_visits - 1)]
        return ages

    def _simulate_individual_parameters(self, model, number_of_simulated_subjects, kernel, ss, get_sources,
                                        df_mean, df_cov):
        """
        Compute the simulated individual parameters and timepoints.

        Parameters
        ----------
        model : a `leaspy.model` class object
        number_of_simulated_subjects : `int`
        kernel : a `scipy.stats.gaussian_kde` class object
        ss : a `sklearn.preprocessing.StandardScaler` class object
        get_sources : `bool`
        df_mean : `torch.Tensor`
        df_cov : `torch.Tensor`

        Returns
        -------
        simulated_parameters : `dict` [`str`, `numpy.ndarray`]
            Contains the simulated parameters.
        timepoints : `list` [`float`]
            Contains the ages of the subjects for all their visits - 2D list with one row per simulated subject.
        """
        samples = kernel.resample(number_of_simulated_subjects).T
        samples = ss.inverse_transform(samples)  # A 2D numpy.ndarray - of shape n_subjects x n_features

        timepoints = list(map(self._get_timepoints, samples.T[0]))
        # timempoints is a 2D list - one row per simulated subject

        simulated_parameters = {'tau': samples.T[1], 'xi': samples.T[2]}
        # xi & tau are 1D array - one value per simulated subject
        if get_sources:
            if self.sources_method == "full_kde":
                simulated_parameters['sources'] = samples.T[3:].T
            elif self.sources_method == "normal_sources":
                # Generate sources
                def simulate_sources(x):
                    return self._sample_sources(x[0], x[1], x[2], model.source_dimension, df_mean, df_cov).numpy()
                simulated_parameters['sources'] = np.apply_along_axis(simulate_sources, axis=1, arr=samples)
                # sources is 2D array - of shape n_subjects x n_sources

        return simulated_parameters, timepoints

    def _simulate_subjects(self, simulated_parameters, timepoints, model):
        """
        Compute the simulated scores.

        Parameters
        ----------
        model : a `leaspy.model` class object
        simulated_parameters : `dict` [`str`, `numpy.ndarray`]
            Contains the simulated parameters.
        timepoints : `list` [`float`]
            Contains the ages of the subjects for all their visits - 2D list with one row per simulated subject.

        Returns
        -------
        features_values : `list` [`numpy.ndarray`]
            Contains the scores of all the subjects for all their visits. Contains one entry per subject, each of
            them is a 2D-numpy.ndarray of shape n_visits x n_features.
        """
        features_values = []
        for i in range(len(timepoints)):
            indiv_param = {key: val[i] for key, val in simulated_parameters.items()}
            indiv_param['sources'] = indiv_param['sources'].tolist()
            observations = model.compute_individual_trajectory(timepoints[i], indiv_param)
            # Add the desired noise
            if self.noise:
                if type(self.noise) in [int, float]:
                    noise = self.noise
                else:
                    noise = torch.distributions.Normal(loc=0, scale=model.parameters['noise_std']).sample(
                        observations.shape)
                observations += noise
                observations = observations.clamp(0, 1)

            observations = observations.squeeze(0).detach().numpy()
            features_values.append(observations)

        return features_values

    @staticmethod
    def _get_bounded_subject(features_values, features_min, features_max):
        """
        Select the subject whose scores are within the features boundaries.

        Parameters
        ----------
        features_values : `list` [`np.ndarray`]
            Contains the scores of all the subjects of all their visits. Each element correspond to a simulated
            subject, these elements are of shape n_vists x n_features.
        features_min : `np.ndarray`
            Lowest score allowed per feature - sorted accordingly to the features in ``result.data.headers``.
        features_max : `np.ndarray`
            Highest score allowed per feature - sorted accordingly to the features in ``result.data.headers``.

        Returns
        -------
        indices_of_accepted_simulated_subjects : `list` [`float`]
        `list` [`np.ndarray`]
            Contains the scores of all the subjects whose scores are within the features boundaries.
        """
        def _test_subject(bl_score, features_min, features_max):
            return (features_min <= bl_score).all() & (bl_score <= features_max).all()

        baseline_scores = np.array([scores[0] for scores in features_values])
        indices_of_accepted_simulated_subjects = [i for i, bl_score in enumerate(baseline_scores)
                                                  if _test_subject(bl_score, features_min, features_max)]
        return indices_of_accepted_simulated_subjects, [val for i, val in enumerate(features_values)
                                                        if i in indices_of_accepted_simulated_subjects]

    def run(self, model, results):
        """
        Run simulation - learn joined distribution of patients' individual parameters and return a results object
        containing the simulated individual parameters and the simulated scores.

        Parameters
        ----------
        model : leaspy.model class object
            Model used to compute the population & individual parameters. It contains the population parameters.
        results : `leaspy.inputs.data.result.Result` class object
            Object containing the computed individual parameters.

        Notes
        -----
        In simulation_settings, one can specify in the parameters the cofactor & cofactor_state. By doing so,
        one can simulate based only on the subject for the given cofactor & cofactor's state.

        By default, all the subject in results.data are used to estimate the joined distribution.

        Returns
        -------
        `leaspy.inputs.data.result.Result` class object
            Contains the simulated individual parameters & individual scores.
        """
        if self.cofactor is not None:
            self._check_cofactors(results.data)

        # Get individual parameters & baseline ages - for joined density estimation
        # Get individual parameters (optional - & the cofactor states)
        df_ind_param = results.get_dataframe_individual_parameters(cofactors=self.cofactor)
        if self.cofactor_state:
            # Select only subjects with the given cofactor state
            df_ind_param = df_ind_param[df_ind_param[self.cofactor] == self.cofactor_state]
            # Remove the cofactor column
            df_ind_param = df_ind_param.loc[:, df_ind_param.columns != self.cofactor_state]
        # Add the baseline ages
        df_ind_param = results.data.to_dataframe().groupby('ID').first()[['TIME']].join(df_ind_param, how='right')
        # At this point, df_ind_param.columns = ['TIME', 'tau', 'xi', 'sources_0', 'sources_1', ..., 'sources_n']
        distribution = df_ind_param.values
        # Get joined density estimation of bl, tau, xi (and the sources if the model is not univariate)

        get_sources = (model.name != 'univariate')
        if get_sources & (self.sources_method == "normal_sources"):
            # Sources are not learned with a kernel density estimator
            distribution = distribution[:, :3]
            # Get mean by variable & covariance matrix
            # Needed to sample new sources from simulated bl, tau & xi
            df_mean, df_cov = self._get_mean_and_covariance_matrix(torch.from_numpy(df_ind_param.values))
        else:
            df_mean, df_cov = None, None

        # Normalize by variable then transpose to learn the joined distribution
        ss = StandardScaler()
        # fit_transform receive an numpy array of shape [n_samples, n_features]
        distribution = ss.fit_transform(distribution).T
        # gaussian_kde receive an numpy array of shape [n_features, n_samples]
        kernel = stats.gaussian_kde(distribution, bw_method=self.bandwidth_method)

        if self.features_bounds:
            number_of_simulated_subjects = 10 * self.number_of_subjects
        else:
            number_of_simulated_subjects = self.number_of_subjects

        simulated_parameters, timepoints = self._simulate_individual_parameters(
            model, number_of_simulated_subjects, kernel, ss, get_sources, df_mean, df_cov)

        features_values = self._simulate_subjects(simulated_parameters, timepoints, model)

        if self.features_bounds:
            # Handle bounds on the generated features
            features_min, features_max = self._get_features_bounds(results)
            #  Test the boundary conditions
            indices_of_accepted_simulated_subjects, features_values = self._get_bounded_subject(
                features_values, features_min, features_max)
            for key, val in simulated_parameters.items():
                simulated_parameters[key] = val[indices_of_accepted_simulated_subjects]

            timepoints = [v for i, v in enumerate(timepoints) if i in indices_of_accepted_simulated_subjects]

            while len(features_values) < self.number_of_subjects:
                # Complete to attain the goal
                number_of_simulated_subjects *= self.number_of_subjects / len(indices_of_accepted_simulated_subjects)

                simulated_parameters_bis, timepoints_bis = self._simulate_individual_parameters(
                    model, number_of_simulated_subjects, kernel, ss, get_sources, df_mean, df_cov)

                features_values_bis = self._simulate_subjects(simulated_parameters_bis, timepoints_bis, model)

                #  Test the boundary conditions
                indices_of_accepted_simulated_subjects_bis, features_values_bis = self._get_bounded_subject(
                    features_values_bis, features_min, features_max)
                for val in simulated_parameters_bis.values():
                    simulated_parameters_bis[key] = val[indices_of_accepted_simulated_subjects_bis]
                timepoints_bis = [v for i, v in enumerate(timepoints_bis)
                                  if i in indices_of_accepted_simulated_subjects_bis]

                # Concatenate with previous generated subjects
                features_values += features_values_bis
                timepoints += timepoints_bis
                for key in simulated_parameters:
                    simulated_parameters[key] = np.concatenate(simulated_parameters[key],
                                                               simulated_parameters_bis[key])

        # Take only the `number_of_simulated_subjects` first subjects
        timepoints = timepoints[:self.number_of_subjects]
        for key, val in simulated_parameters.items():
            if key in ['tau', 'xi']:
                simulated_parameters[key] = torch.from_numpy(val).view(-1, 1)[:self.number_of_subjects]
            if key == 'sources':
                simulated_parameters[key] = torch.from_numpy(val)[:self.number_of_subjects]

        indices = ['Generated_subject_' + str(i) for i in range(1, self.number_of_subjects + 1)]

        simulated_scores = Data.from_individuals(indices=indices,
                                                 timepoints=timepoints,
                                                 values=features_values,
                                                 headers=results.data.headers)
        return Result(data=simulated_scores,
                      individual_parameters=simulated_parameters,
                      noise_std=self.noise)
