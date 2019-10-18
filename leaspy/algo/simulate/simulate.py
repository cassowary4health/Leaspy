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
    bandwidth_method = float, str, callable, optionnal
        Bandwith argument used in scipy.stats.gaussian_kde in order to learn the patients' distribution
    cofactor: str (default = None)
        The cofactor used to select the wanted group of patients (ex - 'genes')
    cofactor_state: str - TODO: check that the loaded cofactors are converted into strings!
        The cofactor state used to select the  wanted group of patients (ex - 'APOE4')
    mean_number_of_visits: int
        Average number of visits of the simulated patients (ex - choose 5 => in average, a simulated patient will have 5 visits)
    noise: float
        Wanted level of noise in the generated scores - noise of zero will lead to patients having "perfect progression"
        of their scores, i.e. following exactly a logistic curve
    number_of_subjects: int
        Number of subject to simulate
    std_number_of_visits: float
        Standard deviation used into the generation of the number of visits per simulated patient

    Methods
    -------
    run(model, results)
        Run the simulation of new patients for some given leaspy object result & model.
    """

    def __init__(self, settings):
        """
        Process initializer function that is called by Leaspy.simulate

        Parameters
        ----------
        settings: leaspy.inputs.algorithm_settings class object
            Set the class attributes
        """

        super().__init__()

        self.bandwidth_method = settings.parameters['bandwidth_method']
        self.cofactor = settings.parameters['cofactor']
        self.cofactor_state = settings.parameters['cofactor_state']
        self.mean_number_of_visits = settings.parameters['mean_number_of_visits']
        self.noise = settings.parameters['noise']
        self.number_of_subjects = settings.parameters['number_of_subjects']
        self.std_number_of_visits = settings.parameters['std_number_of_visits']

    @staticmethod
    def _get_covariance_matrix(m):
        """
        Compute the empirical covariance matrix of the input

        Parameters
        ----------
        m: torch tensor
            Input matrix - one row per individual parameter distribution (xi, tau etc.)

        Returns
        -------
        torch tensor
            covariance matrix
        """
        m_exp = torch.mean(m, dim=1)
        x = m - m_exp[:, None]
        cov = 1 / (x.size(1) - 1) * x.mm(x.t())
        return cov

    def _initialize_kernel(self, results=None):
        return 0

    def _sample_sources(self, xi, tau, bl, source_dimension, df_mean, df_cov):
        """
        Simulate individual sources given log-acceleration xi, time-shift tau, baseline time bl & sources dimension

        Parameters
        ----------
        xi: float
            Log-acceleration of the simulated patient
        tau: float
            Time-shift of the simulated patient
        bl: float
            Baseline age of the simulated patient
        source_dimension: int
            Sources' dimension of the simulated patient
        df_mean: 1D torch tensor
            Mean values per individual parameter type (xi_mean, tau_mean, etc.)
        df_cov: 2D torch tensor
            Empirical covariance matrix of the individual parameters

        Returns
        -------
        1D torch tensor
            sources of the simulated patient
        """

        x_1 = torch.tensor([xi, tau, bl], dtype=torch.float32)

        mu_1 = df_mean[:3].clone()
        mu_2 = df_mean[3:].clone()

        sigma_11 = df_cov.narrow(0, 0, 3).narrow(1, 0, 3).clone()
        sigma_22 = df_cov.narrow(0, 3, source_dimension).narrow(1, 3, source_dimension).clone()
        sigma_12 = df_cov.narrow(0, 3, source_dimension).narrow(1, 0, 3).clone()

        mean_cond = mu_2 + torch.matmul(sigma_12, torch.matmul(sigma_11.inverse(), x_1 - mu_1))
        cov_cond = sigma_22 - torch.matmul(sigma_12.matmul(sigma_11.inverse()), sigma_12.transpose(0, -1))

        return torch.distributions.multivariate_normal.MultivariateNormal(mean_cond, cov_cond).sample()

    def _get_number_of_visits(self):
        """
        Simulate number of visits for a new simulated patient based of attributes `mean_number_of_visits' &
        'std_number_of_visits'

        Returns
        -------
        int
            number of visits
        """
        # Generate a number of visit around the mean_number_of_visits
        number_of_visits = int(self.mean_number_of_visits)
        if self.mean_number_of_visits != 0:
            number_of_visits += int(torch.normal(torch.tensor(0, dtype=torch.float32),
                                                 torch.tensor(self.std_number_of_visits, dtype=torch.float32)).item())
        return number_of_visits

    def _get_xi_tau_sources_bl(self, results, get_sources):
        """
        Get individual parameters

        Parameters
        ----------
        results: leaspy.inputs.data.result class object
            Object obtained at the personalization step in order to compute individual parameters
        get_sources: bool
            Needed in order to differentiate univariate models from other - if univariate, ignores 'sources'

        Returns
        -------
        tuple(list[float], list[float], list(list(float)), list[float])
            Tuple containing (in this order) the log-acceleration xi, the time-shift tau, the sources
            (of shape = Number_of_sources x Number_of_subjects) & the baseline age of the patients. The sources are not
            returned if the model is univariate.
        """

        xi = results.get_parameter_distribution('xi', self.cofactor) # list of float
        tau = results.get_parameter_distribution('tau', self.cofactor) # list of float
        if get_sources:
            sources = results.get_parameter_distribution('sources', self.cofactor)
            # {'source1': list of float, 'source2': ..., ...}
        bl = []
        for idx in results.data.individuals.keys():
            ages = results.data.get_by_idx(idx).timepoints
            bl.append(min(ages))

        if self.cofactor is not None:
            # transform {'state1': xi_list1, 'state2': xi_list2, ...} to xi_list
            xi = xi[self.cofactor_state]
            tau = tau[self.cofactor_state]
            if get_sources:
                sources = sources[self.cofactor_state]
            bl = [bl[i] for i, state in enumerate(results.get_cofactor_distribution(self.cofactor))
                  if state == self.cofactor_state]
        if get_sources:
            sources = [sources[key] for key in sources.keys()]  # [[ float ], [ float ], ... ]
            return xi, tau, sources, bl
        else:
            return xi, tau, bl

    def run(self, model, results):
        """
        Run simulation - learn joined distribution of patients' individual parameters and return a results object
        containing the simulated individual data and parameters.

        Parameters
        ----------
        model: leaspy.model class object
            Model used to compute the population parameters
        results: leaspy.inputs.data.result class object
            Object containing the computed individual parameters

        Returns
        -------
        leaspy.inputs.data.result class object
            Contains the simulated individual data and parameters
        """

        get_sources = (model.name != 'univariate')
        # Get individual parameters - for joined density estimation
        if get_sources:
            xi, tau, sources, bl = self._get_xi_tau_sources_bl(results, get_sources)
        else:
            xi, tau, bl = self._get_xi_tau_sources_bl(results, get_sources)

        # Get joined density estimation (sources are not learn in this fashion)
        distribution = [xi, tau, bl]
        distribution = [list(i) for i in zip(*distribution)]  # Transpose it
        ss = StandardScaler()
        kernel = stats.gaussian_kde(ss.fit_transform(distribution).T,  # fit_transform return a numpy array
                                    bw_method=self.bandwidth_method)

        # Generate individual parameters (except sources)
        samples = kernel.resample(self.number_of_subjects).T  # Resample return a numpy.ndarray
        samples = ss.inverse_transform(samples)

        # Initialize simulated scores
        indices, timepoints, values = [], [], []
        # Simulated parameters
        xi, tau, bl = samples.T.tolist()  # Come back to list objects

        if get_sources:
            # Get mean by variable & covariance matrix
            df = torch.tensor([xi, tau, bl] + sources, dtype=torch.float32)  # Concat in a single object
            df_mean = df.mean(dim=1)
            df_cov = self._get_covariance_matrix(df)
            sources = []

        # Generate individual sources, scores, indices & time-points
        for i in range(self.number_of_subjects):
            if get_sources:
                # Generate sources
                sources.append(self._sample_sources(xi[i], tau[i], bl[i], model.source_dimension, df_mean, df_cov).tolist())
            # Generate time-points
            number_of_visits = self._get_number_of_visits()  # xi[i], tau[i], bl[i] - 1, sources[-1])
            if number_of_visits == 1:
                ages = [bl[i]]
            elif number_of_visits == 2:
                ages = [bl[i], bl[i] + 0.5]
            else:
                ages = [bl[i], bl[i] + 0.5] + [bl[i] + i for i in range(1, number_of_visits - 1)]
            timepoints.append(ages)

            # Generate scores
            indiv_param = {'xi': torch.tensor([xi[i]], dtype=torch.float32).unsqueeze(0),
                           'tau': torch.tensor([tau[i]], dtype=torch.float32).unsqueeze(0)}
            if get_sources:
                indiv_param['sources'] = torch.tensor(sources[-1], dtype=torch.float32).unsqueeze(0)

            observations = model.compute_individual_tensorized(torch.tensor(ages, dtype=torch.float32).unsqueeze(0),
                                                               indiv_param)
            # Add the desired noise
            if self.noise:
                noise = torch.distributions.Normal(loc=0, scale=model.parameters['noise_std']).sample(
                    observations.shape)
                observations += noise
                observations = observations.clamp(0, 1)
            values.append(observations.squeeze(0).detach().tolist())
            # Generate indices
            indices.append(i)

        # Return the leaspy.inputs.data.results object
        simulated_parameters = {'xi': torch.tensor(xi, dtype=torch.float32).view(-1, 1),
                                'tau': torch.tensor(tau, dtype=torch.float32).view(-1, 1)}
        if get_sources:
            simulated_parameters['sources'] = torch.tensor(sources, dtype=torch.float32)
        simulated_scores = Data.from_individuals(indices, timepoints, values, results.data.headers)
        return Result(data=simulated_scores,
                      individual_parameters=simulated_parameters,
                      noise_std=self.noise)

        # TODO : Check with Raphaël if he needs the following
        '''

        param = []
        for a in param_ind:
            for i in range(a.shape[1]):
                param.append(a[:, i].detach().numpy())
        param = np.array(param)

        kernel = scipy.stats.gaussian_kde(param)

        # Get metrics from Data
        n_points = np.mean(dataset.nb_observations_per_individuals)
        data_sim = pd.DataFrame(columns=['ID', 'TIME'] + dataset.headers)
        indiv_param = {}

        noise = torch.distribution.normal.Normal(0, noise_scale * self.model.parameters['noise_std'])
        t0 = model.parameters['tau_mean'].detach().numpy()
        v0 = model.parameters['xi_mean'].detach().numpy()
        for idx in range(N_indiv):
            this_indiv_param = {}
            this_indiv_param[idx] = {}
            sim = kernel.resample(1)[:, 0]
            this_indiv_param[idx]['xi'] = sim[0]
            this_indiv_param[idx]['tau'] = sim[1]
            if model.name != "univariate":
                this_indiv_param[idx]['sources'] = sim[2:]
            indiv_param.update(this_indiv_param)
            age_diag = (ref_age - t0) * np.exp(v0) / np.exp(sim[0]) + sim[1]
            # Draw the number of visits
            n_visits = np.random.randint(max(2, n_points - 3), dataset.max_observations)
            timepoints = np.linspace(age_diag - n_visits * self.interval / 2, age_diag + n_visits * self.interval / 2, n_visits)
            timepoints = torch.tensor(timepoints, dtype=torch.float32).unsqueeze(0)

            values = model.compute_individual_tensorized(timepoints,
                                                              self.model.param_ind_from_dict(this_indiv_param))
            values = values + noise_scale * noise.sample(values.shape)
            values = torch.clamp(values, 0, 1).squeeze(0)

            ten_idx = torch.tensor([idx] * (n_visits), dtype=torch.float32)
            val = torch.cat([ten_idx.unsqueeze(1), timepoints.squeeze(0).unsqueeze(1), values], dim=1)
            truc = pd.DataFrame(val.detach().numpy(), columns=['ID', 'TIME'] + dataset.headers)
            data_sim = data_sim.append(truc)

        return data_sim

        '''
