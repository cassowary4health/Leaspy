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
    algo_parameters: `dict`
        Contains the algorithm's parameters.
    bandwidth_method: `float`, `str`, `callable`, optional
        Bandwidth argument used in scipy.stats.gaussian_kde in order to learn the patients' distribution.
    cofactor: `str` (default = None)
        The cofactor used to select the wanted group of patients (ex - 'genes').
    cofactor_state: `str` - TODO: check that the loaded cofactors are converted into strings!
        The cofactor state used to select the  wanted group of patients (ex - 'APOE4').
    mean_number_of_visits: `int`
        Average number of visits of the simulated patients.
        Examples - choose 5 => in average, a simulated patient will have 5 visits.
    name: `str`
        Algorithm's name.
    noise: `float`
        Wanted level of noise in the generated scores - noise of zero will lead to patients having "perfect progression"
        of their scores, i.e. following exactly a logistic curve.
    number_of_subjects: `int`
        Number of subject to simulate.
    seed: `int`
        Used by numpy.random & torch.random for reproducibility.
    std_number_of_visits: `float`
        Standard deviation used into the generation of the number of visits per simulated patient.

    Methods
    -------
    run(model, results)
        Run the simulation of new patients for some given leaspy object result & model.
    """

    def __init__(self, settings):
        """
        Process initializer function that is called by Leaspy.simulate.Simulate.

        Parameters
        ----------
        settings: `leaspy.inputs.algorithm_settings.AlgorithmSettings` class object
            Set the class attributes.
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
        self.mean_number_of_visits = settings.parameters['mean_number_of_visits']
        self.noise = settings.parameters['noise']
        self.number_of_subjects = settings.parameters['number_of_subjects']
        self.std_number_of_visits = settings.parameters['std_number_of_visits']

    @staticmethod
    def _get_mean_and_covariance_matrix(m):
        """
        Compute the empirical mean and covariance matrix of the input.

        Parameters
        ----------
        m: `torch.Tensor`
            Input matrix - one row per individual parameter distribution (xi, tau etc).

        Returns
        -------
        mean: `torch.Tensor`
            Mean by variable.
        covariance:  `torch.Tensor`
            Covariance matrix.
        """
        m_exp = torch.mean(m, dim=0)
        x = m - m_exp[None, :]
        cov = 1 / (x.size(1) - 1) * x.t() @ x
        return m_exp, cov

    @staticmethod
    def _sample_sources(bl, tau, xi, source_dimension, df_mean, df_cov):
        """
        Simulate individual sources given baseline age bl, time-shift tau, log-acceleration xi & sources dimension.

        Parameters
        ----------
        bl: `float`
            Baseline age of the simulated patient.
        tau: `float`
            Time-shift of the simulated patient.
        xi: `float`
            Log-acceleration of the simulated patient.
        source_dimension: `int`
            Sources' dimension of the simulated patient.
        df_mean: `torch.Tensor`
            Mean values per individual parameter type (bl_mean, tau_mean, xi_mean & sources_means) (1-dimensional).
        df_cov: `torch.Tensor`
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
        number_of_visits: `int`
            Number of visits.
        """
        # Generate a number of visit around the mean_number_of_visits
        number_of_visits = int(self.mean_number_of_visits)
        if self.mean_number_of_visits != 0:
            number_of_visits += int(torch.normal(torch.tensor(0, dtype=torch.float32),
                                                 torch.tensor(self.std_number_of_visits, dtype=torch.float32)).item())
        return number_of_visits

    def run(self, model, results):
        """
        Run simulation - learn joined distribution of patients' individual parameters and return a results object
        containing the simulated individual parameters and the simulated scores.

        Parameters
        ----------
        model: leaspy.model class object
            Model used to compute the population & individual parameters. It contains the population parameters.
        results: `leaspy.inputs.data.result.Result` class object
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

        distribution = torch.from_numpy(df_ind_param.values)
        # Note: torch.tensor always copy data, torch.from_numpy always does not
        #   =>  torch.from_numpy(np.array) 5x faster than torch.tensor(np.array)
        # Note: pandas.DataFrame.values.T 20x faster than pandas.DataFrame.T.values
        # Note: 10x faster to transpose in numpy than in torch

        get_sources = (model.name != 'univariate')
        if get_sources:
            # Get mean by variable & covariance matrix
            # Needed to sample new sources from simulated bl, tau & xi
            df_mean, df_cov = self._get_mean_and_covariance_matrix(distribution)

        # Get joined density estimation of bl, tau & xi (sources are not learn in this fashion)
        distribution = distribution[:, :3].numpy()
        # Normalize by variable then transpose to learn the joined distribution
        ss = StandardScaler()
        # fit_transform receive an numpy array of shape [n_samples, n_features]
        distribution = ss.fit_transform(distribution).T
        # gaussian_kde receive an numpy array of shape [n_features, n_samples]
        kernel = stats.gaussian_kde(distribution, bw_method=self.bandwidth_method)

        # Generate individual parameters (except sources)
        samples = kernel.resample(self.number_of_subjects).T
        samples = ss.inverse_transform(samples)
        # A 2D array - one raw per simulated subject
        bl, tau, xi = samples.T

        # Generate sources
        if get_sources:
            def generate_sources(x):
                return self._sample_sources(x[0], x[1], x[2], model.source_dimension, df_mean, df_cov).numpy()
            sources = np.apply_along_axis(generate_sources, axis=1, arr=samples)
            # A 2D array - one raw per subject

        # Initialize simulated scores
        indices, timepoints, values = [], [], []

        # Generate individual sources, scores, indices & time-points
        for i in range(self.number_of_subjects):
            # Generate time-points
            number_of_visits = self._get_number_of_visits()  # xi[i], tau[i], bl[i] - 1, sources[-1])
            if number_of_visits == 1:
                ages = [bl[i]]
            elif number_of_visits == 2:
                ages = [bl[i], bl[i] + 0.5]
            else:
                ages = [bl[i], bl[i] + 0.5] + [bl[i] + j for j in range(1, number_of_visits - 1)]
            timepoints.append(ages)

            # Generate scores
            indiv_param = {'xi': xi[i], 'tau': tau[i]}
            if get_sources:
                indiv_param['sources'] = sources[i].tolist()
            observations = model.compute_individual_trajectory(ages, indiv_param)

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
        simulated_parameters = {'xi': torch.from_numpy(xi).view(-1, 1),
                                'tau': torch.from_numpy(tau).view(-1, 1)}
        if get_sources:
            simulated_parameters['sources'] = torch.from_numpy(sources)

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
