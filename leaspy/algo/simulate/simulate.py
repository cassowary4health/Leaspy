import numpy as np
import pandas as pd
import scipy
import torch
from scipy import stats
from sklearn.preprocessing import StandardScaler

from leaspy.algo.abstract_algo import AbstractAlgo
from leaspy.inputs.data.data import Data


class SimulationAlgorithm(AbstractAlgo):

    def __init__(self, settings):

        super().__init__()

        self.number_of_subjects = settings.parameters['number_of_subjects']
        self.bandwidth_method = settings.parameters['bandwidth_method']
        self.noise = settings.parameters['noise']
        self.mean_number_of_visits = settings.parameters['mean_number_of_visits']
        self.std_number_of_visits = settings.parameters['std_number_of_visits']

    def _initialize_kernel(self, results):
        return 0

    def _sample_sources(self, xi, tau, bl, source_dimension):
        ind_1 = [0, 1, 2]
        ind_2 = list(range(3, source_dimension + 3))
        x_1 = np.vstack((xi, tau, bl))

        mu_1 = self.df_mean[ind_1][:, np.newaxis]
        mu_2 = self.df_mean[ind_2][:, np.newaxis]
        sigma_11 = self.df_cov[np.ix_(ind_1, ind_1)]
        sigma_22 = self.df_cov[np.ix_(ind_2, ind_2)]
        sigma_12 = self.df_cov[np.ix_(ind_1, ind_2)]

        mean_cond = (mu_2 + np.dot(np.transpose(sigma_12), np.dot(np.linalg.inv(sigma_11), x_1 - mu_1))).ravel()
        cov_cond = sigma_22 - np.dot(np.dot(np.transpose(sigma_12), np.linalg.inv(sigma_11)), sigma_12)

        return np.random.multivariate_normal(mean_cond, cov_cond)

    def _get_number_of_visits(self, xi, tau, bl, sources):
        number_of_visits = int(self.mean_number_of_visits)

        if self.mean_number_of_visits != 0:
            number_of_visits += int(np.random.normal(0, self.std_number_of_visits))

        return number_of_visits

    def run(self, model, results):

        xi, tau, bl, sources = [], [], [], []
        for k, v in results.individual_parameters.items():
            xi.append(v['xi'])
            tau.append(v['tau'])
            sources.append(v['sources'])
            ages = results.data.get_by_idx(k).timepoints
            bl.append(min(ages))

        distribution = np.array([xi, tau, bl]).T
        ss = StandardScaler()
        rescaled_distribution = ss.fit_transform(distribution)
        kernel = stats.gaussian_kde(rescaled_distribution.T, bw_method=self.bandwidth_method)

        df = np.concatenate((np.array([xi, tau, bl]), np.array(sources).T), axis=0).T
        df = pd.DataFrame(data=df)

        self.df_mean = df.mean().values
        self.df_cov = df.cov().values

        samples = np.transpose(kernel.resample(self.number_of_subjects))
        samples = ss.inverse_transform(samples)

        indices, timepoints, values = [], [], []

        for idx, s in enumerate(samples):
            xi, tau, bl = s
            bl = bl - 1
            sources = self._sample_sources(xi, tau, bl, model.source_dimension)
            number_of_visits = self._get_number_of_visits(xi, tau, bl, sources)
            if number_of_visits == 1:
                ages = [bl]
            elif number_of_visits == 2:
                ages = [bl, bl + 0.5]
            else:
                ages = [bl, bl + 0.5] + [bl + i for i in range(1, number_of_visits - 1)]

            indiv_param = torch.tensor([xi], dtype=torch.float32).unsqueeze(0), \
                          torch.tensor([tau], dtype=torch.float32).unsqueeze(0), \
                          torch.tensor(sources, dtype=torch.float32).unsqueeze(0)

            observations = model.compute_individual_tensorized(
                torch.tensor(ages, dtype=torch.float32).unsqueeze(0), indiv_param)

            if self.noise:
                noise = torch.distributions.Normal(loc=0, scale=model.parameters['noise_std']).sample(
                    observations.shape)
                observations += noise
                observations = observations.clamp(0, 1)

            indices.append(idx)
            timepoints.append(ages)
            values.append(observations.squeeze(0).detach().numpy().tolist())

        simulated_data = Data.from_individuals(indices, timepoints, values, results.data.headers)
        return simulated_data

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
