import unittest

import torch
import pandas as pd

from leaspy import Leaspy, Data, AlgorithmSettings
from tests import example_data_path


class LeaspyFitTest(unittest.TestCase):

    # Test MCMC-SAEM
    def test_fit_logistic(self):

        # Inputs
        data = Data.from_csv_file(example_data_path)
        algo_settings = AlgorithmSettings('mcmc_saem', n_iter=10, seed=0)

        # Initialize
        leaspy = Leaspy("logistic")
        leaspy.model.load_hyperparameters({'source_dimension': 2})

        # Fit the model on the data
        leaspy.fit(data, algorithm_settings=algo_settings)

        self.assertAlmostEqual(leaspy.model.parameters['noise_std'], 0.2986, delta=0.01)
        self.assertAlmostEqual(leaspy.model.parameters['tau_mean'], 78.0270, delta=0.01)
        self.assertAlmostEqual(leaspy.model.parameters['tau_std'], 0.9494, delta=0.01)
        self.assertAlmostEqual(leaspy.model.parameters['xi_mean'], 0.0, delta=0.001)
        self.assertAlmostEqual(leaspy.model.parameters['xi_std'], 0.1317, delta=0.001)

        diff_g = leaspy.model.parameters['g'] - torch.tensor([1.9557, 2.5899, 2.5184, 2.2369])
        diff_v = leaspy.model.parameters['v0'] - torch.tensor([-3.5714, -3.5820, -3.5811, -3.5886])
        self.assertAlmostEqual(torch.sum(diff_g**2).item(), 0.0, delta=0.01)
        self.assertAlmostEqual(torch.sum(diff_v**2).item(), 0.0, delta=0.02)

    # Test MCMC-SAEM (1 noise per feature)
    def test_fit_logistic_diag_noise(self):

        # Inputs
        data = Data.from_csv_file(example_data_path)
        algo_settings = AlgorithmSettings('mcmc_saem', loss='MSE_diag_noise', n_iter=10, seed=0)

        # Initialize
        leaspy = Leaspy("logistic")
        leaspy.model.load_hyperparameters({'source_dimension': 2})

        # Fit the model on the data
        leaspy.fit(data, algorithm_settings=algo_settings)

        #leaspy.save("../../fitted_multivariate_model_diag_noise.json")

        self.assertAlmostEqual(leaspy.model.parameters['tau_mean'], 78.0697, delta=0.01)
        self.assertAlmostEqual(leaspy.model.parameters['tau_std'], 1.0275, delta=0.01)
        self.assertAlmostEqual(leaspy.model.parameters['xi_mean'], 0.0, delta=0.001)
        self.assertAlmostEqual(leaspy.model.parameters['xi_std'], 0.1634, delta=0.001)

        diff_noise = leaspy.model.parameters['noise_std'] - torch.tensor([0.3287, 0.2500, 0.2591, 0.2913])
        diff_g = leaspy.model.parameters['g'] - torch.tensor([1.9405, 2.5914, 2.5199, 2.2495])
        diff_v = leaspy.model.parameters['v0'] - torch.tensor([-3.5214, -3.5387, -3.5215, -3.5085])
        self.assertAlmostEqual(torch.sum(diff_noise**2).item(), 0.0, delta=0.01)
        self.assertAlmostEqual(torch.sum(diff_g**2).item(), 0.0, delta=0.01)
        self.assertAlmostEqual(torch.sum(diff_v**2).item(), 0.0, delta=0.02)

        #diff_betas = leaspy.model.parameters['betas'] - torch.tensor([[-0.0165, 0.0077],[-0.0077, 0.0193],[0.0140, 0.0143]])
        #self.assertAlmostEqual(torch.sum(diff_betas**2).item(), 0.0, delta=0.01)

    def test_fit_logisticparallel(self):
        # Inputs
        data = Data.from_csv_file(example_data_path)
        algo_settings = AlgorithmSettings('mcmc_saem', n_iter=10, seed=0)

        # Initialize
        leaspy = Leaspy("logistic_parallel")
        leaspy.model.load_hyperparameters({'source_dimension': 2})

        # Fit the model on the data
        leaspy.fit(data, algorithm_settings=algo_settings)

        self.assertAlmostEqual(leaspy.model.parameters['noise_std'], 0.2641, delta=0.01)
        self.assertAlmostEqual(leaspy.model.parameters['tau_mean'], 70.4093, delta=0.01)
        self.assertAlmostEqual(leaspy.model.parameters['tau_std'], 2.2325, delta=0.01)
        self.assertAlmostEqual(leaspy.model.parameters['xi_mean'], -3.1897, delta=0.001)
        self.assertAlmostEqual(leaspy.model.parameters['xi_std'], 0.1542, delta=0.001)

        self.assertAlmostEqual(leaspy.model.parameters['g'], 1.0160, delta=0.001)

        diff_deltas = leaspy.model.parameters['deltas'] - torch.tensor([-0.0099, -0.0239, -0.0100])
        self.assertAlmostEqual(torch.sum(diff_deltas ** 2).item(), 0.0, delta=0.01)

    def test_fit_logisticparallel_diag_noise(self):
        # Inputs
        data = Data.from_csv_file(example_data_path)
        algo_settings = AlgorithmSettings('mcmc_saem', loss='MSE_diag_noise', n_iter=10, seed=0)

        # Initialize
        leaspy = Leaspy("logistic_parallel")
        leaspy.model.load_hyperparameters({'source_dimension': 2})

        # Fit the model on the data
        leaspy.fit(data, algorithm_settings=algo_settings)

        self.assertAlmostEqual(leaspy.model.parameters['tau_mean'], 70.3955, delta=0.01)
        self.assertAlmostEqual(leaspy.model.parameters['tau_std'], 2.2052, delta=0.01)
        self.assertAlmostEqual(leaspy.model.parameters['xi_mean'], -3.1508, delta=0.001)
        self.assertAlmostEqual(leaspy.model.parameters['xi_std'], 0.1296, delta=0.001)

        self.assertAlmostEqual(leaspy.model.parameters['g'], 1.0097, delta=0.001)

        diff_noise = leaspy.model.parameters['noise_std'] - torch.tensor([0.1917, 0.2906, 0.2802, 0.2785])
        diff_deltas = leaspy.model.parameters['deltas'] - torch.tensor([-0.0372, -0.0024, -0.0329])
        self.assertAlmostEqual(torch.sum(diff_noise**2).item(), 0.0, delta=0.01)
        self.assertAlmostEqual(torch.sum(diff_deltas ** 2).item(), 0.0, delta=0.01)

        #diff_betas = leaspy.model.parameters['betas'] - torch.tensor([[-0.0424,0.0157],[-0.0161,0.0164],[0.0036,0.0399]])
        #self.assertAlmostEqual(torch.sum(diff_betas**2).item(), 0.0, delta=0.01)

    # TODO Univariate Model ???
    def test_fit_univariate_logistic(self):
        # Inputs
        df = pd.read_csv(example_data_path)
        data = Data.from_dataframe(df.iloc[:,:3]) # one feature column
        algo_settings = AlgorithmSettings('mcmc_saem', n_iter=10, seed=0)

        # Initialize
        leaspy = Leaspy("univariate")

        # Fit the model on the data
        leaspy.fit(data, algorithm_settings=algo_settings)

        self.assertAlmostEqual(leaspy.model.parameters['noise_std'], 0.1780, delta=0.01)
        self.assertAlmostEqual(leaspy.model.parameters['tau_mean'], 70.2322, delta=0.01)
        self.assertAlmostEqual(leaspy.model.parameters['tau_std'], 2.0974, delta=0.01)
        self.assertAlmostEqual(leaspy.model.parameters['xi_mean'], -2.8940, delta=0.001)
        self.assertAlmostEqual(leaspy.model.parameters['xi_std'], 0.1063, delta=0.001)

        self.assertAlmostEqual(leaspy.model.parameters['g'], 0.9939, delta=0.001)

    # TODO HMC, Gradient Descent
