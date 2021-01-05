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
        algo_settings = AlgorithmSettings('mcmc_saem', n_iter=100, seed=0)

        # Initialize
        leaspy = Leaspy("logistic")
        leaspy.model.load_hyperparameters({'source_dimension': 2})

        # Fit the model on the data
        leaspy.fit(data, algorithm_settings=algo_settings)
        print(leaspy.model.parameters)

        self.assertAlmostEqual(leaspy.model.parameters['noise_std'], 0.1503, delta=0.01)
        self.assertAlmostEqual(leaspy.model.parameters['tau_mean'], 78.7451, delta=0.01)
        self.assertAlmostEqual(leaspy.model.parameters['tau_std'], 4.4030, delta=0.01)
        self.assertAlmostEqual(leaspy.model.parameters['xi_mean'], 0.0, delta=0.001)
        self.assertAlmostEqual(leaspy.model.parameters['xi_std'], 0.6343, delta=0.001)

        #diff_g = leaspy.model.parameters['g'] - torch.tensor([1.9557, 2.5899, 2.5184, 2.2369])
        diff_g = leaspy.model.parameters['g'] - torch.tensor([0.0230, 2.9281, 2.5348, 1.1416])
        #diff_v = leaspy.model.parameters['v0'] - torch.tensor([-3.5714, -3.5820, -3.5811, -3.5886])
        diff_v = leaspy.model.parameters['v0'] - torch.tensor([-4.3988, -4.5022, -4.4267, -4.4193])
        self.assertAlmostEqual(torch.sum(diff_g**2).item(), 0.0, delta=0.01)
        self.assertAlmostEqual(torch.sum(diff_v**2).item(), 0.0, delta=0.02)

    # Test MCMC-SAEM (1 noise per feature)
    def test_fit_logistic_diag_noise(self):

        # Inputs
        data = Data.from_csv_file(example_data_path)
        algo_settings = AlgorithmSettings('mcmc_saem', loss='MSE_diag_noise', n_iter=100, seed=0)

        # Initialize
        leaspy = Leaspy("logistic")
        leaspy.model.load_hyperparameters({'source_dimension': 2})

        # Fit the model on the data
        leaspy.fit(data, algorithm_settings=algo_settings)
        print(leaspy.model.parameters)

        #leaspy.save("../../fitted_multivariate_model_diag_noise.json")

        self.assertAlmostEqual(leaspy.model.parameters['tau_mean'], 78.6799, delta=0.01)
        self.assertAlmostEqual(leaspy.model.parameters['tau_std'], 5.4379, delta=0.01)
        self.assertAlmostEqual(leaspy.model.parameters['xi_mean'], 0.0, delta=0.001)
        self.assertAlmostEqual(leaspy.model.parameters['xi_std'], 0.8032, delta=0.001)

        #diff_noise = leaspy.model.parameters['noise_std'] - torch.tensor([0.3287, 0.2500, 0.2591, 0.2913])
        diff_noise = leaspy.model.parameters['noise_std'] - torch.tensor([0.1226, 0.0637, 0.0978, 0.2617])
        #diff_g = leaspy.model.parameters['g'] - torch.tensor([1.9405, 2.5914, 2.5199, 2.2495])
        diff_g = leaspy.model.parameters['g'] - torch.tensor([0.0824, 2.8976, 2.4821, 1.1305])
        #diff_v = leaspy.model.parameters['v0'] - torch.tensor([-3.5214, -3.5387, -3.5215, -3.5085])
        diff_v = leaspy.model.parameters['v0'] - torch.tensor([-4.4648, -4.5519, -4.6381, -4.5520])
        self.assertAlmostEqual(torch.sum(diff_noise**2).item(), 0.0, delta=0.01)
        self.assertAlmostEqual(torch.sum(diff_g**2).item(), 0.0, delta=0.01)
        self.assertAlmostEqual(torch.sum(diff_v**2).item(), 0.0, delta=0.02)

        #diff_betas = leaspy.model.parameters['betas'] - torch.tensor([[-0.0165, 0.0077],[-0.0077, 0.0193],[0.0140, 0.0143]])
        #self.assertAlmostEqual(torch.sum(diff_betas**2).item(), 0.0, delta=0.01)

    def test_fit_logisticparallel(self):
        # Inputs
        data = Data.from_csv_file(example_data_path)
        algo_settings = AlgorithmSettings('mcmc_saem', n_iter=100, seed=0)

        # Initialize
        leaspy = Leaspy("logistic_parallel")
        leaspy.model.load_hyperparameters({'source_dimension': 2})

        # Fit the model on the data
        leaspy.fit(data, algorithm_settings=algo_settings)
        print(leaspy.model.parameters)

        self.assertAlmostEqual(leaspy.model.parameters['noise_std'], 0.1561, delta=0.01)
        self.assertAlmostEqual(leaspy.model.parameters['tau_mean'], 77.6668, delta=0.01)
        self.assertAlmostEqual(leaspy.model.parameters['tau_std'], 5.0594, delta=0.01)
        self.assertAlmostEqual(leaspy.model.parameters['xi_mean'], -2.9430, delta=0.001)
        self.assertAlmostEqual(leaspy.model.parameters['xi_std'], 0.8617, delta=0.001)

        self.assertAlmostEqual(leaspy.model.parameters['g'], 1.8490, delta=0.001)

        diff_deltas = leaspy.model.parameters['deltas'] - torch.tensor([0.0011, -0.0800,  0.0736])
        self.assertAlmostEqual(torch.sum(diff_deltas ** 2).item(), 0.0, delta=0.01)

    def test_fit_logisticparallel_diag_noise(self):
        # Inputs
        data = Data.from_csv_file(example_data_path)
        algo_settings = AlgorithmSettings('mcmc_saem', loss='MSE_diag_noise', n_iter=100, seed=0)

        # Initialize
        leaspy = Leaspy("logistic_parallel")
        leaspy.model.load_hyperparameters({'source_dimension': 2})

        # Fit the model on the data
        leaspy.fit(data, algorithm_settings=algo_settings)
        print(leaspy.model.parameters)

        self.assertAlmostEqual(leaspy.model.parameters['tau_mean'], 70.4505, delta=0.01)
        self.assertAlmostEqual(leaspy.model.parameters['tau_std'], 5.0625, delta=0.01)
        self.assertAlmostEqual(leaspy.model.parameters['xi_mean'], -3.1740, delta=0.001)
        self.assertAlmostEqual(leaspy.model.parameters['xi_std'], 1.2120, delta=0.001)

        self.assertAlmostEqual(leaspy.model.parameters['g'], 2.6867, delta=0.001)

        diff_noise = leaspy.model.parameters['noise_std'] - torch.tensor([0.1987, 0.0712, 0.0856, 0.2409])
        diff_deltas = leaspy.model.parameters['deltas'] - torch.tensor([-0.0254,  0.0677,  0.1495])
        self.assertAlmostEqual(torch.sum(diff_noise**2).item(), 0.0, delta=0.01)
        self.assertAlmostEqual(torch.sum(diff_deltas ** 2).item(), 0.0, delta=0.01)

        #diff_betas = leaspy.model.parameters['betas'] - torch.tensor([[-0.0424,0.0157],[-0.0161,0.0164],[0.0036,0.0399]])
        #self.assertAlmostEqual(torch.sum(diff_betas**2).item(), 0.0, delta=0.01)

    def test_fit_univariate_logistic(self):
        # Inputs
        df = pd.read_csv(example_data_path)
        data = Data.from_dataframe(df.iloc[:,:3]) # one feature column
        algo_settings = AlgorithmSettings('mcmc_saem', n_iter=100, seed=0)

        # Initialize
        leaspy = Leaspy("univariate_logistic")

        # Fit the model on the data
        leaspy.fit(data, algorithm_settings=algo_settings)
        print(leaspy.model.parameters)

        self.assertAlmostEqual(leaspy.model.parameters['noise_std'], 0.1288, delta=0.01)
        self.assertAlmostEqual(leaspy.model.parameters['tau_mean'], 78.2409, delta=0.01)
        self.assertAlmostEqual(leaspy.model.parameters['tau_std'], 5.5983, delta=0.01)
        self.assertAlmostEqual(leaspy.model.parameters['xi_mean'], -3.1452, delta=0.001)
        self.assertAlmostEqual(leaspy.model.parameters['xi_std'], 0.4529, delta=0.001)
        self.assertAlmostEqual(leaspy.model.parameters['g'], 0.0982, delta=0.001)

    def test_fit_univariate_linear(self):
        # Inputs
        df = pd.read_csv(example_data_path)
        data = Data.from_dataframe(df.iloc[:,:3]) # one feature column
        algo_settings = AlgorithmSettings('mcmc_saem', n_iter=100, seed=0)

        # Initialize
        leaspy = Leaspy("univariate_linear")

        # Fit the model on the data
        leaspy.fit(data, algorithm_settings=algo_settings)
        print(leaspy.model.parameters)

        self.assertAlmostEqual(leaspy.model.parameters['noise_std'], 0.1114, delta=0.01)
        self.assertAlmostEqual(leaspy.model.parameters['tau_mean'], 78.3471, delta=0.01)
        self.assertAlmostEqual(leaspy.model.parameters['tau_std'], 5.2568, delta=0.01)
        self.assertAlmostEqual(leaspy.model.parameters['xi_mean'], -3.9552, delta=0.01)
        self.assertAlmostEqual(leaspy.model.parameters['xi_std'], 0.8314, delta=0.01)
        self.assertAlmostEqual(leaspy.model.parameters['g'], 0.4936, delta=0.001)

    # TODO HMC, Gradient Descent
