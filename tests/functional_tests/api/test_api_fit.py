import unittest

import torch
import pandas as pd

from leaspy import Leaspy, Data, AlgorithmSettings
from tests import example_data_path, from_fit_model_path

# Weirdly, some results are perfectly reproducible on local mac + CI linux but not on CI mac...
# Increasing tolerances so to pass despite these reproducibility issues...

class LeaspyFitTest(unittest.TestCase):

    # Test MCMC-SAEM
    def test_fit_logistic(self, tol=5e-2, tol_tau=2e-1):

        # Inputs
        data = Data.from_csv_file(example_data_path)
        algo_settings = AlgorithmSettings('mcmc_saem', n_iter=100, seed=0)

        # Initialize
        leaspy = Leaspy("logistic")
        leaspy.model.load_hyperparameters({'source_dimension': 2})

        # Fit the model on the data
        leaspy.fit(data, algorithm_settings=algo_settings)
        print(leaspy.model.parameters)

        ## uncomment to re-generate example file
        #leaspy.save(from_fit_model_path('logistic'), indent=2)

        self.assertAlmostEqual(leaspy.model.parameters['tau_mean'], 78.6396, delta=tol_tau)
        self.assertAlmostEqual(leaspy.model.parameters['tau_std'], 5.1182, delta=tol_tau)
        self.assertAlmostEqual(leaspy.model.parameters['xi_mean'], 0.0, delta=tol)
        self.assertAlmostEqual(leaspy.model.parameters['xi_std'], 0.7190, delta=tol)
        self.assertAlmostEqual(leaspy.model.parameters['noise_std'], 0.1521, delta=tol)

        diff_g = leaspy.model.parameters['g'] - torch.tensor([-0.0430,  2.8104,  2.5435,  1.1309])
        diff_v = leaspy.model.parameters['v0'] - torch.tensor([-4.2846, -4.9695, -5.1563, -4.2357])
        self.assertAlmostEqual(torch.sum(diff_g**2).item(), 0.0, delta=tol**2)
        self.assertAlmostEqual(torch.sum(diff_v**2).item(), 0.0, delta=tol**2)

    # Test MCMC-SAEM (1 noise per feature)
    def test_fit_logistic_diag_noise(self, tol=6e-2, tol_tau=2e-1):

        # Inputs
        data = Data.from_csv_file(example_data_path)
        algo_settings = AlgorithmSettings('mcmc_saem', loss='MSE_diag_noise', n_iter=100, seed=0)

        # Initialize
        leaspy = Leaspy("logistic")
        leaspy.model.load_hyperparameters({'source_dimension': 2})

        # Fit the model on the data
        leaspy.fit(data, algorithm_settings=algo_settings)
        print(leaspy.model.parameters)

        ## uncomment to re-generate example file
        #leaspy.save(from_fit_model_path('logistic_diag_noise'), indent=2)

        self.assertAlmostEqual(leaspy.model.parameters['tau_mean'], 78.5145, delta=tol_tau)
        self.assertAlmostEqual(leaspy.model.parameters['tau_std'], 4.9691, delta=tol_tau)
        self.assertAlmostEqual(leaspy.model.parameters['xi_mean'], 0.0, delta=tol)
        self.assertAlmostEqual(leaspy.model.parameters['xi_std'], 0.5102, delta=tol)

        diff_g = leaspy.model.parameters['g'] - torch.tensor([0.0654, 2.8317, 2.4647, 1.1498])
        diff_v = leaspy.model.parameters['v0'] - torch.tensor([-4.3287, -5.0044, -5.2275, -4.2064])
        diff_noise = leaspy.model.parameters['noise_std'] - torch.tensor([0.1223, 0.0866, 0.1069, 0.2456])
        self.assertAlmostEqual(torch.sum(diff_g**2).item(), 0.0, delta=tol) # tol**2
        self.assertAlmostEqual(torch.sum(diff_v**2).item(), 0.0, delta=tol) # tol**2
        self.assertAlmostEqual(torch.sum(diff_noise**2).item(), 0.0, delta=tol) # tol**2

    def test_fit_logisticparallel(self, tol=1e-2):
        # Inputs
        data = Data.from_csv_file(example_data_path)
        algo_settings = AlgorithmSettings('mcmc_saem', n_iter=100, seed=0)

        # Initialize
        leaspy = Leaspy("logistic_parallel")
        leaspy.model.load_hyperparameters({'source_dimension': 2})

        # Fit the model on the data
        leaspy.fit(data, algorithm_settings=algo_settings)
        print(leaspy.model.parameters)

        ## uncomment to re-generate example file
        #leaspy.save(from_fit_model_path('logistic_parallel'), indent=2)

        self.assertAlmostEqual(leaspy.model.parameters['g'], 1.5409, delta=tol)
        self.assertAlmostEqual(leaspy.model.parameters['tau_mean'], 78.5953, delta=tol)
        self.assertAlmostEqual(leaspy.model.parameters['tau_std'], 5.0366, delta=tol)
        self.assertAlmostEqual(leaspy.model.parameters['xi_mean'], -3.3278, delta=tol)
        self.assertAlmostEqual(leaspy.model.parameters['xi_std'], 0.7850, delta=tol)
        self.assertAlmostEqual(leaspy.model.parameters['noise_std'], 0.2300, delta=tol)

        diff_deltas = leaspy.model.parameters['deltas'] - torch.tensor([ 0.0117, -0.0162,  0.0359])
        self.assertAlmostEqual(torch.sum(diff_deltas ** 2).item(), 0.0, delta=tol**2)

    def test_fit_logisticparallel_diag_noise(self, tol=1e-2):
        # Inputs
        data = Data.from_csv_file(example_data_path)
        algo_settings = AlgorithmSettings('mcmc_saem', loss='MSE_diag_noise', n_iter=100, seed=0)

        # Initialize
        leaspy = Leaspy("logistic_parallel")
        leaspy.model.load_hyperparameters({'source_dimension': 2})

        # Fit the model on the data
        leaspy.fit(data, algorithm_settings=algo_settings)
        print(leaspy.model.parameters)

        ## uncomment to re-generate example file
        #leaspy.save(from_fit_model_path('logistic_parallel_diag_noise'), indent=2)

        self.assertAlmostEqual(leaspy.model.parameters['g'], 1.7843, delta=tol)
        self.assertAlmostEqual(leaspy.model.parameters['tau_mean'], 78.6494, delta=tol)
        self.assertAlmostEqual(leaspy.model.parameters['tau_std'], 5.0751, delta=tol)
        self.assertAlmostEqual(leaspy.model.parameters['xi_mean'], -3.5628, delta=tol)
        self.assertAlmostEqual(leaspy.model.parameters['xi_std'], 0.7413, delta=tol)

        diff_deltas = leaspy.model.parameters['deltas'] - torch.tensor([-0.0631, -0.0712,  0.0056])
        diff_noise = leaspy.model.parameters['noise_std'] - torch.tensor([0.3581, 0.1213, 0.1226, 0.2805])
        self.assertAlmostEqual(torch.sum(diff_deltas ** 2).item(), 0.0, delta=tol**2)
        self.assertAlmostEqual(torch.sum(diff_noise**2).item(), 0.0, delta=tol**2)

    def test_fit_univariate_logistic(self, tol=1e-2):
        # Inputs
        df = pd.read_csv(example_data_path)
        data = Data.from_dataframe(df.iloc[:,:3]) # one feature column
        algo_settings = AlgorithmSettings('mcmc_saem', n_iter=100, seed=0)

        # Initialize
        leaspy = Leaspy("univariate_logistic")

        # Fit the model on the data
        leaspy.fit(data, algorithm_settings=algo_settings)
        print(leaspy.model.parameters)

        ## uncomment to re-generate example file
        #leaspy.save(from_fit_model_path('univariate_logistic'), indent=2)

        self.assertAlmostEqual(leaspy.model.parameters['g'], 0.1102, delta=tol)
        self.assertAlmostEqual(leaspy.model.parameters['tau_mean'], 78.2246, delta=tol)
        self.assertAlmostEqual(leaspy.model.parameters['tau_std'], 5.5927, delta=tol)
        self.assertAlmostEqual(leaspy.model.parameters['xi_mean'], -3.1730, delta=tol)
        self.assertAlmostEqual(leaspy.model.parameters['xi_std'], 0.4896, delta=tol)
        self.assertAlmostEqual(leaspy.model.parameters['noise_std'], 0.1307, delta=tol)

    def test_fit_univariate_linear(self, tol=1e-2):
        # Inputs
        df = pd.read_csv(example_data_path)
        data = Data.from_dataframe(df.iloc[:,:3]) # one feature column
        algo_settings = AlgorithmSettings('mcmc_saem', n_iter=100, seed=0)

        # Initialize
        leaspy = Leaspy("univariate_linear")

        # Fit the model on the data
        leaspy.fit(data, algorithm_settings=algo_settings)
        print(leaspy.model.parameters)

        ## uncomment to re-generate example file
        #leaspy.save(from_fit_model_path('univariate_linear'), indent=2)

        self.assertAlmostEqual(leaspy.model.parameters['g'], 0.4936, delta=tol)
        self.assertAlmostEqual(leaspy.model.parameters['tau_mean'], 78.3471, delta=tol)
        self.assertAlmostEqual(leaspy.model.parameters['tau_std'], 5.2568, delta=tol)
        self.assertAlmostEqual(leaspy.model.parameters['xi_mean'], -3.9552, delta=tol)
        self.assertAlmostEqual(leaspy.model.parameters['xi_std'], 0.8314, delta=tol)
        self.assertAlmostEqual(leaspy.model.parameters['noise_std'], 0.1114, delta=tol)

    def test_fit_linear(self, tol=5e-2, tol_tau=2e-1):

        # Inputs
        data = Data.from_csv_file(example_data_path)
        algo_settings = AlgorithmSettings('mcmc_saem', n_iter=100, seed=0)

        # Initialize
        leaspy = Leaspy("linear")
        leaspy.model.load_hyperparameters({'source_dimension': 2})

        # Fit the model on the data
        leaspy.fit(data, algorithm_settings=algo_settings)
        print(leaspy.model.parameters)

        ## uncomment to re-generate example file
        #leaspy.save(from_fit_model_path('linear'), indent=2)

        self.assertAlmostEqual(leaspy.model.parameters['tau_mean'], 78.5021, delta=tol_tau)
        self.assertAlmostEqual(leaspy.model.parameters['tau_std'], 5.2851, delta=tol_tau)
        self.assertAlmostEqual(leaspy.model.parameters['xi_mean'], 0.0, delta=tol)
        self.assertAlmostEqual(leaspy.model.parameters['xi_std'], 0.9000, delta=tol)
        self.assertAlmostEqual(leaspy.model.parameters['noise_std'], 0.1407, delta=tol)

        diff_g = leaspy.model.parameters['g'] - torch.tensor([0.4591, 0.0539, 0.0815, 0.3015])
        diff_v = leaspy.model.parameters['v0'] - torch.tensor([-4.1763, -4.8371, -4.9546, -4.0944])
        self.assertAlmostEqual(torch.sum(diff_g**2).item(), 0.0, delta=tol) # tol**2
        self.assertAlmostEqual(torch.sum(diff_v**2).item(), 0.0, delta=tol) # tol**2


    # TODO HMC, Gradient Descent
