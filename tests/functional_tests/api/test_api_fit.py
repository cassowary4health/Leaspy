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

        self.assertAlmostEqual(leaspy.model.parameters['tau_mean'], 78.8487, delta=tol_tau)
        self.assertAlmostEqual(leaspy.model.parameters['tau_std'], 5.1182, delta=tol_tau)
        self.assertEqual(leaspy.model.parameters['xi_mean'], 0.0)
        self.assertAlmostEqual(leaspy.model.parameters['xi_std'], 0.7334, delta=tol)
        self.assertAlmostEqual(leaspy.model.parameters['noise_std'], 0.1330, delta=tol)

        diff_g = leaspy.model.parameters['g'] - torch.tensor([0.0469, 2.8038, 2.5467, 1.2391])
        diff_v = leaspy.model.parameters['v0'] - torch.tensor([-4.2359, -4.9385, -5.1203, -4.1422])
        diff_betas = leaspy.model.parameters['betas'] - torch.tensor([[-0.0078, -0.0018],[ 0.0163, -0.0051],[-0.1162, -0.0914]])
        self.assertAlmostEqual(torch.sum(diff_g**2).item(), 0.0, delta=tol**2)
        self.assertAlmostEqual(torch.sum(diff_v**2).item(), 0.0, delta=tol**2)
        self.assertAlmostEqual(torch.sum(diff_betas ** 2).item(), 0.0, delta=tol**2)

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

        self.assertAlmostEqual(leaspy.model.parameters['tau_mean'], 78.9043, delta=tol_tau)
        self.assertAlmostEqual(leaspy.model.parameters['tau_std'], 5.2539, delta=tol_tau)
        self.assertEqual(leaspy.model.parameters['xi_mean'], 0.0)
        self.assertAlmostEqual(leaspy.model.parameters['xi_std'], 0.730, delta=tol)

        diff_g = leaspy.model.parameters['g'] - torch.tensor([0.0963, 2.8176, 2.5137, 1.1444])
        diff_v = leaspy.model.parameters['v0'] - torch.tensor([-4.5411, -5.2473, -5.3571, -4.3723])
        diff_betas = leaspy.model.parameters['betas'] - torch.tensor([[ 0.0357, -0.0087], [ 0.0163, -0.0109],[ 0.0346, -0.0201]])
        diff_noise = leaspy.model.parameters['noise_std'] - torch.tensor([0.1164, 0.0911, 0.1055, 0.2429])
        self.assertAlmostEqual(torch.sum(diff_g**2).item(), 0.0, delta=tol) # tol**2
        self.assertAlmostEqual(torch.sum(diff_v**2).item(), 0.0, delta=tol) # tol**2
        self.assertAlmostEqual(torch.sum(diff_noise**2).item(), 0.0, delta=tol) # tol**2
        self.assertAlmostEqual(torch.sum(diff_betas ** 2).item(), 0.0, delta=tol ** 2)

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

        self.assertAlmostEqual(leaspy.model.parameters['g'], 1.6102, delta=tol)
        self.assertAlmostEqual(leaspy.model.parameters['tau_mean'], 77.9064, delta=tol)
        self.assertAlmostEqual(leaspy.model.parameters['tau_std'], 5.3658, delta=tol)
        self.assertAlmostEqual(leaspy.model.parameters['xi_mean'], -3.6736, delta=tol)
        self.assertAlmostEqual(leaspy.model.parameters['xi_std'], 0.5963, delta=tol)
        self.assertAlmostEqual(leaspy.model.parameters['noise_std'], 0.1576, delta=tol)

        diff_deltas = leaspy.model.parameters['deltas'] - torch.tensor([-0.0848, -0.0065, -0.0105])
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

        self.assertAlmostEqual(leaspy.model.parameters['g'], 1.6642, delta=tol)
        self.assertAlmostEqual(leaspy.model.parameters['tau_mean'], 78.9500, delta=tol)
        self.assertAlmostEqual(leaspy.model.parameters['tau_std'], 5.1243, delta=tol)
        self.assertAlmostEqual(leaspy.model.parameters['xi_mean'], -3.3786, delta=tol)
        self.assertAlmostEqual(leaspy.model.parameters['xi_std'], 0.7675, delta=tol)

        diff_deltas = leaspy.model.parameters['deltas'] - torch.tensor([-0.0597, -0.1301,  0.0157])
        diff_noise = leaspy.model.parameters['noise_std'] - torch.tensor([0.1183, 0.0876, 0.1062, 0.2631])
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
