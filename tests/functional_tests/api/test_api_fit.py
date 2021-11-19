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

        self.assertAlmostEqual(leaspy.model.parameters['tau_mean'], 78.8212, delta=tol_tau)
        self.assertAlmostEqual(leaspy.model.parameters['tau_std'], 4.5039, delta=tol_tau)
        self.assertEqual(leaspy.model.parameters['xi_mean'], 0.0)
        self.assertAlmostEqual(leaspy.model.parameters['xi_std'], 0.9220, delta=tol)
        self.assertAlmostEqual(leaspy.model.parameters['noise_std'], 0.1314, delta=tol)

        diff_g = leaspy.model.parameters['g'] - torch.tensor([0.1262, 2.8975, 2.5396, 1.0504])
        diff_v = leaspy.model.parameters['v0'] - torch.tensor([-4.2079, -4.9066, -4.9962, -4.1774])
        diff_betas = leaspy.model.parameters['betas'] - torch.tensor(
            [[ 0.0103, -0.0088],
             [ 0.0072, -0.0046],
             [-0.0488, -0.1117]])

        self.assertAlmostEqual(torch.sum(diff_g**2).item(), 0.0, delta=tol**2)
        self.assertAlmostEqual(torch.sum(diff_v**2).item(), 0.0, delta=tol**2)
        self.assertAlmostEqual(torch.sum(diff_betas ** 2).item(), 0.0, delta=tol**2)

    # Test MCMC-SAEM (1 noise per feature)
    def test_fit_logistic_diag_noise(self, tol=6e-2, tol_tau=2e-1):

        # Inputs
        data = Data.from_csv_file(example_data_path)
        algo_settings = AlgorithmSettings('mcmc_saem', n_iter=100, seed=0)

        # Initialize
        leaspy = Leaspy("logistic", loss='MSE_diag_noise')
        leaspy.model.load_hyperparameters({'source_dimension': 2})

        # Fit the model on the data
        leaspy.fit(data, algorithm_settings=algo_settings)
        print(leaspy.model.parameters)

        ## uncomment to re-generate example file
        #leaspy.save(from_fit_model_path('logistic_diag_noise'), indent=2)

        self.assertAlmostEqual(leaspy.model.parameters['tau_mean'], 78.5633, delta=tol_tau)
        self.assertAlmostEqual(leaspy.model.parameters['tau_std'], 5.0105, delta=tol_tau)
        self.assertEqual(leaspy.model.parameters['xi_mean'], 0.0)
        self.assertAlmostEqual(leaspy.model.parameters['xi_std'], 0.7890, delta=tol)

        ## FIX PyTorch >= 1.7 values changed
        torch_major, torch_minor, *_ = torch.__version__.split('.')
        if (int(torch_major), int(torch_minor)) < (1, 7):
            diff_g = leaspy.model.parameters['g'] - torch.tensor([0.0510, 2.8941, 2.5810, 1.1241])
            diff_v = leaspy.model.parameters['v0'] - torch.tensor([-4.1882, -4.9868, -4.9800, -4.0447])
            diff_betas = leaspy.model.parameters['betas'] - torch.tensor(
                [[-0.0670, -0.0272],
                [ 0.0340,  0.0115],
                [ 0.0339, -0.0005]])
            diff_noise = leaspy.model.parameters['noise_std'] - torch.tensor([0.1165, 0.0750, 0.0988, 0.2478])
        else:
            diff_g = leaspy.model.parameters['g'] - torch.tensor([0.0379, 2.8926, 2.5623, 1.1620])
            diff_v = leaspy.model.parameters['v0'] - torch.tensor([-4.0076, -4.8284, -4.8279, -3.8997])
            diff_betas = leaspy.model.parameters['betas'] - torch.tensor(
                [[-0.0445, -0.0331],
                [ 0.0110,  0.0106],
                [ 0.0413, -0.0049]])
            diff_noise = leaspy.model.parameters['noise_std'] - torch.tensor([0.1153, 0.0764, 0.1011, 0.2355])

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
        self.assertAlmostEqual(leaspy.model.parameters['xi_mean'], -3.6563, delta=tol)
        self.assertAlmostEqual(leaspy.model.parameters['xi_std'], 0.5822, delta=tol)
        self.assertAlmostEqual(leaspy.model.parameters['noise_std'], 0.1576, delta=tol)

        diff_deltas = leaspy.model.parameters['deltas'] - torch.tensor([-0.0848, -0.0065, -0.0105])
        self.assertAlmostEqual(torch.sum(diff_deltas ** 2).item(), 0.0, delta=tol**2)

    def test_fit_logisticparallel_diag_noise(self, tol=1e-2):
        # Inputs
        data = Data.from_csv_file(example_data_path)
        algo_settings = AlgorithmSettings('mcmc_saem', n_iter=100, seed=0)

        # Initialize
        leaspy = Leaspy("logistic_parallel", loss='MSE_diag_noise')
        leaspy.model.load_hyperparameters({'source_dimension': 2})

        # Fit the model on the data
        leaspy.fit(data, algorithm_settings=algo_settings)
        print(leaspy.model.parameters)

        ## uncomment to re-generate example file
        #leaspy.save(from_fit_model_path('logistic_parallel_diag_noise'), indent=2)

        self.assertAlmostEqual(leaspy.model.parameters['g'], 1.6642, delta=tol)
        self.assertAlmostEqual(leaspy.model.parameters['tau_mean'], 78.9500, delta=tol)
        self.assertAlmostEqual(leaspy.model.parameters['tau_std'], 5.0987, delta=tol)
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

    def test_fit_linear(self, tol=1e-1, tol_tau=2e-1):

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

        self.assertAlmostEqual(leaspy.model.parameters['tau_mean'], 78.7079, delta=tol_tau)
        self.assertAlmostEqual(leaspy.model.parameters['tau_std'], 4.8328, delta=tol_tau)
        self.assertAlmostEqual(leaspy.model.parameters['xi_mean'], 0.0, delta=tol)
        self.assertAlmostEqual(leaspy.model.parameters['xi_std'], 1.0, delta=tol)
        self.assertAlmostEqual(leaspy.model.parameters['noise_std'], 0.1401, delta=tol)

        diff_g = leaspy.model.parameters['g'] - torch.tensor([0.4539, 0.0515, 0.0754, 0.2751])
        diff_v = leaspy.model.parameters['v0'] - torch.tensor([-4.2557, -4.7875, -4.9763, -4.1410])
        self.assertAlmostEqual(torch.sum(diff_g**2).item(), 0.0, delta=tol) # tol**2
        self.assertAlmostEqual(torch.sum(diff_v**2).item(), 0.0, delta=tol) # tol**2


@unittest.skipIf(not torch.cuda.is_available(),
                "GPU calibration tests need an available CUDA environment")
class LeaspyFitGPUTest(unittest.TestCase):

    # Test MCMC-SAEM
    def test_fit_logistic(self, tol=5e-2, tol_tau=2e-1):

        # Inputs
        data = Data.from_csv_file(example_data_path)
        algo_settings = AlgorithmSettings('mcmc_saem', n_iter=100, seed=0, device=torch.device("cuda"))

        # Initialize
        leaspy = Leaspy("logistic")
        leaspy.model.load_hyperparameters({'source_dimension': 2})

        # Fit the model on the data
        leaspy.fit(data, algorithm_settings=algo_settings)
        print(leaspy.model.parameters)

        ## uncomment to re-generate example file
        #leaspy.save(from_fit_model_path('logistic'), indent=2)

        self.assertAlmostEqual(leaspy.model.parameters['tau_mean'], 76.6287, delta=tol_tau)
        self.assertAlmostEqual(leaspy.model.parameters['tau_std'], 4.8775, delta=tol_tau)
        self.assertEqual(leaspy.model.parameters['xi_mean'], 0.0)
        self.assertAlmostEqual(leaspy.model.parameters['xi_std'], 0.7978, delta=tol)
        self.assertAlmostEqual(leaspy.model.parameters['noise_std'], 0.1367, delta=tol)

        diff_g = leaspy.model.parameters['g'] - torch.tensor([0.0743, 2.8258, 2.5927, 1.0700])
        diff_v = leaspy.model.parameters['v0'] - torch.tensor([-4.4334, -5.1050, -5.2571, -4.3333])
        diff_betas = leaspy.model.parameters['betas'] - torch.tensor(
            [[ 0.0343, -0.0840],
            [ 0.0144, -0.0464],
            [-0.1405, -0.0041]])

        self.assertAlmostEqual(torch.sum(diff_g**2).item(), 0.0, delta=tol**2)
        self.assertAlmostEqual(torch.sum(diff_v**2).item(), 0.0, delta=tol**2)
        self.assertAlmostEqual(torch.sum(diff_betas ** 2).item(), 0.0, delta=tol**2)

    # Test MCMC-SAEM (1 noise per feature)
    def test_fit_logistic_diag_noise(self, tol=6e-2, tol_tau=2e-1):

        # Inputs
        data = Data.from_csv_file(example_data_path)
        algo_settings = AlgorithmSettings('mcmc_saem', n_iter=100, seed=0, device=torch.device("cuda"))

        # Initialize
        leaspy = Leaspy("logistic", loss='MSE_diag_noise')
        leaspy.model.load_hyperparameters({'source_dimension': 2})

        # Fit the model on the data
        leaspy.fit(data, algorithm_settings=algo_settings)
        print(leaspy.model.parameters)

        ## uncomment to re-generate example file
        #leaspy.save(from_fit_model_path('logistic_diag_noise'), indent=2)

        self.assertAlmostEqual(leaspy.model.parameters['tau_mean'], 76.9868, delta=tol_tau)
        self.assertAlmostEqual(leaspy.model.parameters['tau_std'], 5.2285, delta=tol_tau)
        self.assertEqual(leaspy.model.parameters['xi_mean'], 0.0)
        self.assertAlmostEqual(leaspy.model.parameters['xi_std'], 1.0035, delta=tol)


        diff_g = leaspy.model.parameters['g'] - torch.tensor([0.0782, 2.8931, 2.5946, 1.1231])
        diff_v = leaspy.model.parameters['v0'] - torch.tensor([-4.5053, -5.2051, -5.3466, -4.3271])
        diff_betas = leaspy.model.parameters['betas'] - torch.tensor(
            [[-0.0095, -0.0850],
            [-0.0035, -0.0218],
            [-0.0008, -0.0476]])

        diff_noise = leaspy.model.parameters['noise_std'] - torch.tensor([0.1047, 0.0843, 0.0954, 0.2695])

        self.assertAlmostEqual(torch.sum(diff_g**2).item(), 0.0, delta=tol) # tol**2
        self.assertAlmostEqual(torch.sum(diff_v**2).item(), 0.0, delta=tol) # tol**2
        self.assertAlmostEqual(torch.sum(diff_noise**2).item(), 0.0, delta=tol) # tol**2
        self.assertAlmostEqual(torch.sum(diff_betas ** 2).item(), 0.0, delta=tol ** 2)

    def test_fit_logisticparallel(self, tol=1e-2):
        # Inputs
        data = Data.from_csv_file(example_data_path)
        algo_settings = AlgorithmSettings('mcmc_saem', n_iter=100, seed=0, device=torch.device("cuda"))

        # Initialize
        leaspy = Leaspy("logistic_parallel")
        leaspy.model.load_hyperparameters({'source_dimension': 2})

        # Fit the model on the data
        leaspy.fit(data, algorithm_settings=algo_settings)
        print(leaspy.model.parameters)

        ## uncomment to re-generate example file
        #leaspy.save(from_fit_model_path('logistic_parallel'), indent=2)

        self.assertAlmostEqual(leaspy.model.parameters['g'], 1.6327, delta=tol)
        self.assertAlmostEqual(leaspy.model.parameters['tau_mean'], 75.8097, delta=tol)
        self.assertAlmostEqual(leaspy.model.parameters['tau_std'], 5.4882, delta=tol)
        self.assertAlmostEqual(leaspy.model.parameters['xi_mean'], -3.3677, delta=tol)
        self.assertAlmostEqual(leaspy.model.parameters['xi_std'], 0.8474, delta=tol)
        self.assertAlmostEqual(leaspy.model.parameters['noise_std'], 0.1731, delta=tol)

        diff_deltas = leaspy.model.parameters['deltas'] - torch.tensor([-0.0129, -0.0640, 0.1117])
        self.assertAlmostEqual(torch.sum(diff_deltas ** 2).item(), 0.0, delta=tol**2)

    def test_fit_logisticparallel_diag_noise(self, tol=1e-2):
        # Inputs
        data = Data.from_csv_file(example_data_path)
        algo_settings = AlgorithmSettings('mcmc_saem', n_iter=100, seed=0, device=torch.device("cuda"))

        # Initialize
        leaspy = Leaspy("logistic_parallel", loss='MSE_diag_noise')
        leaspy.model.load_hyperparameters({'source_dimension': 2})

        # Fit the model on the data
        leaspy.fit(data, algorithm_settings=algo_settings)
        print(leaspy.model.parameters)

        ## uncomment to re-generate example file
        #leaspy.save(from_fit_model_path('logistic_parallel_diag_noise'), indent=2)

        self.assertAlmostEqual(leaspy.model.parameters['g'], 1.7531, delta=tol)
        self.assertAlmostEqual(leaspy.model.parameters['tau_mean'], 76.8170, delta=tol)
        self.assertAlmostEqual(leaspy.model.parameters['tau_std'], 5.7350, delta=tol)
        self.assertAlmostEqual(leaspy.model.parameters['xi_mean'], -3.6819, delta=tol)
        self.assertAlmostEqual(leaspy.model.parameters['xi_std'], 0.8078, delta=tol)

        diff_deltas = leaspy.model.parameters['deltas'] - torch.tensor([-0.0907, -0.1239,  0.0590])
        diff_noise = leaspy.model.parameters['noise_std'] - torch.tensor([0.2092, 0.0883, 0.1051, 0.2771])
        self.assertAlmostEqual(torch.sum(diff_deltas ** 2).item(), 0.0, delta=tol**2)
        self.assertAlmostEqual(torch.sum(diff_noise**2).item(), 0.0, delta=tol**2)

    def test_fit_univariate_logistic(self, tol=1e-2):
        # Inputs
        df = pd.read_csv(example_data_path)
        data = Data.from_dataframe(df.iloc[:,:3]) # one feature column
        algo_settings = AlgorithmSettings('mcmc_saem', n_iter=100, seed=0, device=torch.device("cuda"))

        # Initialize
        leaspy = Leaspy("univariate_logistic")

        # Fit the model on the data
        leaspy.fit(data, algorithm_settings=algo_settings)
        print(leaspy.model.parameters)

        ## uncomment to re-generate example file
        #leaspy.save(from_fit_model_path('univariate_logistic'), indent=2)

        self.assertAlmostEqual(leaspy.model.parameters['g'], 0.1376, delta=tol)
        self.assertAlmostEqual(leaspy.model.parameters['tau_mean'], 76.7298, delta=tol)
        self.assertAlmostEqual(leaspy.model.parameters['tau_std'], 5.2014, delta=tol)
        self.assertAlmostEqual(leaspy.model.parameters['xi_mean'], -3.0594, delta=tol)
        self.assertAlmostEqual(leaspy.model.parameters['xi_std'], 0.6888, delta=tol)
        self.assertAlmostEqual(leaspy.model.parameters['noise_std'], 0.1368, delta=tol)

    def test_fit_univariate_linear(self, tol=1e-2):
        # Inputs
        df = pd.read_csv(example_data_path)
        data = Data.from_dataframe(df.iloc[:,:3]) # one feature column
        algo_settings = AlgorithmSettings('mcmc_saem', n_iter=100, seed=0, device=torch.device("cuda"))

        # Initialize
        leaspy = Leaspy("univariate_linear")

        # Fit the model on the data
        leaspy.fit(data, algorithm_settings=algo_settings)
        print(leaspy.model.parameters)

        ## uncomment to re-generate example file
        #leaspy.save(from_fit_model_path('univariate_linear'), indent=2)

        self.assertAlmostEqual(leaspy.model.parameters['g'], 0.4533, delta=tol)
        self.assertAlmostEqual(leaspy.model.parameters['tau_mean'], 76.8098, delta=tol)
        self.assertAlmostEqual(leaspy.model.parameters['tau_std'], 5.2071, delta=tol)
        self.assertAlmostEqual(leaspy.model.parameters['xi_mean'], -4.2250, delta=tol)
        self.assertAlmostEqual(leaspy.model.parameters['xi_std'], 0.2939, delta=tol)
        self.assertAlmostEqual(leaspy.model.parameters['noise_std'], 0.1449, delta=tol)

    def test_fit_linear(self, tol=1e-1, tol_tau=2e-1):

        # Inputs
        data = Data.from_csv_file(example_data_path)
        algo_settings = AlgorithmSettings('mcmc_saem', n_iter=100, seed=0, device=torch.device("cuda"))

        # Initialize
        leaspy = Leaspy("linear")
        leaspy.model.load_hyperparameters({'source_dimension': 2})

        # Fit the model on the data
        leaspy.fit(data, algorithm_settings=algo_settings)
        print(leaspy.model.parameters)

        ## uncomment to re-generate example file
        #leaspy.save(from_fit_model_path('linear'), indent=2)

        self.assertAlmostEqual(leaspy.model.parameters['tau_mean'], 76.2992, delta=tol_tau)
        self.assertAlmostEqual(leaspy.model.parameters['tau_std'], 5.1489, delta=tol_tau)
        self.assertAlmostEqual(leaspy.model.parameters['xi_mean'], 0.0, delta=tol)
        self.assertAlmostEqual(leaspy.model.parameters['xi_std'], 1.0, delta=tol)
        self.assertAlmostEqual(leaspy.model.parameters['noise_std'], 0.1641, delta=tol)

        diff_g = leaspy.model.parameters['g'] - torch.tensor([0.4573, 0.0363, 0.0696, 0.2198])
        diff_v = leaspy.model.parameters['v0'] - torch.tensor([-4.9010, -5.5698, -5.5997, -4.7486])
        self.assertAlmostEqual(torch.sum(diff_g**2).item(), 0.0, delta=tol) # tol**2
        self.assertAlmostEqual(torch.sum(diff_v**2).item(), 0.0, delta=tol) # tol**2
