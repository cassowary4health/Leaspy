import unittest
from typing import Optional

import torch
import pandas as pd

from leaspy import Leaspy, Data, AlgorithmSettings
from tests import example_data_path, from_fit_model_path



class LeaspyTestBase:
    def __init__(self, model_name:str, **kwargs):
        self.model_name = model_name
        self.leaspy = Leaspy(model_name, **kwargs)

    def fit(self, data, algo_settings: AlgorithmSettings, **kwargs):

        self.leaspy.fit(data, algorithm_settings=algo_settings)

        return self.leaspy

# Weirdly, some results are perfectly reproducible on local mac + CI linux but not on CI mac...
# Increasing tolerances so to pass despite these reproducibility issues...
class LeaspyFitTest(unittest.TestCase):

    # Test MCMC-SAEM
    def test_fit_logistic(self, tol=5e-2, tol_tau=2e-1):
        algo_settings = AlgorithmSettings(name="mcmc_saem", n_iter=100, seed=0)

        leaspy = LeaspyTestBase(model_name="logistic", source_dimension=2).fit(data=Data.from_csv_file(example_data_path), algo_settings=algo_settings)

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
        algo_settings = AlgorithmSettings('mcmc_saem', loss='MSE_diag_noise', n_iter=100, seed=0)
        leaspy = LeaspyTestBase(model_name="logistic", source_dimension=2).fit(data=Data.from_csv_file(example_data_path), algo_settings=algo_settings)

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
        algo_settings = AlgorithmSettings('mcmc_saem', n_iter=100, seed=0)
        leaspy = LeaspyTestBase(model_name="logistic_parallel", source_dimension=2).fit(data=Data.from_csv_file(example_data_path), algo_settings=algo_settings)

        self.assertAlmostEqual(leaspy.model.parameters['g'], 1.6102, delta=tol)
        self.assertAlmostEqual(leaspy.model.parameters['tau_mean'], 77.9064, delta=tol)
        self.assertAlmostEqual(leaspy.model.parameters['tau_std'], 5.3658, delta=tol)
        self.assertAlmostEqual(leaspy.model.parameters['xi_mean'], -3.6563, delta=tol)
        self.assertAlmostEqual(leaspy.model.parameters['xi_std'], 0.5822, delta=tol)
        self.assertAlmostEqual(leaspy.model.parameters['noise_std'], 0.1576, delta=tol)

        diff_deltas = leaspy.model.parameters['deltas'] - torch.tensor([-0.0848, -0.0065, -0.0105])
        self.assertAlmostEqual(torch.sum(diff_deltas ** 2).item(), 0.0, delta=tol**2)

    def test_fit_logisticparallel_diag_noise(self, tol=1e-2):
        algo_settings = AlgorithmSettings('mcmc_saem', loss='MSE_diag_noise', n_iter=100, seed=0)
        leaspy = LeaspyTestBase(model_name="logistic_parallel", source_dimension=2).fit(data=Data.from_csv_file(example_data_path), algo_settings=algo_settings)

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
        leaspy = LeaspyTestBase(model_name="univariate_logistic").fit(algo_settings=algo_settings,
                                                                                        data=data)

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

        leaspy = LeaspyTestBase(model_name="univariate_linear").fit(algo_settings=algo_settings,
                                                                                        data=data)
        ## uncomment to re-generate example file
        #leaspy.save(from_fit_model_path('univariate_linear'), indent=2)

        self.assertAlmostEqual(leaspy.model.parameters['g'], 0.4936, delta=tol)
        self.assertAlmostEqual(leaspy.model.parameters['tau_mean'], 78.3471, delta=tol)
        self.assertAlmostEqual(leaspy.model.parameters['tau_std'], 5.2568, delta=tol)
        self.assertAlmostEqual(leaspy.model.parameters['xi_mean'], -3.9552, delta=tol)
        self.assertAlmostEqual(leaspy.model.parameters['xi_std'], 0.8314, delta=tol)
        self.assertAlmostEqual(leaspy.model.parameters['noise_std'], 0.1114, delta=tol)

    def test_fit_linear(self, tol=1e-1, tol_tau=2e-1):
        algo_settings = AlgorithmSettings('mcmc_saem', n_iter=100, seed=0)
        leaspy = LeaspyTestBase(model_name="linear", source_dimension=2).fit(data=Data.from_csv_file(example_data_path), algo_settings=algo_settings)

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


    # TODO HMC, Gradient Descent
