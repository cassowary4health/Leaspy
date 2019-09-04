import torch
from leaspy.main import Leaspy

import unittest
from leaspy.inputs.data.data import Data

from leaspy.inputs.settings.algorithm_settings import AlgorithmSettings

from tests import example_data_path

class LeaspyFitTest(unittest.TestCase):

    """

    ## Test MCMC-SAEM
    def test_fit_logistic(self):

        # Inputs
        data = Data.from_csv_file(example_data_path)
        algo_settings = AlgorithmSettings('mcmc_saem', n_iter=10, seed=0)

        # Initialize
        leaspy = Leaspy("logistic")
        leaspy.model.load_hyperparameters({'source_dimension': 2})

        # Fit the model on the data
        leaspy.fit(data, algorithm_settings=algo_settings)

        self.assertAlmostEqual(leaspy.model.parameters['noise_std'],  0.2158, delta=0.01)
        self.assertAlmostEqual(leaspy.model.parameters['tau_mean'],  77.6206, delta=0.01)
        self.assertAlmostEqual(leaspy.model.parameters['tau_std'],  5.5511, delta=0.01)
        self.assertAlmostEqual(leaspy.model.parameters['xi_mean'], 0.0, delta=0.001)
        self.assertAlmostEqual(leaspy.model.parameters['xi_std'], 0.1768, delta=0.001)

        diff_g = leaspy.model.parameters['g'] - torch.Tensor([1.9978, 2.5808, 2.5728, 2.2152])
        diff_v = leaspy.model.parameters['v0'] - torch.Tensor([-3.0979, -3.7677, -3.8723, -2.9698])

        self.assertAlmostEqual(torch.sum(diff_g**2), 0.0, delta=0.01)
        self.assertAlmostEqual(torch.sum(diff_v**2), 0.0, delta=0.01)


    def test_fit_logisticparallel(self):
        # Inputs
        data = Data.from_csv_file(example_data_path)
        algo_settings = AlgorithmSettings('mcmc_saem', n_iter=10, seed=0)

        # Initialize
        leaspy = Leaspy("logistic_parallel")
        leaspy.model.load_hyperparameters({'source_dimension': 2})

        # Fit the model on the data
        leaspy.fit(data, algorithm_settings=algo_settings)

        self.assertAlmostEqual(leaspy.model.parameters['noise_std'], 0.2433, delta=0.01)
        self.assertAlmostEqual(leaspy.model.parameters['tau_mean'], 77.8313, delta=0.01)
        self.assertAlmostEqual(leaspy.model.parameters['tau_std'], 5.6973, delta=0.01)
        self.assertAlmostEqual(leaspy.model.parameters['xi_mean'], -2.9848, delta=0.001)
        self.assertAlmostEqual(leaspy.model.parameters['xi_std'], 0.0586, delta=0.001)

        self.assertAlmostEqual(leaspy.model.parameters['g'], 1.0130, delta=0.001)

        diff_deltas = leaspy.model.parameters['deltas'] - torch.Tensor([0.0034, -0.0055, 0.0174])
        self.assertAlmostEqual(torch.sum(diff_deltas ** 2), 0.0, delta=0.01)"""

