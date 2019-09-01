import torch
from leaspy.main import Leaspy

import unittest
from leaspy.inputs.data.data import Data

from leaspy.inputs.settings.algorithm_settings import AlgorithmSettings

from tests import example_data_dir

class LeaspyFitTest(unittest.TestCase):

    ## Test MCMC-SAEM
    def test_fit_logistic(self):

        # Inputs
        data = Data.from_csv_file(example_data_dir)
        algo_settings = AlgorithmSettings('mcmc_saem', n_iter=10, seed=0)

        # Initialize
        leaspy = Leaspy("logistic")
        leaspy.model.load_hyperparameters({'source_dimension': 2})

        # Fit the model on the data
        leaspy.fit(data, algorithm_settings=algo_settings)

        self.assertAlmostEqual(leaspy.model.parameters['noise_std'],  0.3392, delta=0.01)
        self.assertAlmostEqual(leaspy.model.parameters['tau_mean'],   76.0229, delta=0.01)
        self.assertAlmostEqual(leaspy.model.parameters['tau_std'],  0.9095, delta=0.01)
        self.assertAlmostEqual(leaspy.model.parameters['xi_mean'], 0.0, delta=0.001)
        self.assertAlmostEqual(leaspy.model.parameters['xi_std'], 0.1681, delta=0.001)

        diff_g = leaspy.model.parameters['g'] -torch.Tensor([1.9509, 2.5124, 2.4718, 2.1432])
        diff_v = leaspy.model.parameters['v0'] - torch.Tensor([-3.2049, -3.5710, -3.6805, -2.9792])

        self.assertAlmostEqual(torch.sum(diff_g**2), 0.0, delta=0.01)
        self.assertAlmostEqual(torch.sum(diff_v**2), 0.0, delta=0.01)

    def test_fit_logisticparallel(self):
        # Inputs
        data = Data.from_csv_file(example_data_dir)
        algo_settings = AlgorithmSettings('mcmc_saem', n_iter=10, seed=0)

        # Initialize
        leaspy = Leaspy("logistic_parallel")
        leaspy.model.load_hyperparameters({'source_dimension': 2})

        # Fit the model on the data
        leaspy.fit(data, algorithm_settings=algo_settings)

        self.assertAlmostEqual(leaspy.model.parameters['noise_std'],0.2578, delta=0.01)
        self.assertAlmostEqual(leaspy.model.parameters['tau_mean'], 70.1108, delta=0.01)
        self.assertAlmostEqual(leaspy.model.parameters['tau_std'], 1.8394, delta=0.01)
        self.assertAlmostEqual(leaspy.model.parameters['xi_mean'], -3.0908, delta=0.001)
        self.assertAlmostEqual(leaspy.model.parameters['xi_std'], 0.1428, delta=0.001)

        self.assertAlmostEqual(leaspy.model.parameters['g'], 1.0138, delta=0.001)

        diff_deltas = leaspy.model.parameters['deltas'] - torch.Tensor([-0.0279, -0.0090, -0.0172])
        self.assertAlmostEqual(torch.sum(diff_deltas ** 2), 0.0, delta=0.01)

