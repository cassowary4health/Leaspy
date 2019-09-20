import torch
import unittest
from leaspy import Leaspy, Data, AlgorithmSettings
from tests import example_data_path


class LeaspyFitTest(unittest.TestCase):

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

        self.assertAlmostEqual(leaspy.model.parameters['noise_std'],  0.2596, delta=0.01)
        self.assertAlmostEqual(leaspy.model.parameters['tau_mean'],  77.9872, delta=0.01)
        self.assertAlmostEqual(leaspy.model.parameters['tau_std'],  1.0339, delta=0.01)
        self.assertAlmostEqual(leaspy.model.parameters['xi_mean'], 0.0, delta=0.001)
        self.assertAlmostEqual(leaspy.model.parameters['xi_std'], 0.1707, delta=0.001)

        diff_g = leaspy.model.parameters['g'] - torch.Tensor([1.9883, 2.5703, 2.5591, 2.2219])
        diff_v = leaspy.model.parameters['v0'] - torch.Tensor([-3.3347, -3.9911, -4.1306, -3.2095])

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

        self.assertAlmostEqual(leaspy.model.parameters['noise_std'],0.2685, delta=0.01)
        self.assertAlmostEqual(leaspy.model.parameters['tau_mean'], 70.4183, delta=0.01)
        self.assertAlmostEqual(leaspy.model.parameters['tau_std'], 2.0912, delta=0.01)
        self.assertAlmostEqual(leaspy.model.parameters['xi_mean'], -3.0492, delta=0.001)
        self.assertAlmostEqual(leaspy.model.parameters['xi_std'], 0.1003, delta=0.001)

        self.assertAlmostEqual(leaspy.model.parameters['g'], 1.0092, delta=0.001)

        diff_deltas = leaspy.model.parameters['deltas'] - torch.Tensor([-0.0087, 0.0033, 0.0113])
        self.assertAlmostEqual(torch.sum(diff_deltas ** 2), 0.0, delta=0.01)

    # TODO Univariate Model ???
    # TODO HMC, Gradient Descent



