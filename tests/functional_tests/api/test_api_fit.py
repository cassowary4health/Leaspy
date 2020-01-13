import torch
import unittest
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

        self.assertAlmostEqual(leaspy.model.parameters['noise_std'],  0.2986, delta=0.01)
        self.assertAlmostEqual(leaspy.model.parameters['tau_mean'],  78.0270, delta=0.01)
        self.assertAlmostEqual(leaspy.model.parameters['tau_std'],  0.9494, delta=0.01)
        self.assertAlmostEqual(leaspy.model.parameters['xi_mean'], 0.0, delta=0.001)
        self.assertAlmostEqual(leaspy.model.parameters['xi_std'], 0.1317, delta=0.001)

        diff_g = leaspy.model.parameters['g'] - torch.tensor([1.9557, 2.5899, 2.5184, 2.2369])
        diff_v = leaspy.model.parameters['v0'] - torch.tensor([-3.5714, -3.5820, -3.5811, -3.5886])

        self.assertAlmostEqual(torch.sum(diff_g**2).item(), 0.0, delta=0.01)
        self.assertAlmostEqual(torch.sum(diff_v**2).item(), 0.0, delta=0.02)

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

    # TODO Univariate Model ???
    # TODO HMC, Gradient Descent
