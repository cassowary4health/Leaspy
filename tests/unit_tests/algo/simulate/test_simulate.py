import unittest

import numpy as np
import torch

from leaspy.algo.simulate.simulate import SimulationAlgorithm
from leaspy.inputs.settings.algorithm_settings import AlgorithmSettings


class SimulationAlgorithmTest(unittest.TestCase):

    def setUp(self):
        self.settings = AlgorithmSettings('simulation')
        self.algo = SimulationAlgorithm(self.settings)

    def test_construtor(self):
        self.assertEqual(self.settings.parameters['bandwidth_method'], self.algo.bandwidth_method)
        self.assertEqual(self.settings.parameters['noise'], self.algo.noise)
        self.assertEqual(self.settings.parameters['number_of_subjects'], self.algo.number_of_subjects)
        self.assertEqual(self.settings.parameters['mean_number_of_visits'], self.algo.mean_number_of_visits)
        self.assertEqual(self.settings.parameters['std_number_of_visits'], self.algo.std_number_of_visits)
        self.assertEqual(self.settings.parameters['cofactor'], self.algo.cofactor)
        self.assertEqual(self.settings.parameters['cofactor_state'], self.algo.cofactor_state)

    def test_get_number_of_visits(self):
        n_visit = self.algo._get_number_of_visits()
        self.assertTrue(type(n_visit) == int)
        self.assertTrue(n_visit >= 1)

    def test_get_mean_and_covariance_matrix(self):
        values = np.random.rand(100, 5)
        t_mean = torch.tensor(values).mean(dim=0)
        self.assertTrue(np.allclose(values.mean(axis=0),
                                    t_mean.numpy()))
        t_cov = torch.tensor(values) - t_mean[None, :]
        t_cov = 1. / (t_cov.size(0) - 1) * t_cov.t() @ t_cov
        self.assertTrue(np.allclose(np.cov(values.T),
                                    t_cov.numpy()))

    # global behaviour of SimulationAlgorithm class is tested in the functional test test_api.py
