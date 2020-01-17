import os
import unittest

import numpy as np
import pandas as pd
import torch

from leaspy import AlgorithmSettings, Data, Leaspy
from leaspy.algo.simulate.simulate import SimulationAlgorithm
from tests import example_data_path, test_data_dir


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

        settings = AlgorithmSettings('simulation', sources_method="dummy")
        self.assertRaises(ValueError, SimulationAlgorithm, settings)

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

    def test_check_cofactors(self):
        data = Data.from_csv_file(example_data_path)
        cofactors = pd.read_csv(os.path.join(test_data_dir, "inputs/data_tiny_covariate.csv"))
        cofactors.columns = ("ID", "Treatments")
        cofactors['ID'] = cofactors['ID'].apply(lambda x: str(x))
        cofactors = cofactors.set_index("ID")
        data.load_cofactors(cofactors, ["Treatments"])

        model = Leaspy.load(os.path.join(test_data_dir, "model_parameters/multivariate_model_sampler.json"))
        settings = AlgorithmSettings('mode_real')
        results = model.personalize(data, settings)

        settings = AlgorithmSettings('simulation', cofactor="dummy")
        self.assertRaises(ValueError, model.simulate, results, settings)

        settings = AlgorithmSettings('simulation', cofactor="Treatments", cofactor_state="dummy")
        self.assertRaises(ValueError, model.simulate, results, settings)

    # global behaviour of SimulationAlgorithm class is tested in the functional test test_api.py
