import os
import unittest

import torch

from leaspy import Leaspy
from tests import test_data_dir


class LeaspyEstimateTest(unittest.TestCase):

    def test_estimate_different_timepoints(self):
        # Initialize the leaspy model
        model_parameters_path = os.path.join(test_data_dir, 'model_parameters', 'fitted_multivariate_model.json')
        leaspy = Leaspy.load(model_parameters_path)

        # Initialize the individual parameters
        individual_parameters = {
            'xi': 0,
            'tau': 70,
            'sources': [0, 0]
        }

        # Test with one time-point
        timepoints = 74
        output = leaspy.estimate(timepoints, individual_parameters)
        diff_output = output - torch.tensor([[[0.5420, 0.3890, 0.3233, 0.7209]]], dtype=torch.float32)
        self.assertAlmostEqual(torch.sum(diff_output ** 2).item(), 0.0, delta=10e-8)

        # Test with one multiple time-points
        timepoints = [70, 71, 72]
        output = leaspy.estimate(timepoints, individual_parameters)
        diff_output = output - torch.tensor([
            [0.1339, 0.0764, 0.0779, 0.1068],
            [0.2045, 0.1211, 0.1152, 0.2049],
            [0.2996, 0.1867, 0.1673, 0.3572]
        ], dtype=torch.float32)
        self.assertAlmostEqual(torch.sum(diff_output ** 2).item(), 0.0, delta=10e-8)

    def test_estimate_different_individual_parameters(self):
        # Initialize the leaspy model and the time-points
        model_parameters_path = os.path.join(test_data_dir, 'model_parameters', 'fitted_multivariate_model.json')
        leaspy = Leaspy.load(model_parameters_path)
        timepoints = [70, 74]

        # Test with standard individual_parameters
        individual_parameters = {
            'xi': 0,
            'tau': 70,
            'sources': [0, 0]
        }

        output = leaspy.estimate(timepoints, individual_parameters)
        expected_output = torch.tensor([
            [0.1339, 0.0764, 0.0779, 0.1068],
            [0.5420, 0.3890, 0.3233, 0.7209]
        ], dtype=torch.float32)
        diff_output = expected_output - output
        self.assertAlmostEqual(torch.sum(diff_output ** 2).item(), 0.0, delta=10e-8)

        # Test with individual parameters as list
        individual_parameters = {
            'xi': [0],
            'tau': [70],
            'sources': [0, 0]
        }

        output = leaspy.estimate(timepoints, individual_parameters)
        diff_output = expected_output - output
        self.assertAlmostEqual(torch.sum(diff_output ** 2).item(), 0.0, delta=10e-8)

        # Test with individual_parameters as tensors
        individual_parameters = {
            'xi': torch.tensor([0], dtype=torch.float32),
            'tau': torch.tensor([70], dtype=torch.float32),
            'sources': torch.tensor([0, 0], dtype=torch.float32)
        }
        output = leaspy.estimate(timepoints, individual_parameters)
        diff_output = expected_output - output
        self.assertAlmostEqual(torch.sum(diff_output ** 2).item(), 0.0, delta=10e-8)
