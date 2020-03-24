import os
import unittest

import numpy as np
import torch

from leaspy import Leaspy
from tests import test_data_dir


class LeaspyEstimateMultiTest(unittest.TestCase):

    def setUp(self):
        # Initialize the leaspy model
        model_parameters_path = os.path.join(test_data_dir, 'model_parameters', 'fitted_multivariate_model.json')
        self.leaspy = Leaspy.load(model_parameters_path)

    def test_estimate_one_individual(self):
        # Initialize the individual parameters
        for individual_parameters in ({
                'xi': 0,
                'tau': 70,
                'sources': [0, 0]
            },
            {
                'xi': [0],
                'tau': [70],
                'sources': [[0, 0]]
            },
            {
                'xi': [[0]],
                'tau': [[70]],
                'sources': [[0, 0]]
        }):
            # Test with one time-point (scalar or array_like with 1 element)
            n_vis = 1
            for timepoints in (
                #[74],
                [(74,)],
                [[74]],
                #[np.array(74)],
                [np.array([74])],
            ):
                output = list(self.leaspy.estimate_multi(timepoints, individual_parameters))
                #self.assertIsInstance(logs, list)
                self.assertEqual(len(output), 1)
                o = output[0]
                self.assertIsInstance(o, torch.FloatTensor)
                self.assertEqual(o.ndim, 2)
                self.assertEqual(o.shape, (n_vis, self.leaspy.model.dimension))
                diff_output = o - torch.tensor([[0.5420, 0.3890, 0.3233, 0.7209]], dtype=torch.float32)
                self.assertAlmostEqual(torch.sum(diff_output ** 2).item(), 0.0, delta=10e-8)

            # Test with multiple time-points
            n_vis = 3
            for timepoints in (
                [[70, 71, 72]],
                [np.array([70,71,72])]
            ):
                output = list(self.leaspy.estimate_multi(timepoints, individual_parameters))
                #self.assertIsInstance(logs, list)
                self.assertEqual(len(output), 1)
                self.assertEqual(output[0].shape, (n_vis, self.leaspy.model.dimension))
                diff_output = output[0] - torch.tensor([
                    [0.1339, 0.0764, 0.0779, 0.1068],
                    [0.2045, 0.1211, 0.1152, 0.2049],
                    [0.2996, 0.1867, 0.1673, 0.3572]
                ], dtype=torch.float32)
                self.assertAlmostEqual(torch.sum(diff_output ** 2).item(), 0.0, delta=10e-8)

            # Various cases of expected errors due to wrong shape:
            # timepoints must be iterable, even if a single individual
            for tpts in (
                74,
                [74],
                [np.array(74)],
                np.array(74)
            ):
                self.assertRaises(ValueError, lambda: self.leaspy.estimate_multi(tpts, individual_parameters))


    def test_estimate_different_multiple_individuals(self):

        n_ind = 2

        ips_list_1D = {
            'xi': [0,0],
            'tau': [70,70],
            'sources': [[0, 0],
                        [0, 0]]
        }
        ips_list_2D = {
            'xi': [[0],[0]],
            'tau': [[70],[70]],
            'sources': [[0, 0],
                        [0, 0]]
        }
        t_ips = {k: torch.tensor(v) for k,v in ips_list_2D.items()}

        # Initialize the individual parameters
        for individual_parameters in (ips_list_1D, ips_list_2D, t_ips):

            # Test with one time-point (scalar or array_like with 1 element)
            n_vis = 1
            for timepoints in (
                #[74,74],
                [[74],[74]],
            ):
                output = list(self.leaspy.estimate_multi(timepoints, individual_parameters))
                #self.assertIsInstance(logs, list)
                self.assertEqual(len(output), n_ind)
                for o in output:
                    self.assertIsInstance(o, torch.FloatTensor)
                    self.assertEqual(o.ndim, 2)
                    self.assertEqual(o.shape, (n_vis, self.leaspy.model.dimension))
                    diff_output = o - torch.tensor([[0.5420, 0.3890, 0.3233, 0.7209]], dtype=torch.float32)
                    self.assertAlmostEqual(torch.sum(diff_output ** 2).item(), 0.0, delta=10e-8)

            # Test with multiple time-points
            for timepoints in (
                [[71],[70,71,72]],
            ):
                output = list(self.leaspy.estimate_multi(timepoints, individual_parameters))
                #self.assertIsInstance(logs, list)
                self.assertEqual(len(output), n_ind)

                diff_output = output[0] - torch.tensor([[0.2045, 0.1211, 0.1152, 0.2049]], dtype=torch.float32)
                self.assertAlmostEqual(torch.sum(diff_output ** 2).item(), 0.0, delta=10e-8)
                self.assertEqual(output[0].shape, (1, self.leaspy.model.dimension))

                diff_output = output[1] - torch.tensor([
                    [0.1339, 0.0764, 0.0779, 0.1068],
                    [0.2045, 0.1211, 0.1152, 0.2049],
                    [0.2996, 0.1867, 0.1673, 0.3572]
                ], dtype=torch.float32)
                self.assertAlmostEqual(torch.sum(diff_output ** 2).item(), 0.0, delta=10e-8)
                self.assertEqual(output[1].shape, (3, self.leaspy.model.dimension))

            # No timepoint for an individual
            timepoints = [[],[74]]
            output = list(self.leaspy.estimate_multi(timepoints, individual_parameters))
            self.assertEqual(len(output), n_ind)

            o = output[0]
            self.assertIsInstance(o, torch.FloatTensor)
            self.assertEqual(o.ndim, 2)
            self.assertEqual(o.shape, (0, self.leaspy.model.dimension))

            o = output[1]
            diff_output = o - torch.tensor([[0.5420, 0.3890, 0.3233, 0.7209]], dtype=torch.float32)
            self.assertAlmostEqual(torch.sum(diff_output ** 2).item(), 0.0, delta=10e-8)
            self.assertEqual(o.shape, (1, self.leaspy.model.dimension))

            # Various cases of expected errors due to wrong shape:
            # timepoints must be iterable, even if a single individual
            for tpts in (
                [74,74], # old behavior
                74,
                [74],
                [[74]],
                [74,74,74], # 3 individuals...
                [[74],[74],[74]], # 3 individuals...
            ):
                self.assertRaises(ValueError, lambda: self.leaspy.estimate_multi(tpts, individual_parameters))
