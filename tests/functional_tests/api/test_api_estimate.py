import os
import unittest

import torch

from leaspy import Leaspy, IndividualParameters
from tests import test_data_dir


class LeaspyEstimateTest(unittest.TestCase):

    def test_estimate(self):
        model_parameters_path = os.path.join(test_data_dir, 'model_parameters', 'fitted_multivariate_model.json')
        leaspy = Leaspy.load(model_parameters_path)

        ip_path = os.path.join(test_data_dir, 'io', 'outputs', 'ip_save.json')
        ip = IndividualParameters.load(ip_path)

        timepoints = {
            'idx1': [78, 81],
            'idx2': [91]
        }

        estimations = leaspy.estimate(timepoints, ip)

        test = {
            'idx1': [[0.9168074,  0.88841885, 0.80543965, 0.9921461 ],
                     [0.98348546, 0.9773835,  0.9456895,  0.99938035]],
            'idx2': [[0.9999131,  0.9998343,  0.9991264,  0.99999964]]}


        # Test
        self.assertEqual(estimations.keys(), test.keys())
        for k in estimations.keys():
            self.assertTrue(k in test.keys())
            for v1, v2 in zip(estimations[k], test[k]):

                self.assertTrue((v1 - v2 < 10e-8).all())

