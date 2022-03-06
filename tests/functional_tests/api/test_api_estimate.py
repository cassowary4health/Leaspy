import os
import unittest

import torch
import numpy as np

from leaspy import Leaspy, IndividualParameters
from tests import test_data_dir, hardcoded_model_path


class LeaspyEstimateTest(unittest.TestCase):

    def check_almost_equal_for_all_ind_tpts(self, a, b, tol=1e-5):
        self.assertEqual(a.keys(), b.keys())
        for ind_id, a_i_vis in a.items(): # individual
            b_i_vis = b[ind_id]
            self.assertEqual(len(a_i_vis), len(b_i_vis)) # same nb of visits estimated!
            for v1, v2 in zip(a_i_vis, b_i_vis): # visits
                self.assertTrue(np.allclose(v1, v2, atol=tol), a)

    def batch_checks(self, ip, tpts, models, expected_ests):
        for model_name in models:
            with self.subTest(model_name=model_name):
                model_path = hardcoded_model_path(model_name)
                leaspy = Leaspy.load(model_path)

                estimations = leaspy.estimate(tpts, ip)
                print(estimations)

                self.check_almost_equal_for_all_ind_tpts(estimations, expected_ests)

    def test_estimate(self):

        ip_path = os.path.join(test_data_dir, 'leaspy_io', 'outputs', 'ip_save.json')
        ip = IndividualParameters.load(ip_path)

        timepoints = {
            'idx1': [78, 81],
            'idx2': [71]
        }

        # first batch of tests same logistic model but with / without diag noise (no impact in estimation!)
        models = ('logistic', 'logistic_diag_noise_id', 'logistic_diag_noise')
        expected_ests = {
            'idx1': [[0.99641526, 0.34549406, 0.67467   , 0.98959327],
                     [0.9994672 , 0.5080943 , 0.8276345 , 0.99921334]],
            'idx2': [[0.13964376, 0.1367586 , 0.23170303, 0.01551363]]
        }

        self.batch_checks(ip, timepoints, models, expected_ests)

        # TODO logistic parallel?

        # TODO univariate models?

        # TODO linear model?

