import os
import unittest

import torch
import pandas as pd

from leaspy import Leaspy, Data, AlgorithmSettings
from tests import example_data_path, test_data_dir, hardcoded_model_path


class LeaspyPersonalizeTest(unittest.TestCase):

    # Test MCMC-SAEM
    def test_personalize_meanreal_logistic(self, tol_noise=1e-3):
        """
        Load logistic model from file, and personalize it to data from ...
        """
        # Inputs
        data = Data.from_csv_file(example_data_path)

        # Initialize
        leaspy = Leaspy.load(hardcoded_model_path('logistic'))

        # Launch algorithm
        path_settings = os.path.join(os.path.dirname(__file__), "data/settings_mean_real.json")
        algo_personalize_settings = AlgorithmSettings.load(path_settings)
        _, noise_std = leaspy.personalize(data, settings=algo_personalize_settings, return_noise=True)

        self.assertAlmostEqual(noise_std.item(), 0.11631, delta=tol_noise)

    def test_personalize_modereal_logistic(self, tol_noise=1e-3):
        """
        Load logistic model from file, and personalize it to data from ...
        """
        # Inputs
        data = Data.from_csv_file(example_data_path)

        # Initialize
        leaspy = Leaspy.load(hardcoded_model_path('logistic'))

        # Launch algorithm
        path_settings = os.path.join(os.path.dirname(__file__), "data/settings_mode_real.json")
        algo_personalize_settings = AlgorithmSettings.load(path_settings)
        _, noise_std = leaspy.personalize(data, settings=algo_personalize_settings, return_noise=True)

        self.assertAlmostEqual(noise_std.item(), 0.11711, delta=tol_noise)

    def test_personalize_scipy_models(self, tol_noise=1e-3):
        """
        Load data and compute its personalization on various models
        with scipy minimize personalization (with and without jacobian)

        Use hardcoded models rather than the ones from fit functional tests
        in order to isolate functional problems...
        """

        for (model_name, use_jacobian), expected_noise_std in {

            ('logistic', False):               0.118869,
            ('logistic', True):                0.118774,
            ('logistic_diag_noise_id', False): [0.1414, 0.0806, 0.0812, 0.1531],
            ('logistic_diag_noise_id', True):  [0.1414, 0.0804, 0.0811, 0.1529],
            ('logistic_diag_noise', False):    [0.1542, 0.0597, 0.0827, 0.1509],
            ('logistic_diag_noise', True):     [0.1543, 0.0597, 0.0827, 0.1509],
            ('logistic_parallel', False):      0.0960,
            ('logistic_parallel', True):       0.0956,
            ('univariate_logistic', False):    0.134107,
            ('univariate_logistic', True):     0.134116,
            ('univariate_linear', False):      0.081208,
            ('univariate_linear', True):       0.081208,
            ('linear', False):                 0.124072,
            ('linear', True):                  0.124071,

        }.items():

            with self.subTest(model_name=model_name, use_jacobian=use_jacobian):
                # load data
                if 'univariate' not in model_name:
                    data = Data.from_csv_file(example_data_path)
                else:
                    df = pd.read_csv(example_data_path)
                    data = Data.from_dataframe(df.iloc[:,:3]) # one feature column

                # load saved model (hardcoded values)
                leaspy = Leaspy.load(hardcoded_model_path(model_name))

                # scipy algo (with/without jacobian)
                algo_personalize_settings = AlgorithmSettings('scipy_minimize', seed=0, use_jacobian=use_jacobian)

                # only look at residual MSE to detect any regression in personalization
                _, noise_std = leaspy.personalize(data, settings=algo_personalize_settings, return_noise=True)

                if isinstance(expected_noise_std, float):
                    self.assertEqual(noise_std.numel(), 1) # scalar noise
                    self.assertAlmostEqual(noise_std.item(), expected_noise_std, delta=tol_noise)
                else:
                    # vector of noises (for diag_noise)
                    diff_noise = noise_std - torch.tensor(expected_noise_std)
                    self.assertAlmostEqual((diff_noise ** 2).sum(), 0., msg=noise_std, delta=tol_noise**2)


    # TODO : problem with nans
    """
    def test_personalize_gradientdescent(self):
        # Inputs
        data = Data.from_csv_file(example_data_path)

        # Initialize
        leaspy = Leaspy.load(...)

        # Launch algorithm
        algo_personalize_settings = AlgorithmSettings('gradient_descent_personalize', seed=2)
        result = leaspy.personalize(data, settings=algo_personalize_settings)

        self.assertAlmostEqual(result.noise_std,  0.17925, delta=0.01)"""
