import os

import torch

from leaspy import IndividualParameters

from tests import LeaspyTestCase


class LeaspyPersonalizeTest_Mixin(LeaspyTestCase):
    """Mixin holding generic personalization methods that may be safely reused in other tests (no actual test here)."""

    @classmethod
    def generic_personalization(cls, hardcoded_model_name: str, *,
                                algo_path: str = None, algo_name: str = None, **algo_params):
        """Helper for a generic personalization in following tests."""

        # load saved model (hardcoded values)
        leaspy = cls.get_hardcoded_model(hardcoded_model_name)

        # load the right data
        data = cls.get_suited_test_data_for_model(hardcoded_model_name)

        # create the personalize algo settings (from path or name + params)
        algo_settings = cls.get_algo_settings(path=algo_path, name=algo_name, **algo_params)

        # return results of personalization
        ips, noise = leaspy.personalize(data, settings=algo_settings, return_noise=True)

        return ips, noise, leaspy # data?

    def check_consistency_of_personalization_outputs(self, ips, noise_std, expected_noise_std, *, tol_noise = 5e-3):

        self.assertIsInstance(ips, IndividualParameters)
        self.assertIsInstance(noise_std, torch.Tensor)

        if isinstance(expected_noise_std, float):
            self.assertEqual(noise_std.numel(), 1) # scalar noise
            self.assertAlmostEqual(noise_std.item(), expected_noise_std, delta=tol_noise)
        else:
            # vector of noises (for diag_noise)
            self.assertEqual(noise_std.numel(), len(expected_noise_std)) # diagonal noise
            diff_noise = noise_std - torch.tensor(expected_noise_std)
            self.assertAlmostEqual((diff_noise ** 2).sum(), 0., delta=tol_noise**2,
                                   msg=f'Noise != Expected: {noise_std.tolist()} != {expected_noise_std}')

class LeaspyPersonalizeTest(LeaspyPersonalizeTest_Mixin):

    # Test MCMC-SAEM
    def test_personalize_mean_real_logistic(self, tol_noise=1e-3):
        """
        Load logistic model from file, and personalize it to data from ...
        """
        # Load saved algorithm
        path_settings = os.path.join(self.test_data_dir, 'settings', 'algo', 'settings_mean_real.json')
        ips, noise_std, _ = self.generic_personalization('logistic_scalar_noise', algo_path=path_settings)

        self.check_consistency_of_personalization_outputs(ips, noise_std, expected_noise_std=0.11631, tol_noise=tol_noise)

    def test_personalize_mode_real_logistic(self, tol_noise=1e-3):
        """
        Load logistic model from file, and personalize it to data from ...
        """
        # Load saved algorithm
        path_settings = os.path.join(self.test_data_dir, 'settings', 'algo', 'settings_mode_real.json')
        ips, noise_std, _ = self.generic_personalization('logistic_scalar_noise', algo_path=path_settings)

        self.check_consistency_of_personalization_outputs(ips, noise_std, expected_noise_std=0.11711, tol_noise=tol_noise)

    def test_personalize_scipy_minimize(self, tol_noise=5e-3):
        """
        Load data and compute its personalization on various models
        with scipy minimize personalization (with and without jacobian)

        Use hardcoded models rather than the ones from fit functional tests
        in order to isolate functional problems...
        """

        for (model_name, use_jacobian), expected_noise_std in {

            ('logistic_scalar_noise', False):               0.118869,
            ('logistic_scalar_noise', True):                0.118774,
            ('logistic_diag_noise_id', False): [0.1414, 0.0806, 0.0812, 0.1531],
            ('logistic_diag_noise_id', True):  [0.1414, 0.0804, 0.0811, 0.1529],
            ('logistic_diag_noise', False):    [0.156, 0.0595, 0.0827, 0.1515],
            ('logistic_diag_noise', True):     [0.1543, 0.0597, 0.0827, 0.1509],
            ('logistic_parallel_scalar_noise', False):      0.0960,
            ('logistic_parallel_scalar_noise', True):       0.0956,
            ('univariate_logistic', False):    0.134107,
            ('univariate_logistic', True):     0.134116,
            ('univariate_linear', False):      0.081208,
            ('univariate_linear', True):       0.081208,
            ('linear_scalar_noise', False):                 0.124072,
            ('linear_scalar_noise', True):                  0.124071,

            # TODO: binary (crossentropy) models here

        }.items():

            with self.subTest(model_name=model_name, use_jacobian=use_jacobian):

                # only look at residual MSE to detect any regression in personalization
                ips, noise_std, _ = self.generic_personalization(model_name, algo_name='scipy_minimize', seed=0, use_jacobian=use_jacobian)

                self.check_consistency_of_personalization_outputs(ips, noise_std, expected_noise_std=expected_noise_std, tol_noise=tol_noise)

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
