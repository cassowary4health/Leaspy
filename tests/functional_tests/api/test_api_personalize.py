import torch

from leaspy import Data, IndividualParameters

from tests import LeaspyTestCase


class LeaspyPersonalizeTest_Mixin(LeaspyTestCase):
    """Mixin holding generic personalization methods that may be safely reused in other tests (no actual test here)."""

    @classmethod
    def generic_personalization(cls, hardcoded_model_name: str, *,
                                data_path: str = None, data_kws: dict = {},
                                algo_path: str = None, algo_name: str = None, **algo_params):
        """Helper for a generic personalization in following tests."""

        # load saved model (hardcoded values)
        leaspy = cls.get_hardcoded_model(hardcoded_model_name)

        # load the right data
        if data_path is None:
            # automatic (main test data)
            data = cls.get_suited_test_data_for_model(hardcoded_model_name)
        else:
            # relative path to data (csv expected)
            data_full_path = cls.test_data_path('data_mock', data_path)
            data = Data.from_csv_file(data_full_path, **data_kws)

        # force correct feature names for tests
        assert len(leaspy.model.features) == len(data.headers), "Bad dimension"
        leaspy.model.features = data.headers

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
        path_settings = self.test_data_path('settings', 'algo', 'settings_mean_real.json')
        ips, noise_std, _ = self.generic_personalization('logistic_scalar_noise', algo_path=path_settings)

        self.check_consistency_of_personalization_outputs(ips, noise_std, expected_noise_std=0.11631, tol_noise=tol_noise)

    def test_personalize_mode_real_logistic(self, tol_noise=1e-3):
        """
        Load logistic model from file, and personalize it to data from ...
        """
        # Load saved algorithm
        path_settings = self.test_data_path('settings', 'algo', 'settings_mode_real.json')
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

    def test_scipy_minimize_robustness_to_data_sparsity(self, rtol=2e-2, atol=5e-3):
        """
        In this test, we check that estimated individual parameters are the same no matter if data is sparse
        (i.e. multiple really close visits with many missing values) or data is 'merged' in a rounded visit.

        TODO? we could build a mock dataset to also check same property for calibration :)
        """

        for (model_name, use_jacobian), expected_noise_std in {

            ('logistic_scalar_noise', False):               0.1161,
            ('logistic_scalar_noise', True):                0.1162,
            ('logistic_diag_noise_id', False): [0.0865, 0.0358, 0.0564, 0.2049],
            ('logistic_diag_noise_id', True):  [0.0865, 0.0359, 0.0564, 0.2050],
            ('logistic_diag_noise', False):    [0.0824, 0.0089, 0.0551, 0.1819],
            ('logistic_diag_noise', True):     [0.0824, 0.0089, 0.0552, 0.1819],
            ('logistic_parallel_scalar_noise', False):      0.1525,
            ('logistic_parallel_scalar_noise', True):       0.1872,
            ('linear_scalar_noise', False):                 0.1699,
            ('linear_scalar_noise', True):                  0.1699,

            # univariate would be quite useless
            # TODO: binary (crossentropy) models here

        }.items():

            subtest = dict(model_name=model_name, use_jacobian=use_jacobian)
            with self.subTest(**subtest):

                common_params = dict(algo_name='scipy_minimize', seed=0, use_jacobian=use_jacobian)

                ips_sparse, noise_sparse, _ = self.generic_personalization(model_name, **common_params,
                                                                           data_path='missing_data/sparse_data.csv',
                                                                           data_kws={'drop_full_nan': False})

                ips_merged, noise_merged, _ = self.generic_personalization(model_name, **common_params,
                                                                           data_path='missing_data/merged_data.csv')

                indices_sparse, ips_sparse_torch = ips_sparse.to_pytorch()
                indices_merged, ips_merged_torch = ips_merged.to_pytorch()

                # same individuals
                self.assertEqual(indices_sparse, indices_merged, msg=subtest)
                # same individual parameters (up to rounding errors)
                self.assertDictAlmostEqual(ips_sparse_torch, ips_merged_torch, rtol=rtol, atol=atol, msg=subtest)
                # same noise between them and as expected
                self.assertTrue(torch.allclose(noise_sparse, noise_merged, atol=atol), msg=subtest)
                self.assertTrue(torch.allclose(noise_merged, torch.tensor(expected_noise_std), atol=atol), msg=subtest)

    # TODO : problem with nans
    """
    def test_personalize_gradientdescent(self):
        # Inputs
        data = Data.from_csv_file(self.example_data_path)

        # Initialize
        leaspy = Leaspy.load(...)

        # Launch algorithm
        algo_personalize_settings = AlgorithmSettings('gradient_descent_personalize', seed=2)
        result = leaspy.personalize(data, settings=algo_personalize_settings)

        self.assertAlmostEqual(result.noise_std,  0.17925, delta=0.01)"""
