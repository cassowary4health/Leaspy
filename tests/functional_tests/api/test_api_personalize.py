import torch
from numpy import nan
import pandas as pd

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
            data_full_path = cls.get_test_data_path('data_mock', data_path)
            data = Data.from_csv_file(data_full_path, **data_kws)

        # create the personalize algo settings (from path or name + params)
        algo_settings = cls.get_algo_settings(path=algo_path, name=algo_name, **algo_params)

        # return results of personalization
        ips, noise = leaspy.personalize(data, settings=algo_settings, return_noise=True)

        return ips, noise, leaspy # data?

    def check_consistency_of_personalization_outputs(self, ips, noise_std, expected_noise_std, *,
                                                     tol_noise = 5e-3, msg = None):

        self.assertIsInstance(ips, IndividualParameters)
        self.assertIsInstance(noise_std, torch.Tensor)

        if isinstance(expected_noise_std, float):
            self.assertEqual(noise_std.numel(), 1, msg=msg) # scalar noise
            self.assertAlmostEqual(noise_std.item(), expected_noise_std, delta=tol_noise, msg=msg)
        else:
            # vector of noises (for diag_noise)
            self.assertEqual(noise_std.numel(), len(expected_noise_std), msg=msg) # diagonal noise
            self.assertAllClose(noise_std, expected_noise_std, atol=tol_noise, what='noise', msg=msg)

class LeaspyPersonalizeTest(LeaspyPersonalizeTest_Mixin):

    def test_personalize_mean_real_logistic_old(self, tol_noise=1e-3):
        """
        Load logistic model from file, and personalize it to data from ...
        """
        # There was a bug previously in mode & mean real: initial temperature = 10 was used even if
        # no real annealing is implemented for those perso algos. As a consequence regularity term
        # was not equally weighted during all the sampling of individual variables.
        # We test this old "buggy" behavior to check past consistency (but we raise a warning now)
        path_settings = self.get_test_data_path('settings', 'algo', 'settings_mean_real_old_with_annealing.json')
        with self.assertWarnsRegex(UserWarning, r'[Aa]nnealing'):
            ips, noise_std, _ = self.generic_personalization('logistic_scalar_noise', algo_path=path_settings)

        self.check_consistency_of_personalization_outputs(ips, noise_std, expected_noise_std=0.102, tol_noise=tol_noise)

    def test_personalize_mode_real_logistic_old(self, tol_noise=1e-3):
        """
        Load logistic model from file, and personalize it to data from ...
        """
        # cf. mean_real notice
        path_settings = self.get_test_data_path('settings', 'algo', 'settings_mode_real_old_with_annealing.json')
        with self.assertWarnsRegex(UserWarning, r'[Aa]nnealing'):
            ips, noise_std, _ = self.generic_personalization('logistic_scalar_noise', algo_path=path_settings)

        self.check_consistency_of_personalization_outputs(ips, noise_std, expected_noise_std=0.117, tol_noise=tol_noise)

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

    def test_personalize_full_nan(self):
        # test result of personalization with no data at all
        df = pd.DataFrame({
            'ID': ['SUBJ1', 'SUBJ1'],
            'TIME': [75.12, 78.9],
            'Y0': [nan]*2,
            'Y1': [nan]*2,
            'Y2': [nan]*2,
            'Y3': [nan]*2,
        }).set_index(['ID', 'TIME'])

        lsp = self.get_hardcoded_model('logistic_diag_noise')
        algo = self.get_algo_settings(name='scipy_minimize', seed=0, progress_bar=False)

        with self.assertRaisesRegex(ValueError, 'Dataframe should have at least '):
            # drop rows full of nans, nothing is left...
            data_0 = Data.from_dataframe(df)

        with self.assertWarnsRegex(UserWarning, r"These columns only contain nans: \['Y0', 'Y1', 'Y2', 'Y3'\]"):
            data_1 = Data.from_dataframe(df.head(1), drop_full_nan=False)
            data_2 = Data.from_dataframe(df, drop_full_nan=False)

        self.assertEqual(data_1.n_individuals, 1)
        self.assertEqual(data_1.n_visits, 1)

        self.assertEqual(data_2.n_individuals, 1)
        self.assertEqual(data_2.n_visits, 2)

        ips_1 = lsp.personalize(data_1, algo)
        ips_2 = lsp.personalize(data_2, algo)

        indices_1, dict_1 = ips_1.to_pytorch()
        indices_2, dict_2 = ips_2.to_pytorch()

        self.assertEqual(indices_1, ['SUBJ1'])
        self.assertEqual(indices_1, indices_2)
        self.assertDictAlmostEqual(dict_1, dict_2)
        self.assertDictAlmostEqual(dict_1, {
            'tau': [lsp.model.parameters['tau_mean']],
            'xi': [0],
            'sources': lsp.model.source_dimension*[0]
        })

    def test_personalize_same_if_extra_totally_nan_visits(self):

        df = pd.DataFrame({
            'ID': ['SUBJ1']*4,
            'TIME': [75.12, 78.9, 67.1, 76.1],
            'Y0': [nan, .6, nan, .2],
            'Y1': [nan, .4, nan, nan],
            'Y2': [nan, .5, nan, .2],
            'Y3': [nan, .3, nan, .2],
        }).set_index(['ID', 'TIME'])

        lsp = self.get_hardcoded_model('logistic_diag_noise')
        algo = self.get_algo_settings(name='scipy_minimize', seed=0, progress_bar=False)

        data_without_empty_visits = Data.from_dataframe(df)
        data_with_empty_visits = Data.from_dataframe(df, drop_full_nan=False)

        self.assertEqual(data_without_empty_visits.n_individuals, 1)
        self.assertEqual(data_without_empty_visits.n_visits, 2)

        self.assertEqual(data_with_empty_visits.n_individuals, 1)
        self.assertEqual(data_with_empty_visits.n_visits, 4)

        ips_without_empty_visits = lsp.personalize(data_without_empty_visits, algo)
        ips_with_empty_visits = lsp.personalize(data_with_empty_visits, algo)

        indices_1, dict_1 = ips_without_empty_visits.to_pytorch()
        indices_2, dict_2 = ips_with_empty_visits.to_pytorch()

        self.assertEqual(indices_1, ['SUBJ1'])
        self.assertEqual(indices_1, indices_2)
        self.assertDictAlmostEqual(dict_1, dict_2)

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
