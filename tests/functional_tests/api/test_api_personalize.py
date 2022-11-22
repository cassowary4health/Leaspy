import math

import torch
from numpy import nan
import pandas as pd

from leaspy import Data, Dataset, IndividualParameters

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

    def test_personalize_comprehensive(self, tol_noise=5e-4):
        """Tests different personalization algos for many hardcoded models."""

        mode_real_kws = dict()
        mean_real_kws = dict()

        for model_name, perso_name, perso_kws, expected_noise_std in [

            # multivariate logistic models
            ('logistic_scalar_noise', 'scipy_minimize', dict(use_jacobian=False),          0.1189),
            ('logistic_scalar_noise', 'scipy_minimize', dict(use_jacobian=True),           0.1188),
            ('logistic_scalar_noise', 'mode_real', mode_real_kws,                          0.1191),
            ('logistic_scalar_noise', 'mean_real', mean_real_kws,                          0.1200),

            ('logistic_diag_noise_id', 'scipy_minimize', dict(use_jacobian=False),        [0.1414, 0.0806, 0.0812, 0.1531]),
            ('logistic_diag_noise_id', 'scipy_minimize', dict(use_jacobian=True),         [0.1414, 0.0804, 0.0811, 0.1529]),
            ('logistic_diag_noise_id', 'mode_real', mode_real_kws,                        [0.1415, 0.0814, 0.0810, 0.1532]),
            ('logistic_diag_noise_id', 'mean_real', mean_real_kws,                        [0.1430, 0.0789, 0.0775, 0.1578]),

            ('logistic_diag_noise', 'scipy_minimize', dict(use_jacobian=False),           [0.1543, 0.0597, 0.0827, 0.1509]),
            ('logistic_diag_noise', 'scipy_minimize', dict(use_jacobian=True),            [0.1543, 0.0597, 0.0827, 0.1509]),
            ('logistic_diag_noise', 'mode_real', mode_real_kws,                           [0.1596, 0.0598, 0.0824, 0.1507]),
            ('logistic_diag_noise', 'mean_real', mean_real_kws,                           [0.1565, 0.0587, 0.0833, 0.1511]),

            # without source
            ('logistic_diag_noise_no_source', 'scipy_minimize', dict(use_jacobian=False), [0.1053, 0.0404, 0.0699, 0.1992]),
            ('logistic_diag_noise_no_source', 'scipy_minimize', dict(use_jacobian=True),  [0.1053, 0.0404, 0.0699, 0.1991]),
            ('logistic_diag_noise_no_source', 'mode_real', mode_real_kws,                 [0.1053, 0.0404, 0.0700, 0.1990]),
            ('logistic_diag_noise_no_source', 'mean_real', mean_real_kws,                 [0.1067, 0.0406, 0.0691, 0.1987]),

            # multivariate logistic parallel models
            ('logistic_parallel_scalar_noise', 'scipy_minimize', dict(use_jacobian=False), 0.0960),
            ('logistic_parallel_scalar_noise', 'scipy_minimize', dict(use_jacobian=True),  0.0956),
            ('logistic_parallel_scalar_noise', 'mode_real', mode_real_kws,                 0.0959),
            ('logistic_parallel_scalar_noise', 'mean_real', mean_real_kws,                 0.0964),

            ('logistic_parallel_diag_noise', 'scipy_minimize', dict(use_jacobian=False),  [0.0670, 0.0538, 0.1043, 0.1494]),
            ('logistic_parallel_diag_noise', 'scipy_minimize', dict(use_jacobian=True),   [0.0669, 0.0538, 0.1043, 0.1494]),
            ('logistic_parallel_diag_noise', 'mode_real', mode_real_kws,                  [0.0675, 0.0531, 0.1046, 0.1505]),
            ('logistic_parallel_diag_noise', 'mean_real', mean_real_kws,                  [0.0671, 0.0553, 0.1040, 0.1509]),

            # univariate models
            ('univariate_logistic', 'scipy_minimize', dict(use_jacobian=False),            0.1341),
            ('univariate_logistic', 'scipy_minimize', dict(use_jacobian=True),             0.1341),
            ('univariate_logistic', 'mode_real', mode_real_kws,                            0.1346),
            ('univariate_logistic', 'mean_real', mean_real_kws,                            0.1351),

            ('univariate_linear', 'scipy_minimize', dict(use_jacobian=False),              0.0812),
            ('univariate_linear', 'scipy_minimize', dict(use_jacobian=True),               0.0812),
            ('univariate_linear', 'mode_real', mode_real_kws,                              0.0817),
            ('univariate_linear', 'mean_real', mean_real_kws,                              0.0898),

            # multivariate linear models
            ('linear_scalar_noise', 'scipy_minimize', dict(use_jacobian=False),            0.1241),
            ('linear_scalar_noise', 'scipy_minimize', dict(use_jacobian=True),             0.1241),
            ('linear_scalar_noise', 'mode_real', mode_real_kws,                            0.1241),
            ('linear_scalar_noise', 'mean_real', mean_real_kws,                            0.1237),

            ('linear_diag_noise', 'scipy_minimize', dict(use_jacobian=False),             [0.1003, 0.1274, 0.1249, 0.1486]),
            ('linear_diag_noise', 'scipy_minimize', dict(use_jacobian=True),              [0.1002, 0.1276, 0.1249, 0.1486]),
            ('linear_diag_noise', 'mode_real', mode_real_kws,                             [0.1007, 0.1292, 0.1250, 0.1489]),
            ('linear_diag_noise', 'mean_real', mean_real_kws,                             [0.1000, 0.1265, 0.1242, 0.1485]),

            # multivariate binary model
            ('logistic_binary', 'scipy_minimize', dict(use_jacobian=False),               [103.7]),
            ('logistic_binary', 'scipy_minimize', dict(use_jacobian=True),                [103.67]),
            ('logistic_binary', 'mode_real', mode_real_kws,                               [103.96]),
            ('logistic_binary', 'mean_real', mean_real_kws,                               [101.95]),

            # multivariate parallel binary model
            ('logistic_parallel_binary', 'scipy_minimize', dict(use_jacobian=False),      [112.66]),
            ('logistic_parallel_binary', 'scipy_minimize', dict(use_jacobian=True),       [112.63]),
            ('logistic_parallel_binary', 'mode_real', mode_real_kws,                      [111.96]),
            ('logistic_parallel_binary', 'mean_real', mean_real_kws,                      [120.06]),

            # multivariate ordinal model
            ('logistic_ordinal', 'scipy_minimize', dict(use_jacobian=False), [700.55]),
            ('logistic_ordinal', 'scipy_minimize', dict(use_jacobian=True), [638.66]),
            ('logistic_ordinal', 'mode_real', mode_real_kws, [619.64]),
            ('logistic_ordinal', 'mean_real', mean_real_kws, [616.94]),

            # multivariate ordinal ranking model
            ('logistic_ordinal_ranking', 'scipy_minimize', dict(use_jacobian=False), [1014.2]),
            ('logistic_ordinal_ranking', 'scipy_minimize', dict(use_jacobian=True), [1014.1]),
            ('logistic_ordinal_ranking', 'mode_real', mode_real_kws, [1014.9]),
            ('logistic_ordinal_ranking', 'mean_real', mean_real_kws, [1015.0]),

        ]:

            subtest = dict(model_name=model_name, perso_name=perso_name, perso_kws=perso_kws)
            with self.subTest(**subtest):

                # only look at residual MSE to detect any regression in personalization
                ips, noise_std, _ = self.generic_personalization(model_name, algo_name=perso_name, seed=0, **perso_kws)

                tol_noise_sub = tol_noise
                # not noise but NLL (less precise...); some minor exact reproducibility issues MacOS vs. Linux
                if 'binary' in model_name:
                    tol_noise_sub = 0.1
                elif 'ordinal_ranking' in model_name:
                    tol_noise_sub = 0.5
                elif 'ordinal' in model_name:
                    tol_noise_sub = 3.0  # highest reprod. issues

                self.check_consistency_of_personalization_outputs(ips, noise_std, expected_noise_std=expected_noise_std,
                                                                  tol_noise=tol_noise_sub,
                                                                  msg=subtest)

    def test_robustness_to_data_sparsity(self, rtol=2e-2, atol=5e-3):
        """
        In this test, we check that estimated individual parameters are almost same no matter if data is sparse
        (i.e. multiple really close visits with many missing values) or data is 'merged' in a rounded visit.

        TODO? we could build a mock dataset to also check same property for calibration :)
        """

        mode_real_kws = dict()
        mean_real_kws = dict()

        for model_name, perso_name, perso_kws, expected_noise_std in [

            # multivariate logistic models
            ('logistic_scalar_noise', 'scipy_minimize', dict(use_jacobian=False),          0.1161),
            ('logistic_scalar_noise', 'scipy_minimize', dict(use_jacobian=True),           0.1162),

            ('logistic_diag_noise_id', 'scipy_minimize', dict(use_jacobian=False),        [0.0865, 0.0358, 0.0564, 0.2049]),
            ('logistic_diag_noise_id', 'scipy_minimize', dict(use_jacobian=True),         [0.0865, 0.0359, 0.0564, 0.2050]),

            ('logistic_diag_noise', 'scipy_minimize', dict(use_jacobian=False),           [0.0824, 0.0089, 0.0551, 0.1819]),
            ('logistic_diag_noise', 'scipy_minimize', dict(use_jacobian=True),            [0.0824, 0.0089, 0.0552, 0.1819]),
            ('logistic_diag_noise', 'mode_real', mode_real_kws,                           [0.0937, 0.0126, 0.0587, 0.1831]),
            ('logistic_diag_noise', 'mean_real', mean_real_kws,                           [0.0908, 0.0072, 0.0595, 0.1817]),

            # without source
            ('logistic_diag_noise_no_source', 'scipy_minimize', dict(use_jacobian=False), [0.1349, 0.0336, 0.0760, 0.1777]),
            ('logistic_diag_noise_no_source', 'scipy_minimize', dict(use_jacobian=True),  [0.1349, 0.0336, 0.0761, 0.1777]),
            ('logistic_diag_noise_no_source', 'mode_real', mode_real_kws,                 [0.1339, 0.0356, 0.0754, 0.1761]),
            ('logistic_diag_noise_no_source', 'mean_real', mean_real_kws,                 [0.1387, 0.0277, 0.0708, 0.1807]),

            # multivariate logistic parallel models
            ('logistic_parallel_scalar_noise', 'scipy_minimize', dict(use_jacobian=False), 0.1525),
            ('logistic_parallel_scalar_noise', 'scipy_minimize', dict(use_jacobian=True),  0.1872),
            ('logistic_parallel_scalar_noise', 'mode_real', mode_real_kws,                 0.1517),
            ('logistic_parallel_scalar_noise', 'mean_real', mean_real_kws,                 0.2079),

            ('logistic_parallel_diag_noise', 'scipy_minimize', dict(use_jacobian=False),  [0.0178, 0.0120, 0.0509, 0.0939]),
            ('logistic_parallel_diag_noise', 'scipy_minimize', dict(use_jacobian=True),   [0.0178, 0.0120, 0.0508, 0.0940]),
            ('logistic_parallel_diag_noise', 'mode_real', mode_real_kws,                  [0.0193, 0.0179, 0.0443, 0.0971]),
            ('logistic_parallel_diag_noise', 'mean_real', mean_real_kws,                  [0.0385, 0.0153, 0.0433, 0.3016]),

            ('linear_scalar_noise', 'scipy_minimize', dict(use_jacobian=False),            0.1699),
            ('linear_scalar_noise', 'scipy_minimize', dict(use_jacobian=True),             0.1699),

            ('linear_diag_noise', 'scipy_minimize', dict(use_jacobian=False),             [0.1021, 0.1650, 0.2083, 0.1481]),
            ('linear_diag_noise', 'scipy_minimize', dict(use_jacobian=True),              [0.1023, 0.1630, 0.2081, 0.1480]),

            # binary models (noise_std is reported as diagonal)
            ('logistic_binary', 'scipy_minimize', dict(use_jacobian=False),               [8.4722]),
            ('logistic_binary', 'scipy_minimize', dict(use_jacobian=True),                [8.4718]),

            ('logistic_parallel_binary', 'scipy_minimize', dict(use_jacobian=False),      [8.8422]),
            ('logistic_parallel_binary', 'scipy_minimize', dict(use_jacobian=True),       [8.8408]),
        ]:

            subtest = dict(model_name=model_name, perso_name=perso_name, perso_kws=perso_kws)
            with self.subTest(**subtest):

                common_params = dict(algo_name=perso_name, seed=0, **perso_kws)

                ips_sparse, noise_sparse, _ = self.generic_personalization(model_name, **common_params,
                                                                           data_path='missing_data/sparse_data.csv',
                                                                           data_kws={'drop_full_nan': False})

                ips_merged, noise_merged, _ = self.generic_personalization(model_name, **common_params,
                                                                           data_path='missing_data/merged_data.csv')

                indices_sparse, ips_sparse_torch = ips_sparse.to_pytorch()
                indices_merged, ips_merged_torch = ips_merged.to_pytorch()

                # same individuals
                self.assertEqual(indices_sparse, indices_merged, msg=subtest)

                # same noise as expected
                self.assertAllClose(noise_merged, expected_noise_std, atol=atol, what='noise', msg=subtest)
                # same noise between both cases
                self.assertAllClose(noise_sparse, noise_merged, left_desc='sparse', right_desc='merged', what='noise',
                                    atol=atol, msg=subtest)
                # same individual parameters (up to rounding errors)
                self.assertDictAlmostEqual(ips_sparse_torch, ips_merged_torch, left_desc='sparse', right_desc='merged',
                                           rtol=rtol, atol=atol, msg=subtest)

    def test_personalize_full_nan(self, *, general_tol=1e-3):
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

        for perso_algo, perso_kws, coeff_tol_per_param_std in [

            ('scipy_minimize', dict(use_jacobian=False), general_tol),
            ('scipy_minimize', dict(use_jacobian=True), general_tol),

            # the LL landscape is quite flat so tolerance is high here...
            # we may deviate from tau_mean / xi_mean / sources_mean when no data at all
            # (intrinsically represent the incertitude on those individual parameters)
            ('mode_real', {}, .4),
            ('mean_real', {}, .4),
        ]:

            subtest = dict(perso_algo=perso_algo, perso_kws=perso_kws)
            with self.subTest(**subtest):
                algo = self.get_algo_settings(name=perso_algo, seed=0, progress_bar=False, **perso_kws)

                with self.assertRaisesRegex(ValueError, 'Dataframe should have at least '):
                    # drop rows full of nans, nothing is left...
                    Data.from_dataframe(df)

                with self.assertWarnsRegex(UserWarning, r"These columns only contain nans: \['Y0', 'Y1', 'Y2', 'Y3'\]"):
                    data_1 = Data.from_dataframe(df.head(1), drop_full_nan=False)
                    data_2 = Data.from_dataframe(df, drop_full_nan=False)

                dataset_1 = Dataset(data_1)
                dataset_2 = Dataset(data_2)

                self.assertEqual(data_1.n_individuals, 1)
                self.assertEqual(data_1.n_visits, 1)
                self.assertEqual(dataset_1.n_observations_per_ft.tolist(), [0, 0, 0, 0])
                self.assertEqual(dataset_1.n_observations, 0)

                self.assertEqual(data_2.n_individuals, 1)
                self.assertEqual(data_2.n_visits, 2)
                self.assertEqual(dataset_2.n_observations_per_ft.tolist(), [0, 0, 0, 0])
                self.assertEqual(dataset_2.n_observations, 0)

                ips_1 = lsp.personalize(data_1, algo)
                ips_2 = lsp.personalize(data_2, algo)

                indices_1, dict_1 = ips_1.to_pytorch()
                indices_2, dict_2 = ips_2.to_pytorch()

                self.assertEqual(indices_1, ['SUBJ1'])
                self.assertEqual(indices_1, indices_2)

                # replication is OK
                self.assertDictAlmostEqual(dict_1, dict_2, atol=general_tol, msg=subtest)

                # we have no information so high incertitude when stochastic perso algo
                allclose_custom = {
                    p: dict(atol=math.ceil(coeff_tol_per_param_std * lsp.model.parameters[f'{p}_std'].item() / general_tol) * general_tol)
                    for p, pi in lsp.model.random_variable_informations().items()
                    if pi['type'] == 'individual'
                }
                self.assertDictAlmostEqual(dict_1, {
                    'tau': [lsp.model.parameters['tau_mean']],
                    'xi': [0.],
                    'sources': lsp.model.source_dimension*[0.]
                }, allclose_custom=allclose_custom, msg=subtest)

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

        for perso_algo, perso_kws, tol in [

            ('scipy_minimize', dict(use_jacobian=False), 1e-3),
            ('scipy_minimize', dict(use_jacobian=True), 1e-3),
            ('mode_real', {}, 1e-3),
            ('mean_real', {}, 1e-3),
        ]:

            subtest = dict(perso_algo=perso_algo, perso_kws=perso_kws)
            with self.subTest(**subtest):
                algo = self.get_algo_settings(name=perso_algo, seed=0, progress_bar=False, **perso_kws)

                data_without_empty_visits = Data.from_dataframe(df)
                data_with_empty_visits = Data.from_dataframe(df, drop_full_nan=False)

                dataset_without_empty_visits = Dataset(data_without_empty_visits)
                dataset_with_empty_visits = Dataset(data_with_empty_visits)

                self.assertEqual(data_without_empty_visits.n_individuals, 1)
                self.assertEqual(data_without_empty_visits.n_visits, 2)
                self.assertEqual(dataset_without_empty_visits.n_observations_per_ft.tolist(), [2, 1, 2, 2])
                self.assertEqual(dataset_without_empty_visits.n_observations, 7)

                self.assertEqual(data_with_empty_visits.n_individuals, 1)
                self.assertEqual(data_with_empty_visits.n_visits, 4)
                self.assertEqual(dataset_with_empty_visits.n_observations_per_ft.tolist(), [2, 1, 2, 2])
                self.assertEqual(dataset_with_empty_visits.n_observations, 7)

                ips_without_empty_visits = lsp.personalize(data_without_empty_visits, algo)
                ips_with_empty_visits = lsp.personalize(data_with_empty_visits, algo)

                indices_1, dict_1 = ips_without_empty_visits.to_pytorch()
                indices_2, dict_2 = ips_with_empty_visits.to_pytorch()

                self.assertEqual(indices_1, ['SUBJ1'])
                self.assertEqual(indices_1, indices_2)

                self.assertDictAlmostEqual(dict_1, dict_2, atol=tol, msg=subtest)

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
