import math
import warnings

import torch
from numpy import nan
import pandas as pd
from typing import Optional, Union, List
from unittest import skipIf

from leaspy import Data, Dataset, IndividualParameters

from tests import LeaspyTestCase

# Logistic parallel models are currently broken in Leaspy v2.
# Flip to True to test with them.
TEST_LOGISTIC_PARALLEL_MODELS = False
SKIP_LOGISTIC_PARALLEL_MODELS = "Logistic parallel models are currently broken."

TEST_LOGISTIC_MODELS_WITH_JACOBIAN = False
SKIP_LOGISTIC_MODELS_WITH_JACOBIAN = "Jacobian not implemented for logistic model."

TEST_LOGISTIC_BINARY_MODELS = False
SKIP_LOGISTIC_BINARY_MODELS = "Logistic binary models are currently broken."

# Linear models are currently broken in Leaspy v2.
# Flip to True to test with them.
TEST_LINEAR_MODELS = True
SKIP_LINEAR_MODELS = "Linear models are currently broken."

# Ordinal observation models are not yet implemented in Leaspy v2.
# Flip to True to test with them.
TEST_ORDINAL_MODELS = False
SKIP_ORDINAL_MODELS = "Ordinal observation models are not implemented yet."


class LeaspyPersonalizeTestMixin(LeaspyTestCase):
    """
    Mixin holding generic personalization methods that may be safely
    reused in other tests (no actual test here).
    """

    @classmethod
    def generic_personalization(
        cls,
        hardcoded_model_name: str,
        *,
        data_path: Optional[str] = None,
        data_kws: Optional[dict] = None,
        algo_path: Optional[str] = None,
        algo_name: Optional[str] = None,
        **algo_params,
    ):
        """Helper for a generic personalization in following tests."""
        data_kws = data_kws or {}
        leaspy = cls.get_hardcoded_model(hardcoded_model_name)

        if data_path is None:
            # automatic (main test data)
            data = cls.get_suited_test_data_for_model(hardcoded_model_name)
        else:
            # relative path to data (csv expected)
            data_full_path = cls.get_test_data_path('data_mock', data_path)
            data = Data.from_csv_file(data_full_path, **data_kws)

        algo_settings = cls.get_algo_settings(path=algo_path, name=algo_name, **algo_params)
        ips, loss = leaspy.personalize(data, settings=algo_settings, return_loss=True)

        return ips, loss, leaspy  # data?

    def check_consistency_of_personalization_outputs(
        self,
        ips: IndividualParameters,
        loss: torch.Tensor,
        expected_loss: Union[float, list[float]],
        *,
        tol_loss: Optional[float] = 5e-3,
        msg=None,
    ):
        self.assertIsInstance(ips, IndividualParameters)
        self.assertIsInstance(loss, torch.Tensor)

        if isinstance(expected_loss, float):
            self.assertEqual(loss.numel(), 1, msg=msg)  # scalar noise or neg-ll
            self.assertAlmostEqual(loss.item(), expected_loss, delta=tol_loss, msg=msg)
        else:
            # vector of noises (for Gaussian diagonal noise)
            self.assertEqual(loss.numel(), len(expected_loss), msg=msg)  # diagonal noise
            self.assertAllClose(loss, expected_loss, atol=tol_loss, what='noise', msg=msg)


class LeaspyPersonalizeTest(LeaspyPersonalizeTestMixin):

    def test_personalize_mean_real_logistic_old(self, tol_noise=1e-3):
        """
        Load logistic model from file, and personalize it to data from ...
        """
        # There was a bug previously in mode & mean real: initial temperature = 10 was used even if
        # no real annealing is implemented for those perso algos. As a consequence regularity term
        # was not equally weighted during all the sampling of individual variables.
        # We test this old "buggy" behavior to check past consistency (but we raise a warning now)
        path_settings = self.get_test_data_path(
            "settings", "algo", "settings_mean_real_old_with_annealing.json"
        )
        with self.assertWarnsRegex(UserWarning, r'[Aa]nnealing'):
            ips, noise_std, _ = self.generic_personalization(
                "logistic_scalar_noise", algo_path=path_settings
            )
        self.check_consistency_of_personalization_outputs(
            ips,
            noise_std,
            expected_loss=0.102,
            tol_loss=tol_noise,
        )

    def test_personalize_mode_real_logistic_old(self, tol_noise=1e-3):
        """
        Load logistic model from file, and personalize it to data from ...
        """
        # cf. mean_real notice
        path_settings = self.get_test_data_path(
            "settings", "algo", "settings_mode_real_old_with_annealing.json"
        )
        with self.assertWarnsRegex(UserWarning, r'[Aa]nnealing'):
            ips, noise_std, _ = self.generic_personalization(
                "logistic_scalar_noise", algo_path=path_settings
            )
        self.check_consistency_of_personalization_outputs(
            ips,
            noise_std,
            expected_loss=0.117,
            tol_loss=tol_noise,
        )

    def _personalize_generic(
        self,
        model_name: str,
        algo_name: str,
        expected_loss: Union[float, List[float]],
        algo_kws: Optional[dict] = None,
        tol_noise: Optional[float] = 5e-4,
    ):
        algo_kws = algo_kws or {}
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter('always')
            # only look at loss to detect any regression in personalization
            ips, loss, _ = self.generic_personalization(model_name, algo_name=algo_name, seed=0, **algo_kws)

        ws = [str(w.message) for w in ws]
        if 'ordinal' in model_name:
            self.assertEqual(len(ws), 1, msg=ws)
            self.assertIn("Some features have missing codes", ws[0])
        else:
            self.assertEqual(ws, [])

        tol_loss = tol_noise
        # not noise but NLL (less precise...); some minor exact reproducibility issues MacOS vs. Linux
        if 'binary' in model_name:
            tol_loss = 0.1
        elif 'ordinal_ranking' in model_name:
            tol_loss = 0.5
        elif 'ordinal' in model_name:
            tol_loss = 3.0  # highest reprod. issues

        self.check_consistency_of_personalization_outputs(
            ips,
            loss,
            expected_loss=expected_loss,
            tol_loss=tol_loss,
            msg={
                "model_name": model_name,
                "perso_name": algo_name,
                "perso_kws": algo_kws,
            },
        )

    def test_multivariate_logistic_scipy_minimize(self):
        self._personalize_generic(
            "logistic_scalar_noise",
            "scipy_minimize",
            0.1189,
            {"use_jacobian": False},
        )

    @skipIf(not TEST_LOGISTIC_MODELS_WITH_JACOBIAN, SKIP_LOGISTIC_MODELS_WITH_JACOBIAN)
    def test_multivariate_logistic_scipy_minimize_with_jacobian(self):
        self._personalize_generic(
            "logistic_scalar_noise",
            "scipy_minimize",
            0.1188,
            {"use_jacobian": True},
        )

    def test_multivariate_logistic_mode_real(self):
        self._personalize_generic("logistic_scalar_noise", "mode_real", 0.1191)

    def test_multivariate_logistic_mean_real(self):
        self._personalize_generic("logistic_scalar_noise", "mean_real", 0.1200)

    def test_multivariate_logistic_diagonal_id_scipy_minimize(self):
        self._personalize_generic(
            "logistic_diag_noise_id",
            "scipy_minimize",
            [0.1414, 0.0806, 0.0812, 0.1531],
            {"use_jacobian": False},
        )

    @skipIf(not TEST_LOGISTIC_MODELS_WITH_JACOBIAN, SKIP_LOGISTIC_MODELS_WITH_JACOBIAN)
    def test_multivariate_logistic_diagonal_id_scipy_minimize_with_jacobian(self):
        self._personalize_generic(
            "logistic_diag_noise_id",
            "scipy_minimize",
            [0.1414, 0.0804, 0.0811, 0.1529],
            {"use_jacobian": True},
        )

    def test_multivariate_logistic_diagonal_id_mode_real(self):
        self._personalize_generic(
            "logistic_diag_noise_id",
            "mode_real",
            [0.1415, 0.0814, 0.0810, 0.1532],
        )

    def test_multivariate_logistic_diagonal_id_mean_real(self):
        self._personalize_generic(
            "logistic_diag_noise_id",
            "mean_real",
            [0.1430, 0.0789, 0.0775, 0.1578],
        )

    def test_multivariate_logistic_diagonal_scipy_minimize(self):
        self._personalize_generic(
            "logistic_diag_noise",
            "scipy_minimize",
            [0.1537, 0.0597, 0.0827, 0.1513],
            {"use_jacobian": False},
            tol_noise = 0.003
        )

    @skipIf(not TEST_LOGISTIC_MODELS_WITH_JACOBIAN, SKIP_LOGISTIC_MODELS_WITH_JACOBIAN)
    def test_multivariate_logistic_diagonal_scipy_minimize_with_jacobian(self):
        self._personalize_generic(
            "logistic_diag_noise",
            "scipy_minimize",
            [0.1543, 0.0597, 0.0827, 0.1509],
            {"use_jacobian": True},
        )

    def test_multivariate_logistic_diagonal_mode_real(self):
        self._personalize_generic(
            "logistic_diag_noise",
            "mode_real",
            [0.1596, 0.0598, 0.0824, 0.1507],
        )

    def test_multivariate_logistic_diagonal_mean_real(self):
        self._personalize_generic(
            "logistic_diag_noise",
            "mean_real",
            [0.1565, 0.0587, 0.0833, 0.1511],
        )

    @skipIf(not TEST_LOGISTIC_MODELS_WITH_JACOBIAN, SKIP_LOGISTIC_MODELS_WITH_JACOBIAN)
    def test_multivariate_logistic_diagonal_no_source_scipy_minimize_with_jacobian(self):
        self._personalize_generic(
            "logistic_diag_noise_no_source",
            "scipy_minimize",
            [0.1053, 0.0404, 0.0699, 0.1991],
            {"use_jacobian": True},
        )

    def test_multivariate_logistic_diagonal_no_source_mode_real(self):
        self._personalize_generic(
            "logistic_diag_noise_no_source",
            "mode_real",
            [0.1053, 0.0404, 0.0700, 0.1990],
        )

    def test_multivariate_logistic_diagonal_no_source_mean_real(self):
        self._personalize_generic(
            "logistic_diag_noise_no_source",
            "mean_real",
            [0.1067, 0.0406, 0.0691, 0.1987],
        )

    @skipIf(not TEST_LOGISTIC_PARALLEL_MODELS, SKIP_LOGISTIC_PARALLEL_MODELS)
    def test_multivariate_logistic_parallel_scipy_minimize(self):
        self._personalize_generic(
            "logistic_parallel_scalar_noise",
            "scipy_minimize",
            0.0960,
            {"use_jacobian": False},
        )

    @skipIf(
        not TEST_LOGISTIC_PARALLEL_MODELS or not TEST_LOGISTIC_MODELS_WITH_JACOBIAN,
        SKIP_LOGISTIC_PARALLEL_MODELS if TEST_LOGISTIC_MODELS_WITH_JACOBIAN else SKIP_LOGISTIC_MODELS_WITH_JACOBIAN
    )
    def test_multivariate_logistic_parallel_scipy_minimize_with_jacobian(self):
        self._personalize_generic(
            "logistic_parallel_scalar_noise",
            "scipy_minimize",
            0.0956,
            {"use_jacobian": True},
        )

    @skipIf(not TEST_LOGISTIC_PARALLEL_MODELS, SKIP_LOGISTIC_PARALLEL_MODELS)
    def test_multivariate_logistic_parallel_mode_real(self):
        self._personalize_generic(
            "logistic_parallel_scalar_noise",
            "mode_real",
            0.0959,
        )

    @skipIf(not TEST_LOGISTIC_PARALLEL_MODELS, SKIP_LOGISTIC_PARALLEL_MODELS)
    def test_multivariate_logistic_parallel_mean_real(self):
        self._personalize_generic(
            "logistic_parallel_scalar_noise",
            "mean_real",
            0.0964,
        )

    @skipIf(not TEST_LOGISTIC_PARALLEL_MODELS, SKIP_LOGISTIC_PARALLEL_MODELS)
    def test_multivariate_logistic_parallel_diagonal_scipy_minimize(self):
        self._personalize_generic(
            "logistic_parallel_diag_noise",
            "scipy_minimize",
            [0.0670, 0.0538, 0.1043, 0.1494],
            {"use_jacobian": False},
        )

    @skipIf(
        not TEST_LOGISTIC_PARALLEL_MODELS or not TEST_LOGISTIC_MODELS_WITH_JACOBIAN,
        SKIP_LOGISTIC_PARALLEL_MODELS if TEST_LOGISTIC_MODELS_WITH_JACOBIAN else SKIP_LOGISTIC_MODELS_WITH_JACOBIAN
    )
    def test_multivariate_logistic_parallel_diagonal_scipy_minimize_with_jacobian(self):
        self._personalize_generic(
            "logistic_parallel_diag_noise",
            "scipy_minimize",
            [0.0669, 0.0538, 0.1043, 0.1494],
            {"use_jacobian": True},
        )

    @skipIf(not TEST_LOGISTIC_PARALLEL_MODELS, SKIP_LOGISTIC_PARALLEL_MODELS)
    def test_multivariate_logistic_parallel_diagonal_mode_real(self):
        self._personalize_generic(
            "logistic_parallel_diag_noise",
            "mode_real",
            [0.0675, 0.0531, 0.1046, 0.1505],
        )

    @skipIf(not TEST_LOGISTIC_PARALLEL_MODELS, SKIP_LOGISTIC_PARALLEL_MODELS)
    def test_multivariate_logistic_parallel_diagonal_mean_real(self):
        self._personalize_generic(
            "logistic_parallel_diag_noise",
            "mean_real",
            [0.0671, 0.0553, 0.1040, 0.1509],
        )

    ################################################################
    # Univariate logistic

    def test_univariate_logistic_scipy_minimize(self):
        self._personalize_generic(
            "univariate_logistic",
            "scipy_minimize",
            0.1341,
            {"use_jacobian": False},
        )

    @skipIf(not TEST_LOGISTIC_MODELS_WITH_JACOBIAN, SKIP_LOGISTIC_MODELS_WITH_JACOBIAN)
    def test_univariate_logistic_scipy_minimize_with_jacobian(self):
        self._personalize_generic(
            "univariate_logistic",
            "scipy_minimize",
            0.1341,
            {"use_jacobian": True},
        )

    def test_univariate_logistic_mode_real(self):
        self._personalize_generic(
            "univariate_logistic",
            "mode_real",
            0.1346,
        )

    def test_univariate_logistic_mean_real(self):
        self._personalize_generic(
            "univariate_logistic",
            "mean_real",
            0.1351,
        )

    ################################################################
    # Univariate joint

    def test_univariate_joint_scipy_minimize(self):
        self._personalize_generic(
            "univariate_joint",
            "scipy_minimize",
            0.30333825945854187,
            {"use_jacobian": False},
        )

    @skipIf(not TEST_LOGISTIC_MODELS_WITH_JACOBIAN, SKIP_LOGISTIC_MODELS_WITH_JACOBIAN)
    def test_univariate_joint_scipy_minimize_with_jacobian(self):
        self._personalize_generic(
            "univariate_joint",
            "scipy_minimize",
            0.1341,
            {"use_jacobian": True},
        )


    def test_univariate_joint_mode_real(self):
        self._personalize_generic(
            "univariate_joint",
            "mode_real",
            0.30058470368385315,
        )


    def test_univariate_joint_mean_real(self):
        self._personalize_generic(
            "univariate_joint",
            "mean_real",
            0.30380967259407043,
        )

    ################################################################
    # Univariate linear

    @skipIf(not TEST_LINEAR_MODELS, SKIP_LINEAR_MODELS)
    def test_univariate_linear_scipy_minimize(self):
        self._personalize_generic(
            "univariate_linear",
            "scipy_minimize",
            0.0812,
            {"use_jacobian": False},
        )

    @skipIf(not TEST_LINEAR_MODELS, SKIP_LINEAR_MODELS)
    def test_univariate_linear_scipy_minimize_with_jacobian(self):
        self._personalize_generic(
            "univariate_linear",
            "scipy_minimize",
            0.0812,
            {"use_jacobian": True},
        )

    @skipIf(not TEST_LINEAR_MODELS, SKIP_LINEAR_MODELS)
    def test_univariate_linear_mode_real(self):
        self._personalize_generic(
            "univariate_linear",
            "mode_real",
            0.0817,
        )

    @skipIf(not TEST_LINEAR_MODELS, SKIP_LINEAR_MODELS)
    def test_univariate_linear_mean_real(self):
        self._personalize_generic(
            "univariate_linear",
            "mean_real",
            0.0898,
        )

    @skipIf(not TEST_LINEAR_MODELS, SKIP_LINEAR_MODELS)
    def test_multivariate_linear_scipy_minimize(self):
        self._personalize_generic(
            "linear_scalar_noise",
            "scipy_minimize",
            0.1241,
            {"use_jacobian": False},
        )

    @skipIf(not TEST_LINEAR_MODELS, SKIP_LINEAR_MODELS)
    def test_multivariate_linear_scipy_minimize_with_jacobian(self):
        self._personalize_generic(
            "linear_scalar_noise",
            "scipy_minimize",
            0.1241,
            {"use_jacobian": True},
        )

    @skipIf(not TEST_LINEAR_MODELS, SKIP_LINEAR_MODELS)
    def test_multivariate_linear_mode_real(self):
        self._personalize_generic(
            "linear_scalar_noise",
            "mode_real",
            0.1241,
        )

    @skipIf(not TEST_LINEAR_MODELS, SKIP_LINEAR_MODELS)
    def test_multivariate_linear_mean_real(self):
        self._personalize_generic(
            "linear_scalar_noise",
            "mean_real",
            0.1237,
        )

    @skipIf(not TEST_LINEAR_MODELS, SKIP_LINEAR_MODELS)
    def test_multivariate_linear_diagonal_scipy_minimize(self):
        self._personalize_generic(
            "linear_diag_noise",
            "scipy_minimize",
            [0.1003, 0.1274, 0.1249, 0.1486],
            {"use_jacobian": False},
        )

    @skipIf(not TEST_LINEAR_MODELS, SKIP_LINEAR_MODELS)
    def test_multivariate_linear_diagonal_scipy_minimize_with_jacobian(self):
        self._personalize_generic(
            "linear_diag_noise",
            "scipy_minimize",
            [0.1002, 0.1276, 0.1249, 0.1486],
            {"use_jacobian": True},
        )

    @skipIf(not TEST_LINEAR_MODELS, SKIP_LINEAR_MODELS)
    def test_multivariate_linear_diagonal_mode_real(self):
        self._personalize_generic(
            "linear_diag_noise",
            "mode_real",
            [0.1007, 0.1292, 0.1250, 0.1489],
        )

    @skipIf(not TEST_LINEAR_MODELS, SKIP_LINEAR_MODELS)
    def test_multivariate_linear_diagonal_mean_real(self):
        self._personalize_generic(
            "linear_diag_noise",
            "mean_real",
            [0.1000, 0.1265, 0.1242, 0.1485],
        )

    @skipIf(not TEST_LOGISTIC_BINARY_MODELS, SKIP_LOGISTIC_BINARY_MODELS)
    def test_multivariate_binary_scipy_minimize(self):
        self._personalize_generic(
            "logistic_binary",
            "scipy_minimize",
            103.7,
            {"use_jacobian": False},
        )

    @skipIf(not TEST_LOGISTIC_MODELS_WITH_JACOBIAN, SKIP_LOGISTIC_MODELS_WITH_JACOBIAN)
    def test_multivariate_binary_scipy_minimize_with_jacobian(self):
        self._personalize_generic(
            "logistic_binary",
            "scipy_minimize",
            103.67,
            {"use_jacobian": True},
        )

    @skipIf(not TEST_LOGISTIC_BINARY_MODELS, SKIP_LOGISTIC_BINARY_MODELS)
    def test_multivariate_binary_mode_real(self):
        self._personalize_generic(
            "logistic_binary",
            "mode_real",
            103.96,
        )

    @skipIf(not TEST_LOGISTIC_BINARY_MODELS, SKIP_LOGISTIC_BINARY_MODELS)
    def test_multivariate_binary_mean_real(self):
        self._personalize_generic(
            "logistic_binary",
            "mean_real",
            101.95,
        )

    @skipIf(not TEST_LOGISTIC_PARALLEL_MODELS, SKIP_LOGISTIC_PARALLEL_MODELS)
    def test_multivariate_parallel_binary_scipy_minimize(self):
        self._personalize_generic(
            "logistic_parallel_binary",
            "scipy_minimize",
            112.66,
            {"use_jacobian": False},
        )

    @skipIf(
        not TEST_LOGISTIC_PARALLEL_MODELS or not TEST_LOGISTIC_MODELS_WITH_JACOBIAN,
        SKIP_LOGISTIC_PARALLEL_MODELS if TEST_LOGISTIC_MODELS_WITH_JACOBIAN else SKIP_LOGISTIC_MODELS_WITH_JACOBIAN
    )
    def test_multivariate_parallel_binary_scipy_minimize_with_jacobian(self):
        self._personalize_generic(
            "logistic_parallel_binary",
            "scipy_minimize",
            112.63,
            {"use_jacobian": True},
        )

    @skipIf(not TEST_LOGISTIC_PARALLEL_MODELS, SKIP_LOGISTIC_PARALLEL_MODELS)
    def test_multivariate_parallel_binary_mode_real(self):
        self._personalize_generic(
            "logistic_parallel_binary",
            "mode_real",
            111.96,
        )

    @skipIf(not TEST_LOGISTIC_PARALLEL_MODELS, SKIP_LOGISTIC_PARALLEL_MODELS)
    def test_multivariate_parallel_binary_mean_real(self):
        self._personalize_generic(
            "logistic_parallel_binary",
            "mean_real",
            120.06,
        )

    @skipIf(not TEST_ORDINAL_MODELS, SKIP_ORDINAL_MODELS)
    def test_multivariate_ordinal_scipy_minimize(self):
        self._personalize_generic(
            "logistic_ordinal",
            "scipy_minimize",
            700.55,
            {"use_jacobian": False},
        )

    @skipIf(not TEST_ORDINAL_MODELS, SKIP_ORDINAL_MODELS)
    def test_multivariate_ordinal_scipy_minimize_with_jacobian(self):
        self._personalize_generic(
            "logistic_ordinal",
            "scipy_minimize",
            629.97,
            {"use_jacobian": True},
        )

    @skipIf(not TEST_ORDINAL_MODELS, SKIP_ORDINAL_MODELS)
    def test_multivariate_ordinal_mode_real(self):
        self._personalize_generic(
            "logistic_ordinal",
            "mode_real",
            619.64,
        )

    @skipIf(not TEST_ORDINAL_MODELS, SKIP_ORDINAL_MODELS)
    def test_multivariate_ordinal_mean_real(self):
        self._personalize_generic(
            "logistic_ordinal",
            "mean_real",
            616.94,
        )

    @skipIf(not TEST_ORDINAL_MODELS, SKIP_ORDINAL_MODELS)
    def test_multivariate_ordinal_ranking_scipy_minimize(self):
        self._personalize_generic(
            "logistic_ordinal_ranking",
            "scipy_minimize",
            1014.2,
            {"use_jacobian": False},
        )

    @skipIf(not TEST_ORDINAL_MODELS, SKIP_ORDINAL_MODELS)
    def test_multivariate_ordinal_ranking_scipy_minimize_with_jacobian(self):
        self._personalize_generic(
            "logistic_ordinal_ranking",
            "scipy_minimize",
            1014.1,
            {"use_jacobian": True},
        )

    @skipIf(not TEST_ORDINAL_MODELS, SKIP_ORDINAL_MODELS)
    def test_multivariate_ordinal_ranking_mode_real(self):
        self._personalize_generic(
            "logistic_ordinal_ranking",
            "mode_real",
            1014.9,
        )

    @skipIf(not TEST_ORDINAL_MODELS, SKIP_ORDINAL_MODELS)
    def test_multivariate_ordinal_ranking_mean_real(self):
        self._personalize_generic(
            "logistic_ordinal_ranking",
            "mean_real",
            1015.0,
        )


class LeaspyPersonalizeRobustnessDataSparsityTest(LeaspyPersonalizeTestMixin):
    """
    In this test, we check that estimated individual parameters are almost the same
    no matter if data is sparse (i.e. multiple really close visits with many missing
    values) or data is 'merged' in a rounded visit.

    TODO? we could build a mock dataset to also check same property for calibration :)
    """
    def _robustness_to_data_sparsity(
        self,
        model_name: str,
        algo_name: str,
        expected_loss: Union[float, list[float]],
        algo_kws: Optional[dict] = None,
        rtol: float = 2e-2,
        atol: float = 5e-3,
    ) -> None:
        algo_kws = algo_kws or {}
        subtest = {
            "model_name": model_name,
            "perso_name": algo_name,
            "perso_kws": algo_kws,
        }
        common_params = dict(algo_name=algo_name, seed=0, **algo_kws)

        ips_sparse, loss_sparse, _ = self.generic_personalization(
            model_name,
            **common_params,
            data_path="missing_data/sparse_data.csv",
            data_kws={'drop_full_nan': False},
        )
        ips_merged, loss_merged, _ = self.generic_personalization(
            model_name,
            **common_params,
            data_path="missing_data/merged_data.csv",
        )
        indices_sparse, ips_sparse_torch = ips_sparse.to_pytorch()
        indices_merged, ips_merged_torch = ips_merged.to_pytorch()

        # same individuals
        self.assertEqual(indices_sparse, indices_merged, msg=subtest)

        # same loss between both cases
        loss_desc = "nll" if any(kw in model_name for kw in {"binary", "ordinal"}) else "noise"
        self.assertAllClose(
            loss_sparse,
            loss_merged,
            left_desc="sparse",
            right_desc="merged",
            what=loss_desc,
            atol=atol,
            msg=subtest,
        )

        # same individual parameters (up to rounding errors)
        self.assertDictAlmostEqual(
            ips_sparse_torch,
            ips_merged_torch,
            left_desc="sparse",
            right_desc="merged",
            rtol=rtol,
            atol=atol,
            msg=subtest,
        )

        # same loss as expected
        self.assertAllClose(loss_merged, expected_loss, atol=atol, what=loss_desc, msg=subtest)

    def test_multivariate_logistic_scipy_minimize(self):
        self._robustness_to_data_sparsity(
            "logistic_scalar_noise",
            "scipy_minimize",
            0.1161,
            {"use_jacobian": False},
        )

    @skipIf(not TEST_LOGISTIC_MODELS_WITH_JACOBIAN, SKIP_LOGISTIC_MODELS_WITH_JACOBIAN)
    def test_multivariate_logistic_scipy_minimize_with_jacobian(self):
        self._robustness_to_data_sparsity(
            "logistic_scalar_noise",
            "scipy_minimize",
            0.1162,
            {"use_jacobian": True},
        )

    def test_multivariate_logistic_diagonal_id_scipy_minimize(self):
        self._robustness_to_data_sparsity(
            "logistic_diag_noise_id",
            "scipy_minimize",
            [0.0865, 0.0358, 0.0564, 0.2049],
            {"use_jacobian": False},
        )

    @skipIf(not TEST_LOGISTIC_MODELS_WITH_JACOBIAN, SKIP_LOGISTIC_MODELS_WITH_JACOBIAN)
    def test_multivariate_logistic_diagonal_id_scipy_minimize_with_jacobian(self):
        self._robustness_to_data_sparsity(
            "logistic_diag_noise_id",
            "scipy_minimize",
            [0.0865, 0.0359, 0.0564, 0.2050],
            {"use_jacobian": True},
        )

    def test_multivariate_logistic_diagonal_scipy_minimize(self):
        self._robustness_to_data_sparsity(
            "logistic_diag_noise",
            "scipy_minimize",
            [0.0824, 0.0089, 0.0551, 0.1819],
            {"use_jacobian": False},
        )

    @skipIf(not TEST_LOGISTIC_MODELS_WITH_JACOBIAN, SKIP_LOGISTIC_MODELS_WITH_JACOBIAN)
    def test_multivariate_logistic_diagonal_scipy_minimize_with_jacobian(self):
        self._robustness_to_data_sparsity(
            "logistic_diag_noise",
            "scipy_minimize",
            [0.0824, 0.0089, 0.0552, 0.1819],
            {"use_jacobian": True},
        )

    def test_multivariate_logistic_diagonal_mode_real(self):
        self._robustness_to_data_sparsity(
            "logistic_diag_noise",
            "mode_real",
            [0.0937, 0.0126, 0.0587, 0.1831],
        )

    def test_multivariate_logistic_diagonal_mean_real(self):
        self._robustness_to_data_sparsity(
            "logistic_diag_noise",
            "mean_real",
            [0.0908, 0.0072, 0.0595, 0.1817],
        )

    def test_multivariate_logistic_diagonal_no_source_scipy_minimize(self):
        self._robustness_to_data_sparsity(
            "logistic_diag_noise_no_source",
            "scipy_minimize",
            [0.1349, 0.0336, 0.0760, 0.1777],
            {"use_jacobian": False},
        )

    @skipIf(not TEST_LOGISTIC_MODELS_WITH_JACOBIAN, SKIP_LOGISTIC_MODELS_WITH_JACOBIAN)
    def test_multivariate_logistic_diagonal_no_source_scipy_minimize_with_jacobian(self):
        self._robustness_to_data_sparsity(
            "logistic_diag_noise_no_source",
            "scipy_minimize",
            [0.1349, 0.0336, 0.0761, 0.1777],
            {"use_jacobian": True},
        )

    def test_multivariate_logistic_diagonal_no_source_mode_real(self):
        self._robustness_to_data_sparsity(
            "logistic_diag_noise_no_source",
            "mode_real",
            [0.1339, 0.0356, 0.0754, 0.1761],
        )

    def test_multivariate_logistic_diagonal_no_source_mean_real(self):
        self._robustness_to_data_sparsity(
            "logistic_diag_noise_no_source",
            "mean_real",
            [0.1387, 0.0277, 0.0708, 0.1807],
        )

    @skipIf(not TEST_LOGISTIC_PARALLEL_MODELS, SKIP_LOGISTIC_PARALLEL_MODELS)
    def test_multivariate_logistic_parallel_scipy_minimize(self):
        self._robustness_to_data_sparsity(
            "logistic_parallel_scalar_noise",
            "scipy_minimize",
            0.1525,
            {"use_jacobian": False},
        )

    @skipIf(
        not TEST_LOGISTIC_PARALLEL_MODELS or not TEST_LOGISTIC_MODELS_WITH_JACOBIAN,
        SKIP_LOGISTIC_PARALLEL_MODELS if TEST_LOGISTIC_MODELS_WITH_JACOBIAN else SKIP_LOGISTIC_MODELS_WITH_JACOBIAN
    )
    def test_multivariate_logistic_parallel_scipy_minimize_with_jacobian(self):
        self._robustness_to_data_sparsity(
            "logistic_parallel_scalar_noise",
            "scipy_minimize",
            0.1872,
            {"use_jacobian": True},
        )

    @skipIf(not TEST_LOGISTIC_PARALLEL_MODELS, SKIP_LOGISTIC_PARALLEL_MODELS)
    def test_multivariate_logistic_parallel_mode_real(self):
        self._robustness_to_data_sparsity("logistic_parallel_scalar_noise", "mode_real", 0.1517)

    @skipIf(not TEST_LOGISTIC_PARALLEL_MODELS, SKIP_LOGISTIC_PARALLEL_MODELS)
    def test_multivariate_logistic_parallel_mean_real(self):
        self._robustness_to_data_sparsity("logistic_parallel_scalar_noise", "mean_real", 0.2079)

    @skipIf(not TEST_LOGISTIC_PARALLEL_MODELS, SKIP_LOGISTIC_PARALLEL_MODELS)
    def test_multivariate_logistic_parallel_diagonal_scipy_minimize(self):
        self._robustness_to_data_sparsity(
            "logistic_parallel_diag_noise",
            "scipy_minimize",
            [0.0178, 0.0120, 0.0509, 0.0939],
            {"use_jacobian": False},
        )

    @skipIf(
        not TEST_LOGISTIC_PARALLEL_MODELS or not TEST_LOGISTIC_MODELS_WITH_JACOBIAN,
        SKIP_LOGISTIC_PARALLEL_MODELS if TEST_LOGISTIC_MODELS_WITH_JACOBIAN else SKIP_LOGISTIC_MODELS_WITH_JACOBIAN
    )
    def test_multivariate_logistic_parallel_diagonal_scipy_minimize_with_jacobian(self):
        self._robustness_to_data_sparsity(
            "logistic_parallel_diag_noise",
            "scipy_minimize",
            [0.0178, 0.0120, 0.0508, 0.0940],
            {"use_jacobian": True},
        )

    @skipIf(not TEST_LOGISTIC_PARALLEL_MODELS, SKIP_LOGISTIC_PARALLEL_MODELS)
    def test_multivariate_logistic_parallel_diagonal_mode_real(self):
        self._robustness_to_data_sparsity(
            "logistic_parallel_diag_noise",
            "mode_real",
            [0.0193, 0.0179, 0.0443, 0.0971],
        )

    @skipIf(not TEST_LOGISTIC_PARALLEL_MODELS, SKIP_LOGISTIC_PARALLEL_MODELS)
    def test_multivariate_logistic_parallel_diagonal_mean_real(self):
        self._robustness_to_data_sparsity(
            "logistic_parallel_diag_noise",
            "mean_real",
            [0.0385, 0.0153, 0.0433, 0.3016],
        )

    @skipIf(not TEST_LINEAR_MODELS, SKIP_LINEAR_MODELS)
    def test_linear_scipy_minimize(self):
        self._robustness_to_data_sparsity(
            "linear_scalar_noise",
            "scipy_minimize",
            0.1699,
            {"use_jacobian": False},
        )

    @skipIf(not TEST_LINEAR_MODELS, SKIP_LINEAR_MODELS)
    def test_linear_scipy_minimize_with_jacobian(self):
        self._robustness_to_data_sparsity(
            "linear_scalar_noise",
            "scipy_minimize",
            0.1699,
            {"use_jacobian": True},
        )

    @skipIf(not TEST_LINEAR_MODELS, SKIP_LINEAR_MODELS)
    def test_linear_diagonal_scipy_minimize(self):
        self._robustness_to_data_sparsity(
            "linear_diag_noise",
            "scipy_minimize",
            [0.1021, 0.1650, 0.2083, 0.1481],
            {"use_jacobian": False},
        )

    @skipIf(not TEST_LINEAR_MODELS, SKIP_LINEAR_MODELS)
    def test_linear_diagonal_scipy_minimize_with_jacobian(self):
        self._robustness_to_data_sparsity(
            "linear_diag_noise",
            "scipy_minimize",
            [0.1023, 0.1630, 0.2081, 0.1480],
            {"use_jacobian": True},
        )

    @skipIf(not TEST_LOGISTIC_BINARY_MODELS, SKIP_LOGISTIC_BINARY_MODELS)
    def test_logistic_binary_scipy_minimize(self):
        self._robustness_to_data_sparsity(
            "logistic_binary",
            "scipy_minimize",
            8.4722,
            {"use_jacobian": False},
        )

    @skipIf(not TEST_LOGISTIC_BINARY_MODELS, SKIP_LOGISTIC_BINARY_MODELS)
    def test_logistic_binary_scipy_minimize_with_jacobian(self):
        self._robustness_to_data_sparsity(
            "logistic_binary",
            "scipy_minimize",
            8.4718,
            {"use_jacobian": False},
        )

    @skipIf(not TEST_LOGISTIC_BINARY_MODELS, SKIP_LOGISTIC_BINARY_MODELS)
    def test_logistic_parallel_binary_scipy_minimize(self):
        self._robustness_to_data_sparsity(
            "logistic_parallel_binary",
            "scipy_minimize",
            8.8422,
            {"use_jacobian": False},
        )

    @skipIf(not TEST_LOGISTIC_BINARY_MODELS, SKIP_LOGISTIC_BINARY_MODELS)
    def test_logistic_parallel_binary_scipy_minimize_with_jacobian(self):
        self._robustness_to_data_sparsity(
            "logistic_parallel_binary",
            "scipy_minimize",
            8.8408,
            {"use_jacobian": False},
        )


class LeaspyPersonalizeWithNansTest(LeaspyPersonalizeTestMixin):
    def test_personalize_full_nan(self, *, general_tol=1e-3):
        # test result of personalization with no data at all
        df = pd.DataFrame(
            {
                "ID": ["SUBJ1", "SUBJ1"],
                "TIME": [75.12, 78.9],
                "Y0": [nan] * 2,
                "Y1": [nan] * 2,
                "Y2": [nan] * 2,
                "Y3": [nan] * 2,
            }
        ).set_index(["ID", "TIME"])

        lsp = self.get_hardcoded_model("logistic_diag_noise")

        for perso_algo, perso_kws, coeff_tol_per_param_std in [

            ('scipy_minimize', dict(use_jacobian=False), general_tol),
            # ('scipy_minimize', dict(use_jacobian=True), general_tol),

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
                from leaspy.variables.specs import IndividualLatentVariable
                allclose_custom = {
                    p: dict(
                        atol=(
                            math.ceil(
                                coeff_tol_per_param_std * lsp.model.parameters[f'{p}_std'].item() / general_tol
                            ) * general_tol
                        )
                    )
                    for p in lsp.model.dag.sorted_variables_by_type[IndividualLatentVariable]
                }
                self.assertDictAlmostEqual(dict_1, {
                    'tau': [lsp.model.parameters['tau_mean']],
                    'xi': [[0.]],
                    'sources': [lsp.model.source_dimension*[0.]],
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
            # ('scipy_minimize', dict(use_jacobian=True), 1e-3),
            ('mode_real', {}, 1e-3),
            ('mean_real', {}, 1e-3),
        ]:

            subtest = dict(perso_algo=perso_algo, perso_kws=perso_kws)
            with self.subTest(**subtest):
                algo = self.get_algo_settings(name=perso_algo, seed=0, progress_bar=False, **perso_kws)

                data_without_empty_visits = Data.from_dataframe(df)
                data_with_empty_visits = Data.from_dataframe(df, drop_full_nan = False)

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
