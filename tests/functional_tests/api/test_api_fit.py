import os
import warnings
import json
import unittest
import torch

from leaspy import Leaspy

from tests.unit_tests.plots.test_plotter import MatplotlibTestCase


class LeaspyFitTest_Mixin(MatplotlibTestCase):
    """Mixin holding generic fit methods that may be safely reused in other tests (no actual test here)."""

    def generic_fit(self, model_name: str, model_codename: str, *,
                    algo_name='mcmc_saem', algo_params: dict = {},
                    logs_kws: dict = {},
                    print_model: bool = False,
                    check_model: bool = True, check_kws: dict = {},
                    save_model: bool = False,
                    **model_hyperparams):
        """Helper for a generic calibration in following tests."""

        # load the right data
        data = self.get_suited_test_data_for_model(model_codename)

        # create a new leaspy object containing the model
        leaspy = Leaspy(model_name, **model_hyperparams)

        # create the fit algo settings
        algo_settings = self.get_algo_settings(name=algo_name, **algo_params)

        # set logs by default
        if logs_kws is not None:
            auto_path_logs = self.get_test_tmp_path(f'{model_codename}-logs')
            with self.assertWarnsRegex(UserWarning, r" does not exist\. Needed paths will be created"):
                algo_settings.set_logs(path=auto_path_logs, **logs_kws)

        # calibrate model
        leaspy.fit(data, settings=algo_settings)

        # print parameters (for debugging...)
        if print_model:
            print(leaspy.model.parameters)

        # path to expected
        expected_model_path = self.from_fit_model_path(model_codename)
        inexistant_model = not os.path.exists(expected_model_path)

        # check that values in already saved file are same than the ones in fitted model
        if check_model:
            if inexistant_model:
                warnings.warn(f"<!> Consistency of model could not be checked since '{model_codename}' did not exist...")
            else:
                self.check_model_consistency(leaspy, expected_model_path, **check_kws)

        ## set `save_model=True` to re-generate example model
        ## <!> use carefully (only when needed following breaking changes in model / calibration)
        if save_model or inexistant_model:
            leaspy.save(expected_model_path)
            if save_model:
                warnings.warn(f"<!> You overwrote previous '{model_codename}' model...")

        # return leaspy & data objects
        return leaspy, data

    def check_model_consistency(self, leaspy: Leaspy, path_to_backup_model: str, **allclose_kwds):
        # Temporary save parameters and check consistency with previously saved model

        path_to_tmp_saved_model = self.get_test_tmp_path(os.path.basename(path_to_backup_model))
        leaspy.save(path_to_tmp_saved_model)

        with open(path_to_backup_model, 'r') as f1:
            expected_model_parameters = json.load(f1)
            # don't compare leaspy exact version...
            expected_model_parameters['leaspy_version'] = None
        with open(path_to_tmp_saved_model) as f2:
            model_parameters_new = json.load(f2)
            # don't compare leaspy exact version...
            model_parameters_new['leaspy_version'] = None

        # Remove the temporary file saved (before asserts since they may fail!)
        os.remove(path_to_tmp_saved_model)

        self.assertDictAlmostEqual(model_parameters_new, expected_model_parameters, **allclose_kwds)

        ## test consistency of model attributes (only mixing matrix here)
        expected_model = Leaspy.load(path_to_backup_model).model
        if expected_model.dimension != 1:
            self.assertTrue(torch.allclose(expected_model.attributes.mixing_matrix, torch.tensor(expected_model_parameters['parameters']['mixing_matrix']),
                                           rtol=1e-4, atol=1e-6),
                            (expected_model.attributes.mixing_matrix, expected_model_parameters['parameters']['mixing_matrix']))


# Weirdly, some results are perfectly reproducible on local mac + CI linux but not on CI mac...
# Increasing tolerances so to pass despite these reproducibility issues...
class LeaspyFitTest(LeaspyFitTest_Mixin):
    """
    There are many reproducibility issues between Linux & MacOS for calibration... so we increase the tolerances in all those functional checks.
    We might consider having two expected files depending on architecture at a point?
    """

    # Test MCMC-SAEM
    def test_fit_logistic_scalar_noise(self, tol=6e-2, tol_tau=2e-1):

        leaspy, _ = self.generic_fit('logistic', 'logistic_scalar_noise', noise_model='gaussian_scalar', source_dimension=2,
                                     algo_params=dict(n_iter=100, seed=0),
                                     check_model=True,
                                     check_kws=dict(atol=tol, allclose_custom={'tau_mean': dict(atol=tol_tau),
                                                                               'tau_std': dict(atol=tol_tau)}))

    # Test MCMC-SAEM (1 noise per feature)
    def test_fit_logistic_diag_noise(self, tol=2e-1, tol_tau=2e-1):
        # <!> reproducibility gap for PyTorch >= 1.7, only those are supported now
        leaspy, _ = self.generic_fit('logistic', 'logistic_diag_noise', noise_model='gaussian_diagonal', source_dimension=2,
                                     algo_params=dict(n_iter=100, seed=0),
                                     check_model=True,
                                     check_kws=dict(atol=tol, allclose_custom={'tau_mean': dict(atol=tol_tau),
                                                                               'tau_std': dict(atol=tol_tau)}))

    def test_fit_logistic_parallel(self, tol=1e-2):

        leaspy, _ = self.generic_fit('logistic_parallel', 'logistic_parallel_scalar_noise', noise_model='gaussian_scalar', source_dimension=2,
                                     algo_params=dict(n_iter=100, seed=0),
                                     check_model=True,
                                     check_kws=dict(atol=tol))

    def test_fit_logistic_parallel_diag_noise(self, tol=2e-1, tol_tau=1.):

        leaspy, _ = self.generic_fit('logistic_parallel', 'logistic_parallel_diag_noise', noise_model='gaussian_diagonal', source_dimension=2,
                                     algo_params=dict(n_iter=100, seed=0),
                                     check_model=True,
                                     check_kws=dict(atol=tol, allclose_custom={'tau_mean': dict(atol=tol_tau),
                                                                               'tau_std': dict(atol=tol_tau)}))

    def test_fit_univariate_logistic(self, tol=1e-2):

        leaspy, _ = self.generic_fit('univariate_logistic', 'univariate_logistic',
                                     algo_params=dict(n_iter=100, seed=0),
                                     check_model=True,
                                     check_kws=dict(atol=tol))

    def test_fit_univariate_linear(self, tol=1e-2):

        leaspy, _ = self.generic_fit('univariate_linear', 'univariate_linear',
                                     algo_params=dict(n_iter=100, seed=0),
                                     check_model=True,
                                     check_kws=dict(atol=tol))

    def test_fit_linear(self, tol=5e-1, tol_tau=5e-1):

        leaspy, _ = self.generic_fit('linear', 'linear_scalar_noise', noise_model='gaussian_scalar', source_dimension=2,
                                     algo_params=dict(n_iter=100, seed=0),
                                     check_model=True,
                                     check_kws=dict(atol=tol, allclose_custom={'tau_mean': dict(atol=tol_tau),
                                                                               'tau_std': dict(atol=tol_tau)}))

    def test_fit_linear_diag_noise(self, tol=3e-1, tol_tau=1.):

        leaspy, _ = self.generic_fit('linear', 'linear_diag_noise', noise_model='gaussian_diagonal', source_dimension=2,
                                     algo_params=dict(n_iter=100, seed=0),
                                     check_model=True,
                                     check_kws=dict(atol=tol, allclose_custom={'tau_mean': dict(atol=tol_tau),
                                                                               'tau_std': dict(atol=tol_tau)}))

    def test_fit_logistic_binary(self, tol=6e-2, tol_tau=2e-1):

        leaspy, _ = self.generic_fit('logistic', 'logistic_binary', noise_model='bernoulli', source_dimension=2,
                                     algo_params=dict(n_iter=100, seed=0),
                                     check_model=True,
                                     check_kws=dict(atol=tol, allclose_custom={'tau_mean': dict(atol=tol_tau),
                                                                               'tau_std': dict(atol=tol_tau)}))

@unittest.skipIf(not torch.cuda.is_available(),
                "GPU calibration tests need an available CUDA environment")
class LeaspyFitGPUTest(LeaspyFitTest_Mixin):

    # Test MCMC-SAEM
    def test_fit_logistic_scalar_noise(self, tol=5e-2, tol_tau=2e-1):

        leaspy, _ = self.generic_fit('logistic', 'logistic_scalar_noise_gpu', noise_model='gaussian_scalar', source_dimension=2,
                                     algo_params=dict(n_iter=100, seed=0, device='cuda'),
                                     check_model=True,
                                     check_kws=dict(atol=tol, allclose_custom={'tau_mean': dict(atol=tol_tau),
                                                                               'tau_std': dict(atol=tol_tau)}))


    # Test MCMC-SAEM (1 noise per feature)
    def test_fit_logistic_diag_noise(self, tol=6e-2, tol_tau=2e-1):

        leaspy, _ = self.generic_fit('logistic', 'logistic_diag_noise_gpu', noise_model='gaussian_diagonal', source_dimension=2,
                                     algo_params=dict(n_iter=100, seed=0, device='cuda'),
                                     check_model=True,
                                     check_kws=dict(atol=tol, allclose_custom={'tau_mean': dict(atol=tol_tau),
                                                                               'tau_std': dict(atol=tol_tau)}))

    def test_fit_logisticparallel(self, tol=1e-2):

        leaspy, _ = self.generic_fit('logistic_parallel', 'logistic_parallel_scalar_noise_gpu', noise_model='gaussian_scalar', source_dimension=2,
                                     algo_params=dict(n_iter=100, seed=0, device='cuda'),
                                     check_model=True,
                                     check_kws=dict(atol=tol))

    def test_fit_logisticparallel_diag_noise(self, tol=1e-2):

        leaspy, _ = self.generic_fit('logistic_parallel', 'logistic_parallel_diag_noise_gpu', noise_model='gaussian_diagonal', source_dimension=2,
                                     algo_params=dict(n_iter=100, seed=0, device='cuda'),
                                     check_model=True,
                                     check_kws=dict(atol=tol))

    def test_fit_univariate_logistic(self, tol=1e-2):

        leaspy, _ = self.generic_fit('univariate_logistic', 'univariate_logistic_gpu',
                                     algo_params=dict(n_iter=100, seed=0, device='cuda'),
                                     check_model=True,
                                     check_kws=dict(atol=tol))

    def test_fit_univariate_linear(self, tol=1e-2):

        leaspy, _ = self.generic_fit('univariate_linear', 'univariate_linear_gpu',
                                     algo_params=dict(n_iter=100, seed=0, device='cuda'),
                                     check_model=True,
                                     check_kws=dict(atol=tol))

    def test_fit_linear(self, tol=2e-1, tol_tau=1e-1):

        leaspy, _ = self.generic_fit('linear', 'linear_scalar_noise_gpu', noise_model='gaussian_scalar', source_dimension=2,
                                     algo_params=dict(n_iter=100, seed=0, device='cuda'),
                                     check_model=True,
                                     check_kws=dict(atol=tol, allclose_custom={'tau_mean': dict(atol=tol_tau),
                                                                               'tau_std': dict(atol=tol_tau)}))
