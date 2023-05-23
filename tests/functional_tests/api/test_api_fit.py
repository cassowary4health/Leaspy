import os
import warnings
import json
import unittest
import torch

from leaspy import Leaspy
from leaspy.models.obs_models import FullGaussianObs  # WIP --> keyword-based factory needed

from tests.unit_tests.plots.test_plotter import MatplotlibTestCase


class LeaspyFitTest_Mixin(MatplotlibTestCase):
    """Mixin holding generic fit methods that may be safely reused in other tests (no actual test here)."""

    def generic_fit(self, model_name: str, model_codename: str, *,
                    algo_name='mcmc_saem', algo_params: dict = {},
                    # change default parameters for logs so everything is tested despite the very few iterations in tests
                    # TODO reactivate plotting once FitOutputManager & Plotter are ready
                    #logs_kws: dict = dict(console_print_periodicity=50, save_periodicity=20, plot_periodicity=100),
                    logs_kws: dict = dict(console_print_periodicity=50, save_periodicity=None, plot_periodicity=None),
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

    def _tmp_convert_old_to_new(self, old_model_dict, new_model_dict) -> None:
        # TODO/WIP: on-the-fly conversion old<->new models:
        # 1. new obs_models supplanting noise_model
        # 2. modification of some model (hyper-)parameter names & shapes
        # 3. some new/renamed/deleted fit-metrics
        old_mod = old_model_dict
        new_mod = new_model_dict
        has_src = int(old_mod['source_dimension'] >= 1)
        old_mod['parameters']['noise_std'] = old_mod['noise_model']['scale']
        del old_mod['noise_model']
        old_mod['fit_metrics']['nll_regul_ind_sum'] = old_mod['fit_metrics']['nll_regul_tot']
        for ip in ("tau", "xi", "tot") + ("sources",)*has_src:
            del old_mod['fit_metrics'][f'nll_regul_{ip}']
        for pp_old, pp_new in {"g": "log_g_mean", "v0": "log_v0_mean", "betas": "betas_mean"}.items():
            old_mod['parameters'][pp_new] = old_mod['parameters'][pp_old]
            del old_mod['parameters'][pp_old]
        for p in ("tau_mean", "tau_std", "xi_std") + ("sources_mean",)*has_src:
            old_mod['parameters'][p] = torch.tensor(old_mod['parameters'][p]).expand(torch.tensor(new_mod['parameters'][p]).shape).tolist()
        if has_src:
            old_mod['parameters']['mixing_matrix'] = torch.tensor(old_mod['parameters']['mixing_matrix']).t().tolist()
        else:
            for p in ("mixing_matrix", "sources_mean", "sources_std", "betas_mean"):
                del old_mod['parameters'][p]

        del new_mod['obs_models']
        del new_mod['fit_metrics']['nll_regul_pop_sum']
        del new_mod['fit_metrics']['nll_regul_all_sum']
        for pp in ("log_g_std", "log_v0_std") + ("betas_std",)*has_src:
            del new_mod['parameters'][pp]

    def check_model_consistency(self, leaspy: Leaspy, path_to_backup_model: str, **allclose_kwds):
        # Temporary save parameters and check consistency with previously saved model

        allclose_kwds = {'atol': 1e-5, 'rtol': 1e-4, **allclose_kwds}

        path_to_tmp_saved_model = self.get_test_tmp_path(os.path.basename(path_to_backup_model))
        leaspy.save(path_to_tmp_saved_model)

        with open(path_to_backup_model, 'r') as f1:
            expected_model_parameters = json.load(f1)
        with open(path_to_tmp_saved_model, 'r') as f2:
            model_parameters_new = json.load(f2)

        # don't compare leaspy exact version...
        expected_model_parameters['leaspy_version'] = None
        model_parameters_new['leaspy_version'] = None

        # TODO/WIP: on-the-fly conversion old<->new models:
        self._tmp_convert_old_to_new(expected_model_parameters, model_parameters_new)
        # END WIP

        # Remove the temporary file saved (before asserts since they may fail!)
        os.remove(path_to_tmp_saved_model)

        self.assertDictAlmostEqual(model_parameters_new, expected_model_parameters, **allclose_kwds)

        ## test consistency of model attributes (only mixing matrix here)
        expected_model = Leaspy.load(path_to_backup_model).model
        if expected_model.dimension != 1:
            self.assertAllClose(expected_model.attributes.mixing_matrix, expected_model_parameters['parameters']['mixing_matrix'],
                                **allclose_kwds, what='mixing_matrix')

class LeaspyFitTest(LeaspyFitTest_Mixin):
    # <!> reproducibility gap for PyTorch >= 1.7, only those are supported now

    # Test MCMC-SAEM
    def test_fit_logistic_scalar_noise(self):

        obs_model = FullGaussianObs.with_noise_std_as_model_parameter(dimension=1)
        leaspy, _ = self.generic_fit('logistic', 'logistic_scalar_noise', obs_models=obs_model, source_dimension=2,
                                     algo_params=dict(n_iter=100, seed=0),
                                     check_model=True)

    # Test MCMC-SAEM (1 noise per feature)
    def test_fit_logistic_diag_noise(self):

        # some noticeable reproducibility errors btw MacOS and Linux here...
        allclose_custom = dict(
            #nll_regul_tau=dict(atol=2),
            #nll_regul_xi=dict(atol=2),
            #nll_regul_sources=dict(atol=3),
            nll_regul_ind_sum=dict(atol=5),
            nll_attach=dict(atol=10),
            nll_tot=dict(atol=15),
            tau_mean=dict(atol=0.2),
            tau_std=dict(atol=0.2),
        )

        # TODO: dimension should not be needed at this point...
        obs_model = FullGaussianObs.with_noise_std_as_model_parameter(dimension=4)
        leaspy, _ = self.generic_fit('logistic', 'logistic_diag_noise', obs_models=obs_model, source_dimension=2,
                                     algo_params=dict(n_iter=100, seed=0),
                                     check_model=True,
                                     check_kws=dict(atol=0.1, rtol=1e-2, allclose_custom=allclose_custom),
                                    )

    def test_fit_logistic_diag_noise_fast_gibbs(self):

        # TODO: dimension should not be needed at this point...
        obs_model = FullGaussianObs.with_noise_std_as_model_parameter(dimension=4)
        leaspy, _ = self.generic_fit('logistic', 'logistic_diag_noise_fast_gibbs', obs_models=obs_model, source_dimension=2,
                                     algo_params=dict(n_iter=100, seed=0, sampler_pop='FastGibbs'),
                                     check_model=True)

    def test_fit_logistic_diag_noise_mh(self):

        # TODO: dimension should not be needed at this point...
        obs_model = FullGaussianObs.with_noise_std_as_model_parameter(dimension=4)
        leaspy, _ = self.generic_fit('logistic', 'logistic_diag_noise_mh', obs_models=obs_model, source_dimension=2,
                                     algo_params=dict(n_iter=100, seed=0, sampler_pop='Metropolis-Hastings'),
                                     check_model=True)

    def test_fit_logistic_diag_noise_with_custom_tuning_no_sources(self):

        # TODO: dimension should not be needed at this point...
        obs_model = FullGaussianObs.with_noise_std_as_model_parameter(dimension=4)
        leaspy, _ = self.generic_fit('logistic', 'logistic_diag_noise_custom',
                                     obs_models=obs_model, source_dimension=0,
                                     algo_params=dict(
                                         n_iter=100,
                                         burn_in_step_power=0.65,
                                         sampler_pop_params=dict(
                                             acceptation_history_length=10,
                                             mean_acceptation_rate_target_bounds=(.1, .5),
                                             adaptive_std_factor=0.1,
                                         ),
                                         sampler_ind_params=dict(
                                             acceptation_history_length=10,
                                             mean_acceptation_rate_target_bounds=(.1, .5),
                                             adaptive_std_factor=0.1,
                                         ),
                                         annealing=dict(
                                             initial_temperature=5.,
                                             do_annealing=True,
                                             n_plateau=2,
                                         ),
                                         seed=0,
                                     ),
                                     check_model=True)

    def test_fit_logistic_parallel(self):

        leaspy, _ = self.generic_fit('logistic_parallel', 'logistic_parallel_scalar_noise', noise_model='gaussian_scalar', source_dimension=2,
                                     algo_params=dict(n_iter=100, seed=0),
                                     check_model=True)

    def test_fit_logistic_parallel_diag_noise(self):

        leaspy, _ = self.generic_fit('logistic_parallel', 'logistic_parallel_diag_noise', noise_model='gaussian_diagonal', source_dimension=2,
                                     algo_params=dict(n_iter=100, seed=0),
                                     check_model=True)

    def test_fit_logistic_parallel_diag_noise_no_source(self):

        leaspy, _ = self.generic_fit('logistic_parallel', 'logistic_parallel_diag_noise_no_source', noise_model='gaussian_diagonal', source_dimension=0,
                                     algo_params=dict(n_iter=100, seed=0),
                                     check_model=True)

    def test_fit_univariate_logistic(self):

        leaspy, _ = self.generic_fit('univariate_logistic', 'univariate_logistic',
                                     algo_params=dict(n_iter=100, seed=0),
                                     check_model=True)

    def test_fit_univariate_linear(self):

        leaspy, _ = self.generic_fit('univariate_linear', 'univariate_linear',
                                     algo_params=dict(n_iter=100, seed=0),
                                     check_model=True)

    def test_fit_linear(self):

        leaspy, _ = self.generic_fit('linear', 'linear_scalar_noise', noise_model='gaussian_scalar', source_dimension=2,
                                     algo_params=dict(n_iter=100, seed=0),
                                     check_model=True)

    def test_fit_linear_diag_noise(self):

        leaspy, _ = self.generic_fit('linear', 'linear_diag_noise', noise_model='gaussian_diagonal', source_dimension=2,
                                     algo_params=dict(n_iter=100, seed=0),
                                     check_model=True)

    def test_fit_logistic_binary(self):

        leaspy, _ = self.generic_fit('logistic', 'logistic_binary', noise_model='bernoulli', source_dimension=2,
                                     algo_params=dict(n_iter=100, seed=0),
                                     check_model=True)

    def test_fit_logistic_parallel_binary(self):

        leaspy, _ = self.generic_fit('logistic_parallel', 'logistic_parallel_binary', noise_model='bernoulli', source_dimension=2,
                                     algo_params=dict(n_iter=100, seed=0),
                                     check_model=True)

    def test_fit_logistic_ordinal(self):

        leaspy, _ = self.generic_fit('logistic', 'logistic_ordinal', noise_model='ordinal', source_dimension=2,
                                     algo_params=dict(n_iter=100, seed=0),
                                     check_model=True)

    def test_fit_logistic_ordinal_batched(self):

        leaspy, _ = self.generic_fit('logistic', 'logistic_ordinal_b', noise_model='ordinal', source_dimension=2,
                                     algo_params=dict(n_iter=100, seed=0),
                                     check_model=True,
                                     batch_deltas_ordinal = True)   # test if batch sampling of deltas works

    def test_fit_logistic_ordinal_ranking(self):

        leaspy, _ = self.generic_fit('logistic', 'logistic_ordinal_ranking', noise_model='ordinal_ranking', source_dimension=2,
                                     algo_params=dict(n_iter=100, seed=0),
                                     check_model=True)

    def test_fit_logistic_ordinal_ranking_batched_mh(self):

        leaspy, _ = self.generic_fit('logistic', 'logistic_ordinal_ranking_mh', noise_model='ordinal_ranking', source_dimension=2,
                                     algo_params=dict(n_iter=100, seed=0, sampler_pop='Metropolis-Hastings'),
                                     batch_deltas_ordinal=True,  # test if batch sampling of deltas works
                                     check_model=True)

    def test_fit_logistic_ordinal_ranking_batched_fg(self):

        leaspy, _ = self.generic_fit('logistic', 'logistic_ordinal_ranking_fg', noise_model='ordinal_ranking', source_dimension=2,
                                     algo_params=dict(n_iter=100, seed=0, sampler_pop='FastGibbs'),
                                     batch_deltas_ordinal=True,  # test if batch sampling of deltas works
                                     check_model=True)


@unittest.skipIf(not torch.cuda.is_available(),
                "GPU calibration tests need an available CUDA environment")
class LeaspyFitGPUTest(LeaspyFitTest_Mixin):

    # Test MCMC-SAEM
    def test_fit_logistic_scalar_noise(self):

        leaspy, _ = self.generic_fit('logistic', 'logistic_scalar_noise_gpu', noise_model='gaussian_scalar', source_dimension=2,
                                     algo_params=dict(n_iter=100, seed=0, device='cuda'),
                                     check_model=True)


    # Test MCMC-SAEM (1 noise per feature)
    def test_fit_logistic_diag_noise(self):

        leaspy, _ = self.generic_fit('logistic', 'logistic_diag_noise_gpu', noise_model='gaussian_diagonal', source_dimension=2,
                                     algo_params=dict(n_iter=100, seed=0, device='cuda'),
                                     check_model=True)

    def test_fit_logisticparallel(self):

        leaspy, _ = self.generic_fit('logistic_parallel', 'logistic_parallel_scalar_noise_gpu', noise_model='gaussian_scalar', source_dimension=2,
                                     algo_params=dict(n_iter=100, seed=0, device='cuda'),
                                     check_model=True)

    def test_fit_logisticparallel_diag_noise(self):

        leaspy, _ = self.generic_fit('logistic_parallel', 'logistic_parallel_diag_noise_gpu', noise_model='gaussian_diagonal', source_dimension=2,
                                     algo_params=dict(n_iter=100, seed=0, device='cuda'),
                                     check_model=True)

    def test_fit_univariate_logistic(self):

        leaspy, _ = self.generic_fit('univariate_logistic', 'univariate_logistic_gpu',
                                     algo_params=dict(n_iter=100, seed=0, device='cuda'),
                                     check_model=True)

    def test_fit_univariate_linear(self):

        leaspy, _ = self.generic_fit('univariate_linear', 'univariate_linear_gpu',
                                     algo_params=dict(n_iter=100, seed=0, device='cuda'),
                                     check_model=True)

    def test_fit_linear(self):

        leaspy, _ = self.generic_fit('linear', 'linear_scalar_noise_gpu', noise_model='gaussian_scalar', source_dimension=2,
                                     algo_params=dict(n_iter=100, seed=0, device='cuda'),
                                     check_model=True)
