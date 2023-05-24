# <!> NEVER import real tests classes at top-level (otherwise their tests will be duplicated...), only MIXINS!!
from .test_api_fit import LeaspyFitTest_Mixin
from .test_api_personalize import LeaspyPersonalizeTest_Mixin
from .test_api_simulate import LeaspySimulateTest_Mixin


class LeaspyAPITest(LeaspyFitTest_Mixin, LeaspyPersonalizeTest_Mixin, LeaspySimulateTest_Mixin):

    def generic_usecase(self, model_name: str, model_codename: str, *,
                        expected_loss_perso,
                        perso_algo: str, fit_algo='mcmc_saem', simulate_algo='simulation',
                        fit_check_kws = dict(atol=1e-3),
                        fit_algo_params=dict(seed=0), perso_algo_params=dict(seed=0),
                        simulate_algo_params=dict(seed=0), simulate_tol=1e-4,
                        tol_loss=1e-2,
                        **model_hyperparams):
        """
        Functional test of a basic analysis using leaspy package

        1 - Data loading
        2 - Fit logistic model with MCMC algorithm
        3 - Save parameters & reload (remove created files to keep the repo clean)
        4 - Personalize model with 'mode_real' algorithm
        (5 - Plot results)
        6 - Simulate new patients
        """
        filename_expected_model = model_codename + '_for_test_api'

        # WIP
        model_hyperparams['obs_models'] = model_hyperparams.pop('noise_model', 'NONE').replace('_', '-')

        # no loss returned for fit for now
        leaspy, data = self.generic_fit(model_name, filename_expected_model, **model_hyperparams,
                                        algo_name=fit_algo, algo_params=fit_algo_params,
                                        check_model=True, check_kws=fit_check_kws)

        # unlink 1st functional fit test from next steps...
        leaspy = self.get_from_fit_model(filename_expected_model)
        self.assertTrue(leaspy.model.is_initialized)

        # Personalize
        algo_personalize_settings = self.get_algo_settings(name=perso_algo, **perso_algo_params)
        individual_parameters, loss = leaspy.personalize(data, settings=algo_personalize_settings, return_loss=True)
        self.check_consistency_of_personalization_outputs(
                individual_parameters, loss,
                expected_loss=expected_loss_perso, tol_loss=tol_loss)

        # TODO/WIP...
        return

        # Simulate
        simulation_settings = self.get_algo_settings(name=simulate_algo, **simulate_algo_params)
        simulation_results = leaspy.simulate(individual_parameters, data, simulation_settings)

        self.check_consistency_of_simulation_results(simulation_settings, simulation_results, data,
                expected_results_file=f'simulation_results_{model_codename}.csv', model=leaspy.model, tol=simulate_tol)

    def test_usecase_logistic_scalar_noise(self):

        # Simulation parameters
        custom_delays_vis = lambda n: [.5]*min(n, 2) + [1.]*max(0, n-2)  # OLD weird delays between visits
        simul_params = dict(seed=0, delay_btw_visits=custom_delays_vis, number_of_subjects=100)  # noise=...

        self.generic_usecase(
            'logistic', model_codename='logistic_scalar_noise',
            noise_model='gaussian_scalar', source_dimension=2,
            fit_algo_params=dict(n_iter=200, seed=0),
            fit_check_kws=dict(atol=1e-2, rtol=1e-2),
            perso_algo='mode_real',
            perso_algo_params=dict(n_iter=200, seed=0),
            expected_loss_perso=0.0857, # scalar RMSE
            simulate_algo_params=simul_params,
        )

    def test_usecase_logistic_diag_noise(self):

        # Simulation parameters
        custom_delays_vis = dict(mean=1., min=.2, max=2., std=1.)
        simul_params = dict(seed=0, delay_btw_visits=custom_delays_vis, number_of_subjects=100)  # noise=...

        # some noticeable reproducibility errors btw MacOS and Linux here...
        allclose_custom = dict(
            #nll_regul_tau=dict(atol=1),
            #nll_regul_xi=dict(atol=5),
            #nll_regul_sources=dict(atol=1),
            nll_regul_ind_sum=dict(atol=5),
            nll_attach=dict(atol=6),
            nll_tot=dict(atol=5),
            tau_mean=dict(atol=0.3),
            tau_std=dict(atol=0.3),
        )

        self.generic_usecase(
            'logistic', model_codename='logistic_diag_noise',
            noise_model='gaussian_diagonal', source_dimension=2,
            dimension=4, # WIP
            fit_algo_params=dict(n_iter=200, seed=0),
            fit_check_kws=dict(atol=0.1, rtol=1e-2, allclose_custom=allclose_custom),
            perso_algo='scipy_minimize',
            expected_loss_perso=[0.064, 0.037, 0.066, 0.142],  # per-ft RMSE
            simulate_algo_params=simul_params, simulate_tol=2e-3, # Not fully reproducible on Linux below this tol...
        )

    def test_usecase_logistic_binary(self):

        # Simulation parameters
        custom_delays_vis = .5
        simul_params = dict(seed=0, delay_btw_visits=custom_delays_vis, number_of_subjects=100,
                            reparametrized_age_bounds=(50, 85))  # noise=...

        self.generic_usecase(
            'logistic', model_codename='logistic_binary',
            noise_model='bernoulli', source_dimension=2,
            fit_algo_params=dict(n_iter=200, seed=0),
            perso_algo='mean_real',
            expected_loss_perso=105.18,  # logLL, not noise_std
            simulate_algo_params=simul_params,
        )

    def test_usecase_logistic_ordinal(self):

        # Simulation parameters
        custom_delays_vis = .5
        simul_params = dict(seed=0, delay_btw_visits=custom_delays_vis, number_of_subjects=100,
                            reparametrized_age_bounds=(50, 85))  # noise=...

        self.generic_usecase(
            'logistic', model_codename='logistic_ordinal',
            noise_model='ordinal', source_dimension=2,
            fit_algo_params=dict(n_iter=200, seed=0),
            perso_algo='mean_real',
            expected_loss_perso=1029.1,  # logLL, not noise_std
            tol_loss=0.1,
            simulate_algo_params=simul_params,
        )

    def test_usecase_logistic_ordinal_batched(self):
        # Simulation parameters
        # <!> Ordinal simulation may not be fully reproducible on different machines
        #     due to rounding errors when computing MultinomialDistribution.cdf that
        #     can lead to ±1 differences on MLE outcomes in rare cases...
        #     (changing seed, reducing subjects & increasing tol to avoid the problem)
        custom_delays_vis = .5
        simul_params = dict(seed=123, delay_btw_visits=custom_delays_vis, number_of_subjects=10,
                            reparametrized_age_bounds=(50, 85))
        self.generic_usecase(
            'logistic', model_codename='logistic_ordinal_b',
            noise_model='ordinal', source_dimension=2,
            fit_algo_params=dict(n_iter=200, seed=0),
            fit_check_kws=dict(atol=0.005),
            perso_algo='mean_real',
            expected_loss_perso=1132.6,  # logLL, not noise_std
            tol_loss=0.1,
            simulate_algo_params=simul_params,
            simulate_tol=5e-2,
            batch_deltas_ordinal=True,
        )

    def test_usecase_univariate_logistic_ordinal(self):

        # Simulation parameters
        custom_delays_vis = .5
        simul_params = dict(seed=0, delay_btw_visits=custom_delays_vis, number_of_subjects=100,
                            reparametrized_age_bounds=(50, 85))  # noise=...

        self.generic_usecase(
            'univariate_logistic', model_codename='univariate_logistic_ordinal',
            noise_model='ordinal',
            fit_algo_params=dict(n_iter=200, seed=0),
            perso_algo='mean_real',
            expected_loss_perso=169.8,  # logLL, not noise_std
            tol_loss=0.1,
            simulate_algo_params=simul_params,
        )

    def test_usecase_logistic_ordinal_ranking(self):

        # Simulation parameters
        custom_delays_vis = .5
        simul_params = dict(seed=0, delay_btw_visits=custom_delays_vis, number_of_subjects=100,
                            reparametrized_age_bounds=(50, 85))  # noise=...

        self.generic_usecase(
            'logistic', model_codename='logistic_ordinal_ranking',
            noise_model='ordinal_ranking', source_dimension=2,
            fit_algo_params=dict(n_iter=200, seed=0),
            perso_algo='mean_real',
            expected_loss_perso=974.15,  # logLL, not noise_std
            tol_loss=0.1,
            simulate_algo_params=simul_params,
        )

    def test_usecase_logistic_ordinal_ranking_batched(self):

        # Simulation parameters
        custom_delays_vis = .5
        simul_params = dict(seed=0, delay_btw_visits=custom_delays_vis, number_of_subjects=100,
                            reparametrized_age_bounds=(50, 85))  # noise=...

        self.generic_usecase(
            'logistic', model_codename='logistic_ordinal_ranking_b',
            noise_model='ordinal_ranking', source_dimension=2,
            fit_algo_params=dict(n_iter=200, seed=0),
            perso_algo='mode_real',
            expected_loss_perso=971.95,  # logLL, not noise_std
            tol_loss=0.1,
            simulate_algo_params=simul_params,
            batch_deltas_ordinal=True,
        )
