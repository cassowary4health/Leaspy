import unittest

from leaspy import Leaspy, AlgorithmSettings

from tests import from_fit_model_path


class LeaspyTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        # <!> do not import those tests at top-level, otherwise their tests will be duplicated...
        from .test_api_fit import LeaspyFitTest
        from .test_api_personalize import LeaspyPersonalizeTest
        from .test_api_simulate import LeaspySimulateTest

        cls.generic_fit = LeaspyFitTest.generic_fit
        #self.generic_personalize = ...
        #self.generic_simulate = ...

        cls.check_personalize = LeaspyPersonalizeTest().check_consistency_personalize_outputs
        cls.check_simulate = LeaspySimulateTest().check_consistency_of_simulation_results

    def generic_usecase(self, model_name: str, model_codename: str, *,
                        expected_noise_std, # in perso
                        perso_algo: str, fit_algo='mcmc_saem', simulate_algo='simulation',
                        fit_check_kws = dict(atol=1e-3),
                        fit_algo_params=dict(seed=0), perso_algo_params=dict(seed=0), simulate_algo_params=dict(seed=0),
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
        path_to_backup_model = from_fit_model_path(filename_expected_model)

        leaspy, data = self.generic_fit(model_name, filename_expected_model, **model_hyperparams,
                                        algo_name=fit_algo, algo_params=fit_algo_params,
                                        check_model=True, check_kws=fit_check_kws)

        # unlink 1st functional fit test from next steps...
        leaspy = Leaspy.load(path_to_backup_model)
        self.assertTrue(leaspy.model.is_initialized)

        # Personalize
        algo_personalize_settings = AlgorithmSettings(perso_algo, **perso_algo_params)
        individual_parameters, noise_std = leaspy.personalize(data, settings=algo_personalize_settings, return_noise=True)
        self.check_personalize(individual_parameters, noise_std, expected_noise_std=expected_noise_std, tol_noise=1e-2)

        ## Plot TODO
        #path_output = os.path.join(test_data_dir, "plots")
        #plotter = Plotter(path_output)
        # plotter.plot_mean_trajectory(leaspy.model, save_as="mean_trajectory_plot")
        #plt.close()

        # Simulate
        simulation_settings = AlgorithmSettings(simulate_algo, **simulate_algo_params)
        simulation_results = leaspy.simulate(individual_parameters, data, simulation_settings)

        self.check_simulate(simulation_settings, simulation_results, data,
                            expected_results_file=f'simulation_results_{model_codename}.csv')

    def test_usecase_logistic_scalar_noise(self):

        self.generic_usecase(
            'logistic', model_codename='logistic_scalar_noise',
            loss='MSE', source_dimension=2,
            fit_algo_params=dict(n_iter=100, seed=0),
            perso_algo = 'mode_real', expected_noise_std=0.09753, # in perso
        )

    @unittest.skip('TODO')
    def test_usecase_logistic_diag_noise(self):

        self.generic_usecase(
            'logistic', model_codename='logistic_diag_noise',
            loss='MSE_diag_noise', source_dimension=2,
            fit_algo_params=dict(n_iter=100, seed=0),
            perso_algo = 'mode_real', expected_noise_std=[], # in perso
        )

    @unittest.skip('TODO')
    def test_usecase_logistic_binary(self):

        self.generic_usecase(
            'logistic', model_codename='logistic_binary',
            loss='crossentropy', source_dimension=2,
            fit_algo_params=dict(n_iter=100, seed=0),
            perso_algo = 'mode_real', expected_noise_std=[], # in perso
        )

    # TODO? univariate_*, linear
