import numpy as np
import torch

from leaspy.algo.personalize.scipy_minimize import ScipyMinimize
from leaspy.io.settings.algorithm_settings import AlgorithmSettings

from tests import LeaspyTestCase

# test tolerance, lack of precision btw different machines... (no exact reproducibility in scipy.optimize.minimize?)
tol = 3e-3
tol_tau = 1e-2


class ScipyMinimizeTest(LeaspyTestCase):

    def check_individual_parameters(self, ips, *, tau, xi, tol_tau, tol_xi, sources=None, tol_sources=None):

        self.assertAlmostEqual(ips['tau'].item(), tau, delta=tol_tau)
        self.assertAlmostEqual(ips['xi'].item(), xi, delta=tol_xi)

        if sources is not None:
            n_sources = len(sources)
            res_sources = ips['sources'].squeeze().tolist()
            self.assertEqual( len(res_sources), n_sources )
            for s, s_expected in zip(res_sources, sources):
                self.assertAlmostEqual(s, s_expected, delta=tol_sources)

    def test_constructor(self):
        settings = AlgorithmSettings('scipy_minimize')
        algo = ScipyMinimize(settings)

        self.assertEqual(algo.algo_parameters, {'n_iter': 100, 'use_jacobian': True, 'n_jobs': 1, "progress_bar": True})
        self.assertEqual(algo.name, 'scipy_minimize')
        self.assertEqual(algo.seed, None)

    #def test_get_model_name(self):
    #    settings = AlgorithmSettings('scipy_minimize')
    #    algo = ScipyMinimize(settings)
    #    algo._set_model_name('name')
    #
    #    self.assertEqual(algo.model_name, 'name')

    def test_initialize_parameters(self):
        settings = AlgorithmSettings('scipy_minimize')
        algo = ScipyMinimize(settings)

        univariate_model = self.get_hardcoded_model('univariate_logistic')
        param = algo._initialize_parameters(univariate_model.model)

        self.assertEqual(param, [torch.tensor([-1.0/0.01]), torch.tensor([70.0/2.5])])

        multivariate_model = self.get_hardcoded_model('logistic_scalar_noise')
        param = algo._initialize_parameters(multivariate_model.model)
        self.assertEqual(param, [torch.tensor([0.0]), torch.tensor([75.2/7.1]), torch.tensor([0.]), torch.tensor([0.])])

    def test_get_reconstruction_error(self):
        leaspy = self.get_hardcoded_model('logistic_scalar_noise')

        settings = AlgorithmSettings('scipy_minimize')
        algo = ScipyMinimize(settings)
        #algo._set_model_name('logistic')

        times = torch.tensor([70, 80])
        values = torch.tensor([[0.5, 0.4, 0.4, 0.45], [0.3, 0.3, 0.2, 0.4]])

        z = [0.0, 75.2/7.1, 0., 0.]
        individual_parameters = algo._pull_individual_parameters(z, leaspy.model)

        err = algo._get_reconstruction_error(leaspy.model, times, values, individual_parameters)

        output = torch.tensor([
            [-0.4705, -0.3278, -0.3103, -0.4477],
            [0.6059,  0.0709,  0.3537,  0.4523]])
        self.assertEqual(torch.is_tensor(err), True)
        self.assertAlmostEqual(torch.sum((err - output)**2).item(), 0, delta=1e-8)

    def test_get_regularity(self):
        leaspy = self.get_hardcoded_model('logistic_scalar_noise')

        settings = AlgorithmSettings('scipy_minimize')
        algo = ScipyMinimize(settings)
        #algo._set_model_name('logistic')

        z = [0.0, 75.2/7.1, 0., 0.]
        individual_parameters = algo._pull_individual_parameters(z, leaspy.model)

        reg, reg_grads = algo._get_regularity(leaspy.model, individual_parameters)
        self.assertEqual(torch.is_tensor(reg), True)
        output = torch.tensor([4.0264])
        self.assertAlmostEqual(torch.sum((reg - output)**2).item(), 0, delta=1e-8)

        # gradients
        self.assertIsInstance(reg_grads, dict)
        self.assertSetEqual(set(reg_grads.keys()),{'xi','tau','sources'})
        # types & dimensions
        self.assertEqual(torch.is_tensor(reg_grads['xi']), True)
        self.assertEqual(torch.is_tensor(reg_grads['tau']), True)
        self.assertEqual(torch.is_tensor(reg_grads['sources']), True)
        self.assertTupleEqual(reg_grads['xi'].shape, (1,1))
        self.assertTupleEqual(reg_grads['tau'].shape, (1,1))
        self.assertTupleEqual(reg_grads['sources'].shape, (1,2))

    def get_individual_parameters_patient(self, model_name, times, values, *, noise_model, **algo_kwargs):
        # already a functional test in fact...
        leaspy = self.get_hardcoded_model(model_name)
        leaspy.model.load_hyperparameters({'noise_model': noise_model})

        settings = AlgorithmSettings('scipy_minimize', seed=0, **algo_kwargs)
        algo = ScipyMinimize(settings)

        # manually initialize seed since it's not done by algo itself (no call to run afterwards)
        algo._initialize_seed(algo.seed)
        self.assertEqual(algo.seed, np.random.get_state()[1][0])

        # Test without nan
        output = algo._get_individual_parameters_patient(leaspy.model,
                                torch.tensor(times, dtype=torch.float32),
                                torch.tensor(values, dtype=torch.float32))

        return output

    def test_get_individual_parameters_patient_univariate_models(self, tol=tol, tol_tau=tol_tau):

        times = [70, 80]
        values = [[0.5], [0.4]] # no test with nans (done in multivariate models)

        for (model_name, use_jacobian), expected_dict in {

            ('univariate_logistic', False): {'tau': 69.2868, 'xi': -1.0002, 'err': [[-0.1765], [0.5498]]},
            ('univariate_logistic', True ): {'tau': 69.2868, 'xi': -1.0002, 'err': [[-0.1765], [0.5498]]},
            ('univariate_linear',   False): {'tau': 78.1131, 'xi': -4.2035, 'err': [[-0.1212], [0.1282]]},
            ('univariate_linear',   True ): {'tau': 78.0821, 'xi': -4.2016, 'err': [[-0.1210], [0.1287]]},

        }.items():

            individual_parameters, err = self.get_individual_parameters_patient(model_name,
                                                    times, values, noise_model='gaussian_scalar', use_jacobian=use_jacobian)

            self.check_individual_parameters(individual_parameters,
                tau=expected_dict['tau'], tol_tau=tol_tau,
                xi=expected_dict['xi'], tol_xi=tol
            )

            self.assertAlmostEqual(torch.sum((err - torch.tensor(expected_dict['err']))**2).item(), 0, delta=tol**2)

    def test_get_individual_parameters_patient_multivariate_models(self, tol=tol, tol_tau=tol_tau):

        times = [70, 80]
        values = [[0.5, 0.4, 0.4, 0.45], [0.3, 0.3, 0.2, 0.4]] # no nans (separate test)

        for (model_name, use_jacobian), expected_dict in {

            ('logistic_scalar_noise', False): {
                'tau': 78.5750,
                'xi': -0.0919,
                'sources': [0.3517, 0.1662],
                'err': [[-0.49765, -0.31615,  -0.351310, -0.44945],
                        [ 0.00825,  0.06638,  0.139204,  0.00413 ]],
            },
            ('logistic_scalar_noise', True): {
                'tau': 78.5750,
                'xi': -0.0918,
                'sources': [0.3483, 0.1678],
                'err': [[-0.4976, -0.3158, -0.3510, -0.4494],
                        [ 0.0079,  0.0673,  0.1403,  0.0041]]
            },

            # TODO? linear, logistic_parallel

        }.items():

            individual_parameters, err = self.get_individual_parameters_patient(model_name,
                                                    times, values, noise_model='gaussian_scalar', use_jacobian=use_jacobian)

            self.check_individual_parameters(individual_parameters,
                tau=expected_dict['tau'], tol_tau=tol_tau,
                xi=expected_dict['xi'], tol_xi=tol,
                sources=expected_dict['sources'], tol_sources=tol,
            )

            self.assertAlmostEqual(torch.sum((err - torch.tensor(expected_dict['err']))**2).item(), 0, delta=tol**2)

    def test_get_individual_parameters_patient_multivariate_models_with_nans(self, tol=tol, tol_tau=tol_tau):

        times = [70, 80]
        values = [[0.5, 0.4, 0.4, np.nan], [0.3, np.nan, np.nan, 0.4]]

        nan_positions = torch.tensor([
            [False, False, False, True],
            [False, True, True, False]
        ])

        for (model_name, use_jacobian), expected_dict in {

            ('logistic_scalar_noise', False): {
                'tau': 77.5558,
                'xi': -0.0989,
                'sources': [-0.9805,  0.7745],
                'err': [[-0.4981, -0.0895, -0.1161,     0.],
                        [-0.0398,     0.,     0.,  0.0863]]
            },
            ('logistic_scalar_noise', True):  {
                'tau': 77.5555,
                'xi': -0.0990,
                'sources': [-0.9799,  0.7743],
                'err': [[-0.4981, -0.0896, -0.1162,     0.],
                        [-0.0397,     0.,     0.,  0.0863]]
            },

            # TODO? linear, logistic_parallel

        }.items():

            individual_parameters, err = self.get_individual_parameters_patient(model_name,
                                                    times, values, noise_model='gaussian_scalar', use_jacobian=use_jacobian)

            self.check_individual_parameters(individual_parameters,
                tau=expected_dict['tau'], tol_tau=tol_tau,
                xi=expected_dict['xi'], tol_xi=tol,
                sources=expected_dict['sources'], tol_sources=tol,
            )

            self.assertTrue(torch.eq(torch.isnan(err), nan_positions).all())
            err[torch.isnan(err)] = 0.

            self.assertAlmostEqual(torch.sum((err - torch.tensor(expected_dict['err']))**2).item(), 0, delta=tol**2)


    def test_get_individual_parameters_patient_multivariate_models_crossentropy(self, tol=tol, tol_tau=tol_tau):

        times = [70, 80]
        values = [[0, 1, 0, 1], [0, 1, 1, 1]] # no nans (separate test)

        for (model_name, use_jacobian), expected_dict in {

            ('logistic_scalar_noise', False): {
                'tau': 70.6041,
                'xi': -0.0458,
                'sources': [0.9961, 1.2044],
                'err': [[ 1.2993e-03, -6.2189e-02,  4.8657e-01, -4.1219e-01],
                        [ 2.4171e-01, -9.4945e-03, -8.5862e-02, -4.0013e-04]]
            },
            ('logistic_scalar_noise', True): {
                'tau': 70.5971,
                'xi': -0.0471,
                'sources': [0.9984, 1.2037],
                'err': [[ 1.3049e-03, -6.2177e-02,  4.8639e-01, -4.1050e-01],
                        [ 2.4118e-01, -9.5164e-03, -8.6166e-02, -4.0120e-04]]
            },

        }.items():

            individual_parameters, err = self.get_individual_parameters_patient(model_name,
                                                    times, values, use_jacobian=use_jacobian,
                                                    noise_model='bernoulli')

            self.check_individual_parameters(individual_parameters,
                tau=expected_dict['tau'], tol_tau=tol_tau,
                xi=expected_dict['xi'], tol_xi=tol,
                sources=expected_dict['sources'], tol_sources=tol,
            )

            self.assertAlmostEqual(torch.sum((err - torch.tensor(expected_dict['err']))**2).item(), 0, delta=tol**2)

    def test_get_individual_parameters_patient_multivariate_models_with_nans_crossentropy(self, tol=tol, tol_tau=tol_tau):

        times = [70, 80]
        values = [[0, 1, 0, np.nan], [0, np.nan, np.nan, 1]]

        nan_positions = torch.tensor([
            [False, False, False, True],
            [False, True, True, False]
        ])

        for (model_name, use_jacobian), expected_dict in {

            ('logistic_scalar_noise', False): {
                'tau': 75.7494,
                'xi': -0.0043,
                'sources': [0.4151, 1.0180],
                'err': [[ 2.7332e-04, -0.30629,  0.22526,        0.],
                        [ 0.077974,         0.,       0., -0.036894]]
            },
            ('logistic_scalar_noise', True): {
                'tau': 75.7363,
                'xi': -0.0038,
                'sources': [0.4146, 1.0160],
                'err': [[ 2.7735e-04, -0.30730,  0.22526,         0.],
                        [ 0.079207,         0.,         0., -0.036614]]
            },

            # TODO? linear, logistic_parallel

        }.items():

            individual_parameters, err = self.get_individual_parameters_patient(model_name,
                                                    times, values, use_jacobian=use_jacobian,
                                                    noise_model='bernoulli')

            self.check_individual_parameters(individual_parameters,
                tau=expected_dict['tau'], tol_tau=tol_tau,
                xi=expected_dict['xi'], tol_xi=tol,
                sources=expected_dict['sources'], tol_sources=tol,
            )

            self.assertTrue(torch.eq(torch.isnan(err), nan_positions).all())
            err[torch.isnan(err)] = 0.

            self.assertAlmostEqual(torch.sum((err - torch.tensor(expected_dict['err']))**2).item(), 0, delta=tol**2)
