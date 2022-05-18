import numpy as np
import pandas as pd
import torch

from leaspy.io.data.data import Data
from leaspy.io.data.dataset import Dataset
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

    def test_default_constructor(self):
        settings = AlgorithmSettings('scipy_minimize')

        self.assertEqual(settings.parameters, {
            'use_jacobian': True,
            'n_jobs': 1,
            'progress_bar': True,
            'custom_scipy_minimize_params': None,
            'custom_format_convergence_issues': None,
        })

        algo = ScipyMinimize(settings)
        self.assertEqual(algo.name, 'scipy_minimize')
        self.assertEqual(algo.seed, None)

        self.assertEqual(algo.scipy_minimize_params, ScipyMinimize.DEFAULT_SCIPY_MINIMIZE_PARAMS_WITH_JACOBIAN)
        self.assertEqual(algo.format_convergence_issues, ScipyMinimize.DEFAULT_FORMAT_CONVERGENCE_ISSUES)
        self.assertEqual(algo.logger, algo._default_logger)

    def test_default_constructor_no_jacobian(self):
        settings = AlgorithmSettings('scipy_minimize', use_jacobian=False)

        self.assertEqual(settings.parameters, {
            'use_jacobian': False,
            'n_jobs': 1,
            'progress_bar': True,
            'custom_scipy_minimize_params': None,
            'custom_format_convergence_issues': None,
        })

        algo = ScipyMinimize(settings)
        self.assertEqual(algo.name, 'scipy_minimize')
        self.assertEqual(algo.seed, None)

        self.assertEqual(algo.scipy_minimize_params, ScipyMinimize.DEFAULT_SCIPY_MINIMIZE_PARAMS_WITHOUT_JACOBIAN)
        self.assertEqual(algo.format_convergence_issues, ScipyMinimize.DEFAULT_FORMAT_CONVERGENCE_ISSUES)
        self.assertEqual(algo.logger, algo._default_logger)


    def test_custom_constructor(self):

        def custom_logger(msg: str):
            pass

        custom_format_convergence_issues="{patient_id}: {optimization_result_pformat}..."
        custom_scipy_minimize_params={
                                      'method': 'BFGS',
                                      'options': {'gtol': 5e-2, 'maxiter': 100}
                                     }

        settings = AlgorithmSettings('scipy_minimize',
                                     seed=24,
                                     custom_format_convergence_issues=custom_format_convergence_issues,
                                     custom_scipy_minimize_params=custom_scipy_minimize_params)
        settings.logger = custom_logger

        algo = ScipyMinimize(settings)
        self.assertEqual(algo.name, 'scipy_minimize')
        self.assertEqual(algo.seed, 24)

        self.assertEqual(algo.format_convergence_issues, custom_format_convergence_issues)
        self.assertEqual(algo.scipy_minimize_params, custom_scipy_minimize_params)
        self.assertIs(algo.logger, custom_logger)


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

    def test_fallback_without_jacobian(self):
        model = self.get_hardcoded_model('logistic_scalar_noise').model

        # pretend as if compute_jacobian_tensorized was not implemented
        def not_implemented_compute_jacobian_tensorized(tpts, ips, **kws):
            raise NotImplementedError

        model.compute_jacobian_tensorized = not_implemented_compute_jacobian_tensorized

        mini_dataset = Dataset(self.get_suited_test_data_for_model('logistic_scalar_noise'))

        settings = AlgorithmSettings('scipy_minimize') #, use_jacobian=True) # default
        algo = ScipyMinimize(settings)

        with self.assertWarnsRegex(UserWarning, r'`use_jacobian\s?=\s?False`'):
            algo._get_individual_parameters(model, mini_dataset)

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
        z0 = [0.0, 75.2/7.1, 0., 0.]  # for all individual parameters we set `mean/std`

        settings = AlgorithmSettings('scipy_minimize')
        algo = ScipyMinimize(settings)

        individual_parameters = algo._pull_individual_parameters(z0, leaspy.model)

        # regularity constant is not added anymore (useless)
        expected_reg = torch.tensor([0.])
        reg, reg_grads = algo._get_regularity(leaspy.model, individual_parameters)
        self.assertTrue(torch.is_tensor(reg))
        self.assertEqual(reg.shape, expected_reg.shape)
        self.assertAllClose(reg, expected_reg)

        # gradients
        expected_reg_grads = {'tau': torch.tensor([[0.]]), 'xi': torch.tensor([[0.]]), 'sources': torch.tensor([[0., 0.]])}
        self.assertIsInstance(reg_grads, dict)
        self.assertEqual(reg_grads.keys(), expected_reg_grads.keys())
        # types & dimensions
        for ip, expected_reg_grad in expected_reg_grads.items():
            self.assertTrue(torch.is_tensor(reg_grads[ip]))
            self.assertEqual(reg_grads[ip].shape, expected_reg_grad.shape)
        # nice check for all values
        self.assertDictAlmostEqual(reg_grads, expected_reg_grads)

        # second test with a non-zero regularity term
        s = [0.33, -0.59, 0.72, -0.14]  # random shifts to test (in normalized space)
        z = [si + z0i for si, z0i in zip(s, z0)]  # we have to add the z0 by design of `_pull_individual_parameters`
        individual_parameters = algo._pull_individual_parameters(z, leaspy.model)
        expected_reg = 0.5 * (torch.tensor(s) ** 2).sum()  # gaussian regularity (without constant)
        reg, _ = algo._get_regularity(leaspy.model, individual_parameters)
        self.assertAllClose(reg, expected_reg)

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
        if not isinstance(values, torch.Tensor):
            values = torch.tensor(values, dtype=torch.float32)
        output = algo._get_individual_parameters_patient(leaspy.model,
                                torch.tensor(times, dtype=torch.float32),
                                values,
                                with_jac=algo_kwargs['use_jacobian'])

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

    def test_get_individual_parameters_patient_multivariate_models_ordinal(self, tol=tol, tol_tau=tol_tau):

        times = [70, 80]
        values = [[0, 1, 0, 2], [1, 2, 2, 4]] # no nans (separate test)
        values_ohe = torch.nn.functional.one_hot(torch.tensor(values),
                                                 num_classes=1+self.get_hardcoded_model('logistic_ordinal').model.ordinal_infos['max_level'])

        for (model_name, use_jacobian), expected_dict in {

            ('logistic_ordinal', False): {
                'tau': 74.0180,
                'xi': -1.7808,
                'sources': [-0.3543,  0.5963],
                #'err': [[ 0., -1.,  1., 0.],
                #        [ 0., -2., -1., 0.]]
            },
            ('logistic_ordinal', True): {
                'tau': 73.9865,
                'xi': -1.7801,
                'sources': [-0.3506,  0.5917],
                #'err': [[ 0., -1.,  1., 0.],
                #        [ 0., -2., -1., 0.]]
            },

        }.items():

            individual_parameters, _ = self.get_individual_parameters_patient(model_name,
                                                    times, values_ohe, use_jacobian=use_jacobian,
                                                    noise_model='ordinal')
            self.check_individual_parameters(individual_parameters,
                tau=expected_dict['tau'], tol_tau=tol_tau,
                xi=expected_dict['xi'], tol_xi=tol,
                sources=expected_dict['sources'], tol_sources=tol,
            )

            #self.assertAlmostEqual(torch.sum((err - torch.tensor(expected_dict['err']))**2).item(), 0, delta=tol**2)

    def test_get_individual_parameters_patient_multivariate_models_with_nans_ordinal(self, tol=tol, tol_tau=tol_tau):

        times = [70, 80]
        values = torch.tensor([[0, 1, 2, np.nan], [1, np.nan, 2, 4]])

        # `torch.nn.functional.one_hot` cannot handle nans so we have to temporally filled them with 0 and then set them as nans again
        nan_positions = torch.isnan(values)
        values_fillna_0 = values.clone()
        values_fillna_0[nan_positions] = 0

        values_ohe = torch.nn.functional.one_hot(values_fillna_0.long(),
                                                 num_classes=1+self.get_hardcoded_model('logistic_ordinal').model.ordinal_infos['max_level']).float()
        # put again nans
        values_ohe[nan_positions, :] = float('nan')

        for (model_name, use_jacobian), expected_dict in {

            ('logistic_ordinal', False): {
                'tau': 74.2849,
                'xi': -1.7475,
                'sources': [0.0763, 0.8606],
                #'err': [[ 0., -1., -1., 0.],
                #        [ 0., 0., -1.,  0.]]
            },
            ('logistic_ordinal', True): {
                'tau': 74.2808,
                'xi': -1.7487,
                'sources': [0.0754, 0.8551],
                #'err': [[0., -1., -1., 0.],
                #        [0., 0., -1., 0.]]
            },

        }.items():

            individual_parameters, _ = self.get_individual_parameters_patient(model_name,
                                                    times, values_ohe, use_jacobian=use_jacobian,
                                                    noise_model='ordinal')
            self.check_individual_parameters(individual_parameters,
                tau=expected_dict['tau'], tol_tau=tol_tau,
                xi=expected_dict['xi'], tol_xi=tol,
                sources=expected_dict['sources'], tol_sources=tol,
            )

            #self.assertTrue(torch.eq(torch.isnan(err), nan_positions).all())
            #err[torch.isnan(err)] = 0.

            #self.assertAlmostEqual(torch.sum((err - torch.tensor(expected_dict['err']))**2).item(), 0, delta=tol**2)
