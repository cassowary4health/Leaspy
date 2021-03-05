import os
import unittest

import numpy as np
import torch

from leaspy.api import Leaspy
from leaspy.algo.personalize.scipy_minimize import ScipyMinimize
from leaspy.io.settings.algorithm_settings import AlgorithmSettings
from tests import hardcoded_model_path

# test tolerance, lack of precision btw different machines... (no exact reproductibility in scipy.optimize.minimize?)
tol = 3e-3
tol_tau = 1e-2

class ScipyMinimizeTest(unittest.TestCase):

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

        self.assertEqual(algo.algo_parameters, {'n_iter': 100, 'use_jacobian': False, 'n_jobs': 1, "progress_bar": False})
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

        univariate_path = hardcoded_model_path('univariate_logistic')
        univariate_model = Leaspy.load(univariate_path)
        param = algo._initialize_parameters(univariate_model.model)

        self.assertEqual(param, [torch.tensor([-1.0/0.01]), torch.tensor([70.0/2.5])])

        multivariate_path = hardcoded_model_path('logistic')
        multivariate_model = Leaspy.load(multivariate_path)
        param = algo._initialize_parameters(multivariate_model.model)
        self.assertEqual(param, [torch.tensor([0.0]), torch.tensor([75.2/7.1]), torch.tensor([0.]), torch.tensor([0.])])

    def test_get_reconstruction_error(self):
        multivariate_path = hardcoded_model_path('logistic')
        leaspy = Leaspy.load(multivariate_path)

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
        multivariate_path = hardcoded_model_path('logistic')
        leaspy = Leaspy.load(multivariate_path)

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

    def get_individual_parameters_patient(self, model_name, times, values, **algo_kwargs):
        # already a functional test in fact...
        leaspy = Leaspy.load(hardcoded_model_path(model_name))

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
                                                    times, values, use_jacobian=use_jacobian)

            self.check_individual_parameters(individual_parameters,
                tau=expected_dict['tau'], tol_tau=tol_tau,
                xi=expected_dict['xi'], tol_xi=tol
            )

            self.assertAlmostEqual(torch.sum((err - torch.tensor(expected_dict['err']))**2).item(), 0, delta=tol**2)

    def test_get_individual_parameters_patient_multivariate_models(self, tol=tol, tol_tau=tol_tau):

        times = [70, 80]
        values = [[0.5, 0.4, 0.4, 0.45], [0.3, 0.3, 0.2, 0.4]] # no nans (separate test)

        for (model_name, use_jacobian), expected_dict in {

            ('logistic', False): {
                'tau': 78.6033,
                'xi': -0.0919,
                'sources': [0.2967, 0.7069],
                'err': [[-0.4966, -0.3381, -0.3447, -0.4496],
                        [ 0.0969, -0.0049,  0.1711, -0.0611]]
            },
            ('logistic', True): {
                'tau': 78.6033,
                'xi': -0.0918,
                'sources': [0.2997, 0.7000],
                'err': [[-0.4966, -0.3383, -0.3448, -0.4496],
                        [ 0.0981, -0.0056,  0.1706, -0.0616]]
            },

            # TODO? linear, logistic_parallel

        }.items():

            individual_parameters, err = self.get_individual_parameters_patient(model_name,
                                                    times, values, use_jacobian=use_jacobian)

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

            ('logistic', False): {
                'tau': 78.3085,
                'xi': -0.1001,
                'sources': [0.1031, 0.9085],
                'err': [[-0.4963, -0.3286, -0.3354, 0.    ],
                        [ 0.1026,  0.,      0.,    -0.0047]]
            },
            ('logistic', True):  {
                'tau': 78.3124,
                'xi': -0.0998,
                'sources': [0.1073, 0.8990],
                'err': [[-0.4963, -0.3289, -0.3357, 0.    ],
                        [ 0.1039,  0.,      0.,    -0.0060]]
            },

            # TODO? linear, logistic_parallel

        }.items():

            individual_parameters, err = self.get_individual_parameters_patient(model_name,
                                                    times, values, use_jacobian=use_jacobian)

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

            ('logistic', False): {
                'tau': 69.9151,
                'xi': -0.1544,
                'sources': [0.4803, 1.6090],
                'err': [[0.1302, -0.6580,  0.3511, -0.7714],
                        [0.9542, -0.2534, -0.1742, -0.0041]]
            },
            ('logistic', True): {
                'tau': 69.9204,
                'xi': -0.1544,
                'sources': [0.4799, 1.6079],
                'err': [[0.1300, -0.6583,  0.3508, -0.7721],
                        [0.9541, -0.2536, -0.1744, -0.0042]]
            },

        }.items():

            individual_parameters, err = self.get_individual_parameters_patient(model_name,
                                                    times, values, use_jacobian=use_jacobian,
                                                    loss='crossentropy')

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

            ('logistic', False): {
                'tau': 76.3459,
                'xi': -0.0610,
                'sources': [0.2667, 1.0468],
                'err': [[0.0077, -0.8976, 0.0939, 0.    ],
                        [0.6348,  0.,     0.,    -0.2401]]
            },
            ('logistic', True): {
                'tau': 76.3430,
                'xi': -0.0614,
                'sources': [0.2665, 1.0460],
                'err': [[0.0077, -0.8976, 0.0940, 0.    ],
                        [0.6351,  0.,     0.,    -0.2400]]
            },

            # TODO? linear, logistic_parallel

        }.items():

            individual_parameters, err = self.get_individual_parameters_patient(model_name,
                                                    times, values, use_jacobian=use_jacobian,
                                                    loss='crossentropy')

            self.check_individual_parameters(individual_parameters,
                tau=expected_dict['tau'], tol_tau=tol_tau,
                xi=expected_dict['xi'], tol_xi=tol,
                sources=expected_dict['sources'], tol_sources=tol,
            )

            self.assertTrue(torch.eq(torch.isnan(err), nan_positions).all())
            err[torch.isnan(err)] = 0.

            self.assertAlmostEqual(torch.sum((err - torch.tensor(expected_dict['err']))**2).item(), 0, delta=tol**2)
