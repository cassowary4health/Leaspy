import os
import unittest

import numpy as np
import torch

from leaspy.algo.personalize.scipy_minimize import ScipyMinimize
from leaspy.api import Leaspy
from leaspy.inputs.data.data import Data
from leaspy.inputs.settings.algorithm_settings import AlgorithmSettings
from tests import test_data_dir


class ScipyMinimizeTest(unittest.TestCase):

    def test_constructor(self):
        settings = AlgorithmSettings('scipy_minimize')
        algo = ScipyMinimize(settings)

        self.assertEqual(algo.algo_parameters, {'n_iter': 100, 'n_jobs': -1, 'parallel': False})
        self.assertEqual(algo.name, 'scipy_minimize')
        self.assertEqual(algo.seed, None)

    def test_get_model_name(self):
        settings = AlgorithmSettings('scipy_minimize')
        algo = ScipyMinimize(settings)
        algo._set_model_name('name')

        self.assertEqual(algo.model_name, 'name')

    def test_initialize_parameters(self, ):
        settings = AlgorithmSettings('scipy_minimize')
        algo = ScipyMinimize(settings)

        univariate_path = os.path.join(test_data_dir, 'model_parameters', 'example', 'univariate.json')
        univariate_model = Leaspy.load(univariate_path)
        param = algo._initialize_parameters(univariate_model.model)

        self.assertEqual(param, [torch.tensor([-1.0]), torch.tensor([70.0])])

        multivariate_path = os.path.join(test_data_dir, 'model_parameters', 'example', 'logistic.json')
        multivariate_model = Leaspy.load(multivariate_path)
        param = algo._initialize_parameters(multivariate_model.model)
        self.assertEqual(param, [torch.tensor([0.0]), torch.tensor([75.2]), torch.tensor([0.]), torch.tensor([0.])])

    def test_get_attachment(self):
        multivariate_path = os.path.join(test_data_dir, 'model_parameters', 'example', 'logistic.json')
        leaspy = Leaspy.load(multivariate_path)

        settings = AlgorithmSettings('scipy_minimize')
        algo = ScipyMinimize(settings)
        algo._set_model_name('logistic')

        times = torch.tensor([70, 80])
        values = torch.tensor([[0.5, 0.4, 0.4, 0.45], [0.3, 0.3, 0.2, 0.4]])
        individual_parameters = {'xi': torch.zeros(1, dtype=torch.float32),
                                 'tau': torch.tensor(75.2, dtype=torch.float32),
                                 'sources': torch.zeros(2, dtype=torch.float32)}

        err = algo._get_attachment(leaspy.model, times, values, individual_parameters)

        output = torch.tensor([[
            [-0.4705, -0.3278, -0.3103, -0.4477],
            [0.6059,  0.0709,  0.3537,  0.4523]]])
        self.assertEqual(torch.is_tensor(err), True)
        self.assertAlmostEqual(torch.sum((err - output)**2).item(), 0, delta=10e-8)

    def test_get_regularity(self):
        multivariate_path = os.path.join(test_data_dir, 'model_parameters', 'example', 'logistic.json')
        leaspy = Leaspy.load(multivariate_path)

        settings = AlgorithmSettings('scipy_minimize')
        algo = ScipyMinimize(settings)
        algo._set_model_name('logistic')

        individual_parameters = {'xi': torch.zeros(1, dtype=torch.float32),
                                 'tau': torch.tensor(75.2, dtype=torch.float32),
                                 'sources': torch.zeros(2, dtype=torch.float32)}

        reg = algo._get_regularity(leaspy.model, individual_parameters)
        self.assertEqual(torch.is_tensor(reg), True)
        output = torch.tensor([4.0264])
        self.assertAlmostEqual(torch.sum((reg - output)**2).item(), 0, delta=10e-9)

    def test_scipy_minimize_univariate(self):
        data_univariate = Data.from_csv_file(os.path.join(test_data_dir, 'inputs/univariate_data.csv'))
        leaspy_univariate = Leaspy('univariate')
        leaspy_univariate.calibrate(data_univariate, AlgorithmSettings('mcmc_saem', n_iter=100, seed=0))
        result_univariate = leaspy_univariate.personalize(data_univariate,
                                                          AlgorithmSettings('scipy_minimize', seed=0))
        self.assertTrue(torch.allclose(result_univariate.individual_parameters['tau'],
                                       torch.tensor([[70.2492],
                                                     [69.9342],
                                                     [69.9425],
                                                     [70.1529],
                                                     [70.6695],
                                                     [70.5628],
                                                     [70.5859]])
                                       ))
        self.assertTrue(torch.allclose(result_univariate.individual_parameters['xi'],
                                       torch.tensor([[-3.2303],
                                                     [-3.2302],
                                                     [-3.2301],
                                                     [-3.2301],
                                                     [-3.2302],
                                                     [-3.2304],
                                                     [-3.2304]]),
                                       atol=1e-4))

    def test_scipy_minimize_multivariate(self, input_result=None):
        if input_result is None:
            data_multivariate = Data.from_csv_file(os.path.join(test_data_dir, 'inputs/multivariate_data.csv'))
            leaspy_multivariate = Leaspy('logistic')
            leaspy_multivariate.calibrate(data_multivariate, AlgorithmSettings('mcmc_saem', n_iter=100, seed=0))
            result_multivariate = leaspy_multivariate.personalize(data_multivariate,
                                                                  AlgorithmSettings('scipy_minimize', seed=0))
        else:
            result_multivariate = input_result
        self.assertTrue(torch.allclose(result_multivariate.individual_parameters['tau'],
                                       torch.tensor([[76.3797],
                                                     [76.5366],
                                                     [76.2087],
                                                     [76.3607],
                                                     [76.5268]])
                                       ))
        self.assertTrue(torch.allclose(result_multivariate.individual_parameters['xi'],
                                       torch.tensor([[-2.7390e-08],
                                                     [-2.3998e-07],
                                                     [-1.6229e-07],
                                                     [-7.5938e-08],
                                                     [-9.3039e-07]])
                                       ))
        self.assertTrue(torch.allclose(result_multivariate.individual_parameters['sources'],
                                       torch.tensor([[0.0005],
                                                     [-0.0274],
                                                     [0.0096],
                                                     [0.0051],
                                                     [-0.0182]]),
                                       atol=1e-4))

    def test_scipy_minimize_parallel(self):
        data_multivariate = Data.from_csv_file(os.path.join(test_data_dir, 'inputs/multivariate_data.csv'))
        leaspy_multivariate = Leaspy('logistic')
        settings = AlgorithmSettings('mcmc_saem', n_iter=100, seed=0)
        leaspy_multivariate.calibrate(data_multivariate, settings)
        settings = AlgorithmSettings('scipy_minimize', n_iter=100, seed=0, parallel=True, n_jobs=-1)
        result_multivariate = leaspy_multivariate.personalize(data_multivariate, settings)
        self.test_scipy_minimize_multivariate(input_result=result_multivariate)

    def test_get_individual_parameters_patient_multivariate(self):
        multivariate_path = os.path.join(test_data_dir, 'model_parameters', 'example', 'logistic.json')
        leaspy = Leaspy.load(multivariate_path)

        settings = AlgorithmSettings('scipy_minimize', seed=0)
        algo = ScipyMinimize(settings)

        # test tolerance, lack of precision btw different machines...
        # (no exact reproductibility in scipy.optimize.minimize?)
        tol = 5e-3

        # manually initialize seed since it's not done by algo itself (no call to run afterwards)
        algo._initialize_seed(algo.seed)
        self.assertEqual(algo.seed, np.random.get_state()[1][0])

        times = torch.tensor([70, 80])

        # Test without nan
        values = torch.tensor([[0.5, 0.4, 0.4, 0.45], [0.3, 0.3, 0.2, 0.4]])
        individual_parameters = algo._get_individual_parameters_patient(leaspy.model, times, values)
        err = leaspy.model.compute_individual_tensorized(times, individual_parameters) - values

        self.assertAlmostEqual(individual_parameters['tau'], 78.93283994514304, delta=tol)
        self.assertAlmostEqual(individual_parameters['xi'], -0.07679465847751077, delta=tol)
        self.assertAlmostEqual(individual_parameters['sources'].tolist()[0], -0.07733279, delta=tol)
        self.assertAlmostEqual(individual_parameters['sources'].tolist()[1], -0.57428166, delta=tol)

        err_expected = torch.tensor([[
            [-0.4958, -0.3619, -0.3537, -0.4497],
            [0.1650, -0.0948,  0.1361, -0.1050]]])

        self.assertAlmostEqual(torch.sum((err - err_expected)**2).item(), 0, delta=tol)

        # Test with nan
        values = torch.tensor([[0.5, 0.4, 0.4, float('nan')], [0.3, float('nan'), float('nan'), 0.4]])
        individual_parameters = algo._get_individual_parameters_patient(leaspy.model, times, values)
        err = leaspy.model.compute_individual_tensorized(times, individual_parameters) - values

        self.assertAlmostEqual(individual_parameters['tau'], 78.82484683798302, delta=tol)
        self.assertAlmostEqual(individual_parameters['xi'], -0.07808162619234782, delta=tol)
        self.assertAlmostEqual(individual_parameters['sources'].tolist()[0], -0.17007795, delta=tol)
        self.assertAlmostEqual(individual_parameters['sources'].tolist()[1], -0.63483322, delta=tol)

        nan_positions = torch.tensor([
            [False, False, False, True],
            [False, True, True, False]
        ])

        self.assertEqual(torch.all(torch.eq(torch.isnan(err), nan_positions)), True)

        err[err != err] = 0.
        err_expected = torch.tensor([[
            [-0.4957, -0.3613, -0.3516, 0.],
            [0.1718, 0., 0., -0.0796]]])

        self.assertAlmostEqual(torch.sum((err - err_expected)**2).item(), 0, delta=tol)
