import os
import unittest

import numpy as np
import torch

from leaspy.algo.personalize.scipy_minimize import ScipyMinimize
from leaspy.api import Leaspy
from leaspy.io.settings.algorithm_settings import AlgorithmSettings
from tests import test_data_dir


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

    def test_get_reconstruction_error(self):
        multivariate_path = os.path.join(test_data_dir, 'model_parameters', 'example', 'logistic.json')
        leaspy = Leaspy.load(multivariate_path)

        settings = AlgorithmSettings('scipy_minimize')
        algo = ScipyMinimize(settings)
        #algo._set_model_name('logistic')

        times = torch.tensor([70, 80])
        values = torch.tensor([[0.5, 0.4, 0.4, 0.45], [0.3, 0.3, 0.2, 0.4]])

        z = [0.0, 75.2, 0., 0.]
        individual_parameters = algo._pull_individual_parameters(z, leaspy.model)

        err = algo._get_reconstruction_error(leaspy.model, times, values, individual_parameters)

        output = torch.tensor([
            [-0.4705, -0.3278, -0.3103, -0.4477],
            [0.6059,  0.0709,  0.3537,  0.4523]])
        self.assertEqual(torch.is_tensor(err), True)
        self.assertAlmostEqual(torch.sum((err - output)**2).item(), 0, delta=1e-8)

    def test_get_regularity(self):
        multivariate_path = os.path.join(test_data_dir, 'model_parameters', 'example', 'logistic.json')
        leaspy = Leaspy.load(multivariate_path)

        settings = AlgorithmSettings('scipy_minimize')
        algo = ScipyMinimize(settings)
        #algo._set_model_name('logistic')

        z = [0.0, 75.2, 0., 0.]
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


    def test_get_individual_parameters_patient_univariate(self):
        # TODO
        return 0

    def test_get_individual_parameters_patient_multivariate(self):
        multivariate_path = os.path.join(test_data_dir, 'model_parameters', 'example', 'logistic.json')
        leaspy = Leaspy.load(multivariate_path)

        settings = AlgorithmSettings('scipy_minimize', seed=0)
        algo = ScipyMinimize(settings)

        # test tolerance, lack of precision btw different machines... (no exact reproductibility in scipy.optimize.minimize?)
        tol = 5e-3

        # manually initialize seed since it's not done by algo itself (no call to run afterwards)
        algo._initialize_seed(algo.seed)
        self.assertEqual(algo.seed, np.random.get_state()[1][0])

        times = torch.tensor([70, 80])

        # Test without nan
        values = torch.tensor([[0.5, 0.4, 0.4, 0.45], [0.3, 0.3, 0.2, 0.4]])
        output = algo._get_individual_parameters_patient(leaspy.model, times, values)
        individual_parameters = output[0]
        err = output[1]

        self.check_individual_parameters(individual_parameters,
            tau=78.93283994514304, tol_tau=tol,
            xi=-0.07679465847751077, tol_xi=tol,
            sources=[-0.07733279, -0.57428166], tol_sources=tol
        )

        err_expected = torch.tensor([[
            [-0.4958, -0.3619, -0.3537, -0.4497],
            [0.1650, -0.0948,  0.1361, -0.1050]]])

        self.assertAlmostEqual(torch.sum((err - err_expected)**2).item(), 0, delta=tol)

        # Test with nan
        values = torch.tensor([[0.5, 0.4, 0.4, float('nan')], [0.3, float('nan'), float('nan'), 0.4]])
        output = algo._get_individual_parameters_patient(leaspy.model, times, values)
        individual_parameters = output[0]
        err = output[1]

        self.check_individual_parameters(individual_parameters,
            tau=78.82484683798302, tol_tau=tol,
            xi=-0.07808162619234782, tol_xi=tol,
            sources=[-0.17007795, -0.63483322], tol_sources=tol
        )

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

    def test_get_individual_parameters_patient_multivariate_with_jacobian(self):
        multivariate_path = os.path.join(test_data_dir, 'model_parameters', 'example', 'logistic.json')
        leaspy = Leaspy.load(multivariate_path)

        settings = AlgorithmSettings('scipy_minimize', seed=0, use_jacobian=True)
        algo = ScipyMinimize(settings)

        # test tolerance, lack of precision btw different machines... (no exact reproductibility in scipy.optimize.minimize?)
        tol = 5e-3
        tol_tau = 0.01

        # manually initialize seed since it's not done by algo itself (no call to run afterwards)
        algo._initialize_seed(algo.seed)
        self.assertEqual(algo.seed, np.random.get_state()[1][0])

        times = torch.tensor([70, 80])

        # Test without nan
        values = torch.tensor([[0.5, 0.4, 0.4, 0.45], [0.3, 0.3, 0.2, 0.4]])
        output = algo._get_individual_parameters_patient(leaspy.model, times, values)
        individual_parameters = output[0]
        err = output[1]

        self.check_individual_parameters(individual_parameters,
            tau=78.93283994514304, tol_tau=tol_tau,
            xi=-0.07679465847751077, tol_xi=tol,
            sources=[-0.07733279, -0.57428166], tol_sources=tol
        )

        err_expected = torch.tensor([[
            [-0.4958, -0.3619, -0.3537, -0.4497],
            [0.1650, -0.0948,  0.1361, -0.1050]]])

        self.assertAlmostEqual(torch.sum((err - err_expected)**2).item(), 0, delta=tol)

        # Test with nan
        values = torch.tensor([[0.5, 0.4, 0.4, float('nan')], [0.3, float('nan'), float('nan'), 0.4]])
        output = algo._get_individual_parameters_patient(leaspy.model, times, values)
        individual_parameters = output[0]
        err = output[1]

        self.check_individual_parameters(individual_parameters,
            tau=78.83, tol_tau=tol_tau,
            xi=-0.07808162619234782, tol_xi=tol,
            sources=[-0.17007795, -0.63483322], tol_sources=tol
        )

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

    def test_get_individual_parameters_patient_crossentropy(self):
        multivariate_path = os.path.join(test_data_dir, 'model_parameters', 'example', 'logistic.json')
        leaspy = Leaspy.load(multivariate_path)

        settings = AlgorithmSettings('scipy_minimize', seed=0, loss="crossentropy")
        algo = ScipyMinimize(settings)

        # test tolerance, lack of precision btw different machines... (no exact reproductibility in scipy.optimize.minimize?)
        tol = 5e-3
        tol_tau = 0.01

        # manually initialize seed since it's not done by algo itself (no call to run afterwards)
        algo._initialize_seed(algo.seed)
        self.assertEqual(algo.seed, np.random.get_state()[1][0])

        times = torch.tensor([70, 80])

        # Test without nan
        values = torch.tensor([[0., 1., 0., 1.], [0., 1., 1., 1.]])
        output = algo._get_individual_parameters_patient(leaspy.model, times, values)
        individual_parameters = output[0]
        err = output[1]

        self.check_individual_parameters(individual_parameters,
            tau=69.91258208274421, tol_tau=tol_tau,
            xi=-0.1446537485712681, tol_xi=tol,
            sources=[-0.16517799, -0.82381726], tol_sources=tol
        )

        err_expected = torch.tensor([[
            [ 0.3184, -0.8266,  0.2943, -0.8065],
            [ 0.9855, -0.4526, -0.2114, -0.0048]]])

        self.assertAlmostEqual(torch.sum((err - err_expected)**2).item(), 0, delta=tol)

        # Test with nan
        values = torch.tensor([[0., 1., 0., float('nan')], [0., float('nan'), float('nan'), 1.]])
        output = algo._get_individual_parameters_patient(leaspy.model, times, values)
        individual_parameters = output[0]
        err = output[1]

        self.check_individual_parameters(individual_parameters,
            tau=76.57318992643758, tol_tau=tol,
            xi=-0.06489393539830259, tol_xi=tol,
            sources=[-0.08735905, -0.37562645], tol_sources=tol
        )

        nan_positions = torch.tensor([
            [False, False, False, True],
            [False, True, True, False]
        ])

        self.assertEqual(torch.all(torch.eq(torch.isnan(err), nan_positions)), True)

        err[err != err] = 0.
        err_expected = torch.tensor([[
            [ 0.0150, -0.9417,  0.0752,     0.],
            [ 0.7702,    0.,     0., -0.3218]]])

        self.assertAlmostEqual(torch.sum((err - err_expected)**2).item(), 0, delta=tol)

    def test_get_individual_parameters_patient_crossentropy_with_jacobian(self):
        multivariate_path = os.path.join(test_data_dir, 'model_parameters', 'example', 'logistic.json')
        leaspy = Leaspy.load(multivariate_path)

        settings = AlgorithmSettings('scipy_minimize', seed=0, loss="crossentropy", use_jacobian=True)
        algo = ScipyMinimize(settings)

        # test tolerance, lack of precision btw different machines... (no exact reproductibility in scipy.optimize.minimize?)
        tol = 5e-3
        tol_tau = 0.01

        # manually initialize seed since it's not done by algo itself (no call to run afterwards)
        algo._initialize_seed(algo.seed)
        self.assertEqual(algo.seed, np.random.get_state()[1][0])

        times = torch.tensor([70, 80])

        # Test without nan
        values = torch.tensor([[0., 1., 0., 1.], [0., 1., 1., 1.]])
        output = algo._get_individual_parameters_patient(leaspy.model, times, values)
        individual_parameters = output[0]
        err = output[1]

        self.check_individual_parameters(individual_parameters,
            tau=69.91258208274421, tol_tau=tol_tau,
            xi=-0.1446537485712681, tol_xi=tol,
            sources=[-0.16517799, -0.82381726], tol_sources=tol
        )

        err_expected = torch.tensor([[
            [ 0.3184, -0.8266,  0.2943, -0.8065],
            [ 0.9855, -0.4526, -0.2114, -0.0048]]])

        self.assertAlmostEqual(torch.sum((err - err_expected)**2).item(), 0, delta=tol)

        # Test with nan
        values = torch.tensor([[0., 1., 0., float('nan')], [0., float('nan'), float('nan'), 1.]])
        output = algo._get_individual_parameters_patient(leaspy.model, times, values)
        individual_parameters = output[0]
        err = output[1]

        self.check_individual_parameters(individual_parameters,
            tau=76.57318992643758, tol_tau=tol,
            xi=-0.06489393539830259, tol_xi=tol,
            sources=[-0.08735905, -0.37562645], tol_sources=tol
        )

        nan_positions = torch.tensor([
            [False, False, False, True],
            [False, True, True, False]
        ])

        self.assertEqual(torch.all(torch.eq(torch.isnan(err), nan_positions)), True)

        err[err != err] = 0.
        err_expected = torch.tensor([[
            [ 0.0150, -0.9417,  0.0752,     0.],
            [ 0.7702,    0.,     0., -0.3218]]])

        self.assertAlmostEqual(torch.sum((err - err_expected)**2).item(), 0, delta=tol)
