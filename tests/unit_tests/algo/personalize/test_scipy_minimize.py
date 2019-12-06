import os
import unittest
import torch

from tests import test_data_dir
from leaspy.api import Leaspy
from leaspy.algo.personalize.scipy_minimize import ScipyMinimize
from leaspy.inputs.settings.algorithm_settings import AlgorithmSettings


class ScipyMinimizeTest(unittest.TestCase):


    def test_constructor(self):
        settings = AlgorithmSettings('scipy_minimize')
        algo = ScipyMinimize(settings)

        self.assertEqual(algo.algo_parameters, {'n_iter': 100})
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

        self.assertEqual(param, [torch.Tensor([-1.0]), torch.Tensor([70.0])])

        multivariate_path = os.path.join(test_data_dir, 'model_parameters', 'example', 'logistic.json')
        multivariate_model = Leaspy.load(multivariate_path)
        param = algo._initialize_parameters(multivariate_model.model)
        self.assertEqual(param, [torch.Tensor([0.0]), torch.Tensor([75.2]), torch.Tensor([0.]), torch.Tensor([0.])])

    def test_get_attachement(self):
        multivariate_path = os.path.join(test_data_dir, 'model_parameters', 'example', 'logistic.json')
        leaspy = Leaspy.load(multivariate_path)

        settings = AlgorithmSettings('scipy_minimize')
        algo = ScipyMinimize(settings)
        algo._set_model_name('logistic')

        times = torch.Tensor([70, 80])
        values = torch.Tensor([[0.5, 0.4, 0.4, 0.45], [0.3, 0.3, 0.2, 0.4]])
        individual_parameters = [torch.Tensor([0.0]), torch.Tensor([75.2]), torch.Tensor([0.]), torch.Tensor([0.])]

        err = algo._get_attachement(leaspy.model, times, values, individual_parameters)

        output = torch.Tensor([[
            [-0.4705, -0.3278, -0.3103, -0.4477],
            [ 0.6059,  0.0709,  0.3537,  0.4523]]])
        self.assertEqual(torch.is_tensor(err), True)
        self.assertAlmostEqual(torch.sum((err - output)**2), 0, delta=10e-8)


    def test_get_regularity(self):
        multivariate_path = os.path.join(test_data_dir, 'model_parameters', 'example', 'logistic.json')
        leaspy = Leaspy.load(multivariate_path)

        settings = AlgorithmSettings('scipy_minimize')
        algo = ScipyMinimize(settings)
        algo._set_model_name('logistic')

        individual_parameters = [torch.Tensor([0.0]), torch.Tensor([75.2]), torch.Tensor([0.]), torch.Tensor([0.])]

        reg = algo._get_regularity(leaspy.model, individual_parameters)
        self.assertEqual(torch.is_tensor(reg), True)
        output = torch.Tensor([4.0264])
        self.assertAlmostEqual(torch.sum((reg - output)**2), 0, delta=10e-9)

    def test_get_individual_parameters_patient_univariate(self):
        # TODO
        return 0

    def test_get_individual_parameters_patient_multivariate(self):
        multivariate_path = os.path.join(test_data_dir, 'model_parameters', 'example', 'logistic.json')
        leaspy = Leaspy.load(multivariate_path)

        settings = AlgorithmSettings('scipy_minimize')
        algo = ScipyMinimize(settings)

        times = torch.Tensor([70, 80])

        # Test without nan
        values = torch.Tensor([[0.5, 0.4, 0.4, 0.45], [0.3, 0.3, 0.2, 0.4]])
        output = algo._get_individual_parameters_patient(leaspy.model, times, values)
        individual_parameters = output[0]
        err = output[1]

        self.assertAlmostEqual(individual_parameters[0], 78.93283994514304, delta=10e-8)
        self.assertAlmostEqual(individual_parameters[1], -0.07679465847751077, delta=10e-8)
        self.assertAlmostEqual(individual_parameters[2].tolist()[0], -0.07733279, delta=10e-8)
        self.assertAlmostEqual(individual_parameters[2].tolist()[1], -0.57428166, delta=10e-8)

        err_expected = torch.Tensor([[
            [-0.4958, -0.3619, -0.3537, -0.4497],
            [ 0.1650, -0.0948,  0.1361, -0.1050]]])
        self.assertAlmostEqual(torch.sum((err - err_expected)**2), 0, delta=10e-9)

        # Test with nan
        values = torch.Tensor([[0.5, 0.4, 0.4, float('nan')], [0.3, float('nan'), float('nan'), 0.4]])
        output = algo._get_individual_parameters_patient(leaspy.model, times, values)
        individual_parameters = output[0]
        err = output[1]

        self.assertAlmostEqual(individual_parameters[0], 78.82484683798302, delta=10e-8)
        self.assertAlmostEqual(individual_parameters[1], -0.07808162619234782, delta=10e-8)
        self.assertAlmostEqual(individual_parameters[2].tolist()[0], -0.17007795, delta=10e-8)
        self.assertAlmostEqual(individual_parameters[2].tolist()[1], -0.63483322, delta=10e-8)

        nan_positions = torch.Tensor([
            [False, False, False, True],
            [False, True, True, False]
        ])

        self.assertEqual(torch.all(torch.eq(torch.isnan(err), nan_positions)), True)

        err[err != err] = 0.
        err_expected = torch.Tensor([[
            [-0.4957, -0.3613, -0.3516, 0.],
            [0.1718, 0., 0., -0.0796]]])
        self.assertAlmostEqual(torch.sum((err - err_expected)**2), 0, delta=10e-8)

