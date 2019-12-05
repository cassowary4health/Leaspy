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