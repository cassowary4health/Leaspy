import unittest

from src.models.model_factory import ModelFactory
from src.models.univariate_model import UnivariateModel


class ModelFactoryTest(unittest.TestCase):

    def test_factory_return(self):
        model = ModelFactory.model('univariate')
        self.assertEqual(type(model), UnivariateModel)
        self.assertEqual(model.model_parameters['p0'], 0.5)
        self.assertEqual(model.model_parameters['tau_mean'], 70)
        self.assertEqual(model.model_parameters['tau_std'], 5)
        self.assertEqual(model.model_parameters['xi_mean'], -2)
        self.assertEqual(model.model_parameters['xi_std'], 0.1)

