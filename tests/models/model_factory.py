import unittest

from src.models.model_factory import ModelFactory
from src.models.univariate_model import UnivariateModel
from src.models.gaussian_distribution_model import GaussianDistributionModel


class ModelFactoryTest(unittest.TestCase):

    def test_factory_return(self):
        model = ModelFactory.model('univariate')
        self.assertEqual(type(model), UnivariateModel)
        self.assertEqual(model.model_parameters['p0'], 0.5)
        self.assertEqual(model.model_parameters['tau_mean'], 70)
        self.assertEqual(model.model_parameters['tau_std'], 5)
        self.assertEqual(model.model_parameters['xi_mean'], -2)
        self.assertEqual(model.model_parameters['xi_std'], 0.1)

        model = ModelFactory.model('gaussian_distribution')
        self.assertEqual(type(model), GaussianDistributionModel)
        self.assertEqual(model.model_parameters['mean'], 0)
        self.assertEqual(model.model_parameters['std'], 1)

