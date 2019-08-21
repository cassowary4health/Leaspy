import unittest

from leaspy.models.model_factory import ModelFactory
from leaspy.models.univariate_model import UnivariateModel
from leaspy._legacy import GaussianDistributionModel


class ModelFactoryTest(unittest.TestCase):

    def test_factory_return(self):
        model = ModelFactory.model('univariate')
        self.assertEqual(type(model), UnivariateModel)
        self.assertEqual(model.model_parameters['p0'], None)
        self.assertEqual(model.model_parameters['tau_mean'], None)
        self.assertEqual(model.model_parameters['tau_var'], None)
        self.assertEqual(model.model_parameters['xi_mean'], None)
        self.assertEqual(model.model_parameters['xi_var'], None)

        model = ModelFactory.model('gaussian_distribution')
        self.assertEqual(type(model), GaussianDistributionModel)
        self.assertEqual(model.model_parameters['intercept_mean'], None)
        self.assertEqual(model.model_parameters['intercept_var'], None)

