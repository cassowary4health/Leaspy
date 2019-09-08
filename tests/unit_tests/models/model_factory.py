import unittest

from leaspy.models.model_factory import ModelFactory
from leaspy.models.univariate_model import UnivariateModel



class ModelFactoryTest(unittest.TestCase):

    def test_factory_return(self):
        model = ModelFactory.model('univariate')
        self.assertEqual(type(model), UnivariateModel)
        self.assertEqual(model.parameters['g'], None)
        self.assertEqual(model.parameters['tau_mean'], None)
        self.assertEqual(model.parameters['tau_std'], None)
        self.assertEqual(model.parameters['xi_mean'], None)
        self.assertEqual(model.parameters['xi_std'], None)
