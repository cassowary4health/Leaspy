import unittest

from leaspy.models.model_factory import ModelFactory
from leaspy.models.univariate_model import UnivariateModel


class ModelFactoryTest(unittest.TestCase):

    def test_model(self):
        """Test model static method"""
        # Test lower case
        name_exemples = ['univariate', 'uNIVariaTE', 'UNIVARIATE']
        for name in name_exemples:
            model = ModelFactory.model(name)
            # Test model type
            self.assertEqual(type(model), UnivariateModel)

        # Test initialization of UnivariateModel
        self.assertEqual(model.parameters['g'], None)
        self.assertEqual(model.parameters['tau_mean'], None)
        self.assertEqual(model.parameters['tau_std'], None)
        self.assertEqual(model.parameters['xi_mean'], None)
        self.assertEqual(model.parameters['xi_std'], None)

        # Test if raise ValueError if wrong string arg for name
        wrong_arg_exemples = ['lgistic', 'blabla']
        for wrong_arg in wrong_arg_exemples:
            self.assertRaises(ValueError, ModelFactory.model, wrong_arg)

        # Test if raise AttributeError if wrong object in name (not a string)
        wrong_arg_exemples = [3.8, {'truc': .1}]
        for wrong_arg in wrong_arg_exemples:
            self.assertRaises(AttributeError, ModelFactory.model, wrong_arg)
