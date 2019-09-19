import unittest

from leaspy.models.model_factory import ModelFactory
from leaspy.models.univariate_model import UnivariateModel

from .univariate_model import UnivariateModelTest
from .logistic_model import MultivariateModelTest
from .logistic_parallel_model import MultivariateParallelModelTest


class ModelFactoryTest(unittest.TestCase):

    def test_model_factory_constructor(self, name=None):
        """
        Test attribute's initialization of leaspy model
        :param name: string - name of the model
        :return: exit code
        """
        print("Unit-test model factory constructor")
        if name is None:
            for name in ['univariate', 'linear', 'logistic', 'logistic_parallel']:
                self.test_model_factory_constructor(name)
        else:
            model = ModelFactory().model(name)

            if name == 'univariate':
                UnivariateModelTest().test_univariate_constructor(model)
            elif name == 'logistic' or name == 'linear':
                MultivariateModelTest().test_constructor_multivariate(model)
            elif name == 'logistic_parallel':
                MultivariateParallelModelTest().test_constructor_multivariate_parallel(model)

    def test_lower_case(self):
        """Test lower case"""
        name_exemples = ['univariate', 'uNIVariaTE', 'UNIVARIATE']
        for name in name_exemples:
            model = ModelFactory.model(name)
            # Test model type
            self.assertEqual(type(model), UnivariateModel)

    def test_wrong_arg(self):
        """Test if raise error for wrong argument"""
        # Test if raise ValueError if wrong string arg for name
        wrong_arg_exemples = ['lgistic', 'blabla']
        for wrong_arg in wrong_arg_exemples:
            self.assertRaises(ValueError, ModelFactory.model, wrong_arg)

        # Test if raise AttributeError if wrong object in name (not a string)
        wrong_arg_exemples = [3.8, {'truc': .1}]
        for wrong_arg in wrong_arg_exemples:
            self.assertRaises(AttributeError, ModelFactory.model, wrong_arg)
