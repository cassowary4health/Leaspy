import unittest

from leaspy.models.model_factory import ModelFactory
from leaspy.models.multivariate_model import MultivariateModel
from leaspy.models.multivariate_parallel_model import MultivariateParallelModel
from leaspy.models.univariate_model import UnivariateModel


class ModelFactoryTest(unittest.TestCase):

    def test_model_factory_constructor(self, model=None):
        """
        Test initialization of leaspy model.

        Parameters
        ----------
        model : str, optional (default None)
            Name of the model
        """
        if model is None:
            for name in ['univariate', 'linear', 'logistic', 'logistic_parallel']:
                self.test_model_factory_constructor(ModelFactory().model(name))
        else:
            if model.name == 'univariate':
                self.assertEqual(type(model), UnivariateModel)
            elif model.name == 'logistic' or model.name == 'linear':
                self.assertEqual(type(model), MultivariateModel)
            elif model.name == 'logistic_parallel':
                self.assertEqual(type(model), MultivariateParallelModel)

    def test_lower_case(self):
        """Test lower case"""
        name_examples = ['univariate', 'uNIVariaTE', 'UNIVARIATE']
        for name in name_examples:
            model = ModelFactory.model(name)
            # Test model type
            self.assertEqual(type(model), UnivariateModel)

    def test_wrong_arg(self):
        """Test if raise error for wrong argument"""
        # Test if raise ValueError if wrong string arg for name
        wrong_ara_examples = ['lgistic', 'blabla']
        for wrong_arg in wrong_ara_examples:
            self.assertRaises(ValueError, ModelFactory.model, wrong_arg)

        # Test if raise AttributeError if wrong object in name (not a string)
        wrong_ara_examples = [3.8, {'truc': .1}]
        for wrong_arg in wrong_ara_examples:
            self.assertRaises(AttributeError, ModelFactory.model, wrong_arg)
