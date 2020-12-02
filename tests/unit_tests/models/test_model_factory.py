import unittest

from leaspy.models.model_factory import ModelFactory
from leaspy.models.multivariate_model import MultivariateModel
from leaspy.models.multivariate_parallel_model import MultivariateParallelModel
from leaspy.models.univariate_model import UnivariateModel
from leaspy.models.constant_prediction_model import ConstantModel


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
            for name in ['univariate_logistic', 'univariate_linear','linear', 'logistic', 'logistic_parallel','constant']:
                self.test_model_factory_constructor(ModelFactory().model(name))
        else:
            if model.name == 'univariate_logistic':
                self.assertEqual(type(model), UnivariateModel)
            elif model.name == 'univariate_linear':
                self.assertEqual(type(model), UnivariateModel)
            elif model.name == 'logistic' or model.name == 'linear':
                self.assertEqual(type(model), MultivariateModel)
            elif model.name == 'logistic_parallel':
                self.assertEqual(type(model), MultivariateParallelModel)
            elif model.name == 'constant_prediction':
                self.assertEqual(type(model), ConstantModel)

    def test_lower_case(self):
        """Test lower case"""
        name_examples = ['univariate_logistic', 'uNIVariaTE_LogIsTIc', 'UNIVARIATE_LOGISTIC']
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

    def test_load_hyperparameters(self):
        """Test if kwargs are ok"""
        # --- Univariate
        for name in ('univariate_linear', 'univariate_logistic'):
            model = ModelFactory.model(name, features='test', loss='test')
            self.assertEqual(model.features, 'test')
            self.assertEqual(model.loss, 'test')
            with self.assertRaises(ValueError) as err:
                ModelFactory.model('univariate', source_dimension=2, dimension=2)
                hyperparameters = {'source_dimension': 2, 'dimension': 2}
                self.assertEqual(str(err), "Only <features> and <loss> are valid hyperparameters for an UnivariateModel!"
                                           f"You gave {hyperparameters}.")

        # -- Multivariate
        for name in ('linear', 'logistic', 'logistic_parallel'):
            model = ModelFactory.model(name, features='test', loss='test', source_dimension=2, dimension=2)
            self.assertEqual(model.features, 'test')
            self.assertEqual(model.loss, 'test')
            self.assertEqual(model.source_dimension, 2)
            self.assertEqual(model.source_dimension, 2)
            with self.assertRaises(ValueError) as err:
                ModelFactory.model('univariate', blabla=2)
                hyperparameters = {'blabla': 2}
                self.assertEqual(str(err), "Only <features>, <loss>, <dimension> and <source_dimension> are valid "
                                           f"hyperparameters for an AbstractMultivariateModel! You gave {hyperparameters}.")
