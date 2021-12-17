from leaspy.models import all_models, UnivariateModel
from leaspy.models.model_factory import ModelFactory

from tests import LeaspyTestCase

class ModelFactoryTest_Mixin(LeaspyTestCase):

    def check_model_factory_constructor(self, model):
        """
        Test initialization of leaspy model.

        Parameters
        ----------
        model : str, optional (default None)
            Name of the model
        """
        # valid name (preconditon)
        self.assertIn(model.name, all_models)
        self.assertEqual(type(model), all_models[model.name])

class ModelFactoryTest(ModelFactoryTest_Mixin):

    def test_model_factory_constructor(self):
        for name in all_models.keys():
            with self.subTest(model_name=name):
                self.check_model_factory_constructor(model=ModelFactory.model(name))

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
        wrong_arg_examples = ['lgistic', 'blabla']
        for wrong_arg in wrong_arg_examples:
            self.assertRaises(ValueError, ModelFactory.model, wrong_arg)

        # Test if raise AttributeError if wrong object in name (not a string)
        wrong_arg_examples = [3.8, {'truc': .1}]
        for wrong_arg in wrong_arg_examples:
            self.assertRaises(ValueError, ModelFactory.model, wrong_arg)

    def test_load_hyperparameters(self):
        """Test if kwargs are ok"""
        # --- Univariate
        for name in ('univariate_linear', 'univariate_logistic'):
            with self.subTest(model_name=name):
                model = ModelFactory.model(name, features=['t1','t2','t3'], noise_model='gaussian_scalar')
                self.assertEqual(model.features, ['t1','t2','t3'])
                self.assertEqual(model.noise_model, 'gaussian_scalar')
                with self.assertRaises(ValueError) as err:
                    ModelFactory.model(name, source_dimension=2, dimension=3)
                    hyperparameters = {'source_dimension': 2, 'dimension': 3}
                    self.assertEqual(str(err), "Only ('features', 'loss', 'noise_model') are valid hyperparameters for UnivariateModel. "
                                               f"You gave {hyperparameters}.")

        # -- Multivariate
        for name in ('linear', 'logistic', 'logistic_parallel'):
            with self.subTest(model_name=name):
                model = ModelFactory.model(name, features=['t1','t2','t3'], noise_model='gaussian_diagonal', source_dimension=2, dimension=3)
                self.assertEqual(model.features, ['t1','t2','t3'])
                self.assertEqual(model.noise_model, 'gaussian_diagonal')
                self.assertEqual(model.dimension, 3) # TODO: automatic from length of features?
                self.assertEqual(model.source_dimension, 2)
                with self.assertRaises(ValueError) as err:
                    ModelFactory.model(name, blabla=2)
                    hyperparameters = {'blabla': 2}
                    self.assertEqual(str(err), "Only ('features', 'dimension', 'source_dimension', 'loss', 'noise_model') are valid "
                                               f"hyperparameters for AbstractMultivariateModel. You gave {hyperparameters}.")

    def test_bad_noise_model_or_old_loss(self):
        # raise if invalid loss
        with self.assertRaises(Exception):
            ModelFactory.model('logistic', noise_model='bad_noise_model')

        # warns about old loss
        with self.assertWarns(FutureWarning):
            ModelFactory.model('logistic', loss='MSE_diag_noise')

        # raise if bad old loss
        with self.assertRaises(Exception):
            ModelFactory.model('logistic', loss='bad_old_loss')
