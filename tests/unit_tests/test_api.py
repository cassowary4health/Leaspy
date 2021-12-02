import os
import unittest
from glob import glob

import torch

from leaspy.api import Leaspy
from leaspy.models.model_factory import ModelFactory
from leaspy.models.utils import VALID_LOSSES

from tests import hardcoded_model_path, hardcoded_models_folder, from_fit_models_folder


class LeaspyTest(unittest.TestCase):

    def test_constructor(self):
        """
        Test attribute's initialization of leaspy univariate model
        """

        # <!> we should not import this TestCase at top-level otherwise its tests will be duplicated for this file too!
        from tests.unit_tests.models.test_model_factory import ModelFactoryTest

        for name in ['univariate_logistic', 'univariate_linear', 'linear', 'logistic', 'logistic_parallel',
                     'mixed_linear-logistic']:
            leaspy = Leaspy(name, loss='MSE')
            self.assertEqual(leaspy.type, name)
            self.assertEqual(leaspy.model.loss, 'MSE')
            self.assertEqual(type(leaspy.model), type(ModelFactory.model(name)))
            ModelFactoryTest().test_model_factory_constructor(leaspy.model)

        for loss in VALID_LOSSES:
            leaspy = Leaspy('logistic', loss=loss)
            self.assertEqual(leaspy.type, 'logistic')
            self.assertEqual(leaspy.model.loss, loss)

        for name in ['linear', 'logistic', 'logistic_parallel', 'mixed_linear-logistic']:
            leaspy = Leaspy(name, source_dimension=2)
            self.assertEqual(leaspy.model.source_dimension, 2)

        with self.assertRaises(ValueError):
            Leaspy('univariate_logistic', source_dimension=2)
        with self.assertRaises(ValueError):
            Leaspy('univariate_linear', source_dimension=1)
        with self.assertRaises(ValueError):
            Leaspy('univariate') # old name

    def test_load_hyperparameters(self):

        leaspy = Leaspy.load(hardcoded_model_path('logistic_diag_noise'))
        leaspy.model.load_hyperparameters({'source_dimension': 3, 'loss': 'crossentropy'})

        self.assertEqual(leaspy.model.source_dimension, 3)
        self.assertEqual(leaspy.model.loss, 'crossentropy')

    def test_load_logistic_scalar_noise(self):
        """
        Test the initialization of a logistic model from a json file
        """
        leaspy = Leaspy.load(hardcoded_model_path('logistic_scalar_noise'))

        # Test the name
        self.assertEqual(leaspy.type, 'logistic')
        self.assertEqual(type(leaspy.model), type(ModelFactory.model('logistic')))

        # Test the hyperparameters
        self.assertEqual(leaspy.model.dimension, 4)
        self.assertEqual(leaspy.model.features, ['feature_0', 'feature_1', 'feature_2', 'feature_3'])
        self.assertEqual(leaspy.model.source_dimension, 2)
        self.assertEqual(leaspy.model.loss, 'MSE')

        # Test the parameters
        parameters = {
            "g": [0.5, 1.5, 1.0, 2.0],
            "v0": [-2.0, -3.5, -3.0, -2.5],
            "betas": [[0.1, 0.6], [-0.1, 0.4], [0.3, 0.8]],
            "tau_mean": 75.2,
            "tau_std": 7.1,
            "xi_mean": 0.0,
            "xi_std": 0.2,
            "sources_mean": 0.0,
            "sources_std": 1.0,
            "noise_std": 0.2
        }
        for k, v in parameters.items():
            equality = torch.eq(leaspy.model.parameters[k], torch.tensor(v)).all()
            self.assertTrue(equality)

        # Test the initialization
        self.assertEqual(leaspy.model.is_initialized, True)

    def test_load_logistic_parallel_scalar_noise(self):
        """
        Test the initialization of a logistic parallel model from a json file
        """
        leaspy = Leaspy.load(hardcoded_model_path('logistic_parallel_scalar_noise'))

        # Test the name
        self.assertEqual(leaspy.type, 'logistic_parallel')
        self.assertEqual(type(leaspy.model), type(ModelFactory.model('logistic_parallel')))

        # Test the hyperparameters
        self.assertEqual(leaspy.model.dimension, 4)
        self.assertEqual(leaspy.model.features, ['feature_0', 'feature_1', 'feature_2', 'feature_3'])
        self.assertEqual(leaspy.model.source_dimension, 2)
        self.assertEqual(leaspy.model.loss, 'MSE')

        # Test the parameters
        parameters = {
            "g": 1.0,
            "tau_mean": 70.4,
            "tau_std": 7.0,
            "xi_mean": -1.7,
            "xi_std": 1.0,
            "sources_mean": 0.0,
            "sources_std": 1.0,
            "noise_std": 0.1,
            "deltas": [-3, -2.5, -1.0],
            "betas": [[0.1, -0.1], [0.5, -0.3], [0.3, 0.4]],
        }
        for k, v in parameters.items():
            equality = torch.eq(leaspy.model.parameters[k], torch.tensor(v)).all()
            self.assertTrue(equality)

        # Test the initialization
        self.assertEqual(leaspy.model.is_initialized, True)

    def test_load_linear_scalar_noise(self):
        """
        Test the initialization of a linear model from a json file
        """
        leaspy = Leaspy.load(hardcoded_model_path('linear_scalar_noise'))

        # Test the name
        self.assertEqual(leaspy.type, 'linear')
        self.assertEqual(type(leaspy.model), type(ModelFactory.model('linear')))

        # Test the hyperparameters
        self.assertEqual(leaspy.model.dimension, 4)
        self.assertEqual(leaspy.model.features, ['feature_0', 'feature_1', 'feature_2', 'feature_3'])
        self.assertEqual(leaspy.model.source_dimension, 2)
        self.assertEqual(leaspy.model.loss, 'MSE')

        # Test the parameters
        parameters = {
            "g": [0.5, 0.06, 0.1, 0.3],
            "v0": [-0.5, -0.5, -0.5, -0.5],
            "betas": [[0.1, -0.5], [-0.1, 0.1], [-0.8, -0.1]],
            "tau_mean": 75.2,
            "tau_std": 0.9,
            "xi_mean": 0.0,
            "xi_std": 0.3,
            "sources_mean": 0.0,
            "sources_std": 1.0,
            "noise_std": 0.1,
        }
        for k, v in parameters.items():
            equality = torch.eq(leaspy.model.parameters[k], torch.tensor(v)).all()
            self.assertTrue(equality)

        # Test the initialization
        self.assertEqual(leaspy.model.is_initialized, True)

    def test_load_univariate_logistic(self):
        """
        Test the initialization of a linear model from a json file
        """
        leaspy = Leaspy.load(hardcoded_model_path('univariate_logistic'))

        # Test the name
        self.assertEqual(leaspy.type, 'univariate_logistic')
        self.assertEqual(type(leaspy.model), type(ModelFactory.model('univariate_logistic')))

        # Test the hyperparameters
        self.assertEqual(leaspy.model.features, ['feature'])
        self.assertEqual(leaspy.model.loss, 'MSE')

        # Test the parameters
        parameters = {
            "g": 1.0,
            "tau_mean": 70.0,
            "tau_std": 2.5,
            "xi_mean": -1.0,
            "xi_std": 0.01,
            "noise_std": 0.2
        }

        for k, v in parameters.items():
            equality = torch.eq(leaspy.model.parameters[k], torch.tensor(v)).all()
            self.assertTrue(equality)

        # Test the initialization
        self.assertEqual(leaspy.model.is_initialized, True)

    def test_load_univariate_linear(self):
        """
        Test the initialization of a linear model from a json file
        """
        leaspy = Leaspy.load(hardcoded_model_path('univariate_linear'))

        # Test the name
        self.assertEqual(leaspy.type, 'univariate_linear')
        self.assertEqual(type(leaspy.model), type(ModelFactory.model('univariate_linear')))

        # Test the hyperparameters
        self.assertEqual(leaspy.model.features, ['feature'])
        self.assertEqual(leaspy.model.loss, 'MSE')

        # Test the parameters
        parameters = {
            "g": 0.5,
            "tau_mean": 78.0,
            "tau_std": 5.0,
            "xi_mean": -4.0,
            "xi_std": 0.5,
            "noise_std": 0.15
        }

        for k, v in parameters.items():
            equality = torch.eq(leaspy.model.parameters[k], torch.tensor(v)).all()
            self.assertTrue(equality)

        # Test the initialization
        self.assertEqual(leaspy.model.is_initialized, True)

    def test_load_save_load(self, *, atol=1e-4):
        """
        Test loading, saving and loading again all models (hardcoded and functional)
        """
        # <!> we should not import this TestCase at top-level otherwise its tests will be duplicated for this file too!
        from tests.functional_tests.api.test_api_fit import LeaspyFitTest
        check_consistent_load_save = LeaspyFitTest().check_model_consistency

        # hardcoded models
        for model_path in glob(os.path.join(hardcoded_models_folder, '*.json')):
            with self.subTest(model_path=model_path):
                check_consistent_load_save(Leaspy.load(model_path), model_path, atol=atol)

        # functional models (OK because no direct test on values)
        for model_path in glob(os.path.join(from_fit_models_folder, '*.json')):
            with self.subTest(model_path=model_path):
                check_consistent_load_save(Leaspy.load(model_path), model_path, atol=atol)
