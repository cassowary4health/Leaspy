from glob import glob

import pandas as pd
import torch

from leaspy.api import Leaspy
from leaspy.models.model_factory import ModelFactory
from leaspy.models.utils.noise_model import NoiseModel

# backward-compat test
from leaspy.io.data.data import Data
from leaspy.io.settings.algorithm_settings import AlgorithmSettings

# <!> NEVER import real tests classes at top-level (otherwise their tests will be duplicated...), only MIXINS!!
from tests.functional_tests.api.test_api_fit import LeaspyFitTest_Mixin
from tests.unit_tests.models.test_model_factory import ModelFactoryTest_Mixin


class LeaspyTest(LeaspyFitTest_Mixin, ModelFactoryTest_Mixin):

    def test_constructor(self):
        """
        Test attribute's initialization of leaspy univariate model
        """
        for name in ['univariate_logistic', 'univariate_linear', 'linear', 'logistic', 'logistic_parallel',
                     'mixed_linear-logistic']:

            default_noise = 'gaussian_scalar' if 'univariate' in name else 'gaussian_diagonal'
            leaspy = Leaspy(name)
            self.assertEqual(leaspy.type, name)
            self.assertEqual(leaspy.model.noise_model, default_noise)
            self.assertEqual(type(leaspy.model), type(ModelFactory.model(name)))
            self.check_model_factory_constructor(leaspy.model)

            with self.assertRaisesRegex(ValueError, 'not been initialized'):
                leaspy.check_if_initialized()

        for noise_model in NoiseModel.VALID_NOISE_STRUCTS:
            leaspy = Leaspy('logistic', noise_model=noise_model)
            self.assertEqual(leaspy.type, 'logistic')
            self.assertEqual(leaspy.model.noise_model, noise_model)

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

        leaspy = self.get_hardcoded_model('logistic_diag_noise')
        leaspy.model.load_hyperparameters({'source_dimension': 3, 'noise_model': 'bernoulli'})

        self.assertEqual(leaspy.model.source_dimension, 3)
        self.assertEqual(leaspy.model.noise_model, 'bernoulli')

    def test_load_logistic_scalar_noise(self):
        """
        Test the initialization of a logistic model from a json file
        """
        leaspy = self.get_hardcoded_model('logistic_scalar_noise')

        # Test the name
        self.assertEqual(leaspy.type, 'logistic')
        self.assertEqual(type(leaspy.model), type(ModelFactory.model('logistic')))

        # Test the hyperparameters
        self.assertEqual(leaspy.model.dimension, 4)
        self.assertEqual(leaspy.model.features, ['Y0', 'Y1', 'Y2', 'Y3'])
        self.assertEqual(leaspy.model.source_dimension, 2)
        self.assertEqual(leaspy.model.noise_model, 'gaussian_scalar')

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

        # Test that the model attributes were initialized
        attrs = leaspy.model._get_attributes(None)
        self.assertIsNotNone(attrs)
        self.assertIsInstance(attrs, tuple)
        self.assertEqual(len(attrs), 3)
        self.assertTrue(all(attr is not None for attr in attrs))

    def test_load_logistic_parallel_scalar_noise(self):
        """
        Test the initialization of a logistic parallel model from a json file
        """
        leaspy = self.get_hardcoded_model('logistic_parallel_scalar_noise')

        # Test the name
        self.assertEqual(leaspy.type, 'logistic_parallel')
        self.assertEqual(type(leaspy.model), type(ModelFactory.model('logistic_parallel')))

        # Test the hyperparameters
        self.assertEqual(leaspy.model.dimension, 4)
        self.assertEqual(leaspy.model.features, ['Y0', 'Y1', 'Y2', 'Y3'])
        self.assertEqual(leaspy.model.source_dimension, 2)
        self.assertEqual(leaspy.model.noise_model, 'gaussian_scalar')

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

        # Test that the model attributes were initialized
        attrs = leaspy.model._get_attributes(None)
        self.assertIsNotNone(attrs)
        self.assertIsInstance(attrs, tuple)
        self.assertEqual(len(attrs), 3)
        self.assertTrue(all(attr is not None for attr in attrs))

    def test_load_linear_scalar_noise(self):
        """
        Test the initialization of a linear model from a json file
        """
        leaspy = self.get_hardcoded_model('linear_scalar_noise')

        # Test the name
        self.assertEqual(leaspy.type, 'linear')
        self.assertEqual(type(leaspy.model), type(ModelFactory.model('linear')))

        # Test the hyperparameters
        self.assertEqual(leaspy.model.dimension, 4)
        self.assertEqual(leaspy.model.features, ['Y0', 'Y1', 'Y2', 'Y3'])
        self.assertEqual(leaspy.model.source_dimension, 2)
        self.assertEqual(leaspy.model.noise_model, 'gaussian_scalar')

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

        # Test that the model attributes were initialized
        attrs = leaspy.model._get_attributes(None)
        self.assertIsNotNone(attrs)
        self.assertIsInstance(attrs, tuple)
        self.assertEqual(len(attrs), 3)
        self.assertTrue(all(attr is not None for attr in attrs))

    def test_load_univariate_logistic(self):
        """
        Test the initialization of a linear model from a json file
        """
        leaspy = self.get_hardcoded_model('univariate_logistic')

        # Test the name
        self.assertEqual(leaspy.type, 'univariate_logistic')
        self.assertEqual(type(leaspy.model), type(ModelFactory.model('univariate_logistic')))

        # Test the hyperparameters
        self.assertEqual(leaspy.model.features, ['Y0'])
        self.assertEqual(leaspy.model.noise_model, 'gaussian_scalar')

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

        # Test that the model attributes were initialized
        self.assertIsInstance(leaspy.model._get_attributes(None), torch.Tensor)

    def test_load_univariate_linear(self):
        """
        Test the initialization of a linear model from a json file
        """
        leaspy = self.get_hardcoded_model('univariate_linear')

        # Test the name
        self.assertEqual(leaspy.type, 'univariate_linear')
        self.assertEqual(type(leaspy.model), type(ModelFactory.model('univariate_linear')))

        # Test the hyperparameters
        self.assertEqual(leaspy.model.features, ['Y0'])
        self.assertEqual(leaspy.model.noise_model, 'gaussian_scalar')

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

        # Test that the model attributes were initialized
        self.assertIsInstance(leaspy.model._get_attributes(None), torch.Tensor)

    def test_load_save_load(self, *, atol=1e-4):
        """
        Test loading, saving and loading again all models (hardcoded and functional)
        """

        # hardcoded models
        for model_path in glob(self.hardcoded_model_path('*.json')):
            with self.subTest(model_path=model_path):
                self.check_model_consistency(Leaspy.load(model_path), model_path, atol=atol)

        # functional models (OK because no direct test on values)
        for model_path in glob(self.from_fit_model_path('*.json')):
            with self.subTest(model_path=model_path):
                self.check_model_consistency(Leaspy.load(model_path), model_path, atol=atol)

    def test_api_backward_compat_models_saved_before_120_release(self):

        data_full = Data.from_csv_file(self.example_data_path)
        data_bin = Data.from_dataframe(round(pd.read_csv(self.example_data_path, index_col=[0,1])))

        for model_name, (new_noise_model_kwd, data) in {
            'logistic_scal_noise': ('gaussian_scalar', data_full),
            'logistic_diag_noise': ('gaussian_diagonal', data_full),
            'logistic_bin': ('bernoulli', data_bin),
        }.items():

            with self.subTest(model_name=model_name):
                with self.assertWarns(FutureWarning):
                    # manage to load old model_parameters.json, with a warning about old loss kwd
                    lsp = self.get_hardcoded_model(f'backward-compat/{model_name}')

                self.assertFalse(hasattr(lsp.model, 'loss'))
                self.assertTrue(hasattr(lsp.model, 'noise_model'))
                self.assertEqual(lsp.model.noise_model, new_noise_model_kwd)

                # test api main functions when fitting

                # rename headers as model features (strict mode)
                data.headers = lsp.model.features

                perso_settings = AlgorithmSettings('scipy_minimize', seed=0)
                ips = lsp.personalize(data, perso_settings)

                simulate_settings = AlgorithmSettings('simulation', seed=0, number_of_subjects=100)  # noise='inherit_struct'
                r = lsp.simulate(ips, data, simulate_settings)

                simulate_settings = AlgorithmSettings('simulation', seed=0, number_of_subjects=100, noise='model')
                r = lsp.simulate(ips, data, simulate_settings)
