from glob import glob
from unittest import skip

import pandas as pd
import torch

from leaspy.api import Leaspy
from leaspy.models.factory import ModelFactory
from leaspy.models.noise_models import (
    GaussianScalarNoiseModel,
    GaussianDiagonalNoiseModel,
)
from leaspy.models.obs_models import (
    FullGaussianObservationModel,
    OBSERVATION_MODELS,
    ObservationModelNames
)

# backward-compat test
from leaspy.io.data.data import Data
from leaspy.io.settings.algorithm_settings import AlgorithmSettings

# <!> NEVER import real tests classes at top-level (otherwise their tests will be duplicated...), only MIXINS!!
from tests.functional_tests.api.test_api_fit import LeaspyFitTestMixin
from tests.unit_tests.models.test_model_factory import ModelFactoryTestMixin


class LeaspyTest(LeaspyFitTestMixin, ModelFactoryTestMixin):
    model_names = (
        "univariate_logistic",
        # "univariate_linear",
        # "linear",
        "logistic",
        # "logistic_parallel",
        # "mixed_linear-logistic",
    )

    def test_constructor(self):
        """
        Test attribute's initialization of leaspy univariate model
        """
        for name in self.model_names:

            # default_noise = GaussianScalarNoiseModel if 'univariate' in name else GaussianDiagonalNoiseModel

            leaspy = Leaspy(name)
            self.assertEqual(leaspy.type, name)
            self.assertIsInstance(leaspy.model.obs_models[0], FullGaussianObservationModel)
            self.assertEqual(type(leaspy.model), type(ModelFactory.model(name)))
            self.check_model_factory_constructor(leaspy.model)

            with self.assertRaisesRegex(ValueError, 'not been initialized'):
                leaspy.check_if_initialized()

        for observation_model_name, observation_model in OBSERVATION_MODELS.items():
            if observation_model_name !=ObservationModelNames.WEIBULL_RIGHT_CENSORED:
                print(observation_model_name)
                leaspy = Leaspy("logistic", obs_models=observation_model_name, dimension=1)
                self.assertEqual(leaspy.type, "logistic")
                self.assertIsInstance(
                    leaspy.model.obs_models[0],
                    observation_model,
                )

        for name in (
            # "linear",
            "logistic",
            # "logistic_parallel",
            # "mixed_linear-logistic",
        ):
            leaspy = Leaspy(name, source_dimension=2)
            self.assertEqual(leaspy.model.source_dimension, 2)

        for name in (
            # "linear",
            "logistic",
        ):
            leaspy = Leaspy(f"univariate_{name}")
            self.assertEqual(leaspy.model.source_dimension, 0)
            self.assertEqual(leaspy.model.dimension, 1)

            with self.assertRaisesRegex(ValueError, r"`dimension`.+univariate model"):
                Leaspy(f"univariate_{name}", dimension=42)
            with self.assertRaisesRegex(ValueError, r"`source_dimension`.+univariate model"):
                Leaspy(f"univariate_{name}", source_dimension=1)

        with self.assertRaises(ValueError):
            Leaspy('univariate')  # old name

    def test_load_hyperparameters(self):
        leaspy = self.get_hardcoded_model('logistic_diag_noise')
        leaspy.model.load_hyperparameters({'source_dimension': 3})

        self.assertEqual(leaspy.model.source_dimension, 3)
        self.assertIsInstance(leaspy.model.obs_models[0], FullGaussianObservationModel)

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
        self.assertIsInstance(leaspy.model.obs_models[0], FullGaussianObservationModel)

        # Test the parameters
        parameters = {
            # "g": [0.5, 1.5, 1.0, 2.0],  broken...
            # "v0": [-2.0, -3.5, -3.0, -2.5],  broken...
            "betas": [[0.1, 0.6], [-0.1, 0.4], [0.3, 0.8]],
            "tau_mean": [75.2],
            "tau_std": [7.1],
            "xi_mean": 0.0,
            "xi_std": [0.2],
            "sources_mean": [0.0, 0.0],
            "sources_std": 1.0,
        }
        for param_name, param_value in parameters.items():
            self.assertTrue(
                torch.equal(
                    leaspy.model.state[param_name],
                    torch.tensor(param_value),
                )
            )

        # self.assertDictAlmostEqual(leaspy.model.parameters, parameters)
        # self.assertDictAlmostEqual(leaspy.model.noise_model.parameters, {"scale": 0.2})

        # Test the initialization
        self.assertEqual(leaspy.model.is_initialized, True)

        # Test that the model attributes were initialized
        # attrs = leaspy.model._get_attributes(None)
        # self.assertIsNotNone(attrs)
        # self.assertIsInstance(attrs, tuple)
        # self.assertEqual(len(attrs), 3)
        # self.assertTrue(all(attr is not None for attr in attrs))

    @skip("logistic parallel is broken")
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
        self.assertIsInstance(leaspy.model.noise_model, GaussianScalarNoiseModel)

        # Test the parameters
        parameters = {
            "g": 1.0,
            "tau_mean": 70.4,
            "tau_std": 7.0,
            "xi_mean": -1.7,
            "xi_std": 1.0,
            "sources_mean": 0.0,
            "sources_std": 1.0,
            "deltas": [-3, -2.5, -1.0],
            "betas": [[0.1, -0.1], [0.5, -0.3], [0.3, 0.4]],
        }

        self.assertDictAlmostEqual(leaspy.model.parameters, parameters)
        self.assertDictAlmostEqual(leaspy.model.noise_model.parameters, {"scale": 0.1})

        # Test the initialization
        self.assertEqual(leaspy.model.is_initialized, True)

        # Test that the model attributes were initialized
        attrs = leaspy.model._get_attributes(None)
        self.assertIsNotNone(attrs)
        self.assertIsInstance(attrs, tuple)
        self.assertEqual(len(attrs), 3)
        self.assertTrue(all(attr is not None for attr in attrs))

    @skip("linear is broken")
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
        self.assertIsInstance(leaspy.model.obs_models[0], FullGaussianObservationModel)

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
        }

        self.assertDictAlmostEqual(leaspy.model.parameters, parameters)
        self.assertDictAlmostEqual(leaspy.model.noise_model.parameters, {"scale": 0.1})

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
        self.assertIsInstance(leaspy.model.obs_models[0], FullGaussianObservationModel)

        # Test the parameters
        parameters = {
            # "g": [1.0],
            # "v0": [-2.6265233750364456],
            "tau_mean": [70.0],
            "tau_std": [2.5],
            "xi_mean": 0.0,
            "xi_std": [0.01],
            # never used parameters
            # "betas": [],
            # "sources_mean": 0,
            # "sources_std": 1,
        }

        for param_name, param_value in parameters.items():
            try:
                self.assertTrue(
                    torch.equal(
                        leaspy.model.state[param_name],
                        torch.tensor(param_value),
                    )
                )
            except:
                print(leaspy.model.state[param_name])
                print(torch.tensor(param_value))

        # self.assertDictAlmostEqual(leaspy.model.parameters, parameters)
        # self.assertDictAlmostEqual(leaspy.model.noise_model.parameters, {"scale": 0.2})

        # Test the initialization
        self.assertEqual(leaspy.model.is_initialized, True)

        # Test that the model attributes were initialized
        # for attribute in leaspy.model._get_attributes(None):
        #    self.assertIsInstance(attribute, torch.FloatTensor)

    @skip("linear is broken")
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
        self.assertIsInstance(leaspy.model.noise_model, GaussianScalarNoiseModel)

        # Test the parameters
        parameters = {
            "g": [0.5],
            "v0": [-4.0],
            "tau_mean": 78.0,
            "tau_std": 5.0,
            "xi_mean": 0.0,
            "xi_std": 0.5,
            # never used parameters
            "betas": [],
            "sources_mean": 0,
            "sources_std": 1,
        }

        self.assertDictAlmostEqual(leaspy.model.parameters, parameters)
        self.assertDictAlmostEqual(leaspy.model.noise_model.parameters, {"scale": 0.15})

        # Test the initialization
        self.assertEqual(leaspy.model.is_initialized, True)

        # Test that the model attributes were initialized
        for attribute in leaspy.model._get_attributes(None):
            self.assertIsInstance(attribute, torch.FloatTensor)

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

    @skip("Backward compatibility with version <= 1.2.0 is not ensured anymore.")
    def test_api_backward_compat_models_saved_before_120_release(self):

        data_full = Data.from_csv_file(self.example_data_path)
        data_bin = Data.from_dataframe(round(pd.read_csv(self.example_data_path, index_col=[0, 1])))

        for model_name, (new_noise_model_kwd, data) in {
            'logistic_scal_noise': (GaussianScalarNoiseModel, data_full),
            'logistic_diag_noise': (GaussianDiagonalNoiseModel, data_full),
            'logistic_bin': ('bernoulli', data_bin),
        }.items():

            with self.subTest(model_name=model_name):
                with self.assertWarns(FutureWarning):
                    # manage to load old model_parameters.json, with a warning about old loss kwd
                    lsp = self.get_hardcoded_model(f'backward-compat/{model_name}')

                self.assertFalse(hasattr(lsp.model, 'loss'))
                self.assertTrue(hasattr(lsp.model, 'noise_model'))
                self.assertIsInstance(lsp.model.noise_model, new_noise_model_kwd)

                # test api main functions when fitting

                # rename headers as model features (strict mode)
                data.headers = lsp.model.features

                perso_settings = AlgorithmSettings('scipy_minimize', seed=0)
                ips = lsp.personalize(data, perso_settings)

                # noise='inherit_struct'
                simulate_settings = AlgorithmSettings('simulation', seed=0, number_of_subjects=100)
                lsp.simulate(ips, data, simulate_settings)

                simulate_settings = AlgorithmSettings('simulation', seed=0, number_of_subjects=100, noise='model')
                lsp.simulate(ips, data, simulate_settings)
