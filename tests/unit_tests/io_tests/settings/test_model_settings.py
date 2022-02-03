from leaspy.io.settings.model_settings import ModelSettings

from tests import LeaspyTestCase


class ModelSettingsTest(LeaspyTestCase):

    def test_model_settings_univariate(self):
        path_to_model_settings = self.get_test_data_path('settings', 'models', 'model_settings_univariate.json')
        model_settings = ModelSettings(path_to_model_settings)

        self.assertEqual(model_settings.name, "univariate")

        self.assertEqual(model_settings.parameters['p0'], 0.3)
        self.assertEqual(model_settings.parameters['tau_mean'], 50)
        self.assertEqual(model_settings.parameters['tau_var'], 2)
        self.assertEqual(model_settings.parameters['xi_mean'], -10)
        self.assertEqual(model_settings.parameters['xi_var'], 0.8)

        self.assertEqual(model_settings.hyperparameters, {})

    def test_model_settings_multivariate(self):
        path_to_model_settings = self.get_test_data_path('settings', 'models', 'model_settings_multivariate.json')
        model_settings = ModelSettings(path_to_model_settings)

        self.assertEqual(model_settings.name, "multivariate")

        parameters = {
            "p0": 0.3,
            "beta": [[0.1, 0.2, 0.3], [0.5, 0.6, 0.9]],
            "tau_mean": 70,
            "tau_var": 50,
            "xi_mean": -2,
            "xi_var": 4,
            "sources_mean": [0.0, 0.1],
            "sources_var": [1.1, 0.9],
            "noise_var": 0.02
        }

        self.assertEqual(model_settings.parameters, parameters)

        hyperparameters = {
            "dimension": 3,
            "source_dimension": 2
        }

        self.assertEqual(model_settings.hyperparameters, hyperparameters)
