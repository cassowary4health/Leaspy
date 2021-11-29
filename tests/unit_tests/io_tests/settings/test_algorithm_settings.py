import os
import json

from leaspy.io.settings.algorithm_settings import AlgorithmSettings
from leaspy.io.settings import algo_default_data_dir

from tests import LeaspyTestCase


class AlgorithmSettingsTest(LeaspyTestCase):

    def test_default_constructor(self):

        # Default constructor
        name = 'scipy_minimize'
        path = os.path.join(algo_default_data_dir, 'default_' + name + '.json')

        with open(path) as fp:
            json_data = json.load(fp)

        settings = AlgorithmSettings(name)
        self.assertEqual(settings.name, name)
        self.assertEqual(settings.parameters, json_data['parameters'])
        self.assertEqual(settings.parameters['use_jacobian'], True)
        self.assertEqual(settings.seed, None)

    def test_jacobian_personalization(self):
        settings = AlgorithmSettings('scipy_minimize', use_jacobian=False)
        self.assertEqual(settings.parameters['use_jacobian'], False)

    def test_constant_prediction_algorithm(self):
        settings = AlgorithmSettings('constant_prediction')
        self.assertEqual(settings.name, 'constant_prediction')
        self.assertDictEqual(settings.parameters, {'prediction_type': 'last'})

        for prediction_type in ['last', 'last_known', 'max', 'mean']:
            settings = AlgorithmSettings('constant_prediction', prediction_type=prediction_type)
            self.assertEqual(settings.name, 'constant_prediction')
            self.assertDictEqual(settings.parameters, {'prediction_type': prediction_type})

    def test_lme_fit_algorithm(self):
        settings = AlgorithmSettings('lme_fit')
        self.assertEqual(settings.name, 'lme_fit')

    def test_lme_personalize_algorithm(self):
        settings = AlgorithmSettings('lme_personalize')
        self.assertEqual(settings.name, 'lme_personalize')

    def test_default_constructor_with_kwargs(self):
        # Default constructor with kwargs
        name = 'mcmc_saem'
        path = os.path.join(algo_default_data_dir, 'default_' + name + '.json')

        with open(path) as fp:
            json_data = json.load(fp)

        settings = AlgorithmSettings(name, n_iter=2100, seed=10)
        json_data['parameters']['n_iter'] = 2100
        json_data['parameters']['n_burn_in_iter'] = int(0.9*2100)
        json_data['parameters']['annealing']['n_iter'] = int(0.5*2100)
        self.assertEqual(settings.name, name)
        self.assertEqual(settings.parameters, json_data['parameters'])
        self.assertEqual(settings.seed, 10)
        self.assertEqual(settings.parameters['progress_bar'], True)

    def test_constructor_by_loading_json(self):
        # Constructor by loading a json file
        path = self.test_data_path('settings', 'algo', 'mcmc_saem_settings.json')

        with open(path) as fp:
            json_data = json.load(fp)

        settings = AlgorithmSettings.load(path)
        self.assertEqual(settings.name, 'mcmc_saem')
        self.assertEqual(settings.parameters, json_data['parameters'])
