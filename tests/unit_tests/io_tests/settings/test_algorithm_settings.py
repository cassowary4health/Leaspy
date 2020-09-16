import os
import json
import unittest

from tests import test_data_dir, default_algo_dir
from leaspy.io.settings.algorithm_settings import AlgorithmSettings


class AlgorithmSettingsTest(unittest.TestCase):

    def test_default_constructor(self):

        # Default constructor
        name = 'scipy_minimize'
        path = os.path.join(default_algo_dir, 'default_' + name + '.json')

        with open(path) as fp:
            json_data = json.load(fp)

        settings = AlgorithmSettings(name)
        self.assertEqual(settings.name, name)
        self.assertEqual(settings.parameters, json_data['parameters'])
        self.assertEqual(settings.seed, None)

    def test_default_constructor_with_kwargs(self):
        # Default constructor with kwargs
        name = 'mcmc_saem'
        path = os.path.join(default_algo_dir, 'default_' + name + '.json')

        with open(path) as fp:
            json_data = json.load(fp)

        settings = AlgorithmSettings(name, n_iter=2100, seed=10)
        json_data['parameters']['n_iter'] = 2100
        json_data['parameters']['n_burn_in_iter'] = int(0.9*2100)
        json_data['parameters']['annealing']['n_iter'] = int(0.5*2100)
        self.assertEqual(settings.name, name)
        self.assertEqual(settings.parameters, json_data['parameters'])
        self.assertEqual(settings.seed, 10)

    def test_constructor_by_loading_json(self):
        # Constructor by loading a json file
        path = os.path.join(test_data_dir, 'io', 'settings', 'mcmc_saem_settings.json')

        with open(path) as fp:
            json_data = json.load(fp)

        settings = AlgorithmSettings.load(path)
        self.assertEqual(settings.name, 'mcmc_saem')
        self.assertEqual(settings.parameters, json_data['parameters'])


