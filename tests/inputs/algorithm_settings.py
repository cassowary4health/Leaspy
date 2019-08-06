import os
import unittest

from tests import test_data_dir
from src.inputs.settings.algorithm_settings import AlgorithmSettings


class AlgorithmSettingsTest(unittest.TestCase):

    def test_algorithm_settings(self):
        path_to_algorithm_settings = os.path.join(test_data_dir, 'inputs', 'algorithm_settings.json')
        algorithm_settings = AlgorithmSettings(path_to_algorithm_settings)

        self.assertEqual(algorithm_settings.name, "gradient_descent")
        self.assertEqual(algorithm_settings.seed, 0)

        parameters = {
            "n_iter": 100,
            "learning_rate": .005
        }

        self.assertEqual(algorithm_settings.parameters, parameters)

        self.assertEqual(algorithm_settings.output, {"path": None})
