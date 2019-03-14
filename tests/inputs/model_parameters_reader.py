import os
import unittest

from tests import test_data_dir
from src.inputs.model_parameters_reader import ModelParametersReader


class ModelParametersReaderTest(unittest.TestCase):

    def test_model_parameters(self):
        path_to_model_parameters = os.path.join(test_data_dir, 'model_parameters.json')
        model_parameters = ModelParametersReader(path_to_model_parameters)
        self.assertEqual(model_parameters.model_type, "univariate")
        self.assertEqual(model_parameters.parameters['p0'], 0.3)
        self.assertEqual(model_parameters.parameters['tau_mean'], 50)
        self.assertEqual(model_parameters.parameters['tau_std'], 2)
        self.assertEqual(model_parameters.parameters['xi_mean'], -10)
        self.assertEqual(model_parameters.parameters['xi_std'], 0.8)
