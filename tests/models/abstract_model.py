import os
import unittest

from tests import test_data_dir
from tests import default_data_dir
from src.inputs.model_parameters_reader import ModelParametersReader
from src.models.abstract_model import AbstractModel


class AbstractModelTest(unittest.TestCase):

    def test_constructor(self):
        abstract_model = AbstractModel()
        self.assertEqual(abstract_model.model_parameters, {})

    def test_load_parameters(self):
        abstract_model = AbstractModel()
        path_to_model_parameters = os.path.join(default_data_dir, 'default_univariate_parameters.json')
        reader = ModelParametersReader(path_to_model_parameters)

        abstract_model.load_parameters(reader.parameters)
        self.assertEqual(abstract_model.model_parameters['p0'], 0.5)
        self.assertEqual(abstract_model.model_parameters['tau_mean'], 0)
        self.assertEqual(abstract_model.model_parameters['tau_var'], 1)
        self.assertEqual(abstract_model.model_parameters['xi_mean'], 0)
        self.assertEqual(abstract_model.model_parameters['xi_var'], 1)