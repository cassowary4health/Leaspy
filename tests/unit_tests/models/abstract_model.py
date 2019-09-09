import unittest

from leaspy.models.abstract_model import AbstractModel


class AbstractModelTest(unittest.TestCase):

    #TODO

    def test_constructor(self):
        abstract_model = AbstractModel("dummy_abstractmodel")
        self.assertEqual(abstract_model.parameters, None)

    """
    def test_load_parameters(self):
        abstract_model = AbstractModel()
        path_to_model_parameters = os.path.join(default_data_dir, 'model_settings_univariate.json')
        reader = ModelSettings(path_to_model_parameters)

        abstract_model.load_parameters(reader.parameters)
        self.assertEqual(abstract_model.model_parameters['p0'], [0.5])
        self.assertEqual(abstract_model.model_parameters['tau_mean'], 0)
        self.assertEqual(abstract_model.model_parameters['tau_var'], 1)
        self.assertEqual(abstract_model.model_parameters['xi_mean'], 0)
        self.assertEqual(abstract_model.model_parameters['xi_var'], 1)"""