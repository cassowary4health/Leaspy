import unittest

from leaspy.models.abstract_model import AbstractModel


class AbstractModelTest(unittest.TestCase):

    def test_abstract_model_constructor(self):
        """
        Test initialization of abstract model class object
        :return: exit code
        """
        print("Unit-test AbstractModel")

        model = AbstractModel("dummy_abstractmodel")
        self.assertEqual(model.parameters, None)

        # Test the presence of all these essential methods
        main_methods = ['load_parameters', 'get_individual_variable_name', 'compute_sum_squared_tensorized',
                        'compute_individual_attachment_tensorized_mcmc', 'compute_individual_attachment_tensorized',
                        'update_model_parameters', 'update_model_parameters_burn_in',
                        'get_population_realization_names', 'get_individual_realization_names',
                        'compute_regularity_realization', 'compute_regularity_variable', 'get_realization_object']

        present_attributes = [_ for _ in dir(model) if _[:2] != '__']  # Get the present method

        for attribute in main_methods:
            self.assertTrue(attribute in present_attributes)

    #TODO
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
