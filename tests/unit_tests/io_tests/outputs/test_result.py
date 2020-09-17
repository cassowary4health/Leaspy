import filecmp
import os
import unittest

import pandas as pd
import torch
from numpy import allclose

from leaspy import Data, Leaspy, Result
from tests import example_data_path, test_data_dir


class ResultTest(unittest.TestCase):

    def setUp(self, get=False):
        # Inputs
        data = Data.from_csv_file(example_data_path)

        cofactors_path = os.path.join(test_data_dir,
                                      "io",
                                      "data",
                                      "data_tiny_covariate.csv")
        self.cofactors = pd.read_csv(cofactors_path, index_col=0)
        self.cofactors.index = [str(i) for i in self.cofactors.index.values]
        data.load_cofactors(self.cofactors, ['Treatments'])

        load_individual_parameters_path = os.path.join(test_data_dir,
                                                       "individual_parameters",
                                                       "data_tiny-individual_parameters.json")
        self.results = Result.load_result(data, load_individual_parameters_path)
        if get:
            return self.results

    def test_constructor(self):
        self.assertTrue(type(self.results.data) == Data)
        self.assertTrue(type(self.results.individual_parameters) == dict)
        self.assertEqual(list(self.results.individual_parameters.keys()), ['tau', 'xi', 'sources'])
        for key in self.results.individual_parameters.keys():
            self.assertEqual(len(self.results.individual_parameters[key]), 17)
        self.assertEqual(self.results.noise_std, None)

    def test_save_individual_parameters_json(self):
        path_original = os.path.join(test_data_dir, "individual_parameters", "data_tiny-individual_parameters.json")
        path_copy = os.path.join(test_data_dir, "individual_parameters", "data_tiny-individual_parameters-copy.json")
        self.results.save_individual_parameters_json(path_copy)
        self.assertTrue(filecmp.cmp(path_original, path_copy, shallow=False))

        # Test to save only several subjects
        idx = ['116', '142', '169']
        path_original = os.path.join(test_data_dir, "individual_parameters",
                                     "data_tiny-individual_parameters-3subjects.json")
        path_copy = os.path.join(test_data_dir, "individual_parameters",
                                 "data_tiny-individual_parameters-3subjects-copy.json")
        self.results.save_individual_parameters_json(path_copy, idx)
        self.assertTrue(filecmp.cmp(path_original, path_copy, shallow=False))

        # test if run with an **args of json.dump
        path_indent_4 = os.path.join(test_data_dir, "individual_parameters",
                                     "data_tiny-individual_parameters-indent_4.json")
        self.results.save_individual_parameters_json(path_indent_4, idx, indent=4)

    def test_save_individual_parameters_csv(self):
        individual_parameters_path = os.path.join(test_data_dir, "individual_parameters",
                                                  "data_tiny-individual_parameters.csv")
        path_original = os.path.join(test_data_dir, "individual_parameters",
                                     "data_tiny-individual_parameters-original.csv")

        self.results.save_individual_parameters_csv(individual_parameters_path)
        self.assertTrue(filecmp.cmp(path_original, path_original, shallow=False))

        # Test to save only several subjects
        idx = ['116', '142', '169']
        path_original = os.path.join(test_data_dir, "individual_parameters",
                                     "data_tiny-individual_parameters-3subjects.csv")
        path_copy = os.path.join(test_data_dir, "individual_parameters",
                                 "data_tiny-individual_parameters-3subjects-copy.csv")
        self.results.save_individual_parameters_csv(path_copy, idx)
        self.assertTrue(filecmp.cmp(path_original, path_copy, shallow=False))

        for idx in ('116', 116, ('116',), ('116', '142')):
            self.assertRaises(TypeError, self.results.save_individual_parameters_csv,
                              individual_parameters_path, idx=idx)

    def test_save_individual_parameters_torch(self):
        path_original = os.path.join(test_data_dir, "individual_parameters", "data_tiny-individual_parameters.pt")
        path_copy = os.path.join(test_data_dir, "individual_parameters", "data_tiny-individual_parameters-copy.pt")
        self.results.save_individual_parameters_torch(path_copy)
        self.assertTrue(filecmp.cmp(path_original, path_copy, shallow=False))

        # Test to save only several subjects
        idx = ['116', '142', '169']
        path_original = os.path.join(test_data_dir, "individual_parameters",
                                     "data_tiny-individual_parameters-3subjects.pt")
        path_copy = os.path.join(test_data_dir, "individual_parameters",
                                 "data_tiny-individual_parameters-3subjects-copy.pt")
        self.results.save_individual_parameters_torch(path_copy, idx)
        self.assertTrue(filecmp.cmp(path_original, path_copy, shallow=False))

    def test_load_individual_parameters(self, ind_param=None):
        if ind_param is None:
            ind_param = self.results.individual_parameters
        self.assertEqual(type(ind_param), dict)
        self.assertEqual(list(ind_param.keys()), ['tau', 'xi', 'sources'])
        for key in ind_param.keys():
            self.assertEqual(type(ind_param[key]), torch.Tensor)
            self.assertEqual(ind_param[key].dtype, torch.float32)
            self.assertEqual(ind_param[key].dim(), 2)
            self.assertEqual(ind_param[key].shape[0], 17)

    def test_load_result(self):
        ind_param_path_json = os.path.join(test_data_dir, "individual_parameters",
                                           "data_tiny-individual_parameters.json")
        ind_param_path_csv = os.path.join(test_data_dir, "individual_parameters",
                                          "data_tiny-individual_parameters.csv")
        ind_param_path_torch = os.path.join(test_data_dir, "individual_parameters",
                                            "data_tiny-individual_parameters.pt")

        cofactors_path = os.path.join(test_data_dir,
                                      "io",
                                      "data",
                                      "data_tiny_covariate.csv")

        data = self.results.data
        df = data.to_dataframe()

        ind_param_input_list = [ind_param_path_csv, ind_param_path_json, ind_param_path_torch]
        data_input_list = [data, df, example_data_path]

        for data_input in data_input_list:
            for ind_param_input in ind_param_input_list:
                self.launch_test(ind_param_input, data_input, cofactors_path)

    def launch_test(self, ind_param, data, cofactors):
        results = Result.load_result(data, ind_param, cofactors)
        df = results.data.to_dataframe()
        df2 = self.results.data.to_dataframe()
        self.assertTrue(allclose(df.loc[:, df.columns != 'ID'].values,
                                 df2.loc[:, df2.columns != 'ID'].values))
        self.test_load_individual_parameters(ind_param=results.individual_parameters)

    def test_get_error_distribution_dataframe(self):
        model_path = os.path.join(test_data_dir, "model_parameters",
                                  "fitted_multivariate_model.json")
        leaspy_session = Leaspy.load(model_path)
        self.results.get_error_distribution_dataframe(leaspy_session.model)

    ###############################################################
    # DEPRECATION WARNINGS
    # The corresponding methods will be removed in a future release
    ###############################################################

    def test_get_cofactor_distribution(self):
        self.assertEqual(self.results.get_cofactor_distribution('Treatments'),
                         self.cofactors.values.reshape(self.cofactors.shape[0]).tolist())

    def test_get_patient_individual_parameters(self):
        self.assertAlmostEqual(self.results.get_patient_individual_parameters('116')['tau'].tolist()[0],
                               79.124, delta=1e-3)
        self.assertAlmostEqual(self.results.get_patient_individual_parameters('116')['xi'].tolist()[0],
                               0.5355, delta=1e-3)
        self.assertAlmostEqual(self.results.get_patient_individual_parameters('116')['sources'].tolist()[0],
                               3.7742, delta=1e-3)
        self.assertAlmostEqual(self.results.get_patient_individual_parameters('116')['sources'].tolist()[1],
                               5.0088, delta=1e-3)

    def test_get_parameter_distribution(self):
        self.assertEqual(self.results.get_parameter_distribution('xi'),
                         self.results.individual_parameters['xi'].view(-1).tolist())
        self.assertEqual(self.results.get_parameter_distribution('tau'),
                         self.results.individual_parameters['tau'].view(-1).tolist())
        self.assertEqual(self.results.get_parameter_distribution('sources'),
                         {'sources' + str(i): val
                          for i, val in
                          enumerate(self.results.individual_parameters['sources'].transpose(1, 0).tolist())})

        xi_treatment_param = self.results.get_parameter_distribution('xi', 'Treatments')
        self.assertEqual(list(xi_treatment_param.keys()), ["Treatment_A", "Treatment_B"])
        self.assertEqual(len(xi_treatment_param['Treatment_A']),
                         self.cofactors[self.cofactors['Treatments'] == 'Treatment_A'].shape[0])
