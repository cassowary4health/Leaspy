import os
import unittest
import pandas as pd
import torch
import numpy as np
import json

from leaspy.io.outputs.individual_parameters import IndividualParameters
from tests import test_data_dir

class IndividualParametersTest(unittest.TestCase):

    def test_constructor(self):

        ip = IndividualParameters()
        self.assertEqual(ip._indices, [])
        self.assertEqual(ip._individual_parameters, {})
        self.assertEqual(ip._parameters_shape, {})
        self.assertEqual(ip._default_saving_type, "csv")

    def test_individual_parameters(self):

        ip = IndividualParameters()
        p1 = {"xi": 0.1, "tau": 70, "sources": [0.1, -0.3]}
        p2 = {"xi": 0.2, "tau": 73, "sources": [-0.4, 0.1]}
        p3 = {"xi": 0.3, "tau": 58, "sources": [-0.6, 0.2]}

        ip.add_individual_parameters("idx1", p1)
        ip.add_individual_parameters("idx2", p2)
        ip.add_individual_parameters("idx3", p3)

        self.assertEqual(ip._indices, ["idx1", "idx2", "idx3"])
        self.assertEqual(ip._individual_parameters, {"idx1": p1, "idx2": p2, "idx3": p3})
        self.assertEqual(ip._parameters_shape, {"xi": 1, "tau": 1, "sources": 2})

    def test_to_dataframe(self):
        ip = IndividualParameters()

        p1 = {"xi": 0.1, "tau": 70, "sources": [0.1, -0.3]}
        p2 = {"xi": 0.2, "tau": 73, "sources": [-0.4, 0.1]}

        ip.add_individual_parameters("idx1", p1)
        ip.add_individual_parameters("idx2", p2)

        df = ip.to_dataframe()
        df_test = pd.DataFrame(data=[[0.1, 70, 0.1, -0.3], [0.2, 73, -0.4, 0.1]],
                               index=["idx1", "idx2"],
                               columns=["xi", "tau", "sources_0", "sources_1"])

        self.assertTrue((df.values == df_test.values).all())
        self.assertTrue((df.index == df_test.index).all())
        for n1, n2 in zip(df.columns, df_test.columns):
            self.assertEqual(n1, n2)


    def test_from_dataframe(self):

        p1 = {"xi": 0.1, "tau": 70, "sources": [0.1, -0.3]}
        p2 = {"xi": 0.2, "tau": 73, "sources": [-0.4, 0.1]}

        df = pd.DataFrame(data=[[0.1, 70, 0.1, -0.3], [0.2, 73, -0.4, 0.1]],
                          index=["idx1", "idx2"],
                          columns=["xi", "tau", "sources_0", "sources_1"])

        ip = IndividualParameters.from_dataframe(df)

        self.assertEqual(ip._indices, ["idx1", "idx2"])
        self.assertEqual(ip._individual_parameters, {"idx1": p1, "idx2": p2})
        self.assertEqual(ip._parameters_shape, {"xi": 1, "tau": 1, "sources": 2})

    def test_to_from_dataframe(self):
        df1 = pd.DataFrame(data=[[0.1, 70, 0.1, -0.3], [0.2, 73, -0.4, 0.1]],
                           index=["idx1", "idx2"],
                           columns=["xi", "tau", "sources_0", "sources_1"])

        ip1 = IndividualParameters.from_dataframe(df1)
        df2 = ip1.to_dataframe()
        ip2 = IndividualParameters.from_dataframe(df2)

        # Test between individual parameters
        self.assertEqual(ip1._indices, ip2._indices)
        self.assertDictEqual(ip1._individual_parameters, ip2._individual_parameters)
        self.assertDictEqual(ip1._parameters_shape, ip2._parameters_shape)

        # Test between dataframes
        self.assertTrue((df1.values == df2.values).all())
        self.assertTrue((df1.index == df2.index).all())
        for n1, n2 in zip(df1.columns, df2.columns):
            self.assertEqual(n1, n2)

    def test_to_pytorch(self):
        """

        """
        ip = IndividualParameters()

        p1 = {"xi": 0.1, "tau": 70, "sources": [0.1, -0.3]}
        p2 = {"xi": 0.2, "tau": 73, "sources": [-0.4, 0.1]}

        ip.add_individual_parameters("idx1", p1)
        ip.add_individual_parameters("idx2", p2)

        ip_pytorch = ip.to_pytorch()

        dict_test = {
            "xi": torch.tensor([[0.1], [0.2]], dtype=torch.float32),
            "tau": torch.tensor([[70], [73]], dtype=torch.float32),
            "sources": torch.tensor([[0.1, -0.3], [-0.4, 0.1]], dtype=torch.float32)
        }

        self.assertEqual(ip_pytorch.keys(), dict_test.keys())
        for k in dict_test.keys():
            self.assertTrue((ip_pytorch[k] == dict_test[k]).all())


    def test_from_pytorch(self):
        """

        """

        ip_pytorch = {
            "xi": torch.tensor([[0.1], [0.2]], dtype=torch.float32),
            "tau": torch.tensor([[70], [73]], dtype=torch.float32),
            "sources": torch.tensor([[0.1, -0.3], [-0.4, 0.1]], dtype=torch.float32)
        }

        ip = IndividualParameters.from_pytorch(ip_pytorch)

        dict_test = {
            0: {"xi": 0.1, "tau": 70., "sources": [0.1, -0.3]},
            1: {"xi": 0.2, "tau": 73., "sources": [-0.4, 0.1]}
        }

        self.assertEqual(ip._indices, [0, 1])
        self.assertEqual(ip._individual_parameters.keys(), dict_test.keys())
        for k, v in dict_test.items():
            for kk, vv in dict_test[k].items():
                self.assertTrue(kk in ip._individual_parameters[k].keys())
                if np.ndim(vv) == 0:
                    self.assertAlmostEqual(ip._individual_parameters[k][kk], vv, delta=10e-8)
                else:
                    l2 = ip._individual_parameters[k][kk]
                    for s1, s2 in zip(vv, l2):
                        self.assertAlmostEqual(s1, s2, delta=10e-8)

    def test_from_to_pytorch(self):

        ip_pytorch = {
            "xi": torch.tensor([[0.1], [0.2]], dtype=torch.float32),
            "tau": torch.tensor([[70], [73]], dtype=torch.float32),
            "sources": torch.tensor([[0.1, -0.3], [-0.4, 0.1]], dtype=torch.float32)
        }

        ip = IndividualParameters.from_pytorch(ip_pytorch)
        ip_pytorch2 = ip.to_pytorch()
        ip2 = IndividualParameters.from_pytorch(ip_pytorch2)

        # Test Individual parameters
        self.assertEqual(ip._indices, ip2._indices)
        self.assertDictEqual(ip._individual_parameters, ip2._individual_parameters)
        self.assertDictEqual(ip._parameters_shape, ip2._parameters_shape)


        # Test Pytorch dictionaries
        self.assertEqual(ip_pytorch.keys(), ip_pytorch2.keys())
        for k in ip_pytorch.keys():
            for v1, v2 in zip(ip_pytorch[k], ip_pytorch2[k]):
                self.assertTrue((v1.numpy() - v2.numpy() == 0).all())



    def test_check_and_get_extension(self):
        tests = [
            ('file.csv', 'csv'),
            ('path/to/file.csv', 'csv'),
            ('file.json', 'json'),
            ('path/to/file.json', 'json'),
            ('nopath', False),
            ('bad_path.bad', 'bad')
        ]

        for test in tests:
            ext = IndividualParameters._check_and_get_extension(test[0])
            self.assertEqual(ext, test[1])

    def test_save_csv(self):

        ip = IndividualParameters()
        ip.add_individual_parameters("idx1", {"xi": 0.1, "tau": 70, "sources": [0.1, -0.3]})
        ip.add_individual_parameters("idx2", {"xi": 0.2, "tau": 73, "sources": [-0.4, 0.1]})

        path = os.path.join(test_data_dir, "io", "outputs", "ip_save_csv.csv")

        test_path = os.path.join(test_data_dir, "io", "outputs", "ip_save_csv_test.csv")
        ip._save_csv(test_path)

        with open(path, 'r') as f1, open(test_path, 'r') as f2:
            file1 = f1.readlines()
            file2 = f2.readlines()

        for l1, l2 in zip(file1, file2):
            self.assertTrue(l1 == l2)

        os.remove(test_path)

    def test_save_json(self):

        def ordered(obj):
            if isinstance(obj, dict):
                return sorted((k, ordered(v)) for k, v in obj.items())
            if isinstance(obj, list):
                return sorted(ordered(x) for x in obj)
            else:
                return obj

        ip = IndividualParameters()
        ip.add_individual_parameters("idx1", {"xi": 0.1, "tau": 70, "sources": [0.1, -0.3]})
        ip.add_individual_parameters("idx2", {"xi": 0.2, "tau": 73, "sources": [-0.4, 0.1]})

        path = os.path.join(test_data_dir, "io", "outputs", "ip_save_json.json")

        test_path = os.path.join(test_data_dir, "io", "outputs", "ip_save_json_test.json")
        ip._save_json(test_path)

        with open(path, 'r') as f1, open(test_path, 'r') as f2:
            file1 = json.load(f1)
            file2 = json.load(f2)

        self.assertTrue(ordered(file1) == ordered(file2))

        os.remove(test_path)


    def test_load_csv(self):
        path = os.path.join(test_data_dir, "io", "outputs", "ip_save_csv.csv")
        ip = IndividualParameters._load_csv(path)

        p1 = {"xi": 0.1, "tau": 70, "sources": [0.1, -0.3]}
        p2 = {"xi": 0.2, "tau": 73, "sources": [-0.4, 0.1]}

        self.assertEqual(ip._indices, ["idx1", "idx2"])
        self.assertEqual(ip._individual_parameters, {"idx1": p1, "idx2": p2})
        self.assertEqual(ip._parameters_shape, {"xi": 1, "tau": 1, "sources": 2})

    def test_load_json(self):
        path = os.path.join(test_data_dir, "io", "outputs", "ip_save_json.json")
        ip = IndividualParameters._load_json(path)

        p1 = {"xi": 0.1, "tau": 70, "sources": [0.1, -0.3]}
        p2 = {"xi": 0.2, "tau": 73, "sources": [-0.4, 0.1]}

        self.assertEqual(ip._indices, ["idx1", "idx2"])
        self.assertEqual(ip._individual_parameters, {"idx1": p1, "idx2": p2})
        self.assertEqual(ip._parameters_shape, {"xi": 1, "tau": 1, "sources": 2})


    def test_load_individual_parameters(self):
        # Parameters
        p1 = {"xi": 0.1, "tau": 70, "sources": [0.1, -0.3]}
        p2 = {"xi": 0.2, "tau": 73, "sources": [-0.4, 0.1]}
        indices = ["idx1", "idx2"]
        individual_parameters = {"idx1": p1, "idx2": p2}
        parameters_shape = {"xi": 1, "tau": 1, "sources": 2}

        # Test json
        path_json = os.path.join(test_data_dir, "io", "outputs", "ip_save_json.json")
        ip_json = IndividualParameters.load_individual_parameters(path_json)

        self.assertEqual(ip_json._indices, indices)
        self.assertEqual(ip_json._individual_parameters, individual_parameters)
        self.assertEqual(ip_json._parameters_shape, parameters_shape)

        # Test csv
        path_csv = os.path.join(test_data_dir, "io", "outputs", "ip_save_csv.csv")
        ip_csv = IndividualParameters.load_individual_parameters(path_csv)

        self.assertEqual(ip_csv._indices, indices)
        self.assertEqual(ip_csv._individual_parameters, individual_parameters)
        self.assertEqual(ip_csv._parameters_shape, parameters_shape)

    def test_save_individual_parameters(self):

        # Utils
        def ordered(obj):
            if isinstance(obj, dict):
                return sorted((k, ordered(v)) for k, v in obj.items())
            if isinstance(obj, list):
                return sorted(ordered(x) for x in obj)
            else:
                return obj

        # Parameters
        ip = IndividualParameters()
        ip.add_individual_parameters("idx1", {"xi": 0.1, "tau": 70, "sources": [0.1, -0.3]})
        ip.add_individual_parameters("idx2", {"xi": 0.2, "tau": 73, "sources": [-0.4, 0.1]})

        path_json = os.path.join(test_data_dir, "io", "outputs", "ip_save_json.json")
        path_json_test = os.path.join(test_data_dir, "io", "outputs", "ip_save_json_test.json")

        path_csv = os.path.join(test_data_dir, "io", "outputs", "ip_save_csv.csv")
        path_csv_test = os.path.join(test_data_dir, "io", "outputs", "ip_save_csv_test.csv")

        path_default = os.path.join(test_data_dir, "io", "outputs", "ip_save_default")

        # Test json
        ip.save_individual_parameters(path_json_test)

        with open(path_json, 'r') as f1, open(path_json_test, 'r') as f2:
            file1 = json.load(f1)
            file2 = json.load(f2)

        self.assertTrue(ordered(file1) == ordered(file2))

        os.remove(path_json_test)

        # Test csv
        ip.save_individual_parameters(path_csv_test)

        with open(path_csv, 'r') as f1, open(path_csv_test, 'r') as f2:
            file1 = f1.readlines()
            file2 = f2.readlines()

        for l1, l2 in zip(file1, file2):
            self.assertTrue(l1 == l2)

        os.remove(path_csv_test)

        # Test default
        ip.save_individual_parameters(path_default)
        path_default_with_extension = path_default + '.' + ip._default_saving_type

        with open(path_csv, 'r') as f1, open(path_default_with_extension, 'r') as f2:
            file1 = f1.readlines()
            file2 = f2.readlines()

        for l1, l2 in zip(file1, file2):
            self.assertTrue(l1 == l2)

        os.remove(path_default_with_extension)






