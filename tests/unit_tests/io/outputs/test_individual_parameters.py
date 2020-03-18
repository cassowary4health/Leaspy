import unittest
import pandas as pd
import torch
import numpy as np

from leaspy.io.outputs.individual_parameters import IndividualParameters

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
        #TODO
        return 0

    def test_check_and_get_extension(self):
        #TODO
        return 0

    def test_save_csv(self):
        #TODO
        return 0

    def test_save_json(self):
        #TODO
        return 0

    def test_load_csv(self):
        #TODO
        return 0

    def test_load_json(self):
        #TODO
        return 0

    def test_load_individual_parameters(self):
        #TODO
        return 0

    def test_save_individual_parameters(self):
        #TODO
        return 0










