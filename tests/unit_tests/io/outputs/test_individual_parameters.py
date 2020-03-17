import unittest
import pandas as pd

from leaspy.io.outputs.individual_parameters import IndividualParameters

class IndividualParametersTest(unittest.TestCase):

    def test_constructor(self):

        ip = IndividualParameters()
        self.assertEqual(ip.indices, [])
        self.assertEqual(ip.individual_parameters, {})
        self.assertEqual(ip.parameters_shape, {})

    def test_individual_parameters(self):

        ip = IndividualParameters()
        p1 = {"xi": 0.1, "tau": 70, "sources": [0.1, -0.3]}
        p2 = {"xi": 0.2, "tau": 73, "sources": [-0.4, 0.1]}
        p3 = {"xi": 0.3, "tau": 58, "sources": [-0.6, 0.2]}

        ip.add_individual_parameters("idx1", p1)
        ip.add_individual_parameters("idx2", p2)
        ip.add_individual_parameters("idx3", p3)

        self.assertEqual(ip.indices, ["idx1", "idx2", "idx3"])
        self.assertEqual(ip.individual_parameters, {"idx1": p1, "idx2": p2, "idx3": p3})
        self.assertEqual(ip.parameters_shape, {"xi": 1, "tau": 1, "sources": 2})

    def test_dataframe(self):
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
