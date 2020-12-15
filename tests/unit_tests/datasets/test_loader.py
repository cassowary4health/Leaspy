import unittest

from leaspy.datasets.loader import load_dataset, paths


class DataTest(unittest.TestCase):

    def test_load_dataset(self):
        """
        Check ID and dtype of ID, TIME and values.
        """
        for name in paths.keys():
            df = load_dataset(name)
            self.assertEqual(df.index.names, ['ID', 'TIME'])
            self.assertEqual(df.index.get_level_values('ID').unique().tolist(),
                             ['GS-' + '0'*(3 - len(str(i))) + str(i) for i in range(1, 201)])
            self.assertTrue(df.index.get_level_values('TIME').dtype in ('float64', 'float32'))
            self.assertTrue(all(df.dtypes.values == 'float64'))
