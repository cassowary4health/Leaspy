import os
import unittest

from tests import test_data_dir
from src.inputs.data_reader import DataReader


class DataReaderTest(unittest.TestCase):

    def test_constructor(self):
        reader = DataReader()
        self.assertEqual(reader.headers, None)

    def test_reading(self):
        path = os.path.join(test_data_dir, 'univariate_data.csv')
        reader = DataReader()
        data = reader.read(path)
        self.assertEqual(reader.headers, ['MMSE'])
        #self.assertEqual(len(data.indices))
        # TO FINISH