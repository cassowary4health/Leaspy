import unittest

from tests import example_data_path

from leaspy.models.constant_model import ConstantModel
from leaspy.leaspy_io.data.data import Data
from leaspy.leaspy_io.data.dataset import Dataset

class ConstantModelTest(unittest.TestCase):

    def test_constructor(self):
        model = ConstantModel('constant')
        self.assertEqual(model.is_initialized, True)
        self.assertEqual(model.name, 'constant')
        self.assertEqual(model.features, None)
        self.assertEqual(model.dimension, None)




