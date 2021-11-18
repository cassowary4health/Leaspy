import unittest

from tests import example_data_path

from leaspy.models.constant_model import ConstantModel
from leaspy.io.data.data import Data
from leaspy.io.data.dataset import Dataset

class ConstantModelTest(unittest.TestCase):

    def test_constructor(self):
        model = ConstantModel('constant')
        self.assertEqual(model.name, 'constant')
        self.assertTrue(model.is_initialized)  # no need for a fit
        self.assertEqual(model.features, None)
        self.assertEqual(model.dimension, None)




