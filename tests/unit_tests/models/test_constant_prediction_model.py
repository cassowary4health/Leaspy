import unittest

from tests import example_data_path

from leaspy.models.constant_prediction_model import ConstantModel
from leaspy.io.data.data import Data
from leaspy.io.data.dataset import Dataset

class ConstantPredictionModelTest(unittest.TestCase):

    def test_constructor(self):
        model = ConstantModel('constant_prediction')
        self.assertEqual(model.is_initialized, True)
        self.assertEqual(model.name, 'constant_prediction')
        self.assertEqual(model.features, None)
        self.assertEqual(model.dimension, None)




