import unittest
from leaspy.models.lme_model import LMEModel


class LMEModelTest(unittest.TestCase):

    def test_constructor(self):
        model = LMEModel('lme')
        self.assertEqual(model.is_initialized, True)
        self.assertEqual(model.name, 'lme')
        self.assertEqual(model.features, None)
        self.assertEqual(model.dimension, None)
