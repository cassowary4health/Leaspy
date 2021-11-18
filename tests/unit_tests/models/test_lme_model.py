import unittest
from leaspy.models.lme_model import LMEModel


class LMEModelTest(unittest.TestCase):

    def test_constructor(self):
        model = LMEModel('lme')
        self.assertEqual(model.name, 'lme')
        self.assertFalse(model.is_initialized)  # new: more coherent (needs a fit)
        self.assertEqual(model.features, None)
        self.assertEqual(model.dimension, None)
