from leaspy.models.lme_model import LMEModel

from tests import LeaspyTestCase


class LMEModelTest(LeaspyTestCase):

    def test_constructor(self):
        model = LMEModel('lme')
        self.assertEqual(model.name, 'lme')
        self.assertFalse(model.is_initialized)  # new: more coherent (needs a fit)
        self.assertEqual(model.features, None)
        self.assertEqual(model.dimension, None)
