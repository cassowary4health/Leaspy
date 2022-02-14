from leaspy.models.utils.attributes import LinearAttributes, LogisticAttributes

from tests import LeaspyTestCase


class AttributesUnivariateTest(LeaspyTestCase):

    @classmethod
    def setUpClass(cls):
        """Set up the object for all the tests"""
        # for tmp handling
        super().setUpClass()

        cls.to_test = {
            'univariate_logistic': LogisticAttributes,
            'univariate_linear': LinearAttributes
        }

    def test_constructor(self):

        """Test the initialization"""
        for attr_name, klass in self.to_test.items():
            attr = klass(attr_name, dimension=1, source_dimension=None)
            self.assertEqual(attr.positions, None)
            self.assertFalse(hasattr(attr, 'velocities'))
            self.assertEqual(attr.name, attr_name)
            self.assertEqual(attr.update_possibilities, ('all', 'g'))
            #self.assertRaises(TypeError, AttributesUnivariate, 5, 2)  # with arguments for dimension & source_dimension

            """Test if raise a ValueError if asking to update a wrong arg"""
            self.assertRaises(ValueError, attr._check_names, ['blabla1', 3.8, None]) # totally false
            self.assertRaises(ValueError, attr._check_names, ['betas']) # false iff univariate
            self.assertRaises(ValueError, attr._check_names, ['deltas']) # false if univariate
            self.assertRaises(ValueError, attr._check_names, ['xi_mean']) # was USELESS so removed
            self.assertRaises(ValueError, attr._check_names, ['v0']) # only for multivariate
            self.assertRaises(ValueError, attr._check_names, ['v0_collinear']) # only for multivariate

