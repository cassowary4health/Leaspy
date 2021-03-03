import unittest

from leaspy.models.utils.attributes import AttributesLinear, AttributesLogistic


class AttributesUnivariateTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up the object for all the tests"""
        cls.to_test = {
            'univariate_logistic': AttributesLogistic,
            'univariate_linear': AttributesLinear
        }

    def test_constructor(self):

        """Test the initialization"""
        for attr_name, klass in self.to_test.items():
            attr = klass(attr_name, dimension=1, source_dimension=None)
            self.assertEqual(attr.positions, None)
            self.assertEqual(attr.velocities, None)
            self.assertEqual(attr.name, attr_name)
            self.assertEqual(attr.update_possibilities, ('all', 'g', 'xi_mean'))
            #self.assertRaises(TypeError, AttributesUnivariate, 5, 2)  # with arguments for dimension & source_dimension

            """Test if raise a ValueError if asking to update a wrong arg"""
            self.assertRaises(ValueError, attr._check_names, ['blabla1', 3.8, None]) # totally false
            self.assertRaises(ValueError, attr._check_names, ['betas']) # false iff univariate
            self.assertRaises(ValueError, attr._check_names, ['deltas']) # false if univariate

