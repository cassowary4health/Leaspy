import unittest

from leaspy.models.utils.attributes.attributes_univariate import AttributesUnivariate


class AttributesUnivariateTest(unittest.TestCase):

    def setUp(self):
        """Set up the object for all the tests"""
        self.attributes = AttributesUnivariate()

    def test_constructor(self):
        """Test the initialization"""
        self.assertEqual(self.attributes.positions, None)
        self.assertEqual(self.attributes.velocities, None)
        self.assertEqual(self.attributes.name, 'univariate')
        self.assertEqual(self.attributes.update_possibilities, ('all', 'g', 'xi_mean'))
        self.assertRaises(TypeError, AttributesUnivariate, 5, 2)  # with arguments for dimension & source_dimension

    def test_check_names(self):
        """Test if raise a ValueError if wrong arg"""
        wrong_arg_exemples = ['blabla1', 3.8, {'truc': 0.1}]
        # for wrong_arg in wrong_arg_exemples:
        #     self.assertRaises(ValueError, self.attributes._check_names, wrong_arg)
        self.assertRaises(ValueError, self.attributes._check_names, wrong_arg_exemples)