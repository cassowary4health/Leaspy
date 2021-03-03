import unittest

from leaspy.models.utils.attributes.attributes_factory import AttributesFactory
from leaspy.models.utils.attributes.attributes_logistic import AttributesLogistic


class AttributesFactoryTest(unittest.TestCase):

    def test_attributes(self):
        """Test attributes static method"""
        # Test if raise ValueError if wrong string arg for name
        wrong_arg_exemples = ['lgistic', 'blabla']
        for wrong_arg in wrong_arg_exemples:
            self.assertRaises(ValueError,
                              lambda name: AttributesFactory.attributes(name, 4, 2),
                              wrong_arg)

        # Test if raise AttributeError if wrong object in name (not a string)
        wrong_arg_exemples = [3.8, {'truc': .1}]
        for wrong_arg in wrong_arg_exemples:
            self.assertRaises(AttributeError,
                              lambda name: AttributesFactory.attributes(name, 4, 2),
                              wrong_arg)

        # Test if lower name:
        name_exemples = ['logistic', 'LogIStiC', 'LOGISTIC']
        for name in name_exemples:
            self.assertTrue(type(AttributesFactory.attributes(name, 4, 2)) == AttributesLogistic)
