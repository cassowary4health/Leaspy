import unittest

from leaspy.api import Leaspy
from leaspy.models.model_factory import ModelFactory


class LeaspyTest(unittest.TestCase):

    def test_constructor_univariate(self, leaspy=None):
        """
        Test attribute's initialization of leaspy univariate model
        :param leaspy: leaspy class object - used by test_constructor_multivariate function for common attributes
        :return: exit code
        """
        for name in ['univariate', 'linear', 'logistic', 'logistic_parallel']:
            leaspy = Leaspy(name)
            self.assertEqual(leaspy.type, name)
            self.assertEqual(type(leaspy.model), ModelFactory.model(name))





#TODO
