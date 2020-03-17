import unittest

from leaspy.algo.algo_factory import AlgoFactory
from leaspy.io.settings.algorithm_settings import AlgorithmSettings
from leaspy.algo.fit.tensor_mcmcsaem import TensorMCMCSAEM


class Test(unittest.TestCase):

    def test_algo(self):
        """Test attributes static method"""
        # Test for one name
        settings = AlgorithmSettings('mcmc_saem')
        algo = AlgoFactory.algo('fit', settings)
        self.assertTrue(type(algo) == TensorMCMCSAEM)

        # Test if raise ValueError if wrong string arg for name
        wrong_arg_exemples = ['mcmc', 'blabla']
        for wrong_arg in wrong_arg_exemples:
            settings.name = wrong_arg
            self.assertRaises(ValueError, AlgoFactory.algo, 'fit', settings)
