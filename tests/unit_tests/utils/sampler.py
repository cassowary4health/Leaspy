import unittest

from leaspy.utils.sampler import Sampler
import numpy as np


class SamplerTest(unittest.TestCase):



    def test_sample(self):

        var_infos = {
            "name": "x",
            "shape": (1, 1),
            "type": "individual",
            "rv_type": "gaussian"
        }

        sampler = Sampler(var_infos,1, temp_length=100)

        reals = []
        for i in range(1000):
            reals.append(sampler.sample())
        reals = np.array(reals)

        self.assertAlmostEqual(np.mean(reals), 0., delta=0.08)
        self.assertAlmostEqual(np.std(reals), 0.1, delta=0.08)


    def test_acceptation(self):

        var_infos = {
            "name": "x",
            "shape": (1, 1),
            "type": "individual",
            "rv_type": "gaussian"
        }

        # Case where likelihood is improved
        sampler = Sampler(var_infos, 1, temp_length=1000)
        alpha = 1.5
        accepted = sampler.acceptation(alpha)
        self.assertEqual(accepted, True)

        # Case where likelihood is decreased
        sampler = Sampler(var_infos, 1,  temp_length=1000)
        alpha = 0.5
        accepted_list = []
        for i in range(1000):
            accepted_list.append(sampler.acceptation(alpha))
        accepted_list = np.array(accepted_list)
        self.assertAlmostEqual(np.mean(accepted_list), 0.5, delta=0.05)


    #TODO why does noit it work ?
    def test_adaptative_proposition_variance(self):

        # Create sampler
        var_infos = {
            "name": "x",
            "shape": (1, 1),
            "type": "population",
            "rv_type": "gaussian"
        }

        sampler = Sampler(var_infos, 1, temp_length=50)

        # Create random variable
        std = np.random.uniform(low=2, high=10)
        mu = np.random.uniform(low=-10, high=10)

        # iterate
        accepted_list = []
        real = mu
        for i in range(10000):
            prop_real = real + sampler.sample()

            likelihood_new = np.exp(-prop_real ** 2 / (2 * std ** 2))
            likelihood_old = np.exp(-real ** 2 / (2 * std ** 2))


            alpha = likelihood_new/likelihood_old

            accepted = sampler.acceptation(alpha)
            accepted_list.append(accepted)

            if accepted:
                real = prop_real

        self.assertAlmostEqual(np.mean(accepted_list[-200:]), 0.3, delta=0.2)




