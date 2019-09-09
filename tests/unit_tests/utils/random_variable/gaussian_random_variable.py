import unittest
import torch
from leaspy.utils.random_variable.gaussian_random_variable import GaussianRandomVariable
import numpy as np


class GaussianRandomVariableTest(unittest.TestCase):

    def test_construction(self):
        mu = 1
        variance = 2

        rv_infos = {
            "name": "tau",
            "shape": (10, 1),
            "type": "individual",
            "rv_type": "gaussian"
        }

        gaussian_rv = GaussianRandomVariable(rv_infos)
        gaussian_rv._from_parameters(mu=mu, variance=variance)

        self.assertAlmostEqual(gaussian_rv.mu, 1, delta=1e-5)
        self.assertAlmostEqual(gaussian_rv.variance, 2, delta=1e-5)
        self.assertAlmostEqual(gaussian_rv.variance_inverse, 0.5, delta=1e-5)
        self.assertAlmostEqual(gaussian_rv.log_constant, 1.2655121234846454, delta=1e-5)

    def test_getters_setters_variance(self):
        mu = 1
        variance = 2
        rv_infos = {
            "name": "tau",
            "shape": (10, 1),
            "type": "individual",
            "rv_type": "gaussian"
        }

        gaussian_rv = GaussianRandomVariable(rv_infos)
        gaussian_rv._from_parameters(mu=mu, variance=variance)
        self.assertAlmostEqual(gaussian_rv.log_constant, 1.26551212, delta=1e-5)

        gaussian_rv.variance = 1
        self.assertAlmostEqual(gaussian_rv.variance, 1, delta=1e-6)
        self.assertAlmostEqual(gaussian_rv.variance_inverse, 1, delta=1e-6)
        self.assertAlmostEqual(gaussian_rv.log_constant, 0.91893853320, delta=1e-6)

    def test_compute_likelihood(self):
        mu = 1
        variance = 2
        rv_infos = {
            "name": "tau",
            "shape": (10, 1),
            "type": "individual",
            "rv_type": "gaussian"
        }

        gaussian_rv = GaussianRandomVariable(rv_infos)
        gaussian_rv._from_parameters(mu=mu, variance=variance)
        self.assertAlmostEqual(gaussian_rv.compute_negativeloglikelihood(torch.Tensor([1])), 1.26551, delta=1e-5)
        self.assertAlmostEqual(gaussian_rv.compute_negativeloglikelihood(torch.Tensor([2])), 1.7655121, delta=1e-5)
        self.assertAlmostEqual(gaussian_rv.compute_negativeloglikelihood(torch.Tensor([3])), 3.265512, delta=1e-5)
        self.assertAlmostEqual(gaussian_rv.compute_negativeloglikelihood(torch.Tensor([1000])), 499001.7655121235, delta=1e-5)