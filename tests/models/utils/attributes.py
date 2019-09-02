import os
import numpy as np
import unittest

from leaspy.models.utils.attributes.attributes_logistic_parallel import Attributes_LogisticParallel


class AttributesTest(unittest.TestCase):


    # TODO : update; pass to torch, do other attributes
    """
    def test_constructor(self):
        attributes = Attributes_LogisticParallel(6, 2)
        self.assertEqual(attributes.dimension, 6)
        self.assertEqual(attributes.g0, None)
        self.assertEqual(attributes.deltas, None)
        self.assertEqual(attributes.orthonormal_basis, None)
        self.assertEqual(attributes.mixing_matrix, None)

    def test_mixing_matrix_utils(self):

        basis = [[1., 5., 3.],
                 [0., 2., 1.],
                 [30., 0., 1.],
                 [0., 0., 2.]]
        betas = [[1., -1.],
                 [4., -10.],
                 [100., -100.]]

        basis = np.array(basis)
        betas = np.array(betas)

        result = Attributes_LogisticParallel._mixing_matrix_utils(betas, basis)"""

