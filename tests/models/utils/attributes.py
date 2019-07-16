import os
import numpy as np
import unittest

from src.models.utils.attributes import Attributes


class AttributesTest(unittest.TestCase):

    def test_constructor(self):
        attributes = Attributes(6, 2)
        self.assertEqual(attributes.dimension, 6)
        self.assertEqual(attributes.p0, None)
        self.assertEqual(attributes.f_p0, None)
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

        result = Attributes._mixing_matrix_utils(betas, basis)

