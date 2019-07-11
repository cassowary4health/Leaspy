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

        A = [[1., 5., 0.],
             [0., 2., 11.],
             [30., 0., 3.],
             [0., 0., 0.]]
        B = [[1., -1., 4.],
             [4., -10., 80.],
             [100., -100., 400.]]

        A = np.array(A)
        B = np.array(B)

        result = Attributes._mixing_matrix_utils(A, B)
        print(result.shape)
        print(result)
