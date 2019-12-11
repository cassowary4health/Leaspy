import torch
import unittest

from leaspy.models.utils.attributes.attributes_logistic import AttributesLogistic


class AttributesLogisticTest(unittest.TestCase):

    def test_constructor(self):
        attributes = AttributesLogistic(6, 2)
        self.assertEqual(attributes.dimension, 6)
        self.assertEqual(attributes.g, None)
        self.assertEqual(attributes.orthonormal_basis, None)
        self.assertEqual(attributes.mixing_matrix, None)

    def test_compute_orthonormal_basis(self):
        names = ['all']
        values = {
            'g': torch.Tensor([-3, 2, 0, 3]),
            'betas': torch.Tensor([[1, 2, 3], [0.1, 0.2, 0.3], [-1, -2, -3]]),
            'v0': torch.Tensor([-3, 1, 0, -1])
        }
        attributes = AttributesLogistic(4, 2)
        attributes.update(names, values)

        # Test the orthogonality condition
        metric_normalization = attributes.g.pow(2) / (1 + attributes.g).pow(2)

        orthonormal_basis = attributes.orthonormal_basis
        for orthonormal_vector in orthonormal_basis.permute(1, 0):
            self.assertAlmostEqual(torch.norm(orthonormal_vector).data.numpy().tolist(), 1, delta=0.000001)
            self.assertAlmostEqual(torch.dot(orthonormal_vector, attributes.v0 * metric_normalization), 0, delta=10**-6)

    def test_mixing_matrix_utils(self):
        names = ['all']
        values = {
            'g': torch.Tensor([-3, 2, 0, 3]),
            'betas': torch.Tensor([[1, 2, 3], [0.1, 0.2, 0.3], [-1, -2, -3]]),
            'v0': torch.Tensor([-3, 1, 0, -1])
        }
        attributes = AttributesLogistic(4, 2)
        attributes.update(names, values)

        metric_normalization = attributes.g.pow(2) / (1 + attributes.g).pow(2)

        mixing_matrix = attributes.mixing_matrix
        for mixing_column in mixing_matrix.permute(1, 0):
            self.assertAlmostEqual(torch.dot(mixing_column, attributes.v0 * metric_normalization), 0, delta=10**-6)
