import torch
import unittest

from leaspy.models.utils.attributes.attributes_logistic import AttributesLogistic


class AttributesLogisticTest(unittest.TestCase):

    def test_constructor(self):
        attributes = AttributesLogistic(6, 2)
        self.assertEqual(attributes.dimension, 6)
        self.assertEqual(attributes.positions, None)
        self.assertEqual(attributes.orthonormal_basis, None)
        self.assertEqual(attributes.mixing_matrix, None)
        self.assertEqual(attributes.name, 'logistic')
        self.assertEqual(attributes.update_possibilities, ('all', 'g', 'v0', 'betas'))
        self.assertRaises(ValueError, AttributesLogistic, '4', 3.2)  # with bad type arguments
        self.assertRaises(TypeError, AttributesLogistic)  # without argument

    def test_compute_orthonormal_basis(self):
        names = ['all']
        values = {
            'g': torch.tensor([-3, 2, 0, 3], dtype=torch.float32),
            'betas': torch.tensor([[1, 2, 3], [0.1, 0.2, 0.3], [-1, -2, -3]], dtype=torch.float32),
            'v0': torch.tensor([-3, 1, 0, -1], dtype=torch.float32)
        }
        attributes = AttributesLogistic(4, 2)
        attributes.update(names, values)

        # Test the orthogonality condition
        metric_normalization = attributes.positions.pow(2) / (1 + attributes.positions).pow(2)

        orthonormal_basis = attributes.orthonormal_basis
        for orthonormal_vector in orthonormal_basis.permute(1, 0):
            self.assertAlmostEqual(torch.norm(orthonormal_vector).item(), 1, delta=1e-6)
            self.assertAlmostEqual(torch.dot(orthonormal_vector, attributes.velocities * metric_normalization).item(),
                                   0, delta=1e-6)

    def test_mixing_matrix_utils(self):
        names = ['all']
        values = {
            'g': torch.tensor([-3, 2, 0, 3], dtype=torch.float32),
            'betas': torch.tensor([[1, 2, 3], [0.1, 0.2, 0.3], [-1, -2, -3]], dtype=torch.float32),
            'v0': torch.tensor([-3, 1, 0, -1], dtype=torch.float32)
        }
        attributes = AttributesLogistic(4, 2)
        attributes.update(names, values)

        metric_normalization = attributes.positions.pow(2) / (1 + attributes.positions).pow(2)

        mixing_matrix = attributes.mixing_matrix
        for mixing_column in mixing_matrix.permute(1, 0):
            self.assertAlmostEqual(torch.dot(mixing_column, attributes.velocities * metric_normalization).item(),
                                   0, delta=1e-6)
