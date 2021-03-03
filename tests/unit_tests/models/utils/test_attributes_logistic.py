import torch
import unittest

from leaspy.models.utils.attributes.attributes_logistic import AttributesLogistic


class AttributesLogisticTest(unittest.TestCase):

    def test_constructor(self):
        attributes = AttributesLogistic('logistic', 6, 2)
        self.assertEqual(attributes.dimension, 6)
        self.assertEqual(attributes.positions, None)
        self.assertEqual(attributes.orthonormal_basis, None)
        self.assertEqual(attributes.mixing_matrix, None)
        self.assertEqual(attributes.name, 'logistic')
        self.assertEqual(attributes.update_possibilities, ('all', 'g', 'v0', 'betas'))
        self.assertRaises(ValueError, AttributesLogistic, 'name', '4', 3.2)  # with bad type arguments
        self.assertRaises(TypeError, AttributesLogistic)  # without argument

    def test_compute_orthonormal_basis(self):
        names = ['all']
        values = {
            'g': torch.tensor([-3, 2, 0, 3], dtype=torch.float32),
            'betas': torch.tensor([[1, 2, 3], [-0.1, 0.2, 0.3], [-1, 2, -3]], dtype=torch.float32),
            'v0': torch.tensor([-3, 1, 0, -1], dtype=torch.float32)
        }
        attributes = AttributesLogistic('logistic', 4, 2)
        attributes.update(names, values)

        # Test the orthogonality condition
        gamma_t0 = 1/(1+attributes.positions)
        dgamma_t0 = attributes.velocities
        #sqrt_metric_norm = attributes.positions / (1 + attributes.positions).pow(2)
        sqrt_metric_norm = gamma_t0 * (1 - gamma_t0)

        orthonormal_basis = attributes.orthonormal_basis
        for i in range(4-1):
            orthonormal_vector = orthonormal_basis[:, i] # column vector
            # Test normality (metric inner-product)
            self.assertAlmostEqual(torch.norm(orthonormal_vector/sqrt_metric_norm).item(), 1, delta=1e-5)
            # Test orthogonality to dgamma_t0 (metric inner-product)
            self.assertAlmostEqual(torch.dot(orthonormal_vector / sqrt_metric_norm,
                                             dgamma_t0 / sqrt_metric_norm).item(), 0, delta=1e-5)
            # Test orthogonality to other vectors (metric inner-product)
            for j in range(i+1, 4-1):
                self.assertAlmostEqual(torch.dot(orthonormal_vector / sqrt_metric_norm,
                                                 orthonormal_basis[:, j] / sqrt_metric_norm).item(), 0, delta=1e-5)

    def test_mixing_matrix_utils(self):
        names = ['all']
        values = {
            'g': torch.tensor([-3, 2, 0, 3], dtype=torch.float32),
            'betas': torch.tensor([[1, 2, 3], [-0.1, 0.2, 0.3], [-1, 2, -3]], dtype=torch.float32),
            'v0': torch.tensor([-3, 1, 0, -1], dtype=torch.float32)
        }
        attributes = AttributesLogistic('logistic', 4, 2)
        attributes.update(names, values)

        gamma_t0 = 1/(1 + attributes.positions)
        dgamma_t0 = attributes.velocities
        #sqrt_metric_norm = attributes.positions / (1 + attributes.positions).pow(2)
        sqrt_metric_norm = gamma_t0 * (1 - gamma_t0)
        self.assertAlmostEqual(torch.norm(sqrt_metric_norm - attributes.positions / (1 + attributes.positions).pow(2)), 0, delta=1e-6)

        mixing_matrix = attributes.mixing_matrix
        for mixing_column in mixing_matrix.permute(1, 0):
            self.assertAlmostEqual(torch.dot(mixing_column, dgamma_t0 / sqrt_metric_norm**2).item(),
                                   0, delta=1e-5)
