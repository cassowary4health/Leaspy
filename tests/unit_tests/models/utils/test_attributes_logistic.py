import torch
import unittest

from leaspy.models.utils.attributes.logistic_attributes import LogisticAttributes


class AttributesLogisticTest(unittest.TestCase):

    def test_constructor(self):
        attributes = LogisticAttributes('logistic', 6, 2)
        self.assertEqual(attributes.dimension, 6)
        self.assertEqual(attributes.positions, None)
        self.assertEqual(attributes.orthonormal_basis, None)
        self.assertEqual(attributes.mixing_matrix, None)
        self.assertEqual(attributes.name, 'logistic')
        self.assertEqual(attributes.update_possibilities, ('all', 'g', 'v0', 'betas'))
        self.assertRaises(ValueError, LogisticAttributes, 'name', '4', 3.2)  # with bad type arguments
        self.assertRaises(TypeError, LogisticAttributes)  # without argument

    def test_compute_orthonormal_basis(self, tol=5e-5):
        names = ['all']
        values = {
            'g': torch.tensor([-3, 2, 0, 3], dtype=torch.float32),
            'betas': torch.tensor([[1, 2, 3], [-0.1, 0.2, 0.3], [-1, 2, -3]], dtype=torch.float32),
            'v0': torch.tensor([-3, 1, 0, -1], dtype=torch.float32)
        }
        attributes = LogisticAttributes('logistic', 4, 2)
        attributes.update(names, values)

        # Test the orthogonality condition
        gamma_t0 = 1/(1+attributes.positions)
        dgamma_t0 = attributes.velocities
        #sqrt_metric_norm = attributes.positions / (1 + attributes.positions).pow(2)
        sqrt_metric_norm = gamma_t0 * (1 - gamma_t0)

        orthonormal_basis = attributes.orthonormal_basis
        for i in range(4-1):
            orthonormal_vector = orthonormal_basis[:, i] # column vector
            # Test orthogonality to dgamma_t0 (metric inner-product)
            self.assertAlmostEqual(torch.dot(orthonormal_vector,
                                             dgamma_t0 / sqrt_metric_norm**2).item(), 0, delta=tol)
            # Test normality (canonical inner-product)
            self.assertAlmostEqual(torch.norm(orthonormal_vector).item(), 1, delta=tol) # /sqrt_metric_norm
            # Test orthogonality to other vectors (canonical inner-product)
            for j in range(i+1, 4-1):
                self.assertAlmostEqual(torch.dot(orthonormal_vector,
                                                 orthonormal_basis[:, j]).item(), 0, delta=tol) # / sqrt_metric_norm
    def test_mixing_matrix_utils(self, tol=5e-5):
        names = ['all']
        values = {
            'g': torch.tensor([-3., 2., 0., 1.], dtype=torch.float32),
            'betas': torch.tensor([[1, 2, 3], [-0.1, 0.2, 0.3], [-1, 2, -3]], dtype=torch.float32),
            'v0': torch.tensor([-3, 1, 0, -1], dtype=torch.float32)
        }
        attributes = LogisticAttributes('logistic', 4, 2)
        attributes.update(names, values)

        gamma_t0 = 1/(1 + attributes.positions)
        dgamma_t0 = attributes.velocities
        #sqrt_metric_norm = attributes.positions / (1 + attributes.positions).pow(2)
        sqrt_metric_norm = gamma_t0 * (1 - gamma_t0)
        self.assertAlmostEqual(torch.norm(sqrt_metric_norm - attributes.positions / (1 + attributes.positions).pow(2)), 0, delta=tol)

        mixing_matrix = attributes.mixing_matrix
        for mixing_column in mixing_matrix.T:
            self.assertAlmostEqual(torch.dot(mixing_column, dgamma_t0 / sqrt_metric_norm**2).item(),
                                   0, delta=tol)
