import torch
import unittest

from leaspy.models.utils.attributes.attributes_logistic_parallel import AttributesLogisticParallel


class AttributesLogisticParallelTest(unittest.TestCase):

    def test_constructor(self):
        attributes = AttributesLogisticParallel(6, 2)
        self.assertEqual(attributes.dimension, 6)
        self.assertEqual(attributes.positions, None)
        self.assertEqual(attributes.deltas, None)
        self.assertEqual(attributes.orthonormal_basis, None)
        self.assertEqual(attributes.mixing_matrix, None)
        self.assertEqual(attributes.name, 'logistic_parallel')
        self.assertEqual(attributes.update_possibilities, ('g', 'deltas', 'betas', 'xi_mean', 'all'))
        self.assertRaises(ValueError, AttributesLogisticParallel, '4', 3.2)  # with bad type arguments
        self.assertRaises(TypeError, AttributesLogisticParallel)  # without argument

    def compute_instance_and_variables(self):
        names = ['all']
        values = {
            'g': torch.tensor([0.]),
            'deltas': torch.tensor([-1., 0., 2.]),
            'betas': torch.tensor([[1., 2., 3.], [0.1, 0.2, 0.3], [-1., -2., -3.]]),
            'xi_mean': torch.tensor([-3.])
        }
        attributes = AttributesLogisticParallel(4, 2)
        attributes.update(names, values)

        # Test the first value of the derivative of gamma at t0
        p0 = 1. / (1. + torch.exp(values['g']))
        standard_v0 = torch.exp(values['xi_mean']) * p0 * (1 - p0)
        dgamma_t0 = attributes._compute_dgamma_t0()

        # Test the orthogonality condition
        gamma_t0 = 1. / (1 + attributes.positions * torch.exp(-attributes.deltas))
        metric_normalization = gamma_t0.pow(2) * (1 - gamma_t0).pow(2)

        return attributes, dgamma_t0, metric_normalization, standard_v0

    def test_compute_orthonormal_basis(self):
        attributes, dgamma_t0, metric_normalization, standard_v0 = self.compute_instance_and_variables()

        self.assertEqual(dgamma_t0[0], standard_v0)
        self.assertEqual(dgamma_t0[2], standard_v0)

        orthonormal_basis = attributes.orthonormal_basis
        for orthonormal_vector in orthonormal_basis.permute(1, 0):
            # Test normality
            self.assertAlmostEqual(torch.norm(orthonormal_vector).item(), 1, delta=1e-6)
            # Test orthogonality
            self.assertAlmostEqual(torch.dot(orthonormal_vector, dgamma_t0 / metric_normalization).item(),
                                   0, delta=1e-6)

    def test_mixing_matrix_utils(self):
        attributes, dgamma_t0, metric_normalization, _ = self.compute_instance_and_variables()

        mixing_matrix = attributes.mixing_matrix
        for mixing_column in mixing_matrix.permute(1, 0):
            self.assertAlmostEqual(torch.dot(mixing_column, dgamma_t0 / metric_normalization).item(), 0, delta=1e-6)
