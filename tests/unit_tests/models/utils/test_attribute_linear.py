import torch
import unittest

from leaspy.models.utils.attributes.attributes_linear import AttributesLinear


class AttributesLinearTest(unittest.TestCase):

    def setUp(self):
        """Set up the object for all the tests"""
        self.attributes = AttributesLinear(4, 2)

    def test_constructor(self):
        """Test the initialization"""
        self.assertEqual(self.attributes.dimension, 4)
        self.assertEqual(self.attributes.source_dimension, 2)
        self.assertEqual(self.attributes.positions, None)
        self.assertEqual(self.attributes.velocities, None)
        self.assertEqual(self.attributes.orthonormal_basis, None)
        self.assertEqual(self.attributes.mixing_matrix, None)
        self.assertEqual(self.attributes.name, 'linear')
        self.assertEqual(self.attributes.update_possibilities, ('all', 'g', 'v0', 'betas'))
        self.assertRaises(ValueError, AttributesLinear, '4', 3.2)  # with bad type arguments
        self.assertRaises(TypeError, AttributesLinear)  # without argument

    def test_check_names(self):
        """Test if raise a ValueError if wrong arg"""
        wrong_arg_exemples = ['blabla1', 3.8, {'truc': 0.1}]
        # for wrong_arg in wrong_arg_exemples:
        #     self.assertRaises(ValueError, self.attributes._check_names, wrong_arg)
        self.assertRaises(ValueError, self.attributes._check_names, wrong_arg_exemples)

    def test_compute_orthonormal_basis(self):
        """Test the orthonormality condition"""
        names = ['all']
        values = {
            'g': torch.tensor([0.]),
            'v0': torch.tensor([1., 0., 2., 1.]),
            'betas': torch.tensor([[1., 2., 3., 4.], [.1, .2, .3, .4], [-1., -2., -3., -4.]])
        }
        self.attributes.update(names, values)

        # Test the orthonormality condition
        dgamma_t0 = self.attributes.velocities
        orthonormal_basis = self.attributes.orthonormal_basis

        for orthonormal_vector in orthonormal_basis.permute(1, 0):
            # Test normality
            self.assertAlmostEqual(torch.norm(orthonormal_vector).item(), 1, delta=1e-6)
            # Test orthogonality
            self.assertAlmostEqual(torch.dot(orthonormal_vector, dgamma_t0).item(), 0, delta=1e-6)

    def test_mixing_matrix_utils(self):
        """Test the orthogonality condition"""
        names = ['all']
        values = {
            'g': torch.tensor([0.]),
            'v0': torch.tensor([1., 0., 2., 1.]),
            'betas': torch.tensor([[1., 2., 3., 4.], [.1, .2, .3, .4], [-1., -2., -3., -4.]])
        }
        self.attributes.update(names, values)
        dgamma_t0 = self.attributes.velocities
        mixing_matrix = self.attributes.mixing_matrix

        for mixing_column in mixing_matrix.permute(1, 0):
            self.assertAlmostEqual(torch.dot(mixing_column, dgamma_t0).item(), 0, delta=1e-6)
