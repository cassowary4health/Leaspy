import torch

from leaspy.models.utils.attributes.linear_attributes import LinearAttributes

from tests import LeaspyTestCase


class AttributesLinearTest(LeaspyTestCase):

    def setUp(self):
        """Set up the object for all the tests (reinit for all tests)"""
        self.attributes = LinearAttributes('linear', 4, 2)

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
        self.assertRaises(ValueError, LinearAttributes, 'name', '4', 3.2)  # with bad type arguments
        self.assertRaises(TypeError, LinearAttributes)  # without argument

    def test_check_names(self):
        """Test if raise a ValueError if wrong arg"""
        wrong_arg_exemples = ['blabla1', 3.8, None]
        # for wrong_arg in wrong_arg_exemples:
        #     self.assertRaises(ValueError, self.attributes._check_names, wrong_arg)
        self.assertRaises(ValueError, self.attributes._check_names, wrong_arg_exemples)

    def test_compute_orthonormal_basis(self):
        """Test the orthonormality condition"""
        names = ['all']
        values = {
            'g': torch.tensor([0.]),
            'v0': torch.tensor([-3., 1, 0, -1]), # as for logistic (too high v0values [exp'd] implies a precision a bit coarser)
            'betas': torch.tensor([[1, 2, 3], [-0.1, 0.2, 0.3], [-1, 2, -3]]), # dim=4, nb_source=3
        }
        self.attributes.update(names, values)

        # Test the orthonormality condition
        dgamma_t0 = self.attributes.velocities
        orthonormal_basis = self.attributes.orthonormal_basis
        for i in range(4-1):
            orthonormal_vector = orthonormal_basis[:, i] # column vector
            # Test normality (canonical inner-product)
            self.assertAlmostEqual(torch.norm(orthonormal_vector).item(), 1, delta=1e-6)
            # Test orthogonality to dgamma_t0 (canonical inner-product)
            self.assertAlmostEqual(torch.dot(orthonormal_vector, dgamma_t0).item(), 0, delta=1e-6)
            # Test orthogonality to other vectors (canonical inner-product)
            for j in range(i+1, 4-1):
                self.assertAlmostEqual(torch.dot(orthonormal_vector, orthonormal_basis[:, j]).item(), 0, delta=1e-6)

    def test_mixing_matrix_utils(self):
        """Test the orthogonality condition"""
        names = ['all']
        values = {
            'g': torch.tensor([0.]),
            'v0': torch.tensor([-3., 1, 0, -1]),
            'betas': torch.tensor([[1, 2, 3], [-0.1, 0.2, 0.3], [-1, 2, -3]]), # dim=4, nb_source=3
        }
        self.attributes.update(names, values)
        dgamma_t0 = self.attributes.velocities
        mixing_matrix = self.attributes.mixing_matrix

        for mixing_column in mixing_matrix.permute(1, 0):
            # Test orthogonality to dgamma_t0 (canonical inner-product)
            self.assertAlmostEqual(torch.dot(mixing_column, dgamma_t0).item(), 0, delta=1e-6)
