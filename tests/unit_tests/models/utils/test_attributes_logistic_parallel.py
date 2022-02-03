import torch

from leaspy.models.utils.attributes.logistic_parallel_attributes import LogisticParallelAttributes

from tests import LeaspyTestCase


class AttributesLogisticParallelTest(LeaspyTestCase):

    def test_constructor(self):
        attributes = LogisticParallelAttributes('logistic_parallel', 6, 2)
        self.assertEqual(attributes.dimension, 6)
        self.assertEqual(attributes.source_dimension, 2)
        self.assertEqual(attributes.positions, None)
        self.assertEqual(attributes.deltas, None)
        self.assertEqual(attributes.orthonormal_basis, None)
        self.assertEqual(attributes.mixing_matrix, None)
        self.assertEqual(attributes.name, 'logistic_parallel')
        self.assertEqual(attributes.update_possibilities, ('all', 'g', 'xi_mean', 'betas', 'deltas'))
        self.assertRaises(ValueError, LogisticParallelAttributes, 'name', '4', 3.2)  # with bad type arguments
        self.assertRaises(TypeError, LogisticParallelAttributes)  # without argument

    def check_values_and_get_dimensions(self, values):
        dimension = 1 + len(values['deltas'])
        # positions & velocities are univariate in this model!
        self.assertEqual(len(values['g']), 1)
        self.assertEqual(len(values['xi_mean']), 1)
        self.assertEqual(values['betas'].shape[0], dimension-1)
        source_dimension = values['betas'].shape[1]
        self.assertTrue(0 <= source_dimension <= dimension-1)
        return dimension, source_dimension

    def compute_instance_and_variables(self):
        names = ['all']
        values = {
            'g': torch.tensor([0.]),
            'deltas': torch.tensor([-1., 0., 2.]),
            'betas': torch.tensor([[.1, .2, .3], [-.1, .2, .3], [-.1, .2, -.3]]),
            'xi_mean': torch.tensor([-3.])
        }

        dimension, source_dimension = self.check_values_and_get_dimensions(values)
        attributes = LogisticParallelAttributes('logistic_parallel', dimension, source_dimension)
        attributes.update(names, values)

        # Test the first value of the derivative of gamma at t0
        p0 = 1. / (1. + torch.exp(values['g']))
        standard_v0 = p0 * (1 - p0)  # do not multiply by a scalar constant... torch.exp(values['xi_mean'])
        gamma_t0, dgamma_t0 = attributes._compute_gamma_dgamma_t0()

        # Test the orthogonality condition
        #gamma_t0 = 1. / (1 + attributes.positions * torch.exp(-attributes.deltas))
        sqrt_metric_normalization = gamma_t0 * (1 - gamma_t0) # not squared

        return attributes, dgamma_t0, sqrt_metric_normalization, standard_v0

    def test_compute_orthonormal_basis(self):
        attributes, dgamma_t0, sqrt_metric_norm, standard_v0 = self.compute_instance_and_variables()

        # v0 for delta=0 (at dimension 0 and 2 by construction, cf. deltas above)
        self.assertEqual(dgamma_t0[0], standard_v0)
        self.assertEqual(dgamma_t0[2], standard_v0)

        orthonormal_basis = attributes.orthonormal_basis
        for i in range(attributes.dimension-1):
            orthonormal_vector = orthonormal_basis[:, i] # column vector
            # Test orthogonality to dgamma_t0 (metric inner-product)
            self.assertAlmostEqual(torch.dot(orthonormal_vector,
                                             dgamma_t0 / sqrt_metric_norm**2).item(), 0, delta=1e-6) # / sqrt_metric_norm
            # Test normality (canonical inner-product)
            self.assertAlmostEqual(torch.norm(orthonormal_vector).item(), 1, delta=1e-6) # /sqrt_metric_norm
            # Test orthogonality to other vectors (canonical inner-product)
            for j in range(i+1, attributes.dimension-1):
                self.assertAlmostEqual(torch.dot(orthonormal_vector,
                                                 orthonormal_basis[:, j]).item(), 0, delta=1e-6) # / sqrt_metric_norm


    def test_mixing_matrix_utils(self):
        attributes, dgamma_t0, sqrt_metric_norm, _ = self.compute_instance_and_variables()

        mixing_matrix = attributes.mixing_matrix
        for mixing_column in mixing_matrix.permute(1, 0):
            # Test orthogonality to dgamma_t0 (metric inner-product)
            self.assertAlmostEqual(torch.dot(mixing_column, dgamma_t0 / sqrt_metric_norm**2).item(), 0, delta=1e-6)

    def test_orthonormal_basis_consistency(self):

        values = {
            'g': torch.tensor([0.3], dtype=torch.float32),
            'deltas': torch.tensor([-1., 0., 2.]),
            'betas': torch.tensor([[0.1, 0.2, 0.3], [-0.1, 0.2, 0.3], [-0.1, 0.2, -0.3]], dtype=torch.float32),
            'xi_mean': torch.tensor([-3.5], dtype=torch.float32)
        }
        dimension, source_dimension = self.check_values_and_get_dimensions(values)

        attributes = LogisticParallelAttributes('logistic_parallel', dimension, source_dimension)
        attributes.update(['all'], values)
        old_velocities = attributes.velocities
        old_BON = attributes.orthonormal_basis
        old_A = attributes.mixing_matrix

        # check shape of BON & A
        self.assertEqual(old_BON.shape, (dimension, dimension - 1))
        self.assertEqual(old_A.shape, (dimension, source_dimension))

        # shift v0 (log of velocities), so the resulting v0 should be collinear to previous one
        # and so the orthonormal basis should be the same!
        new_v0 = values['xi_mean'] + 0.3
        attributes.update(['xi_mean'], {'xi_mean': new_v0})
        new_velocities = attributes.velocities
        new_BON = attributes.orthonormal_basis
        new_A = attributes.mixing_matrix

        # velocities are different (scalars...)
        self.assertFalse(torch.allclose(old_velocities, new_velocities))
        # but they are collinear (scalars...)
        self.assertTrue(attributes._check_collinearity_vectors(old_velocities, new_velocities))
        # and the orthonormal basis (and mixing matrix) are the same!
        self.assertTrue(torch.allclose(old_BON, new_BON))
        self.assertTrue(torch.allclose(old_A, new_A))
