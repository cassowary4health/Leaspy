from .attributes_abstract import AttributesAbstract


# TODO 2 : Add some individual attributes -> Optimization on the w_i = A * s_i
class AttributesLinear(AttributesAbstract):
    """
    Contains the common attributes & methods to update the linear model's attributes.

    Attributes
    ----------
    dimension: int
    source_dimension: int
    betas: `torch.Tensor` (default None)
    mixing_matrix: `torch.Tensor` (default None)
        Matrix A such that w_i = A * s_i
    orthonormal_basis: `torch.Tensor` (default None)
    positions: `torch.Tensor` (default None)
    velocities: `torch.Tensor` (default None)
    name: str (default 'linear')
        Name of the associated leaspy model. Used by ``update`` method.
    update_possibilities: tuple [str] (default ('all', 'g', 'v0', 'betas') )
        Contains the available parameters to update. Different models have different parameters.
    """

    def __init__(self, name, dimension, source_dimension):
        """
        Instantiate a AttributesLinear class object.

        Parameters
        ----------
        dimension: int
        source_dimension: int
        """
        super().__init__(name, dimension, source_dimension)

    def _compute_orthonormal_basis(self):
        """
        Compute the attribute ``orthonormal_basis`` which is an orthonormal basis, w.r.t the canonical inner product,
        of the sub-space orthogonal, w.r.t the inner product implied by the metric, to the time-derivative of the geodesic at initial time.
        In linear case, this inner product corresponds to canonical Euclidean one.
        """
        if not self.has_sources:
            return

        dgamma_t0 = self.velocities

        # Householder decomposition in Euclidean case, updates `orthonormal_basis` in-place
        self._compute_Q(dgamma_t0, 1.)
