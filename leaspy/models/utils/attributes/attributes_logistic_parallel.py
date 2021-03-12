import torch

from .attributes_abstract import AttributesAbstract


# TODO 2 : Add some individual attributes -> Optimization on the w_i = A * s_i
class AttributesLogisticParallel(AttributesAbstract):
    """
    Contains the common attributes & methods of the logistic parallel models' attributes.

    Attributes
    ----------
    dimension: int
    source_dimension: int
    betas: `torch.Tensor` (default None)
    deltas: `torch.Tensor` (default None)
        deltas = [0, delta_2_realization, ..., delta_n_realization]
    mixing_matrix: `torch.Tensor` (default None)
        Matrix A such that w_i = A * s_i
    orthonormal_basis: `torch.Tensor` (default None)
    positions: `torch.Tensor` (default None)
        positions = exp(realizations['g']) such that p0 = 1 / (1+exp(g))
    velocities: `torch.Tensor` (default None)
    name: str (default 'logistic_parallel')
        Name of the associated leaspy model. Used by ``update`` method.
    update_possibilities: tuple [str] (default ('all', 'g', 'xi_mean', 'betas', 'deltas') )
        Contains the available parameters to update. Different models have different parameters.
    """

    def __init__(self, name, dimension, source_dimension):
        """
        Instantiate a `AttributesLogisticParallel` class object.

        Parameters
        ----------
        dimension: int
        source_dimension: int
        """
        super().__init__(name, dimension, source_dimension)
        assert self.dimension >= 2

        self.deltas = None  # deltas = [0, delta_2_realization, ..., delta_n_realization]
        self.update_possibilities = ('all', 'g', 'xi_mean', 'betas', 'deltas')

    def get_attributes(self):
        """
        Returns the following attributes: ``positions``, ``deltas`` & ``mixing_matrix``.

        Returns
        -------
        positions: `torch.Tensor`
        deltas: `torch.Tensor`
        mixing_matrix: `torch.Tensor`
        """
        return self.positions, self.deltas, self.mixing_matrix

    def _compute_velocities(self, values):
        """
        Update the attribute ``velocities``.

        Parameters
        ----------
        values: dict [str, `torch.Tensor`]
        """
        self.velocities = torch.exp(values['xi_mean'])

    def _compute_deltas(self, values):
        """
        Update` the attribute ``deltas``.

        Parameters
        ----------
        values: dict [str, `torch.Tensor`]
        """
        self.deltas = torch.cat((torch.tensor([0], dtype=torch.float32), values['deltas']))

    def _compute_dgamma_t0(self):
        """
        Computes the derivative of gamma_0 at time t0.

        Returns
        -------
        dgamma_t0: `torch.Tensor`
        """
        exp_d = torch.exp(-self.deltas)
        sub = 1. + self.positions * exp_d
        dgamma_t0 = self.velocities * self.positions * exp_d / (sub * sub)
        return dgamma_t0

    def _compute_orthonormal_basis(self):
        """
        Compute the attribute ``orthonormal_basis`` which is a basis orthogonal to velocities for the inner product implied by
        the metric. It is equivalent to be a base orthogonal to velocities / (p0^2 (1-p0)^2 for the euclidean norm.
        """
        if self.source_dimension == 0:
            return

        # Compute the derivative of gamma_0 at t0
        dgamma_t0 = self._compute_dgamma_t0()

        # Compute regularizer to work in the euclidean space
        gamma_t0 = 1. / (1 + self.positions * torch.exp(-self.deltas))
        metric_normalization = gamma_t0.pow(2) * (1 - gamma_t0).pow(2)
        dgamma_t0 = dgamma_t0 / metric_normalization

        # Compute Q
        self._compute_Q(dgamma_t0)
