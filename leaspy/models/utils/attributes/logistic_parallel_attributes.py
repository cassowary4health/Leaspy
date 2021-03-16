import torch

from .abstract_manifold_model_attributes import AbstractManifoldModelAttributes



class LogisticParallelAttributes(AbstractManifoldModelAttributes):
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

    def update(self, names_of_changed_values, values):
        """
        Update model group average parameter(s).

        Parameters
        ----------
        names_of_changed_values: list [str]
            Must be one of - "all", "g", "v0", "betas". Raise an error otherwise.
            "g" correspond to the attribute ``positions``.
            "v0" correspond to the attribute ``velocities``.
        values: dict [str, `torch.Tensor`]
            New values used to update the model's group average parameters
        """
        self._check_names(names_of_changed_values)

        compute_betas = False
        compute_deltas = False
        compute_positions = False
        compute_velocities = False

        if 'all' in names_of_changed_values:
            names_of_changed_values = self.update_possibilities  # make all possible updates
        if 'betas' in names_of_changed_values:
            compute_betas = True
        if 'deltas' in names_of_changed_values:
            compute_deltas = True
        if 'g' in names_of_changed_values:
            compute_positions = True
        if ('v0' in names_of_changed_values) or ('xi_mean' in names_of_changed_values):
            compute_velocities = True

        if compute_betas:
            self._compute_betas(values)
        if compute_deltas:
            self._compute_deltas(values)
        if compute_positions:
            self._compute_positions(values)
        if compute_velocities:
            self._compute_velocities(values)

        if self.has_sources:
            recompute_ortho_basis = compute_positions or compute_velocities or compute_deltas

            if recompute_ortho_basis:
                self._compute_orthonormal_basis()
            if recompute_ortho_basis or compute_betas:
                self._compute_mixing_matrix()

    def _compute_positions(self, values):
        """
        Update the attribute ``positions``.

        Parameters
        ----------
        values: dict [str, `torch.Tensor`]
        """
        self.positions = torch.exp(values['g'])


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

    def _compute_gamma_dgamma_t0(self):
        """
        Computes both gamma:
        - value at t0
        - derivative w.r.t. time at time t0

        Returns
        -------
        2-tuple:
            gamma_t0: `torch.Tensor` 1D
            dgamma_t0: `torch.Tensor` 1D
        """
        exp_d = torch.exp(-self.deltas)
        denom = 1. + self.positions * exp_d
        gamma_t0 = 1. / denom

        dgamma_t0 = self.velocities * self.positions * exp_d / (denom * denom)

        return gamma_t0, dgamma_t0

    def _compute_orthonormal_basis(self):
        """
        Compute the attribute ``orthonormal_basis`` which is an orthonormal basis, w.r.t the canonical inner product,
        of the sub-space orthogonal, w.r.t the inner product implied by the metric, to the time-derivative of the geodesic at initial time.
        """
        if not self.has_sources:
            return

        # Compute value and time-derivative of gamma at t0
        gamma_t0, dgamma_t0 = self._compute_gamma_dgamma_t0()

        # Compute the diagonal of metric matrix (cf. `_compute_Q`)
        G_metric = ( gamma_t0 * (1 - gamma_t0) )** -2

        # Householder decomposition in non-Euclidean case, updates `orthonormal_basis` in-place
        self._compute_Q(dgamma_t0, G_metric)

