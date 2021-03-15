from .abstract_manifold_model_attributes import AbstractManifoldModelAttributes



class LinearAttributes(AbstractManifoldModelAttributes):
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
        compute_positions = False
        compute_velocities = False

        if 'all' in names_of_changed_values:
            names_of_changed_values = self.update_possibilities  # make all possible updates
        if 'betas' in names_of_changed_values:
            compute_betas = True
        if 'g' in names_of_changed_values:
            compute_positions = True
        if ('v0' in names_of_changed_values) or ('xi_mean' in names_of_changed_values):
            compute_velocities = True

        if compute_betas:
            self._compute_betas(values)
        if compute_positions:
            self._compute_positions(values)
        if compute_velocities:
            self._compute_velocities(values)

        if self.has_sources:
            if compute_velocities:
                self._compute_orthonormal_basis()
            if compute_velocities or compute_betas:
                self._compute_mixing_matrix()

    def _compute_positions(self, values):
        """
        Update the attribute ``positions``.

        Parameters
        ----------
        values: dict [str, `torch.Tensor`]
        """
        self.positions = values['g'].clone()


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
