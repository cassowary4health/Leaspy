import torch


class AttributesLinear:
    """
    Attributes
    ----------
    dimension: `int`
    source_dimension: `int`
    betas: `torch.Tensor` (default None)
    mixing_matrix: `torch.Tensor` (default None)
        Matrix A such that w_i = A * s_i
    orthonormal_basis: `torch.Tensor` (default None)
    positions: `torch.Tensor` (default None)
    velocities: `torch.Tensor` (default None)

    Methods
    -------
    get_attributes()
        Returns the following attributes: ``g``, ``deltas`` & ``mixing_matrix``.
    update(names_of_changed_values, values)
        Update model group average parameter(s).
    """

    def __init__(self, dimension, source_dimension):
        """
        Instantiate a AttributesLogisticParallel class object.

        Parameters
        ----------
        dimension: `int`
        source_dimension: `int`
        """
        self.dimension = dimension
        self.source_dimension = source_dimension
        self.positions = None
        self.velocities = None
        self.orthonormal_basis = None
        self.mixing_matrix = None

    def get_attributes(self):
        """
        Returns the following attributes: ``positions``, ``velocities`` & ``mixing_matrix``.

        Returns
        -------
        - positions: `torch.Tensor`
        - velocities: `torch.Tensor`
        - mixing_matrix: `torch.Tensor`
        """
        return self.positions, self.velocities, self.mixing_matrix

    def update(self, names_of_changed_values, values):
        """
        Update model group average parameter(s).

        Parameters
        ----------
        names_of_changed_values: `list` [`str`]
            Must be one of - "all", "betas", "deltas", "g", "xi_mean". Raise an error otherwise.
        values: `dict` [`str`, `torch.Tensor`]
        """
        self._check_names(names_of_changed_values)

        compute_positions = False
        compute_velocities = False
        compute_betas = False

        for name in names_of_changed_values:
            if name == 'g':
                compute_positions = True
            elif name == 'v0':
                compute_velocities = True
            elif name == 'betas':
                compute_betas = True
            elif name == 'all':
                compute_positions = True
                compute_velocities = True
                compute_betas = True

        if compute_positions:
            self._compute_positions(values)
        if compute_velocities:
            self._compute_velocities(values)
        if compute_betas:
            self._compute_betas(values)

        # TODO : Check if the condition is enough
        if compute_velocities:
            self._compute_orthonormal_basis()
        if compute_velocities or compute_betas:
            self._compute_mixing_matrix()

    def _check_names(self, names_of_changed_values):
        """
        Check if the name of the parameter(s) to update are one of 'all', 'g', 'v0', 'betas'.

        Parameters
        ----------
        names_of_changed_values: `list` [`str`]

        Raises
        -------
        ValueError
        """
        def raise_err(name):
            raise ValueError("The name {} is not in the attributes that are used to be updated".format(name))

        possibilities = ['all', 'g', 'v0', 'betas']
        [raise_err(n) for n in names_of_changed_values if n not in possibilities]

    def _compute_positions(self, values):
        """
        Update the attribute ``g``.

        Parameters
        ----------
        values: `dict` [`str`, `torch.Tensor`]
        """
        self.positions = values['g'].clone()

    def _compute_velocities(self, values):
        """
        Update the attribute ``v0``.

        Parameters
        ----------
        values: `dict` [`str`, `torch.Tensor`]
        """
        self.velocities = values['v0'].clone()

    def _compute_betas(self, values):
        """
        Update the attribute ``betas``.

        Parameters
        ----------
        values: `dict` [`str`, `torch.Tensor`]
        """
        self.betas = values['betas'].clone()

    def _compute_orthonormal_basis(self):
        """
        Compute the attribute ``orthonormal_basis`` which is a basis orthogonal to v0 for the inner product implied by
        the metric. It is equivalent to be a base orthogonal to v0 / (p0^2 (1-p0)^2 for the euclidean norm.
        """
        dgamma_t0 = self.velocities

        e1 = torch.zeros(self.dimension)
        e1[0] = 1
        alpha = torch.sign(dgamma_t0[0]) * torch.norm(dgamma_t0)
        u_vector = dgamma_t0 - alpha * e1
        v_vector = u_vector / torch.norm(u_vector)
        v_vector = v_vector.reshape(1, -1)

        q_matrix = torch.eye(self.dimension) - 2 * v_vector.permute(1, 0) * v_vector
        self.orthonormal_basis = q_matrix[:, 1:]

    @staticmethod
    def _mixing_matrix_utils(linear_combination_values, matrix):
        """
        Intermediate function used to test the good behaviour of the class' methods.

        Parameters
        ----------
        linear_combination_values: `torch.Tensor`
        matrix: `torch.Tensor`

        Returns
        -------
        `torch.Tensor`
        """
        return torch.mm(matrix, linear_combination_values)

    def _compute_mixing_matrix(self):
        """
        Update the attribute ``mixing_matrix``.
        """
        if self.source_dimension == 0:
            return
        self.mixing_matrix = self._mixing_matrix_utils(self.betas, self.orthonormal_basis)
