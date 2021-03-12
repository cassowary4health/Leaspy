import torch


# TODO 2 : Add some individual attributes -> Optimization on the w_i = A * s_i
# TODO: refact, this is not abstract but multivariate & logistic oriented (cf. exp ...)
class AttributesAbstract:
    """
    Contains the common attributes & methods of the different attributes classes.
    Such classes are used to update the models' attributes.

    Parameters
    ----------
    dimension: int (default None)
    source_dimension: int (default None)

    Attributes
    ----------
    dimension: int
    source_dimension: int
    betas : :class:`torch.Tensor` (default None)
    mixing_matrix : :class:`torch.Tensor` (default None)
        Matrix A such that w_i = A * s_i.
    positions : :class:`torch.Tensor` (default None)
        Previously noted "g".
    orthonormal_basis : :class:`torch.Tensor` (default None)
    velocities : :class:`torch.Tensor` (default None)
        Previously noted "v0".
    name: str (default None)
        Name of the associated leaspy model. Used by ``update`` method.
    update_possibilities: tuple [str], (default ('all', 'g', 'v0', 'betas') )
        Contains the available parameters to update. Different models have different parameters.
    """

    def __init__(self, name, dimension=None, source_dimension=None):
        """
        Instantiate a AttributesAbstract class object.
        """
        self.name = name

        if not isinstance(dimension, int):
            raise ValueError("In AttributesAbstract you must provide integer for the parameters `dimension`.")

        self.dimension = dimension
        self.univariate = dimension == 1

        self.source_dimension = source_dimension
        self.has_sources = bool(source_dimension) # False iff None or == 0

        self.positions = None
        self.velocities = None

        if self.univariate:
            assert not self.has_sources

            self.update_possibilities = ('all', 'g', 'xi_mean')
        else:
            if not isinstance(source_dimension, int):
                raise ValueError("In AttributesAbstract you must provide integer for the parameters `source_dimension` for non univariate models.")

            self.betas = None
            self.mixing_matrix = None
            self.orthonormal_basis = None

            self.update_possibilities = ('all', 'g', 'v0', 'betas')

    def get_attributes(self):
        """
        Returns the following attributes: ``positions``, ``velocities`` & ``mixing_matrix``.

        Returns
        -------
        positions: `torch.Tensor`
        velocities: `torch.Tensor`
        mixing_matrix: `torch.Tensor`
        """
        if self.univariate:
            return self.positions
        else:
            return self.positions, self.velocities, self.mixing_matrix

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

        # TODO : Check if the condition is enough
        if self.has_sources and (compute_positions or compute_velocities):
            self._compute_orthonormal_basis()
        if self.has_sources and (compute_positions or compute_velocities or compute_betas):
            self._compute_mixing_matrix()

    def _check_names(self, names_of_changed_values):
        """
        Check if the name of the parameter(s) to update are in the possibilities allowed by the model.

        Parameters
        ----------
        names_of_changed_values: list [str]

        Raises
        -------
        ValueError
        """
        unknown_update_possibilities = set(names_of_changed_values).difference(self.update_possibilities)
        if len(unknown_update_possibilities) > 0:
            raise ValueError(f"{unknown_update_possibilities} not in the attributes that can be updated")

    def _compute_positions(self, values):
        """
        Update the attribute ``positions``.

        Parameters
        ----------
        values: dict [str, `torch.Tensor`]
        """
        if 'linear' in self.name:
            self.positions = values['g'].clone()
        elif 'logistic' in self.name:
            self.positions = torch.exp(values['g'])
        else:
            raise ValueError

    def _compute_velocities(self, values):
        """
        Update the attribute ``velocities``.

        Parameters
        ----------
        values: dict [str, `torch.Tensor`]
        """
        if self.univariate:
            self.velocities = torch.exp(values['xi_mean'])
        else:
            if 'linear' in self.name or 'logistic' in self.name:
                self.velocities = torch.exp(values['v0'])
            else:
                raise ValueError

    def _compute_betas(self, values):
        """
        Update the attribute ``betas``.

        Parameters
        ----------
        values: dict [str, `torch.Tensor`]
        """
        if not self.has_sources:
            return
        self.betas = values['betas'].clone()

    def _compute_deltas(self, values):
        raise NotImplementedError

    def _compute_orthonormal_basis(self):
        """
        Compute the attribute ``orthonormal_basis`` which is a basis orthogonal to velocities v0 for the inner product
        implied by the metric. It is equivalent to be a base orthogonal to v0 / (p0^2 (1-p0)^2 for the euclidean norm.
        """
        raise NotImplementedError

    def _compute_Q(self, dgamma_t0):
        """
        Used in non-abstract attributes classes to compute the ``orthonormal_basis`` attribute given the
        differentiation of the geodesic at initial time.

        Parameters
        ----------
        dgamma_t0: `torch.Tensor`
        """
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
        if not self.has_sources:
            return
        self.mixing_matrix = self._mixing_matrix_utils(self.betas, self.orthonormal_basis)
