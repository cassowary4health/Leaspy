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
        For univariate models:
            positions: `torch.Tensor`

        For not univariate models:
            * positions: `torch.Tensor`
            * velocities: `torch.Tensor`
            * mixing_matrix: `torch.Tensor`
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

        if self.has_sources:
            # TODO more generic: add a method `should_recompute_ortho_basis(names_of_changed_values) -> bool` in sub-classes?
            if 'linear' in self.name:
                # Euclidean inner prod so only velocities count (not positions unlike logist)
                recompute_ortho_basis = compute_velocities
            else:
                # add deltas for logistic parallel
                recompute_ortho_basis = compute_positions or compute_velocities or compute_deltas

            if recompute_ortho_basis:
                self._compute_orthonormal_basis()
            if recompute_ortho_basis or compute_betas:
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
        Compute the attribute ``orthonormal_basis`` which is an orthonormal basis, w.r.t the canonical inner product,
        of the sub-space orthogonal, w.r.t the inner product implied by the metric, to the time-derivative of the geodesic at initial time.
        """
        raise NotImplementedError

    def _compute_Q(self, dgamma_t0, G_metric, strip_col=0):
        """
        Householder decomposition, adapted for a non-Euclidean inner product defined by:
        (1) :math:`< x, y >Metric(p) = < x, G(p) y >Eucl = xT G(p) y`, where:
        :math:`G(p)` is the symmetric positive-definite (SPD) matrix defining the metric at point `p`.

        The Euclidean case is the special case where `G` is the identity matrix.
        Product-metric is a special case where `G(p)` is a diagonal matrix (identified to a vector) whose components are all > 0.

        It is used in child classes to compute and set in-place the ``orthonormal_basis`` attribute
        given the time-derivative of the geodesic at initial time and the `G_metric`.
        The first component of the full orthonormal basis is a vector collinear `G_metric x dgamma_t0` that we get rid of.

        The orthonormal basis we construct is always orthonormal for the Euclidean canonical inner product.
        But all (but first) vectors of it lie in the sub-space orthogonal (for canonical inner product) to `G_metric * dgamma_t0`
        which is the same thing that being orthogonal to `dgamma_t0` for the inner product implied by the metric.

        [We could do otherwise if we'd like a full orthonormal basis, w.r.t. the non-Euclidean inner product.
         But it'd imply to compute G^(-1/2) & G^(1/2) which may be computationally costly in case we don't have direct access to them
         (for the special case of product-metric it is easy - just the component-wise inverse (sqrt'ed) of diagonal)
         TODO are there any advantages/drawbacks of one method over the other except this one?
              are there any biases between features when only considering Euclidean orthonormal basis?]

        Parameters
        ----------
        dgamma_t0: `torch.Tensor` 1D
            Time-derivative of the geodesic at initial time

        G_metric: scalar, `torch.Tensor` 0D, 1D or 2D-square
            The `G(p)` defining the metric as refered in equation (1) just before.
            If 0D / scalar: `G` is proportional to the identity matrix
            If 1D (vector): `G` is a diagonal matrix (diagonal components > 0)
            If 2D (square matrix): `G` is general (SPD)

        strip_col: int in 0..model_dimension-1 (default 0)
            Which column of the basis should be the one collinear to `dgamma_t0` (that we get rid of)
        """

        # enforce `G_metric` to be a tensor
        if not isinstance(G_metric, torch.Tensor):
            G_metric = torch.tensor(G_metric) # convert from scalar...

        # compute the vector that others columns should be orthogonal to, w.r.t canonical inner product
        G_shape = G_metric.shape
        if len(G_shape) == 0: # 0D
            assert G_metric.item() > 0
            dgamma_t0 = G_metric.item() * dgamma_t0 # homothetic
        elif len(G_shape) == 1: # 1D
            assert (G_metric > 0).all()
            assert len(G_metric) == self.dimension
            dgamma_t0 = G_metric * dgamma_t0 # component-wise multiplication of vectors
        elif len(G_shape) == 2: # matrix (general case)
            assert G_shape == (self.dimension, self.dimension)
            dgamma_t0 = G_metric @ dgamma_t0 # matrix multiplication
        else:
            raise NotImplementedError

        """
        Automatically choose the best column to strip?
        <!> Not a good idea because it could fluctuate over iterations making mixing_matrix unstable!
            (betas should slowly readapt to the permutation...)
        #strip_col = dgamma_t0.abs().argmax().item()
        #strip_col = v_metric_normalization.argmin().item()
        """

        assert 0 <= strip_col < self.dimension
        ej = torch.zeros(self.dimension, dtype=torch.float32)
        ej[strip_col] = 1.

        alpha = -torch.sign(dgamma_t0[strip_col]) * torch.norm(dgamma_t0)
        u_vector = dgamma_t0 - alpha * ej
        v_vector = u_vector / torch.norm(u_vector)

        ## Classical Householder method (to get an orthonormal basis for the canonical inner product)
        ## Q = I_n - 2 v • v'
        q_matrix = torch.eye(self.dimension) - 2 * v_vector.view(-1,1) * v_vector

        # first component of basis is a unit vector (for metric norm) collinear to `dgamma_t0`
        #self.orthonormal_basis = q_matrix[:, 1:]

        # concat columns (get rid of the one collinear to `dgamma_t0`)
        self.orthonormal_basis = torch.cat((
            q_matrix[:, :strip_col],
            q_matrix[:, strip_col+1:]
        ), dim=1)

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
