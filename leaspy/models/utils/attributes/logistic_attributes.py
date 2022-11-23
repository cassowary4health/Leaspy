import torch

from .abstract_manifold_model_attributes import AbstractManifoldModelAttributes


class LogisticAttributes(AbstractManifoldModelAttributes):
    """
    Attributes of leaspy logistic models.

    Contains the common attributes & methods to update the logistic model's attributes.

    Parameters
    ----------
    name : str
    dimension : int
    source_dimension : int
    use_householder (optional, default True): bool

    Attributes
    ----------
    name : str (default 'logistic')
        Name of the associated leaspy model.
    dimension : int
    source_dimension : int
    univariate : bool
        Whether model is univariate or not (i.e. dimension == 1)
    has_sources : bool
        Whether model has sources or not (not univariate and source_dimension >= 1)
    update_possibilities : tuple [str] (default ('all', 'g', 'v0', 'betas') )
        Contains the available parameters to update. Different models have different parameters.
    positions : :class:`torch.Tensor` [dimension] (default None)
        positions = exp(realizations['g']) such that "p0" = 1 / (1 + positions)
    velocities : :class:`torch.Tensor` [dimension] (default None)
        Always positive: exp(realizations['v0'])
    orthonormal_basis : :class:`torch.Tensor` [dimension, dimension - 1] (default None)
    betas : :class:`torch.Tensor` [dimension - 1, source_dimension] (default None)
    mixing_matrix : :class:`torch.Tensor` [dimension, source_dimension] (default None)
        Matrix A such that w_i = A * s_i.

    See Also
    --------
    :class:`~leaspy.models.univariate_model.UnivariateModel`
    :class:`~leaspy.models.multivariate_model.MultivariateModel`
    """

    def __init__(self, name, dimension, source_dimension, **kwargs):

        super().__init__(name, dimension, source_dimension, **kwargs)

        if not self.univariate:
            self.velocities: torch.FloatTensor = None

    def update(self, names_of_changed_values, values):
        """
        Update model group average parameter(s).

        Parameters
        ----------
        names_of_changed_values : list [str]
            Elements of list must be either:
                * ``all`` (update everything)
                * ``g`` correspond to the attribute :attr:`positions`.
                * ``v0`` (only for multivariate models) correspond to the attribute :attr:`velocities`.
                  When we are sure that the v0 change is only a scalar multiplication
                  (in particular, when we reparametrize log(v0) <- log(v0) + mean(xi)),
                  we may update velocities using ``v0_collinear``, otherwise
                  we always assume v0 is NOT collinear to previous value
                  (no need to perform the verification it is - would not be really efficient)
                * ``betas`` correspond to the linear combination of columns from the orthonormal basis
                  to derive the :attr:`mixing_matrix`.
                * ``unprojected_directions`` correspond to directions sampled, which will be projected
                  onto the orthogonal complement of Span(v0) to derive the :attr:`mixing_matrix`.
        values : dict [str, `torch.Tensor`]
            New values used to update the model's group average parameters

        Raises
        ------
        :exc:`.LeaspyModelInputError`
            If `names_of_changed_values` contains unknown parameters.
        """
        self._check_names(names_of_changed_values)

        compute_betas = False
        compute_positions = False
        compute_velocities = False
        compute_unprojected_directions = False
        dgamma_t0_not_collinear_to_previous = False

        if 'all' in names_of_changed_values:
            # make all possible updates
            names_of_changed_values = self.update_possibilities

        if 'betas' in names_of_changed_values:
            compute_betas = True
        if 'g' in names_of_changed_values:
            compute_positions = True
        if ('v0' in names_of_changed_values) or ('v0_collinear' in names_of_changed_values):
            compute_velocities = True
            dgamma_t0_not_collinear_to_previous = 'v0' in names_of_changed_values
        if 'unprojected_directions' in names_of_changed_values:
            compute_unprojected_directions = True

        if compute_positions:
            self._compute_positions(values)
        if compute_velocities:
            self._compute_velocities(values)

        if compute_unprojected_directions:
            self._compute_unprojected_directions(values)

        # only for models with sources beyond this point
        if not self.has_sources:
            return

        if self._use_householder:
            if compute_betas:
                self._compute_betas(values)

            # do not recompute orthonormal basis when we know dgamma_t0 is collinear
            # to previous velocities to avoid useless computations!
            recompute_ortho_basis = compute_positions or dgamma_t0_not_collinear_to_previous

            if recompute_ortho_basis:
                self._compute_orthonormal_basis()
            if recompute_ortho_basis or compute_betas:
                self._compute_mixing_matrix()

        else:
            if compute_positions or compute_unprojected_directions or dgamma_t0_not_collinear_to_previous:
                # without Householder, computing the mixing matrix amounts to
                # projecting its columns on Orthogonal_g(Span(v0))
                self._compute_mixing_matrix()

    def _compute_metric(self, position: torch.FloatTensor) -> torch.FloatTensor:
        """
        Compute the metric matrix at the point defined by 1/(1+position). This is usually
        applied to the position attribute, which is updated as exp(g). This allows
        to convert g which is sampled over the real line to a value in [0,1].

        Parameters:
            position : :class:`torch.FloatTensor` 1D

        """
        G_metric = (1 + position).pow(4) / position.pow(2) # = "1/(p * (1-p))**2"
        return G_metric

    def _compute_positions(self, values):
        """
        Update the attribute ``positions``.

        Parameters
        ----------
        values : dict [str, `torch.Tensor`]
        """
        self.positions = torch.exp(values['g'])
