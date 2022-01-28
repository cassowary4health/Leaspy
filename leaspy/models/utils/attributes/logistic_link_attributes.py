import torch

from .abstract_manifold_model_link_attributes import AbstractManifoldModelLinkAttributes


class LogisticLinkAttributes(AbstractManifoldModelLinkAttributes):
    """
    Attributes of leaspy logistic models.

    Contains the common attributes & methods to update the logistic model's attributes.

    Parameters
    ----------
    name : str
    dimension : int
    source_dimension : int
    device : torch.device (defaults to torch.device("cpu")

    Attributes
    ----------
    name : str (default 'logistic')
        Name of the associated leaspy model.
    dimension : int
    source_dimension : int
    device: torch.device
        Torch device on which tensors are stored during the algorithm run
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

    def __init__(self, name, dimension, source_dimension, device=None):

        super().__init__(name, dimension, source_dimension, device)

    def update(self, names_of_changed_values, values):
        """
        Update model group average parameter(s).

        Parameters
        ----------
        names_of_changed_values : list [str]
            Elements of list must be either:
                * ``all`` (update everything)
                * ``g`` correspond to the attribute :attr:`positions`.
                * ``v0`` (``xi_mean`` if univariate) correspond to the attribute :attr:`velocities`.
                * ``betas`` correspond to the linear combinaison of columns from the orthonormal basis so
                  to derive the :attr:`mixing_matrix`.
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
        compute_link_v0 = False
        compute_link_g = False
        compute_link_t_mean = False

        if 'all' in names_of_changed_values:
            names_of_changed_values = self.update_possibilities  # make all possible updates
        if 'betas' in names_of_changed_values:
            compute_betas = True
        if 'link_v0' in names_of_changed_values:
            compute_link_v0 = True
        if 'link_g' in names_of_changed_values:
            compute_link_g = True
        if 'link_t_mean' in names_of_changed_values:
            compute_link_t_mean = True

        if compute_betas:
            self._compute_betas(values)
        if compute_link_v0:
            self._compute_link_v0(values)
        if compute_link_g:
            self._compute_link_g(values)
        if compute_link_t_mean:
            self._compute_link_t_mean(values)

        if self.has_sources:
            return
            recompute_ortho_basis = compute_positions or compute_velocities

            if recompute_ortho_basis:
                self._compute_orthonormal_basis()
            if recompute_ortho_basis or compute_betas:
                self._compute_mixing_matrix()

    def _compute_positions(self, values):
        """
        Update the attribute ``positions``.

        Parameters
        ----------
        values : dict [str, `torch.Tensor`]
        """
        self.positions = torch.exp(values['g'])

    def _compute_link_v0(self, values):
        """
        Update the attribute ``link``.

        Parameters
        ----------
        values : dict [str, `torch.Tensor`]
        """        
        self.link_v0 = values['link_v0'].clone()

    def _compute_link_g(self, values):
        """
        Update the attribute ``link``.

        Parameters
        ----------
        values : dict [str, `torch.Tensor`]
        """        
        self.link_g = values['link_g'].clone()

    def _compute_link_t_mean(self, values):
        """
        Update the attribute ``link``.

        Parameters
        ----------
        values : dict [str, `torch.Tensor`]
        """        
        self.link_t_mean = values['link_t_mean'].clone()

    def _compute_orthonormal_basis(self):
        """
        Compute the attribute ``orthonormal_basis`` which is an orthonormal basis, w.r.t the canonical inner product,
        of the sub-space orthogonal, w.r.t the inner product implied by the metric, to the time-derivative of the geodesic at initial time.
        """
        if not self.has_sources:
            return

        # Compute the diagonal of metric matrix (cf. `_compute_Q`)
        G_metric = (1 + self.positions).pow(4) / self.positions.pow(2) # = "1/(p0 * (1-p0))**2"

        dgamma_t0 = self.velocities

        # Householder decomposition in non-Euclidean case, updates `orthonormal_basis` in-place
        self._compute_Q(dgamma_t0, G_metric)
