import torch

from .abstract_manifold_model_attributes import AbstractAttributes


class SurvivalAttributes(AbstractAttributes):
    """
    Attributes of leaspy logistic models.

    Contains the common attributes & methods to update the logistic model's attributes.

    Parameters
    ----------
    name : str
    dimension : int
    source_dimension : int

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

    def __init__(self, name, dimension, source_dimension):

        super().__init__(name, dimension, source_dimension)

        if self.survival:
            self.update_possibilities = ('all', 'rho', 'nu', 'nu_collinear')
            self.rho = None
            self.nu = None


    def get_attributes(self):
        """
        Returns the attributes of the model.

        It is either a tuple of torch tensors or a single torch tensor if there is
        only one attribute for the model (e.g.: univariate models). For the precise
        definitions of those attributes please refer to the exact attributes class
        associated to your model.

        Returns
        -------
        For univariate models:
            positions: `torch.Tensor`

        For multivariate (but not parallel) models:
            * positions: `torch.Tensor`
            * velocities: `torch.Tensor`
            * mixing_matrix: `torch.Tensor`
        """
        if self.survival:
            return self.rho, self.nu

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
                * ``betas`` correspond to the linear combination of columns from the orthonormal basis so
                  to derive the :attr:`mixing_matrix`.
        values : dict [str, `torch.Tensor`]
            New values used to update the model's group average parameters

        Raises
        ------
        :exc:`.LeaspyModelInputError`
            If `names_of_changed_values` contains unknown parameters.
        """
        self._check_names(names_of_changed_values)

        compute_rho = False
        compute_nu = False

        if 'all' in names_of_changed_values:
            # make all possible updates
            names_of_changed_values = self.update_possibilities

        if 'rho' in names_of_changed_values:
            compute_rho = True
        if ('nu' in names_of_changed_values) or ('nu_collinear' in names_of_changed_values):
            compute_nu = True

        if compute_rho:
            self.rho = torch.exp(values['rho'])
        if compute_nu:
            self.nu = torch.exp(values['nu'])
