import torch

from .logistic_attributes import LogisticAttributes

class LogisticOrdinalAttributes(LogisticAttributes):

    """
    Attributes of leaspy logistic models with ordinal noise model.

    Contains the common attributes & methods to update the logistic ordinal model's attributes.

    Parameters
    ----------
    name : str
    dimension : int
    source_dimension : int
    ordinal_infos : dict[str, Any]
        Dictionary containing the required informations for all the ordinal levels of each item

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
    max_level: int
        Maximum level of ordinal features
    batch_deltas : bool
        True if samplers for deltas are batched or False if deltas are sampled separately.
        If True attribute deltas is a Tensor.
        Otherwise deltas are a dictionary associating names of features to the Tensor of corresponding deltas.
    positions : :class:`torch.Tensor` [dimension] (default None)
        positions = exp(realizations['g']) such that "p0" = 1 / (1 + positions)
    velocities : :class:`torch.Tensor` [dimension] (default None)
        Always positive: exp(realizations['v0'])
    deltas : Union[:class:`torch.Tensor` [dimension, max_level], dict[str, :class:`torch.Tensor`]] (default None)
        Always positive: exp(realizations['deltas'])
    orthonormal_basis : :class:`torch.Tensor` [dimension, dimension - 1] (default None)
    betas : :class:`torch.Tensor` [dimension - 1, source_dimension] (default None)
    mixing_matrix : :class:`torch.Tensor` [dimension, source_dimension] (default None)
        Matrix A such that w_i = A * s_i.

    See Also
    --------
    :class:`~leaspy.models.univariate_model.UnivariateModel`
    :class:`~leaspy.models.multivariate_model.MultivariateModel`
    """

    def __init__(self, name, dimension, source_dimension, ordinal_infos):
        super().__init__(name, dimension, source_dimension)

        self.batched_deltas = ordinal_infos['batch_deltas']
        if self.batched_deltas:
            self.deltas = None
            self.update_possibilities = self.update_possibilities + ("deltas",)
        else:
            self.deltas = {"deltas_" + feat["name"]: None for feat in ordinal_infos["features"]}
            self.max_level = ordinal_infos["max_level"]
            for feat in ordinal_infos["features"]:
                self.update_possibilities = self.update_possibilities + ("deltas_" + feat["name"],)

    def get_deltas(self):
        '''
        Returns the deltas attributes stacked in one tensor

        Returns
        -------
        The deltas concatenated in a big tensor
        '''
        if self.batched_deltas:
            return self.deltas
        deltas = torch.zeros((len(self.deltas), self.max_level - 1))
        for i, name in enumerate(self.deltas):
            deltas[i, :self.deltas[name].shape[0]] = self.deltas[name]
        return deltas

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
                * ``deltas`` correspong to the attribute :attr: `deltas`.
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
        compute_deltas = False
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
        if any('deltas' in c for c in names_of_changed_values):
            compute_deltas = True

        if compute_betas:
            self._compute_betas(values)
        if compute_positions:
            self._compute_positions(values)
        if compute_velocities:
            self._compute_velocities(values)
        if compute_deltas:
            self._compute_deltas(values, [c for c in names_of_changed_values if 'deltas' in c])

        if self.has_sources:

            # do not recompute orthonormal basis when we know dgamma_t0 is collinear
            # to previous velocities to avoid useless computations!
            recompute_ortho_basis = compute_positions or dgamma_t0_not_collinear_to_previous

            if recompute_ortho_basis:
                self._compute_orthonormal_basis()
            if recompute_ortho_basis or compute_betas:
                self._compute_mixing_matrix()

    def _compute_deltas(self, values, names):
        """
        Updates the deltas which have been changed (those included in names)
        Parameters
        ----------
        values
        names

        Returns
        -------

        """
        if self.batched_deltas:
            self.deltas = torch.exp(values['deltas'])
        else:
            for name in names:
                self.deltas[name] = torch.exp(values[name])
