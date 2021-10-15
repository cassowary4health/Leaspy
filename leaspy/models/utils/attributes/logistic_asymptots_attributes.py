import torch

from .abstract_manifold_model_attributes import AbstractManifoldModelAttributes


# TODO 2 : Add some individual attributes -> Optimization on the w_i = A * s_i
class LogisticAsymptotsAttributes(AbstractManifoldModelAttributes):
    """
    Contains the common attributes & methods to update the logistic_asymptots model's attributes.

    Attributes
    ----------
    dimension: `int`
    source_dimension: `int`
    betas: `torch.Tensor` (default None)
    positions: `torch.Tensor` (default None)
        positions = exp(realizations['g']) such that p0 = 1 / (1+exp(g))
    mixing_matrix: `torch.Tensor` (default None)
        Matrix A such that w_i = A * s_i
    orthonormal_basis: `torch.Tensor` (default None)
    velocities: `torch.Tensor` (default None)
    name: `str` (default 'logistic')
        Name of the associated leaspy model. Used by ``update`` method.
    update_possibilities: `tuple` [`str`] (default ('all', 'g', 'v0', 'asymptots', 'betas') )
        Contains the available parameters to update. Different models have different parameters.

    Methods
    -------
    get_attributes()
        Returns the following attributes: ``positions``, ``velocities``, ``asymptots`` & ``mixing_matrix``.
    update(names_of_changed_values, values)
        Update model group average parameter(s).
    """

    def __init__(self, name, dimension, source_dimension):
        """
        Instantiate a AttributesLogistic class object.

        Parameters
        ----------
        dimension: `int`
        source_dimension: `int`
        """
        super().__init__(name, dimension, source_dimension)
        self.asymptots = None
        self.update_possibilities=('all', 'g', 'velocities', 'asymptots', 'betas')

    def get_attributes(self):
        """
        Returns the following attributes: ``positions``, ``velocities``, ``asymptots`` & ``mixing_matrix``.

        Returns
        -------
        - positions: `torch.Tensor`
        - velocities: `torch.Tensor`
        - asymptots: `torch.Tensor`
        - mixing_matrix: `torch.Tensor`
        """
       
        return self.positions, self.velocities, self.asymptots, self.mixing_matrix

    def update(self, names_of_changed_values, values):
        """
        Update model group average parameter(s).

        Parameters
        ----------
        names_of_changed_values: `list` [`str`]
            Must be one of - "all", "g", "v0", "asymptots", "betas". Raise an error otherwise.
            "g" correspond to the attribute ``positions``.
            "v0" correspond to the attribute ``velocities``.
        values: `dict` [`str`, `torch.Tensor`]
            New values used to update the model's group average parameters
        """
        self._check_names(names_of_changed_values)

        compute_betas = False
        compute_positions = False
        compute_velocities = False
        compute_asymptots = False
        

        if 'all' in names_of_changed_values:
            
            names_of_changed_values = self.update_possibilities  # make all possible updates

        if 'betas' in names_of_changed_values:
            compute_betas = True
        if 'g' in names_of_changed_values:
            compute_positions = True
        if ('velocities' in names_of_changed_values) or ('xi_mean' in names_of_changed_values):
            compute_velocities = True
        if 'asymptots' in names_of_changed_values:
            compute_asymptots = True

        if compute_betas:
            self._compute_betas(values)
        if compute_positions:
            self._compute_positions(values)
        if compute_asymptots:
            self._compute_asymptots(values)
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
        names_of_changed_values: `list` [`str`]

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
        values: `dict` [`str`, `torch.Tensor`]
        """
        self.positions = torch.exp(values['g'])
       

    def _compute_asymptots(self, values):
        """
        Update the attribute ``asymptots``.

        Parameters
        ----------
        values: `dict` [`str`, `torch.Tensor`]
        """
        self.asymptots = values['asymptots'].clone()

    def _compute_orthonormal_basis(self):
        self.orthonormal_basis = torch.eye(self.dimension)