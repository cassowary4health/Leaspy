import torch

from .attributes_abstract import AttributesAbstract


# TODO 2 : Add some individual attributes -> Optimization on the w_i = A * s_i
class AttributesLinearVari(AttributesAbstract):
    """
    Contains the common attributes & methods to update the logistic_asymp model's attributes.

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
    update_possibilities: `tuple` [`str`] (default ('all', 'g', 'v0', 'betas') )
        Contains the available parameters to update. Different models have different parameters.

    Methods
    -------
    get_attributes()
        Returns the following attributes: ``positions``, ``deltas`` & ``mixing_matrix``.
    update(names_of_changed_values, values)
        Update model group average parameter(s).
    """

    def __init__(self, name, dimension, source_dimension,source_dimension_direction):
        """
        Instantiate a AttributesLogistic class object.

        Parameters
        ----------
        dimension: `int`
        source_dimension: `int`
        """
        self.Param=None
        
        super().__init__(name, dimension, source_dimension)
        self.update_possibilities=('all', 'g', 'Param', 'betas')


 
    def _compute_orthonormal_basis(self):
        """
        Compute the attribute ``orthonormal_basis`` which is a basis orthogonal to velocities v0 for the inner product
        implied by the metric..
        """
        dgamma_t0 = self.Param
        self._compute_Q(dgamma_t0)
        


    def get_attributes(self):#à changer
        """
        Returns the following attributes: ``positions``, ``Param`` & ``mixing_matrix``.

        Returns
        -------
        - positions: `torch.Tensor`
        - velocities: `torch.Tensor`
        - mixing_matrix: `torch.Tensor`
        """
       
        return self.positions, self.Param, self.mixing_matrix

    def update(self, names_of_changed_values, values):
        """
        Update model group average parameter(s).

        Parameters
        ----------
        names_of_changed_values: `list` [`str`]
            Must be one of - "all", "g", "v0", "betas". Raise an error otherwise.
            "g" correspond to the attribute ``positions``.
            "v0" correspond to the attribute ``velocities``.
        values: `dict` [`str`, `torch.Tensor`]
            New values used to update the model's group average parameters
        """
        self._check_names(names_of_changed_values)

        compute_betas = False
        compute_positions = False
        compute_Param=False 
        

        if 'all' in names_of_changed_values:
            
            names_of_changed_values = self.update_possibilities  # make all possible updates

        if 'betas' in names_of_changed_values:
            compute_betas = True
        if 'g' in names_of_changed_values:
            compute_positions = True
        if ('Param' in names_of_changed_values) or ('xi_mean' in names_of_changed_values):
            compute_Param = True

        if compute_betas:
            self._compute_betas(values)
        if compute_positions:
            self._compute_positions(values)

        
        
        if compute_Param:
            self._compute_Param(values)

        # TODO : Check if the condition is enough
        if self.has_sources and (compute_positions or compute_Param):
            self._compute_orthonormal_basis()
        if self.has_sources and (compute_positions or compute_Param or compute_betas):
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
        self.positions=torch.exp(values['g']) #on a échantilloné suivant une loi normale
       

    def _compute_Param(self, values):
        """
        Update the attribute ``Param``.

        Parameters
        ----------
        values: `dict` [`str`, `torch.Tensor`]
        """
       
        self.Param=torch.exp(values['Param'])
        

    #overwrite les compute_positions get_attributes