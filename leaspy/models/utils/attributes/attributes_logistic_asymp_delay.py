import torch

from .attributes_abstract import AttributesAbstract


# TODO 2 : Add some individual attributes -> Optimization on the w_i = A * s_i
class AttributesLogisticAsympDelay(AttributesAbstract):
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

    def __init__(self, name, dimension, source_dimension):
        """
        Instantiate a AttributesLogistic class object.

        Parameters
        ----------
        dimension: `int`
        source_dimension: `int`
        """
        
        self.Param=None
        self.Asymp=None
        
        self.betas_asymp =None
        
        self.orthonormal_basis_delay =None
        super().__init__(name, dimension, source_dimension)
        self.update_possibilities=('all', 'g', 'Param', 'betas','Asymp','betas_asymp')
        

    def _compute_orthonormal_basis(self):
        """
        Compute the attribute ``orthonormal_basis`` which is a free family orthogonal to Param and is composed of vectors orthogonal
        to Param on all Plan induce by infection_k,guerison_k. 
        
        return a matrix with dim(Param) lines and self.dimension columns (number of lines of beta)
    
        """
        infection=self.Param
        ej = torch.zeros(self.dimension, dtype=torch.float32)
        ej[0] = 1.

        alpha = -torch.sign(infection[0]) * torch.norm(infection)
        u_vector = infection - alpha * ej
        v_vector = u_vector / torch.norm(u_vector)

        ## Classical Householder method (to get an orthonormal basis for the canonical inner product)
        ## Q = I_n - 2 v • v'
        q_matrix = torch.eye(self.dimension) - 2 * v_vector.view(-1,1) * v_vector

        # first component of basis is a unit vector (for metric norm) collinear to `dgamma_t0`
        #self.orthonormal_basis = q_matrix[:, 1:]

        # concat columns (get rid of the one collinear to `dgamma_t0`)
        self.orthonormal_basis_delay =q_matrix[:, 1:]
        
        

    def _mixing_matrix_utils(self,linear_combination_values1, matrix1):
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
        

        #on range les deux bases les unes à la suite des autres
        return torch.mm(matrix1, linear_combination_values1)

    def _compute_mixing_matrix(self):
        """
        Update the attribute ``mixing_matrix``. de dimension 3n x dim source
        """
        if not self.has_sources:
            return
        
        self.mixing_matrix = self._mixing_matrix_utils(self.betas, self.orthonormal_basis_delay)
    

    def get_attributes(self):#à changer
        """
        Returns the following attributes: ``positions``, ``velocities`` & ``mixing_matrix``.

        Returns
        -------
        - positions: `torch.Tensor`
        - velocities: `torch.Tensor`
        - mixing_matrix: `torch.Tensor`
        """
       
        return self.positions, self.Param,self.Asymp, self.mixing_matrix,self.betas_asymp#We can give betas_asymp directly because moving the asymptots is orthogonal movement by definition

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
        compute_Asymp=False 
        compute_betas_asymp=False
        

        if 'all' in names_of_changed_values:
            
            names_of_changed_values = self.update_possibilities  # make all possible updates

        if 'betas' in names_of_changed_values:
            compute_betas = True
        if 'betas_asymp' in names_of_changed_values:
            compute_betas_asymp = True
        if 'Asymp' in names_of_changed_values:
            compute_Asymp = True
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
        if compute_Asymp:
            self._compute_Asymp(values)
        if compute_betas_asymp:
            self._compute_betas_asymp(values)

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
    def _compute_Asymp(self, values):
        """
        Update the attribute ``positions``.

        Parameters
        ----------
        values: `dict` [`str`, `torch.Tensor`]
        """
        self.Asymp=torch.exp(values['Asymp']) #on a échantilloné suivant une loi normale
       
    def _compute_betas_asymp(self, values):
        """
        Update the attribute ``betas``.

        Parameters
        ----------
        values: `dict` [`str`, `torch.Tensor`]
        """
        
        self.betas_asymp = values['betas_asymp'].clone()
    def _compute_Param(self, values):
        """
        Update the attribute ``Param``.

        Parameters
        ----------
        values: `dict` [`str`, `torch.Tensor`]
        """
        
        self.Param=torch.exp(values['Param'])
        

    #overwrite les compute_positions get_attributes