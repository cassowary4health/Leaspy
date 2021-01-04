import torch

from .attributes_abstract import AttributesAbstract


# TODO 2 : Add some individual attributes -> Optimization on the w_i = A * s_i
class AttributesUnivariate(AttributesAbstract):
    """
    Contains the common attributes & methods to update the univariate model's attributes.

    Attributes
    ----------
    positions: `torch.Tensor` (default None)
        Previously noted "g".
    velocities: `torch.Tensor` (default None)
        Previously noted "v0".
    name: `str` (default 'univariate')
        Name of the associated leaspy model. Used by ``update`` method.
    update_possibilities: `tuple` [`str`] (default ('g', 'xi_mean', 'all') )
        Contains the available parameters to update. Different models have different parameters.

    Methods
    -------
    get_attributes()
        Returns the following attributes: ``positions``.
    update(names_of_changed_values, values)
        Update model group average parameter(s).
    """

    def __init__(self):
        """
        Instantiate a AttributesUnivariate class object.
        """
        super().__init__()
        self.update_possibilities = ('all', 'g', 'xi_mean')
        self.name = 'univariate'

        delattr(self, 'dimension')
        delattr(self, 'source_dimension')
        delattr(self, 'betas')
        delattr(self, 'mixing_matrix')
        delattr(self, 'orthonormal_basis')

    def get_attributes(self):
        """
        Returns the following attributes: ``positions``.

        Returns
        -------
        positions: `torch.Tensor`
        """
        return self.positions

    def _compute_velocities(self, values):
        """
        Update the attribute ``velocities``.

        Parameters
        ----------
        values: `dict` [`str`, `torch.Tensor`]
        """
        self.velocities = torch.exp(values['xi_mean'])
