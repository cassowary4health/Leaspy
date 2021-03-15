import torch



class AbstractAttributes:
    """
    Contains the common attributes & methods of the different attributes classes.
    Such classes are used to update the models' attributes.

    Parameters
    ----------
    name: str
    dimension: int (default None)
    source_dimension: int (default None)

    Attributes
    ----------
    dimension: int
    source_dimension: int
    univariate: bool
    has_sources: bool
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

    def get_attributes(self):
        raise NotImplementedError('The `get_attributes` method should be implemented in each child class of AbstractAttribute')

    def update(self, names_of_changes_values, values):
        """
        Update model group average parameter(s).

        Parameters
        ----------
        names_of_changed_values: list [str]
           Values to be updated
        values: dict [str, `torch.Tensor`]
           New values used to update the model's group average parameters
        """
        raise NotImplementedError('The `update` method should be implemented in each child class of AbstractAttribute')

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
