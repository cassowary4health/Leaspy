import torch



class AbstractAttributes:
    """
    Abstract base class for attributes of models.

    Contains the common attributes & methods of the different attributes classes.
    Such classes are used to update the models' attributes.

    Parameters
    ----------
    name: str
    dimension: int (default None)
    source_dimension: int (default None)
    univariate: bool
        Whether model is univariate or not (i.e. dimension == 1)
    has_sources: bool
        Whether model has sources or not (not univariate and source_dimension >= 1)

    Attributes
    ----------
    name: str
        Name of the associated leaspy model.
    dimension: int
        Number of features of the model
    source_dimension: int
        Number of sources of the model
        TODO? move to AbstractManifoldModelAttributes?
    univariate: bool
        Whether model is univariate or not (i.e. dimension == 1)
    has_sources: bool
        Whether model has sources or not (not univariate and source_dimension >= 1)
        TODO? move to AbstractManifoldModelAttributes?
    update_possibilities: tuple[str] (default empty)
        Contains the available parameters to update. Different models have different parameters.
    """

    def __init__(self, name, dimension=None, source_dimension=None):
        """
        Instantiate a AbstractAttributes class object.
        """
        self.name = name

        if not isinstance(dimension, int):
            raise ValueError("In AbstractAttributes you must provide integer for the parameters `dimension`.")

        self.dimension = dimension
        self.univariate = dimension == 1

        self.source_dimension = source_dimension
        self.has_sources = bool(source_dimension) # False iff None or == 0
        assert not (self.univariate and self.has_sources)

        self.update_possibilities = () # empty tuple

    def get_attributes(self):
        """
        Returns the essential attributes of a given model.

        Returns
        -------
        Depends on the subclass, please refer to each specific class.
        """
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

        Raises
        ------
        ValueError
            If `names_of_changed_values` contains unknown values to update.
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
