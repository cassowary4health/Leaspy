from leaspy.models import AbstractModel, all_models

from leaspy.exceptions import LeaspyModelInputError


class ModelFactory:
    """
    Return the wanted model given its name.
    """

    @staticmethod
    def model(name: str, **kwargs) -> AbstractModel:
        """
        Return the model object corresponding to 'name' arg with possible `kwargs`

        Check name type and value.

        Parameters
        ----------
        name : str
            The model's name.
        **kwargs
            Contains model's hyper-parameters. Raise an error if the keyword is inappropriate for the given model's name.

        Returns
        -------
        :class:`.AbstractModel`
            A child class object of :class:`.models.AbstractModel` class object determined by 'name'.

        Raises
        ------
        :exc:`.LeaspyModelInputError`
            if incorrect model requested.

        See Also
        --------
        :class:`~leaspy.api.Leaspy`
        """
        if isinstance(name, str):
            name = name.lower()
        else:
            raise LeaspyModelInputError("The `name` argument must be a string!")

        if name not in all_models:
            raise LeaspyModelInputError("The name of the model you are trying to create does not exist! "
                                       f"It should be in {{{repr(tuple(all_models.keys()))[1:-1]}}}")

        # instantiate model with optional keyword arguments
        return all_models[name](name, **kwargs)
