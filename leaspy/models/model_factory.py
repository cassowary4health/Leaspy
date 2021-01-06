from . import all_models

class ModelFactory:
    """
    Return the wanted model given its name.

    Methods
    -------
    model(name)
        Return the model object corresponding to 'name' arg
    """

    @staticmethod
    def model(name, **kwargs):
        """
        Return the model object corresponding to 'name' arg - check name type and value.

        Parameters
        ----------
        name: str
            The model's name.
        **kwargs:
            Contains model's hyper-parameters. Raise an error if the keyword is inapropriate for the given model's name.

        Returns
        -------
        model: leaspy.model.AbstractModel
            A child class object of leaspy.model.AbstractModel class object determined by 'name'.
        """
        if type(name) == str:
            name = name.lower()
        else:
            raise AttributeError("The `name` argument must be a string!")

        if name not in all_models:
            raise ValueError("The name of the model you are trying to create does not exist! "
                             f"It should be in {{{repr(tuple(all_models.keys()))[1:-1]}}}")

        # instantiate model with optional keyword arguments
        return all_models[name](name, **kwargs)
