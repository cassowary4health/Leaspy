from . import UnivariateModel, MultivariateModel, MultivariateParallelModel, ConstantModel


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

        if name == 'univariate_logistic' or name == "univariate_linear":
            model = UnivariateModel(name, **kwargs)
        elif name == 'logistic' or name == 'linear' or name == 'mixed_linear-logistic':
            model = MultivariateModel(name, **kwargs)
        elif name == 'logistic_parallel':
            model = MultivariateParallelModel(name, **kwargs)
        elif name == 'constant':
            model = ConstantModel(name)
        else:
            raise ValueError("The name of the model you are trying to create does not exist! " +
                             "It should be `univariate_linear`, `univariate_logsitic, `linear`, `logistic`, `logistic_parallel` or `constant`")

        return model