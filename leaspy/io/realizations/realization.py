import torch

class Realization:
    """
    Contains the realization of a given parameter.
    """
    def __init__(self, name, shape, variable_type):
        self.name = name
        self.shape = shape
        self.variable_type = variable_type
        self._tensor_realizations = None

    @classmethod
    def from_tensor(cls, name, shape, variable_type, tensor_realization):
        """
        Create realization from variable infos and torch tensor object
        :param name:
        :param shape:
        :param variable_type:
        :param tensor_realization:
        :return:
        """
        # TODO : a check of shapes
        realization = cls(name, shape, variable_type)
        realization._tensor_realizations = tensor_realization.clone().detach()
        return realization

    def initialize(self, n_individuals, model, scale_individual=1.0):
        # print("Initialize realizations of {0}".format(self.name))
        if self.variable_type == "population":
            self._tensor_realizations: torch.Tensor = model.parameters[self.name].reshape(self.shape) # avoid 0D / 1D tensors mix
        elif self.variable_type == 'individual':

            distribution = torch.distributions.normal.Normal(loc=model.parameters["{0}_mean".format(self.name)],
                                                             scale=scale_individual * model.parameters["{0}_std".format(
                                                                 self.name)])  # TODO change later, to have low variance when initialized
            self._tensor_realizations: torch.Tensor = distribution.sample(sample_shape=(n_individuals, *self.shape))
        else:
            raise ValueError("Variable type not known")

    @property
    def tensor_realizations(self) -> torch.Tensor:
        return self._tensor_realizations

    @tensor_realizations.setter
    def tensor_realizations(self, tensor_realizations: torch.Tensor):
        # TODO, check that it is a torch tensor (not variable for example)
        self._tensor_realizations = tensor_realizations

    def set_tensor_realizations_element(self, element, dim):
        # TODO, check that it is a torch tensor (not variable for example) when assigning
        self._tensor_realizations[dim] = element

    def __str__(self):
        str = "Realization of {0} \n".format(self.name)
        str += "Shape : {0} \n".format(self.shape)
        str += "Variable type : {0} \n".format(self.variable_type)
        return str

    def set_autograd(self):
        if not self._tensor_realizations.requires_grad:
            self._tensor_realizations.requires_grad_(True) # in-place
        else:
            raise ValueError("Realizations are already using autograd")

    def unset_autograd(self):
        if self._tensor_realizations.requires_grad:
            #self._tensor_realizations = self._tensor_realizations.detach()
            self._tensor_realizations.requires_grad_(False) # in-place (or `detach_()` )
        else:
            raise ValueError("Realizations are already detached")

    def copy(self):
        """
        PyTorch `Tensor.clone` doc:
            Unlike copy_(), this function is recorded in the computation graph.
            Gradients propagating to the cloned tensor will propagate to the original tensor.
        """
        new_realization = Realization(self.name, self.shape, self.variable_type)
        new_realization.tensor_realizations = self.tensor_realizations.clone()
        return new_realization
